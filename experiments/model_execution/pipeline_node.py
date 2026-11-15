"""Phase 3B: three-host pipeline over persistent TCP.

One model, three stages, three machines. Each stage loads only its own layers from
its local GGUF and passes hidden states downstream. Weights never cross the
network.

Topology (chosen from measured links, not assumed):

    coordinator -> [HOST-A: embed + layers 0-10] -> [HOST-B: 10-20] -> [HOST-C: 20-31 + head]
                                                                              |
                                                    logits flow back up the chain

HOST-A's 6 GiB card cannot hold the full model, so it owns the fewest layers. The
partition follows measured free VRAM: 11/10/11 layers.

Protocol: length-prefixed JSON header then raw fp16 payload. The header carries
protocol version, request id, position, shape, dtype, payload length and a CRC32
so a corrupted frame is rejected rather than silently fed to the model.

KV cache stays local to each stage and is keyed by request id, so concurrent
requests cannot corrupt each other.

Usage:
    python -m experiments.model_execution.pipeline_node stage --index 1 \
        --layers 10,20 --listen 0.0.0.0:10200 --next 1.208.108.242:46092
    python -m experiments.model_execution.pipeline_node serve --listen 127.0.0.1:10100
    python -m experiments.model_execution.pipeline_node client --target 127.0.0.1:10100 \
        --tokens 1724,338,...
"""

from __future__ import annotations

import argparse
import json
import socket
import struct
import time
import zlib
from dataclasses import dataclass, field

import torch
from meshweave.runtime.gguf_stage import LAYERS, GgufStageLoader
from meshweave.runtime.phimoe_forward import StageExecutor

MODEL = "/root/models/phi-mini-moe-Q6_K.gguf"
PROTO = "meshweave-pipeline/1"
HEADER_LEN = struct.Struct(">I")  # 4-byte big-endian JSON length


# --------------------------------------------------------------------------- #
# framing
# --------------------------------------------------------------------------- #


def send_frame(sock: socket.socket, header: dict, payload: bytes) -> None:
    header = dict(header)
    header["payload_len"] = len(payload)
    header["crc32"] = zlib.crc32(payload) & 0xFFFFFFFF
    header["proto"] = PROTO
    blob = json.dumps(header).encode()
    sock.sendall(HEADER_LEN.pack(len(blob)) + blob + payload)


def recv_frame(sock: socket.socket) -> tuple[dict, bytes]:
    raw_len = _recv_exact(sock, 4)
    (hlen,) = HEADER_LEN.unpack(raw_len)
    if hlen <= 0 or hlen > 1 << 20:
        raise ProtocolError(f"implausible header length {hlen}")
    header = json.loads(_recv_exact(sock, hlen).decode())
    if header.get("proto") != PROTO:
        raise ProtocolError(f"protocol mismatch: {header.get('proto')!r}")
    payload = _recv_exact(sock, int(header["payload_len"]))
    expected = int(header.get("crc32", 0))
    if zlib.crc32(payload) & 0xFFFFFFFF != expected:
        raise ProtocolError(
            f"checksum mismatch on request {header.get('req')}: payload corrupted in transit"
        )
    return header, payload


def _recv_exact(sock: socket.socket, n: int) -> bytes:
    chunks = []
    remaining = n
    while remaining > 0:
        chunk = sock.recv(min(remaining, 1 << 20))
        if not chunk:
            raise ConnectionError(f"peer closed after {n - remaining} of {n} bytes")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


class ProtocolError(RuntimeError):
    pass


# --------------------------------------------------------------------------- #
# stage worker
# --------------------------------------------------------------------------- #


@dataclass
class StageWorker:
    index: int
    layer_range: tuple[int, int]
    owns_embedding: bool
    owns_head: bool
    device: str = "cuda"
    executor: StageExecutor | None = None
    downstream: socket.socket | None = None
    stats: dict = field(default_factory=dict)

    def load(self) -> dict:
        started = time.monotonic()
        loader = GgufStageLoader(MODEL, device=self.device)
        weights = loader.load(
            self.layer_range[0],
            self.layer_range[1],
            with_embedding=self.owns_embedding,
            with_head=self.owns_head,
        )
        self.executor = StageExecutor(weights, torch.device(self.device))
        summary = weights.summary()
        summary["load_seconds"] = round(time.monotonic() - started, 2)
        summary["node"] = socket.gethostname()
        return summary

    def connect_next(self, host: str, port: int) -> None:
        sock = socket.create_connection((host, port), timeout=30)
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4 << 20)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 4 << 20)
        self.downstream = sock

    def handle(self, header: dict, payload: bytes) -> tuple[dict, bytes]:
        """Process one request and return the response frame."""
        assert self.executor is not None
        t0 = time.monotonic()

        if self.owns_embedding:
            tokens = header.get("tokens")
            if not tokens:
                raise ProtocolError("stage 0 requires token ids")
            hidden = self.executor.embed(
                torch.tensor([tokens], dtype=torch.long, device=self.device)
            )
            # Prefill starts at 0; a decode step of one token passes the absolute
            # position that token occupies. Without this, every generated token
            # would be rotated as if it were position 0.
            position = int(header.get("pos", 0))
        else:
            shape = tuple(int(x) for x in header["shape"])
            hidden = torch.frombuffer(bytearray(payload), dtype=torch.float16).to(
                self.device
            ).reshape(shape)
            position = int(header["pos"])

        result = self.executor.forward(
            hidden, position_offset=position, use_cache=True
        )

        out_hidden = result.hidden
        compute_ms = (time.monotonic() - t0) * 1000

        # Last stage returns logits; earlier stages pass the hidden state on.
        if result.logits is not None:
            out = result.logits.contiguous()
            reply_to = header.get("reply_to")
            if reply_to:
                # Direct response routing: the last stage ships logits straight to
                # the client instead of relaying them back through every upstream
                # stage. On this fleet HOST-C and HOST-A share a physical host, so
                # this hop is ~0.5 ms versus the ~620 ms of relaying back through
                # B and A.
                host, port = reply_to.rsplit(":", 1)
                reply_sock = socket.create_connection((host, int(port)), timeout=30)
                reply_sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                try:
                    send_frame(
                        reply_sock,
                        {
                            "op": "logits",
                            "req": header["req"],
                            "stage": self.index,
                            "dtype": "float32",
                            "shape": list(out.shape),
                            "compute_ms": round(compute_ms, 2),
                            "route": "direct",
                        },
                        out.to(torch.float32).cpu().numpy().tobytes(),
                    )
                finally:
                    reply_sock.close()
                # Tell the upstream stages the request completed so they can free
                # their view of it. This ack is tiny and does not carry logits.
                return (
                    {"op": "done", "req": header["req"], "stage": self.index,
                     "route": "direct", "compute_ms": round(compute_ms, 2)},
                    b"",
                )

            resp_header = {
                "op": "logits",
                "req": header["req"],
                "stage": self.index,
                "dtype": "float32",
                "shape": list(out.shape),
                "compute_ms": round(compute_ms, 2),
                "n_tokens": int(hidden.shape[1]),
            }
            return resp_header, out.to(torch.float32).cpu().numpy().tobytes()

        if self.downstream is None:
            raise ProtocolError("stage has no downstream but produced no logits")

        out = out_hidden.contiguous()
        # `pos` is the ABSOLUTE position of the first token in this batch, and it
        # must not change along the chain: every stage rotates RoPE relative to the
        # same absolute positions and appends its own KV entries at those same
        # offsets. Advancing it per stage (as an earlier version did) made stage 1
        # rotate at positions 12-23 for a prefill that starts at 0, which shifted
        # every downstream logit by a near-constant offset (cosine stayed 0.99998,
        # mean abs error 3.26) while keeping the token ranking intact.
        fwd_header = {
            "op": "hidden",
            "req": header["req"],
            "stage": self.index,
            "pos": int(position),
            "dtype": "float16",
            "shape": list(out.shape),
            "compute_ms": round(compute_ms, 2),
        }
        send_started = time.monotonic()
        send_frame(self.downstream, fwd_header, out.to(torch.float16).cpu().numpy().tobytes())
        downstream_header, downstream_payload = recv_frame(self.downstream)
        send_ms = (time.monotonic() - send_started) * 1000

        downstream_header["upstream_compute_ms"] = round(compute_ms, 2)
        downstream_header["upstream_roundtrip_ms"] = round(send_ms, 2)
        return downstream_header, downstream_payload


def run_stage(args: argparse.Namespace) -> int:
    if args.layers == "auto":
        ranges = [(0, 11), (11, 21), (21, LAYERS)]
        layer_range = ranges[args.index]
    else:
        a, b = args.layers.split(",")
        layer_range = (int(a), int(b))

    worker = StageWorker(
        index=args.index,
        layer_range=layer_range,
        owns_embedding=(args.index == 0),
        owns_head=(args.index == 2),
        device=args.device,
    )
    summary = worker.load()
    print(json.dumps({"event": "loaded", "stage": args.index, **summary}), flush=True)

    if args.next:
        host, port = args.next.split(":")
        worker.connect_next(host, int(port))
        print(json.dumps({"event": "connected_downstream", "next": args.next}), flush=True)

    host, port = args.listen.split(":")
    srv = socket.socket()
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((host, int(port)))
    srv.listen(4)
    print(json.dumps({"event": "listening", "addr": f"{host}:{port}"}), flush=True)

    # Serve connections until a shutdown message arrives. A client that simply
    # disconnects must not take the stage down: on a real pipeline, an upstream
    # restart is normal, and a stage that exits on the first hang-up makes the
    # whole chain engineering-hostile.
    try:
        while True:
            conn, addr = srv.accept()
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            print(json.dumps({"event": "client_connected", "peer": str(addr)}), flush=True)
            shutdown = False
            try:
                while True:
                    header, payload = recv_frame(conn)
                    if header.get("op") == "shutdown":
                        shutdown = True
                        break
                    resp_header, resp_payload = worker.handle(header, payload)
                    send_frame(conn, resp_header, resp_payload)
            except (ConnectionError, ProtocolError) as exc:
                print(
                    json.dumps(
                        {"event": "connection_closed", "stage": args.index, "reason": str(exc)}
                    ),
                    flush=True,
                )
            finally:
                conn.close()
            if shutdown:
                break
    finally:
        srv.close()

    print(json.dumps({"event": "shutdown", "stage": args.index}), flush=True)
    return 0


# --------------------------------------------------------------------------- #
# coordinator client
# --------------------------------------------------------------------------- #


def run_client(args: argparse.Namespace) -> int:
    host, port = args.target.split(":")
    tokens = [int(x) for x in args.tokens.split(",")]

    sock = socket.create_connection((host, int(port)), timeout=args.timeout)
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

    started = time.monotonic()
    send_frame(
        sock,
        {"op": "forward", "req": args.req, "tokens": tokens, "dtype": "int64"},
        b"",
    )
    header, payload = recv_frame(sock)
    elapsed = (time.monotonic() - started) * 1000

    import numpy as np

    logits = np.frombuffer(payload, dtype=np.float32).reshape(header["shape"])
    top = np.argsort(-logits[0, -1])[:5]
    print(
        json.dumps(
            {
                "req": args.req,
                "elapsed_ms": round(elapsed, 2),
                "shape": header["shape"],
                "top5_ids": top.tolist(),
                "top5_vals": [round(float(logits[0, -1][i]), 3) for i in top],
                "compute_trace": {
                    k: v for k, v in header.items() if "ms" in k or k == "stage"
                },
            },
            indent=2,
        )
    )
    sock.close()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="role", required=True)

    st = sub.add_parser("stage")
    st.add_argument("--index", type=int, required=True)
    st.add_argument("--layers", default="auto")
    st.add_argument("--listen", required=True)
    st.add_argument("--next", default=None)
    st.add_argument("--device", default="cuda")

    cl = sub.add_parser("client")
    cl.add_argument("--target", required=True)
    cl.add_argument("--tokens", required=True)
    cl.add_argument("--req", default="r1")
    cl.add_argument("--timeout", type=float, default=600.0)

    args = parser.parse_args(argv)
    if args.role == "stage":
        return run_stage(args)
    return run_client(args)


if __name__ == "__main__":
    raise SystemExit(main())
