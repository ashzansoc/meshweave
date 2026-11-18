#!/usr/bin/env python3
"""Run one stage of a sharded model on this machine.

This process is started by the MESHWEAVE agent from an assignment, never by a
person typing a command into a node. It holds only the layers it was given.

Topology, for world=N:

    stage 0 --dial--> stage 1 --dial--> ... --> stage N-1
       ^                                          |
       +--------------- token return -------------+

Only rank 0 keeps the token loop and the tokenizer. The last stage samples the
next token from its own logits and returns the id directly to rank 0, which is
what removes a full round trip per token instead of relaying it back along the
chain.

Rank 0 exposes a JSON-lines control socket bound to 127.0.0.1 so the agent on the
same machine can submit prompts. Nothing outside the host can reach it, and the
process accepts no shell input of any kind.

Usage is deliberately narrow:

    stage_runner.py --repo Qwen/Qwen2.5-0.5B-Instruct --start 0 --end 12 \
        --rank 0 --world 2 --listen 0.0.0.0:10100 --next 1.2.3.4:10100
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import socket
import sys
import threading
import time
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
# In the repo the script lives in deploy/ and the package is one level up; on a
# node both sit together in /opt/meshweave.
if (_HERE.parent / "meshweave").is_dir():
    sys.path.insert(0, str(_HERE.parent))

from meshweave.runtime.dense import (  # noqa: E402
    DenseStage,
    load_stage,
    sample_token,
)
from meshweave.runtime.framing import FramingError, recv_frame, send_frame  # noqa: E402
from meshweave.runtime.model_config import DenseConfig  # noqa: E402

DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def log(message: str) -> None:
    print(message, flush=True)


def parse_host_port(value: str) -> tuple[str, int]:
    host, _, port = value.rpartition(":")
    if not host or not port.isdigit():
        raise SystemExit(f"expected host:port, got {value!r}")
    return host, int(port)


# --------------------------------------------------------------------------- #
# model
# --------------------------------------------------------------------------- #


def ensure_model(repo: str, cache_dir: Path) -> Path:
    """Download config, tokenizer and safetensors shards into the local cache."""
    from huggingface_hub import snapshot_download

    log(f"fetching {repo}")
    path = snapshot_download(
        repo_id=repo,
        cache_dir=str(cache_dir),
        allow_patterns=[
            "*.json",
            "*.safetensors",
            "*.txt",
            "*.model",
            "*.tiktoken",
        ],
    )
    log(f"model files at {path}")
    return Path(path)


def build_stage(
    args: argparse.Namespace, config: DenseConfig, model_dir: Path
) -> tuple[DenseStage, int]:
    dtype_name = args.dtype
    if (
        dtype_name == "bfloat16"
        and args.device == "cuda"
        and torch.cuda.is_available()
        and not torch.cuda.is_bf16_supported()
    ):
        # Turing and older have no bf16 kernels. The planner should have picked
        # fp16 already; this is the last line of defence so the stage starts.
        dtype_name = "float16"
        log(f"bfloat16 is not supported on this GPU; using {dtype_name}")
    dtype = DTYPES[dtype_name]
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda was requested but CUDA is not available")

    device = torch.device(args.device)
    owns_embed = args.rank == 0
    owns_head = args.rank == args.world - 1

    stage_weights = load_stage(
        model_dir,
        config,
        args.start,
        args.end,
        owns_embed=owns_embed,
        owns_head=owns_head,
        device=device,
        dtype=dtype,
    )
    stage = DenseStage(stage_weights, device, capacity=args.max_seq, dtype=dtype)

    resident = stage_weights.bytes
    log(
        f"rank {args.rank}/{args.world}: layers [{args.start},{args.end}) "
        f"on {args.device} in {dtype_name}, {resident / 1024**2:.1f} MiB resident"
    )
    if device.type == "cuda":
        allocated = torch.cuda.memory_allocated(device)
        log(f"rank {args.rank}: cuda memory allocated {allocated / 1024**2:.1f} MiB")

    return stage, resident


# --------------------------------------------------------------------------- #
# transport
# --------------------------------------------------------------------------- #


def dial(
    host: str, port: int, role: str, *, token: str | None = None, timeout: float = 600.0
) -> socket.socket:
    """Connect to a peer, retrying while it is still starting up.

    A socket is created per attempt. Reusing one across retries is a trap: after
    a refused connection the descriptor is unusable, so every later attempt fails
    on the same socket and the stage appears to hang even once the peer is up.

    The default waits ten minutes, because a peer may still be downloading the
    model: a 1 GiB checkpoint on a slow link takes minutes, and giving up after
    three meant the first stage failed the deployment while the second was still
    fetching.
    """
    deadline = time.monotonic() + timeout
    delay = 0.5
    while True:
        shell = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        shell.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        try:
            shell.connect((host, port))
            send_frame(shell, {"kind": "hello", "role": role, "token": token})
            log(f"connected to {host}:{port} as {role}")
            return shell
        except OSError:
            shell.close()
            if time.monotonic() >= deadline:
                raise
            time.sleep(delay)
            delay = min(delay * 1.5, 5.0)


def listen(host: str, port: int) -> socket.socket:
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((host, port))
    server.listen(8)
    log(f"listening on {host}:{port}")
    return server


def accept_as(
    server: socket.socket, role: str, *, token: str | None = None, timeout: float = 180.0
) -> socket.socket:
    """Accept until a peer declares the role we are waiting for.

    When a stage token is configured, a peer that does not present it is refused.
    The stage port is published to the internet, so this is what stops a stranger
    from feeding activations into a running pipeline.
    """
    server.settimeout(timeout)
    while True:
        conn, addr = server.accept()
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        header, _ = recv_frame(conn)
        if header.get("kind") != "hello":
            conn.close()
            continue
        if header.get("role") != role:
            conn.close()
            continue
        if token and header.get("token") != token:
            log(f"refused {role} from {addr[0]}:{addr[1]}: bad stage token")
            conn.close()
            continue
        log(f"accepted {role} from {addr[0]}:{addr[1]}")
        return conn


# --------------------------------------------------------------------------- #
# serving a stage
# --------------------------------------------------------------------------- #


class ChainStage:
    """A non-head stage: transform hidden states and pass them along."""

    def __init__(self, stage: DenseStage, downstream: socket.socket) -> None:
        self.stage = stage
        self.downstream = downstream
        self.upstream: socket.socket | None = None

    def bind_upstream(self, sock: socket.socket) -> None:
        self.upstream = sock

    def loop(self) -> None:
        try:
            while True:
                header, payload = recv_frame(self._incoming())
                kind = header.get("kind")
                if kind == "forward":
                    # The payload arrives on the host; this stage's weights are
                    # on the GPU, so move it before the first matmul.
                    hidden = _decode(payload, header, self.stage.device)
                    result = self.stage.forward(hidden, position_offset=int(header["pos"]))
                    self._send_forward(result.hidden, int(header["pos"]), header["req"], header)
                elif kind == "reset":
                    self.stage.reset()
                    send_frame(self.downstream, {"kind": "reset"})
                elif kind == "shutdown":
                    send_frame(self.downstream, {"kind": "shutdown"})
                    return
                else:
                    raise FramingError(f"unexpected frame {kind!r}")
        except (FramingError, OSError) as exc:
            # A peer going away is a normal end for a stage, not a crash. The
            # agent reports the process exit; logging the reason is enough here.
            log(f"upstream closed: {exc}")

    def _incoming(self) -> socket.socket:
        if self.upstream is None:
            raise RuntimeError("no upstream connection bound")
        return self.upstream

    def _send_forward(
        self, hidden: torch.Tensor, pos: int, req: str, header: dict | None = None
    ) -> None:
        payload = _to_wire(hidden)
        outgoing = {
            "kind": "forward",
            "req": req,
            "pos": pos,
            "shape": list(hidden.shape),
            "dtype": "float16",
        }
        # Carry the sampler settings through. Rebuilding the header without them
        # reset temperature and top_p to their defaults at every middle stage, so
        # sampling survived only a two-stage deployment.
        if header:
            for key in ("temperature", "top_p"):
                if key in header:
                    outgoing[key] = header[key]
        send_frame(self.downstream, outgoing, payload)


class LastStage(ChainStage):
    """The final stage: sample the next token and return it straight to rank 0."""

    def __init__(self, stage: DenseStage, downstream: socket.socket | None, reply: socket.socket):
        super().__init__(stage, downstream)  # type: ignore[arg-type]
        self.reply = reply

    def loop(self) -> None:
        try:
            while True:
                header, payload = recv_frame(self._incoming())
                kind = header.get("kind")
                if kind == "forward":
                    hidden = _decode(payload, header, self.stage.device)
                    result = self.stage.forward(hidden, position_offset=int(header["pos"]))
                    if result.logits is None:
                        raise FramingError("last stage produced no logits")
                    token = sample_token(
                        result.logits,
                        temperature=float(header.get("temperature", 0.0)),
                        top_p=float(header.get("top_p", 1.0)),
                    )
                    send_frame(
                        self.reply,
                        {"kind": "token", "req": header["req"], "token": token},
                    )
                elif kind == "reset":
                    self.stage.reset()
                elif kind == "shutdown":
                    return
                else:
                    raise FramingError(f"unexpected frame {kind!r}")
        except (FramingError, OSError) as exc:
            log(f"upstream closed: {exc}")


def _decode(payload: bytes, header: dict, device: torch.device) -> torch.Tensor:
    """Rebuild an activation and put it on the device that owns the weights.

    The move belongs here rather than at each call site, because forgetting it
    fails only on a GPU: a CPU stage happens to work either way.
    """
    import numpy as np

    shape = tuple(int(v) for v in header["shape"])
    array = np.frombuffer(payload, dtype=np.float16).reshape(shape)
    return torch.from_numpy(array.copy()).to(device)


def _to_wire(tensor: torch.Tensor) -> bytes:
    """Serialise an activation for the next stage.

    Detach, move to host, narrow to fp16. numpy cannot read a CUDA tensor, so the
    .cpu() is not optional -- it raised on the first real two-node request.
    """
    return tensor.detach().cpu().contiguous().to(torch.float16).numpy().tobytes()


# --------------------------------------------------------------------------- #
# generation (rank 0)
# --------------------------------------------------------------------------- #


class RankZero:
    def __init__(
        self,
        stage: DenseStage,
        tokenizer,
        downstream: socket.socket | None,
        *,
        world: int = 1,
        rank: int = 0,
    ):
        self.stage = stage
        self.tokenizer = tokenizer
        self.downstream = downstream
        self.world = world
        self.rank = rank
        self.returns: queue.Queue[dict] = queue.Queue()
        self.lock = threading.Lock()
        self._counter = 0

    def start_return_reader(self, sock: socket.socket) -> None:
        def read() -> None:
            while True:
                try:
                    header, _ = recv_frame(sock)
                except (FramingError, OSError):
                    self.returns.put({"kind": "closed"})
                    return
                self.returns.put(header)

        threading.Thread(target=read, daemon=True).start()

    # ---------------------------------------------------------------- generate

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 32,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ) -> dict:
        with self.lock:
            return self._generate(prompt, max_new_tokens, temperature, top_p)

    def _generate(
        self, prompt: str, max_new_tokens: int, temperature: float, top_p: float
    ) -> dict:
        self._counter += 1
        req = f"r{self._counter}"
        started = time.perf_counter()

        encoded = self.tokenizer.encode(prompt)
        input_ids = list(encoded.ids)
        if not input_ids:
            raise RuntimeError("tokenizer produced no tokens for this prompt")

        self.stage.reset()
        if self.downstream is not None:
            send_frame(self.downstream, {"kind": "reset"})

        eos = self.tokenizer.token_to_id("<|im_end|>") or self.tokenizer.token_to_id("<|endoftext|>")
        generated: list[int] = []
        ttft_ms: float | None = None
        step_times: list[float] = []
        round_trips: list[float] = []
        bytes_sent = 0

        hidden = self.stage.embed(torch.tensor([input_ids], dtype=torch.long, device=self.stage.device))
        position = 0
        prefill_start = time.perf_counter()
        hop_start = time.perf_counter()
        result = self._forward(hidden, position, req, temperature, top_p)
        prefill_ms = (time.perf_counter() - prefill_start) * 1000.0
        round_trips.append((time.perf_counter() - hop_start) * 1000.0)
        bytes_sent += hidden.numel() * 2  # fp16 on the wire
        position += len(input_ids)
        step_start = time.perf_counter()

        for _ in range(max_new_tokens):
            token = result
            if ttft_ms is None:
                ttft_ms = (time.perf_counter() - started) * 1000.0
            generated.append(token)
            if eos is not None and token == eos:
                break
            if len(generated) >= max_new_tokens:
                break

            token_hidden = self.stage.embed(
                torch.tensor([[token]], dtype=torch.long, device=self.stage.device)
            )
            hop_start = time.perf_counter()
            result = self._forward(token_hidden, position, req, temperature, top_p)
            round_trips.append((time.perf_counter() - hop_start) * 1000.0)
            bytes_sent += token_hidden.numel() * 2
            position += 1
            now = time.perf_counter()
            step_times.append((now - step_start) * 1000.0)
            step_start = now

        total_s = time.perf_counter() - started
        text = self.tokenizer.decode(generated)
        decode_round_trips = round_trips[1:] if len(round_trips) > 1 else round_trips
        return {
            "prompt": prompt,
            "text": text,
            "prompt_tokens": len(input_ids),
            "output_tokens": len(generated),
            "ttft_ms": round(ttft_ms, 1) if ttft_ms is not None else None,
            "ms_per_token": round(sum(step_times) / len(step_times), 1) if step_times else None,
            "total_s": round(total_s, 3),
            "tokens_per_second": round(len(generated) / total_s, 3) if total_s > 0 else None,
            # Evidence that a distributed step really is distributed: the round
            # trip covers every stage plus both network hops, and bytes_sent is
            # the activation payload this stage actually put on the wire.
            "stage_roundtrip_ms": (
                round(sum(decode_round_trips) / len(decode_round_trips), 1)
                if decode_round_trips
                else None
            ),
            "activation_bytes_sent": bytes_sent,
            "activation_bytes_per_token": (
                round(bytes_sent / max(1, len(generated)), 1) if generated else None
            ),
            "world": self.world,
            "rank": self.rank,
            # Prefill and decode are different workloads; reporting one combined
            # rate hides which of them a change actually helped.
            "prefill_ms": round(prefill_ms, 2),
            "step_ms": [round(v, 2) for v in step_times],
        }

    def _forward(
        self, hidden: torch.Tensor, position: int, req: str, temperature: float, top_p: float
    ) -> int:
        # Rank 0 owns the embedding and its own layers, so it computes first and
        # only the transformed hidden state travels.
        result = self.stage.forward(hidden, position_offset=position)

        if self.downstream is None:
            if result.logits is None:
                raise RuntimeError("single-stage run has no head")
            return sample_token(result.logits, temperature=temperature, top_p=top_p)

        payload = _to_wire(result.hidden)
        send_frame(
            self.downstream,
            {
                "kind": "forward",
                "req": req,
                "pos": position,
                "shape": list(result.hidden.shape),
                "dtype": "float16",
                "temperature": temperature,
                "top_p": top_p,
            },
            payload,
        )
        message = self.returns.get()
        if message.get("kind") != "token":
            raise RuntimeError(f"pipeline returned {message.get('kind')!r}")
        return int(message["token"])


# --------------------------------------------------------------------------- #
# control socket
# --------------------------------------------------------------------------- #


def serve_control(sock_path: Path, engine: RankZero, on_result) -> None:
    """Accept JSON-lines requests on a local unix socket."""
    sock_path.parent.mkdir(parents=True, exist_ok=True)
    if sock_path.exists():
        sock_path.unlink()
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(str(sock_path))
    os.chmod(sock_path, 0o600)
    server.listen(4)
    log(f"control socket at {sock_path}")

    while True:
        conn, _ = server.accept()
        threading.Thread(
            target=_handle_control, args=(conn, engine, on_result), daemon=True
        ).start()


def _handle_control(conn: socket.socket, engine: RankZero, on_result) -> None:
    with conn:
        buffer = b""
        while True:
            chunk = conn.recv(65536)
            if not chunk:
                return
            buffer += chunk
            while b"\n" in buffer:
                line, buffer = buffer.split(b"\n", 1)
                if not line.strip():
                    continue
                try:
                    request = json.loads(line)
                    if request.get("op") == "shutdown":
                        conn.sendall((json.dumps({"ok": True}) + "\n").encode())
                        os._exit(0)
                    result = engine.generate(
                        str(request["prompt"]),
                        max_new_tokens=int(request.get("max_new_tokens", 32)),
                        temperature=float(request.get("temperature", 0.0)),
                        top_p=float(request.get("top_p", 1.0)),
                    )
                    payload = {"ok": True, "result": result}
                except Exception as exc:  # noqa: BLE001 - reported to the caller
                    payload = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
                conn.sendall((json.dumps(payload) + "\n").encode())
                if on_result is not None:
                    on_result(payload)


# --------------------------------------------------------------------------- #
# entry point
# --------------------------------------------------------------------------- #


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="MESHWEAVE pipeline stage")
    ap.add_argument("--repo", default=None, help="Hugging Face repo id")
    ap.add_argument(
        "--model-dir",
        default=None,
        help="local snapshot directory; skips the download and needs no --repo",
    )
    ap.add_argument("--cache", default="/opt/meshweave/models")
    ap.add_argument("--rank", type=int, required=True)
    ap.add_argument("--world", type=int, required=True)
    ap.add_argument("--start", type=int, required=True, help="first layer, inclusive")
    ap.add_argument("--end", type=int, required=True, help="last layer, exclusive")
    ap.add_argument("--listen", default=None, help="host:port this stage binds")
    ap.add_argument("--next", dest="next_addr", default=None, help="host:port of rank+1")
    ap.add_argument("--reply", default=None, help="host:port of rank 0")
    ap.add_argument("--control", default="/run/meshweave/control.sock")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16", choices=sorted(DTYPES))
    ap.add_argument("--max-seq", type=int, default=2048)
    ap.add_argument(
        "--dial-timeout",
        type=float,
        default=600.0,
        help="seconds to wait for the next stage to come up",
    )
    ap.add_argument(
        "--stage-token",
        default=None,
        help="shared secret each stage must present in its hello frame",
    )
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.world < 1:
        raise SystemExit("--world must be >= 1")
    if not 0 <= args.rank < args.world:
        raise SystemExit("--rank must be within [0, world)")

    if not args.repo and not args.model_dir:
        raise SystemExit("pass --repo or --model-dir")
    model_dir = (
        Path(args.model_dir).expanduser().resolve()
        if args.model_dir
        else ensure_model(args.repo, Path(args.cache))
    )
    config = DenseConfig.from_file(model_dir / "config.json")
    stage, _ = build_stage(args, config, model_dir)

    tokenizer = None
    if args.rank == 0:
        from tokenizers import Tokenizer

        tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))

    if args.world == 1:
        engine = RankZero(stage, tokenizer, None, world=args.world, rank=args.rank)
        # The agent infers a stage's phase from this line. Without a "ready"
        # marker a single-stage process looked like it was still loading forever.
        log("rank 0 ready: single stage runnable")
        serve_control(Path(args.control), engine, None)
        return 0

    server = listen(*parse_host_port(args.listen))

    if args.rank == args.world - 1:
        reply = dial(*parse_host_port(args.reply), role="return", token=args.stage_token, timeout=args.dial_timeout)
        chain = LastStage(stage, None, reply)
        chain.bind_upstream(accept_as(server, "forward", token=args.stage_token))
        threading.Thread(target=chain.loop, daemon=True).start()
        log("last stage ready: sampling tokens and returning them to rank 0")
        threading.Event().wait()
        return 0

    if args.rank == 0:
        downstream = dial(*parse_host_port(args.next_addr), role="forward", token=args.stage_token, timeout=args.dial_timeout)
        engine = RankZero(stage, tokenizer, downstream, world=args.world, rank=args.rank)
        engine.start_return_reader(accept_as(server, "return", token=args.stage_token))
        log("rank 0 ready: runnable")
        serve_control(Path(args.control), engine, None)
        return 0

    # A middle stage: it accepts from the stage before it and dials the next one.
    downstream = dial(*parse_host_port(args.next_addr), role="forward", token=args.stage_token, timeout=args.dial_timeout)
    chain = ChainStage(stage, downstream)
    chain.bind_upstream(accept_as(server, "forward", token=args.stage_token))
    threading.Thread(target=chain.loop, daemon=True).start()
    log(f"stage {args.rank} ready")
    threading.Event().wait()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
