"""Raw TCP echo endpoint used by the network benchmark.

This is the only non-HTTP listener an agent opens. It echoes bytes and nothing
else: no parsing, no execution, no state. The benchmark measures throughput and
stability by sending known payloads and timing the drain.

The listener authenticates nothing because it can do nothing -- an attacker who
reaches it can only make the host echo bytes it already sent.
"""

from __future__ import annotations

import asyncio
import contextlib
import signal
from dataclasses import dataclass


@dataclass
class EchoServer:
    host: str
    port: int

    async def handle(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        writer.get_extra_info("peername")
        try:
            while True:
                chunk = await reader.read(1024 * 1024)
                if not chunk:
                    break
                writer.write(chunk)
                await writer.drain()
        except (ConnectionError, asyncio.CancelledError):
            pass
        finally:
            with contextlib.suppress(Exception):
                writer.close()
                await writer.wait_closed()

    async def serve_forever(self) -> None:
        server = await asyncio.start_server(self.handle, self.host, self.port)
        addrs = ", ".join(str(sock.getsockname()) for sock in server.sockets or [])
        print(f"[meshweave.echo] listening on {addrs}", flush=True)
        async with server:
            await server.serve_forever()


async def _run(host: str, port: int) -> None:
    server = EchoServer(host=host, port=port)
    stop = asyncio.Event()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        with contextlib.suppress(NotImplementedError):
            loop.add_signal_handler(sig, stop.set)

    serve_task = asyncio.create_task(server.serve_forever())
    await stop.wait()
    serve_task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await serve_task


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="MESHWEAVE benchmark echo endpoint")
    parser.add_argument("--host", default="0.0.0.0")  # noqa: S104
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()
    asyncio.run(_run(args.host, args.port))


if __name__ == "__main__":  # pragma: no cover
    main()
