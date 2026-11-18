"""Prove the pipeline actually runs distributed: one stage, then two stages.

    .venv/bin/python tools/check_pipeline.py

Builds a tiny Qwen2 checkpoint plus a small word-level tokenizer, starts the real
``deploy/stage_runner.py`` as separate processes, and submits a prompt over its
control socket. The single-process answer and the two-process answer must be the
same tokens; anything else means the stage boundary changed the computation.

This uses sockets, a real process boundary and a real serialised activation, so
it exercises the transport rather than calling the maths in-process.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
TINY = {
    "hidden_size": 64,
    "num_hidden_layers": 4,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 16,
    "intermediate_size": 128,
    "vocab_size": 32,
    "rms_norm_eps": 1e-6,
    "rope_theta": 1_000_000.0,
    "tie_word_embeddings": False,
    "attention_bias": True,
    "max_position_embeddings": 64,
}
WORDS = ["[UNK]", "[PAD]", "the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog"]


def build_model(dest: Path) -> None:
    from transformers import Qwen2Config, Qwen2ForCausalLM

    config = Qwen2Config(**TINY)
    import torch

    torch.manual_seed(7)
    model = Qwen2ForCausalLM(config).float().eval()
    dest.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(dest, safe_serialization=True)

    vocab = {word: index for index, word in enumerate(WORDS)}
    # Built with the library rather than hand-written JSON: the serialised form
    # is versioned and a hand-rolled file fails to load.
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace

    tokenizer = Tokenizer(WordLevel(vocab=vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.save(str(dest / "tokenizer.json"))


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def wait_for_socket(path: Path, timeout: float = 120.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.exists():
            return
        time.sleep(0.2)
    raise TimeoutError(f"control socket {path} never appeared")


def ask(control: Path, request: dict, timeout: float = 120.0) -> dict:
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
        s.settimeout(timeout)
        s.connect(str(control))
        s.sendall((json.dumps(request) + "\n").encode())
        buffer = b""
        while b"\n" not in buffer:
            chunk = s.recv(65536)
            if not chunk:
                raise RuntimeError("control socket closed with no reply")
            buffer += chunk
        return json.loads(buffer.split(b"\n")[0])


def start(args: list[str], log_path: Path) -> subprocess.Popen:
    handle = log_path.open("w")
    return subprocess.Popen(
        [sys.executable, str(REPO / "deploy" / "stage_runner.py"), *args],
        stdout=handle,
        stderr=subprocess.STDOUT,
        env={**os.environ, "PYTHONPATH": str(REPO)},
    )


def stop(proc: subprocess.Popen) -> None:
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()


def main() -> int:
    prompt = "the quick brown fox"
    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw)
        model_dir = root / "model"
        build_model(model_dir)
        print(f"tiny checkpoint: {model_dir}")

        request = {"prompt": prompt, "max_new_tokens": 6, "temperature": 0.0}

        # ---- one process, whole model --------------------------------
        single_ctl = root / "single.sock"
        single = start(
            [
                "--model-dir", str(model_dir),
                "--rank", "0", "--world", "1",
                "--start", "0", "--end", "4",
                "--device", "cpu", "--dtype", "float32",
                "--control", str(single_ctl),
            ],
            root / "single.log",
        )
        try:
            wait_for_socket(single_ctl)
            one = ask(single_ctl, request)
        finally:
            stop(single)
        if not one.get("ok"):
            print("single failed:", one)
            return 1
        print(f"single  process: {one['result']['text']!r}"
              f"  ({one['result']['output_tokens']} tokens, "
              f"{one['result']['ms_per_token']} ms/token)")

        # ---- two processes, four layers split 2 + 2 -----------------
        # Stage 0 listens for the token return; stage 1 listens for the forward
        # chain. They cannot share a port: each dials the other's listener.
        rank0_port = free_port()
        rank1_port = free_port()
        ctl = root / "split.sock"
        stage0 = start(
            [
                "--model-dir", str(model_dir),
                "--rank", "0", "--world", "2",
                "--start", "0", "--end", "2",
                "--listen", f"0.0.0.0:{rank0_port}",
                "--next", f"127.0.0.1:{rank1_port}",
                "--device", "cpu", "--dtype", "float32",
                "--control", str(ctl),
            ],
            root / "stage0.log",
        )
        stage1 = start(
            [
                "--model-dir", str(model_dir),
                "--rank", "1", "--world", "2",
                "--start", "2", "--end", "4",
                "--listen", f"0.0.0.0:{rank1_port}",
                "--reply", f"127.0.0.1:{rank0_port}",
                "--device", "cpu", "--dtype", "float32",
            ],
            root / "stage1.log",
        )
        try:
            wait_for_socket(ctl)
            two = ask(ctl, request)
        finally:
            stop(stage0)
            stop(stage1)

        for name in ("stage0", "stage1"):
            log = (root / f"{name}.log").read_text()
            for line in log.splitlines():
                print(f"  {name}: {line}")

        if not two.get("ok"):
            print("split failed:", two)
            return 1
        print(f"two stages     : {two['result']['text']!r}"
              f"  ({two['result']['output_tokens']} tokens, "
              f"{two['result']['ms_per_token']} ms/token)")

    same = one["result"]["text"] == two["result"]["text"]
    same_tokens = one["result"]["output_tokens"] == two["result"]["output_tokens"]
    print()
    print(f"identical output: {same and same_tokens}")
    return 0 if same and same_tokens else 1


if __name__ == "__main__":
    raise SystemExit(main())
