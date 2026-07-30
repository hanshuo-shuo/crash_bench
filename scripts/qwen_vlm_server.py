#!/usr/bin/env python
"""Minimal node-local HTTP server for Qwen3-VL collision-monitor inference.

Run this in the modern transformers environment on one GPU; the OpenVLA evaluator
runs in its legacy environment on a second GPU and sends base64 JPEG requests over
localhost.  The server is deliberately tiny and exposes only ``/health`` and
``/predict``.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
from http.server import BaseHTTPRequestHandler, HTTPServer

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-VL-32B-Instruct")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=18765)
    ap.add_argument("--max-new-tokens", type=int, default=8)
    args = ap.parse_args()

    print(f"loading {args.model}", flush=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        device_map="cuda:0",
        low_cpu_mem_usage=True,
        local_files_only=True,
        attn_implementation="sdpa",
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
    resolved = getattr(getattr(model, "config", None), "_commit_hash", None)
    model_label = f"{args.model}@{resolved}" if resolved else args.model
    print(f"ready {model_label} on {model.device}", flush=True)

    class Handler(BaseHTTPRequestHandler):
        def _json(self, status: int, payload: dict) -> None:
            body = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):  # noqa: N802 - BaseHTTPRequestHandler API
            if self.path == "/health":
                self._json(200, {"ready": True, "model": model_label})
            else:
                self._json(404, {"error": "not found"})

        def do_POST(self):  # noqa: N802 - BaseHTTPRequestHandler API
            if self.path != "/predict":
                self._json(404, {"error": "not found"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                request = json.loads(self.rfile.read(length).decode())
                image = Image.open(io.BytesIO(base64.b64decode(request["image_b64"]))).convert("RGB")
                messages = [{
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": str(request["prompt"])},
                    ],
                }]
                inputs = processor.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_dict=True,
                    return_tensors="pt",
                )
                inputs = inputs.to(model.device)
                with torch.inference_mode():
                    generated = model.generate(
                        **inputs,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=False,
                        use_cache=True,
                    )
                trimmed = [out[len(src):] for src, out in zip(inputs.input_ids, generated)]
                text = processor.batch_decode(
                    trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )[0].strip()
                print(json.dumps({"text": text}), flush=True)
                self._json(200, {"text": text, "model": model_label})
            except Exception as exc:
                print(f"request error: {type(exc).__name__}: {exc}", flush=True)
                self._json(500, {"error": f"{type(exc).__name__}: {exc}"})

        def log_message(self, fmt, *values):
            print(f"http: {fmt % values}", flush=True)

    HTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
