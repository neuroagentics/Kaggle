"""Offline-only Transformers transport for Kaggle model attachments."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Mapping


class OfflineModelError(RuntimeError):
    """Raised when the attached neural model cannot be loaded or queried."""


class OfflineTransformersTransport:
    """Expose a local Hugging Face causal model through the solver transport API.

    The class deliberately has no download fallback. Both tokenizer and weights
    must already exist below the attached Kaggle model path.
    """

    def __init__(
        self,
        model_path: str | Path,
        *,
        device: str = "auto",
        max_input_tokens: int = 24_000,
    ) -> None:
        path = Path(model_path).resolve()
        if not path.is_dir():
            raise OfflineModelError(f"Offline model directory does not exist: {path}")
        if not (path / "config.json").is_file():
            raise OfflineModelError(f"Offline model config.json is missing: {path}")
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer
        except ImportError as exc:
            raise OfflineModelError(
                "torch and transformers must be present in the Kaggle image"
            ) from exc

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            raise OfflineModelError("CUDA was requested but is unavailable")
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        try:
            try:
                self.processor = AutoProcessor.from_pretrained(
                    str(path),
                    local_files_only=True,
                    trust_remote_code=False,
                )
                self.tokenizer = getattr(self.processor, "tokenizer", self.processor)
            except Exception:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    str(path),
                    local_files_only=True,
                    trust_remote_code=False,
                )
                self.processor = self.tokenizer
            self.model = AutoModelForCausalLM.from_pretrained(
                str(path),
                local_files_only=True,
                trust_remote_code=False,
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
                attn_implementation="sdpa",
            )
            self.model.to(device)
            self.model.eval()
        except Exception as exc:
            raise OfflineModelError(f"Unable to load attached model at {path}: {exc}") from exc
        self.torch = torch
        self.path = path
        self.device = device
        self.max_input_tokens = max_input_tokens

    def __call__(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        messages = payload.get("messages")
        if not isinstance(messages, list) or not messages:
            raise OfflineModelError("Model request requires chat messages")
        messages = [dict(message) for message in messages]
        schema = payload.get("format")
        if isinstance(schema, Mapping):
            # Transformers does not implement Ollama's `format` option. Include
            # the contract explicitly; parsing + execution remain authoritative.
            # This is prompt guidance, NOT constrained/grammar decoding.
            messages[-1]["content"] += (
                "\nReturn only JSON conforming to this schema:\n" + json.dumps(schema)
            )
        options = payload.get("options", {})
        if not isinstance(options, Mapping):
            options = {}
        seed = int(options.get("seed", 0))
        temperature = float(options.get("temperature", 0.0))
        max_new_tokens = min(max(int(options.get("num_predict", 1024)), 32), 4096)
        max_time = float(options.get("max_time", 120.0))
        if max_time <= 0:
            raise OfflineModelError("Inference deadline exhausted")
        started = time.monotonic()
        self.torch.manual_seed(seed)
        if self.device == "cuda":
            self.torch.cuda.manual_seed_all(seed)

        try:
            rendered = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=bool(payload.get("think", False)),
            )
            encoded = self.tokenizer(
                rendered,
                return_tensors="pt",
                truncation=False,
                add_special_tokens=False,
            )
            encoded = {key: value.to(self.device) for key, value in encoded.items()}
            if encoded["input_ids"].shape[-1] > self.max_input_tokens:
                raise OfflineModelError("Prompt exceeds input budget; refusing to truncate demonstrations")
            generation = {
                "max_new_tokens": max_new_tokens,
                "do_sample": temperature > 0,
                "pad_token_id": self.tokenizer.eos_token_id,
                "max_time": max(0.001, max_time - (time.monotonic() - started)),
            }
            if temperature > 0:
                generation.update({"temperature": temperature, "top_p": 0.92})
            with self.torch.inference_mode():
                output = self.model.generate(**encoded, **generation)
            if time.monotonic() - started > max_time:
                raise OfflineModelError("Inference deadline exceeded")
            prompt_length = encoded["input_ids"].shape[-1]
            content = self.processor.decode(
                output[0, prompt_length:], skip_special_tokens=True
            ).strip()
        except Exception as exc:
            raise OfflineModelError(f"Offline inference failed: {exc}") from exc
        if not content:
            raise OfflineModelError("Offline model returned empty content")
        return {
            "message": {"role": "assistant", "content": content},
            "done": True,
            "offline_model_path": str(self.path),
            "request_schema": json.dumps(payload.get("format", {}), sort_keys=True),
        }
