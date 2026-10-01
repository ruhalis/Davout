"""HRM-Text-1B backend: one forward pass per prompt, logits over answer letters.

The prompt is wrapped as `<|im_start|><|object_ref_start|>{instruction}<|im_end|>`
(`<|object_ref_start|>` is the model's `direct` condition) and the answer is the
first response token, so the logits are read at the `<|im_end|>` position.
"""
from __future__ import annotations

import os
import re
import threading
from typing import Any, Sequence

from .base import LETTERS, Prompt, PromptTooLongError, Readout

# Must be set before torch is imported; torch and transformers are imported lazily below.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

MODEL_ID = "sapientinc/HRM-Text-1B"
IM_START = "<|im_start|>"
IM_END = "<|im_end|>"
DIRECT = "<|object_ref_start|>"  # the `direct` condition token

_ZWSP = "\u200b"  # zero-width space
_ELLIPSIS = "… "
_TRUNCATION_MARGINS = (4, 16, 64)  # tokens of slack per attempt when re-encoding a cut state
_MIN_FILL = 0.75  # a row joins a batch only if it fills this share of the padded width
_DTYPES = {
    "bf16": "bfloat16",
    "bfloat16": "bfloat16",
    "fp16": "float16",
    "float16": "float16",
    "fp32": "float32",
    "float32": "float32",
}


def download(model_id: str = MODEL_ID) -> str:
    """Download the model snapshot (about 2.4 GB) into the Hugging Face cache; return its path."""
    from huggingface_hub import snapshot_download

    return snapshot_download(model_id)


def _pick_device(torch: Any, device: str) -> Any:
    if device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _pick_dtype(torch: Any, dtype: Any, device: Any) -> Any:
    if dtype is None:
        return torch.float32 if device.type == "cpu" else torch.bfloat16
    if isinstance(dtype, str):
        if dtype not in _DTYPES:
            raise ValueError(f"unknown dtype {dtype!r}; use one of {sorted(_DTYPES)}")
        return getattr(torch, _DTYPES[dtype])
    return dtype


class HrmBackend:
    """`LabelBackend` over HRM-Text-1B (or any `hrm_text` model)."""

    name = "hrm-text-1b"
    max_labels = len(LETTERS)

    def __init__(
        self,
        model_id_or_path: str = MODEL_ID,
        device: str = "auto",
        dtype: Any = None,
        max_tokens: int = 4096,
        batch_size: int = 8,
        prefix_lm: bool = True,
        local_files_only: bool = True,
        model: Any = None,
        tokenizer: Any = None,
        max_batch_tokens: int = 8192,
    ) -> None:
        """Load the model, or wrap an injected `model`/`tokenizer` as they are.

        `max_batch_tokens` caps rows x padded length per forward pass, so long
        prompts run in smaller batches (attention memory grows with length squared).
        """
        import torch

        if batch_size < 1 or max_batch_tokens < 1:
            raise ValueError("batch_size and max_batch_tokens must be >= 1")
        self._torch = torch
        self.model_id = model_id_or_path
        if tokenizer is None or model is None:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            try:
                if tokenizer is None:
                    tokenizer = AutoTokenizer.from_pretrained(
                        model_id_or_path, local_files_only=local_files_only
                    )
                if model is None:
                    self.device = _pick_device(torch, device)
                    model = AutoModelForCausalLM.from_pretrained(
                        model_id_or_path,
                        dtype=_pick_dtype(torch, dtype, self.device),
                        local_files_only=local_files_only,
                    ).to(self.device)
            except OSError as e:
                raise RuntimeError(
                    f"could not load {model_id_or_path!r} (local_files_only={local_files_only}); "
                    "download it first with davout.backends.hrm.download()"
                ) from e
        param = next(model.parameters())
        self.device, self.dtype = param.device, param.dtype
        self._model = model.eval()
        self._tok = tokenizer
        self._config = model.config
        self.max_tokens = min(max_tokens, getattr(self._config, "max_position_embeddings", max_tokens))
        self.batch_size = batch_size
        self.max_batch_tokens = max_batch_tokens
        self.prefix_lm = prefix_lm
        self._lock = threading.Lock()

        self._pad_id = tokenizer.pad_token_id
        if self._pad_id is None:
            self._pad_id = self._config.pad_token_id
        if self._pad_id is None:
            raise RuntimeError("tokenizer and model config define no pad token")
        self.letter_ids = self._resolve_letters()
        self._check_wrapper()
        specials = set(tokenizer.get_added_vocab()) | set(tokenizer.all_special_tokens)
        self._special_re = re.compile(
            "|".join(re.escape(s) for s in sorted(specials, key=len, reverse=True))
        )
        head = self._model.lm_head
        ids = torch.tensor(self.letter_ids, device=head.weight.device)
        # The head is applied in fp32 to the last position and the letter rows only.
        self._head_w = head.weight.detach()[ids].float()
        self._head_b = None if head.bias is None else head.bias.detach()[ids].float()

    def _resolve_letters(self) -> list[int]:
        """Token id of each letter A..Z, written with no leading space."""
        ids = []
        for ch in LETTERS:
            enc = self._tok.encode(ch, add_special_tokens=False)
            if len(enc) != 1:
                raise RuntimeError(f"letter {ch!r} is not a single token: {enc}")
            ids.append(enc[0])
        if len(set(ids)) != len(ids):
            raise RuntimeError("answer letters do not map to distinct tokens")
        return ids

    def _check_wrapper(self) -> None:
        want = [self._tok.convert_tokens_to_ids(t) for t in (IM_START, DIRECT, IM_END)]
        got = self._tok.encode(IM_START + DIRECT + IM_END, add_special_tokens=False)
        if got != want or None in want or len(set(want)) != 3:
            raise RuntimeError(f"prompt wrapper does not encode to three control tokens: {got}")

    def info(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "model_id": self.model_id,
            "device": str(self.device),
            "dtype": str(self.dtype).replace("torch.", ""),
            "max_tokens": self.max_tokens,
            "batch_size": self.batch_size,
            "prefix_lm": self.prefix_lm,
            "h_cycles": self._config.H_cycles,
            "l_cycles": self._config.L_cycles,
        }

    def _neutralise(self, text: str) -> str:
        """Break control-token strings in user text so they stay plain text."""

        def split(m: re.Match[str]) -> str:
            s = m.group(0)
            cut = 2 if s.startswith("<|") else 1
            return s[:cut] + _ZWSP + s[cut:]

        while self._special_re.search(text):
            text = self._special_re.sub(split, text)
        return text

    @staticmethod
    def _wrap(body: str) -> str:
        return f"{IM_START}{DIRECT}{body}{IM_END}"

    def _ids(self, body: str) -> list[int]:
        return self._tok.encode(self._wrap(body), add_special_tokens=False)

    def _truncate(self, prefix: str, state: str, suffix: str) -> list[int]:
        """Cut `state` from the left (token level) until the prompt fits `max_tokens`."""
        budget = self.max_tokens - len(self._ids(prefix + suffix))
        if budget < 0:
            raise PromptTooLongError(
                f"prompt without the state needs {self.max_tokens - budget} tokens, "
                f"more than max_tokens={self.max_tokens}"
            )
        state_ids = self._tok.encode(state, add_special_tokens=False)
        for margin in _TRUNCATION_MARGINS:
            keep = max(0, budget - margin)
            tail = self._tok.decode(state_ids[len(state_ids) - keep :]) if keep else ""
            # A cut inside a multi-byte character decodes to U+FFFD; drop it.
            tail = self._neutralise(tail.lstrip("\ufffd").lstrip())
            ids = self._ids(prefix + _ELLIPSIS + tail + suffix)
            if len(ids) <= self.max_tokens:
                return ids
            if not keep:
                break
        raise PromptTooLongError(
            f"prompt without the state needs {self.max_tokens - budget} tokens, "
            f"leaving no room for the state within max_tokens={self.max_tokens}"
        )

    def _encode(self, prompts: Sequence[Prompt]) -> list[tuple[list[int], bool]]:
        parts = [tuple(self._neutralise(s) for s in (p.prefix, p.state, p.suffix)) for p in prompts]
        rows = self._tok([self._wrap("".join(p)) for p in parts], add_special_tokens=False)["input_ids"]
        out = []
        for ids, (prefix, state, suffix) in zip(rows, parts):
            if len(ids) <= self.max_tokens:
                out.append((list(ids), False))
            else:
                out.append((self._truncate(prefix, state, suffix), True))
        return out

    def _batches(self, order: list[int], lengths: list[int]) -> list[list[int]]:
        """Split `order` (longest first) into batches bounded by rows, padded tokens and padding waste."""
        batches: list[list[int]] = []
        for i in order:
            cur = batches[-1] if batches else None
            if (
                cur
                and len(cur) < self.batch_size
                and (len(cur) + 1) * lengths[cur[0]] <= self.max_batch_tokens
                and lengths[i] >= _MIN_FILL * lengths[cur[0]]
            ):
                cur.append(i)
            else:
                batches.append([i])
        return batches

    def _forward(self, rows: list[list[int]]) -> Any:
        """Letter logits at the last position after each H cycle: fp32 cpu tensor [cycles, rows, 26]."""
        torch = self._torch
        width = max(len(r) for r in rows)
        ids = torch.full((len(rows), width), self._pad_id, dtype=torch.long)
        mask = torch.zeros((len(rows), width), dtype=torch.long)
        for i, r in enumerate(rows):  # left padding: the answer position is last in every row
            ids[i, width - len(r) :] = torch.tensor(r, dtype=torch.long)
            mask[i, width - len(r) :] = 1
        kwargs = {"input_ids": ids.to(self.device), "attention_mask": mask.to(self.device), "use_cache": False}
        if self.prefix_lm:
            # 1 = bidirectional prefix; without it the model silently runs causal attention.
            kwargs["token_type_ids"] = kwargs["attention_mask"]

        states: list[Any] = []

        def grab(_module: Any, _inputs: Any, output: Any) -> None:
            hidden = output[0] if isinstance(output, tuple) else output
            states.append(hidden[:, -1])

        with self._lock, torch.inference_mode():
            hook = self._model.model.H_module.register_forward_hook(grab)
            try:
                self._model.model(**kwargs)
            finally:
                hook.remove()
            if len(states) != self._config.H_cycles:
                raise RuntimeError(f"expected {self._config.H_cycles} H cycles, saw {len(states)}")
            hidden = torch.stack(states).float().to(self._head_w.device)
            logits = torch.nn.functional.linear(hidden, self._head_w, self._head_b)
        return logits.cpu()

    def read(self, prompts: Sequence[Prompt]) -> list[Readout]:
        """Letter logits for each prompt, in input order."""
        if not prompts:
            return []
        for p in prompts:
            if not 1 <= p.n_labels <= self.max_labels:
                raise ValueError(f"n_labels must be 1..{self.max_labels}, got {p.n_labels}")
        encoded = self._encode(prompts)
        lengths = [len(ids) for ids, _ in encoded]
        order = sorted(range(len(prompts)), key=lambda i: -lengths[i])
        out: list[Readout | None] = [None] * len(prompts)
        for batch in self._batches(order, lengths):
            logits = self._forward([encoded[i][0] for i in batch])
            if not self._torch.isfinite(logits).all():
                raise RuntimeError("model produced non-finite logits")
            for row, i in enumerate(batch):
                cycles = logits[:, row, : prompts[i].n_labels].tolist()
                out[i] = Readout(
                    logits=list(cycles[-1]),
                    prompt_tokens=lengths[i],
                    truncated=encoded[i][1],
                    cycle_logits=cycles,
                )
        return out  # type: ignore[return-value]
