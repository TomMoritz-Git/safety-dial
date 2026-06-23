"""GPU-side model runner: fp16/eager loading, activation capture, steered gen.

This is the only module that holds weights. It encapsulates the Pascal-specific
choices (fp16, eager attention) and the two hard-won details from the pilot:

* read activations at ``hidden_states[layer + 1]`` (output of decoder block
  ``layer``), at the **last prompt token** (batch size 1, so no padding);
* steer by adding ``coeff * raw_direction`` to a decoder layer's residual
  output at **every** position throughout generation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import MAX_NEW_TOKENS, READ_POSITION, ModelSpec


@dataclass(frozen=True)
class Steer:
    """A steering intervention: add ``coeff * vector`` at ``layer``'s output."""

    layer: int
    vector: np.ndarray  # raw diff-of-means (natural units)
    coeff: float


def _decoder_layers(model: torch.nn.Module) -> torch.nn.ModuleList:
    """Locate the decoder ``ModuleList`` across common HF architectures."""
    for path in ("model.layers", "model.model.layers", "transformer.h"):
        obj = model
        try:
            for attr in path.split("."):
                obj = getattr(obj, attr)
            return obj  # type: ignore[return-value]
        except AttributeError:
            continue
    raise AttributeError("could not locate decoder layers on this model")


class ModelRunner:
    """Loaded model + tokenizer with activation and generation helpers."""

    def __init__(self, spec: ModelSpec, tokenizer, model, device: str):
        """Wrap a loaded tokenizer/model pair (use :meth:`load` instead)."""
        self.spec = spec
        self.tok = tokenizer
        self.model = model
        self.device = device
        self.layers = _decoder_layers(model)
        self.n_layers = len(self.layers)

    @classmethod
    def load(cls, spec: ModelSpec, device: str = "cuda") -> ModelRunner:
        """Load ``spec`` in fp16 with eager attention (Pascal-safe).

        Args:
            spec: The model to load.
            device: Torch device string.

        Returns:
            A ready :class:`ModelRunner`.
        """
        tok = AutoTokenizer.from_pretrained(spec.hf_id)
        kwargs: dict = {
            "attn_implementation": spec.attn_implementation,
            "dtype": torch.float16,
        }
        if spec.load_in_4bit:
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )
            kwargs["device_map"] = {"": device}
            model = AutoModelForCausalLM.from_pretrained(spec.hf_id, **kwargs)
        else:
            model = AutoModelForCausalLM.from_pretrained(spec.hf_id, **kwargs).to(device)
        model.eval()
        return cls(spec, tok, model, device)

    def format(self, prompt: str) -> str:
        """Render a user prompt through the chat template (no safety system msg).

        Honors ``spec.chat_template_kwargs`` (e.g. ``enable_thinking=False``) and
        a minimal ``spec.system`` used solely to disable thinking traces.
        """
        messages = []
        if self.spec.system:
            messages.append({"role": "system", "content": self.spec.system})
        messages.append({"role": "user", "content": prompt})
        return self.tok.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            **self.spec.chat_template_kwargs,
        )

    @torch.no_grad()
    def activations(self, prompt: str) -> np.ndarray:
        """All-layer residual activations at the last prompt token.

        Args:
            prompt: User request (formatted internally).

        Returns:
            ``[n_layers, hidden]`` float32 array; row ``i`` is the output of
            decoder block ``i`` (``hidden_states[i + 1]``).
        """
        enc = self.tok(self.format(prompt), return_tensors="pt").to(self.device)
        hs = self.model(**enc, output_hidden_states=True).hidden_states
        # hs[0] is the embedding output; hs[1:] are the per-block outputs.
        stacked = torch.stack([h[0, READ_POSITION] for h in hs[1:]])
        return stacked.float().cpu().numpy()

    def _steer_hook(self, steer: Steer):
        vec = torch.as_tensor(steer.vector, device=self.device)

        def hook(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            hidden[:] = hidden + steer.coeff * vec.to(hidden.dtype)
            return output

        return self.layers[steer.layer].register_forward_hook(hook)

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        steer: Steer | None = None,
        max_new_tokens: int = MAX_NEW_TOKENS,
    ) -> str:
        """Greedily generate a completion, optionally under a steering vector.

        Args:
            prompt: User request.
            steer: Optional steering intervention; if its ``coeff`` is 0 the hook
                is skipped entirely (identical to no steering).
            max_new_tokens: Generation budget.

        Returns:
            The decoded completion (special tokens stripped, whitespace-collapsed).
        """
        enc = self.tok(self.format(prompt), return_tensors="pt").to(self.device)
        handle = self._steer_hook(steer) if (steer and steer.coeff) else None
        try:
            out = self.model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tok.pad_token_id or self.tok.eos_token_id,
            )
        finally:
            if handle is not None:
                handle.remove()
        text = self.tok.decode(out[0, enc["input_ids"].shape[1] :], skip_special_tokens=True)
        return " ".join(text.split())

    def unload(self) -> None:
        """Free GPU memory held by this runner."""
        del self.model
        torch.cuda.empty_cache()
