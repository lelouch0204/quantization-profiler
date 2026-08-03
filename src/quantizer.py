"""Produce quantized, vLLM-servable checkpoints from HuggingFace models.

Output is written in compressed-tensors format, which records its quantization
config in ``config.json``. vLLM auto-detects that, so the resulting directory can
be handed straight to ``VLLMServerManager.start_server`` with no extra flags.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path

# Bump to invalidate every cached checkpoint after a recipe change.
_KEY_VERSION = 1


class QuantScheme(Enum):
    """Supported quantization schemes.

    To add one: add a member here and a row to ``_RECIPES``. Members without a
    recipe row are placeholders and raise ``NotImplementedError`` when used.
    """

    W4A16 = "W4A16"
    W8A8_INT8 = "W8A8-INT8"
    AWQ_W4A16 = "AWQ-W4A16"
    FP8_BLOCK = "FP8-BLOCK"

    # -- Placeholders: no recipe yet ------------------------------------------
    W4A8 = "W4A8"
    NVFP4 = "NVFP4"


@dataclass(frozen=True)
class CalibrationConfig:
    """Calibration data settings, folded into the cache key of schemes that use it."""

    dataset: str = "HuggingFaceH4/ultrachat_200k"
    split: str = "train_sft"
    num_samples: int = 512  # llm-compressor docs: "512 samples is a good place to start"
    max_seq_length: int = 2048
    seed: int = 42


# ----------------------------------------------------------------------------
# Recipes — imports are local so this module stays importable without the ML stack
# ----------------------------------------------------------------------------


def _recipe_w4a16():
    from llmcompressor.modifiers.gptq import GPTQModifier

    return GPTQModifier(targets="Linear", scheme="W4A16", ignore=["lm_head"])


def _recipe_w8a8_int8():
    from llmcompressor.modifiers.gptq import GPTQModifier
    from llmcompressor.modifiers.transform import SmoothQuantModifier

    # SmoothQuant first: it makes activations easier to quantize for GPTQ.
    return [
        SmoothQuantModifier(smoothing_strength=0.8),
        GPTQModifier(targets="Linear", scheme="W8A8", ignore=["lm_head"]),
    ]


def _recipe_awq_w4a16():
    from llmcompressor.modifiers.quantization import QuantizationModifier
    from llmcompressor.modifiers.transform import AWQModifier

    return [
        AWQModifier(),
        QuantizationModifier(targets=["Linear"], scheme="W4A16_ASYM", ignore=["lm_head"]),
    ]


def _recipe_fp8_block():
    from llmcompressor.modifiers.quantization import QuantizationModifier

    return QuantizationModifier(targets="Linear", scheme="FP8_BLOCK", ignore=["lm_head"])


# scheme -> (recipe builder, needs calibration data)
_RECIPES = {
    QuantScheme.W4A16: (_recipe_w4a16, True),
    QuantScheme.W8A8_INT8: (_recipe_w8a8_int8, True),
    QuantScheme.AWQ_W4A16: (_recipe_awq_w4a16, True),
    QuantScheme.FP8_BLOCK: (_recipe_fp8_block, False),  # round-to-nearest, data-free
}


def _needs_calibration(scheme: QuantScheme) -> bool:
    """Whether ``scheme`` consumes calibration data. False for unimplemented schemes."""
    return scheme in _RECIPES and _RECIPES[scheme][1]


def _load_backend():
    """Import the heavy ML stack.

    Single seam: patching this lets the unit tests run without llmcompressor,
    transformers, or a GPU. Also keeps cache hits free of a multi-second import.
    """
    from datasets import load_dataset
    from llmcompressor import oneshot
    from transformers import AutoModelForCausalLM, AutoTokenizer

    return oneshot, AutoModelForCausalLM, AutoTokenizer, load_dataset


class Quantizer:
    """Quantizes HuggingFace models into cached, vLLM-servable checkpoints."""

    def __init__(
        self,
        output_root: str | Path = "quantized_models",
        calibration: CalibrationConfig | None = None,
    ):
        self.output_root = Path(output_root)
        self.calibration = calibration or CalibrationConfig()

    def cache_key(self, model_id: str, scheme: QuantScheme) -> str:
        """Hash of everything that changes the output checkpoint."""
        key = {"v": _KEY_VERSION, "model_id": model_id, "scheme": scheme.value}
        if _needs_calibration(scheme):
            # Excluded otherwise, so retuning calibration doesn't invalidate
            # checkpoints produced by schemes that never read it.
            key["calibration"] = asdict(self.calibration)
        return hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()[:12]

    def output_dir(self, model_id: str, scheme: QuantScheme) -> Path:
        """Where ``model_id`` at ``scheme`` lands. Readable prefix, hash for correctness."""
        name = model_id.rstrip("/").split("/")[-1]
        return self.output_root / f"{name}-{scheme.value}-{self.cache_key(model_id, scheme)}"

    def quantize(self, model_id: str, scheme: QuantScheme, force: bool = False) -> Path:
        """Quantize ``model_id`` and return the checkpoint directory.

        Returns the cached directory immediately if it already exists, unless
        ``force``. Raises ``NotImplementedError`` for placeholder schemes.
        """
        out = self.output_dir(model_id, scheme)
        if out.exists() and not force:
            return out

        try:
            build_recipe, needs_calibration = _RECIPES[scheme]
        except KeyError:
            supported = ", ".join(s.name for s in _RECIPES)
            raise NotImplementedError(
                f"{scheme.name} has no recipe yet. Supported: {supported}"
            ) from None

        oneshot, AutoModelForCausalLM, AutoTokenizer, load_dataset = _load_backend()

        # ponytail: no device_map/dtype -- matches llm-compressor's own example, and
        # torch_dtype was renamed in Transformers v5. Models too large for one GPU
        # need device_map="auto" or compressed_tensors.offload.dispatch_model.
        model = AutoModelForCausalLM.from_pretrained(model_id)
        tokenizer = AutoTokenizer.from_pretrained(model_id)

        calibration_kwargs = {}
        if needs_calibration:
            calibration_kwargs = {
                "dataset": self._calibration_dataset(tokenizer, load_dataset),
                "max_seq_length": self.calibration.max_seq_length,
                "num_calibration_samples": self.calibration.num_samples,
            }

        # Build in a temp dir and rename on success. A crash mid-save must not leave
        # a half-written directory that the cache check above would later trust.
        # String concat, not with_suffix(): model names contain dots.
        tmp = out.parent / (out.name + ".tmp")
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir(parents=True)

        oneshot(model=model, recipe=build_recipe(), **calibration_kwargs)

        model.save_pretrained(tmp, save_compressed=True)
        tokenizer.save_pretrained(tmp)
        (tmp / "quant_key.json").write_text(
            json.dumps(
                {
                    "model_id": model_id,
                    "scheme": scheme.value,
                    "calibration": asdict(self.calibration) if needs_calibration else None,
                },
                indent=2,
            )
        )

        if out.exists():  # force=True
            shutil.rmtree(out)
        tmp.rename(out)
        return out

    def _calibration_dataset(self, tokenizer, load_dataset):
        """Load, format, and tokenize the calibration set.

        ponytail: assumes a chat dataset with a ``messages`` column (the default).
        Swapping to a plain-text corpus means editing the two formatters below.
        """
        cfg = self.calibration
        ds = load_dataset(cfg.dataset, split=f"{cfg.split}[:{cfg.num_samples}]")
        ds = ds.shuffle(seed=cfg.seed)

        if tokenizer.chat_template:
            def to_text(example):
                return tokenizer.apply_chat_template(example["messages"], tokenize=False)
        else:
            # Base models have no template -- flatten the turns to plain text.
            def to_text(example):
                return "\n".join(m["content"] for m in example["messages"])

        ds = ds.map(lambda example: {"text": to_text(example)})

        # add_special_tokens=False: the chat template already inserted BOS.
        return ds.map(
            lambda example: tokenizer(
                example["text"],
                padding=False,
                truncation=True,
                max_length=cfg.max_seq_length,
                add_special_tokens=False,
            ),
            remove_columns=ds.column_names,
        )
