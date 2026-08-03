import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from quantizer import CalibrationConfig, QuantScheme, Quantizer


def fake_backend():
    """Stand-in for _load_backend()'s 4-tuple, with save_pretrained that writes files."""
    oneshot = MagicMock()
    model = MagicMock()
    tokenizer = MagicMock()

    def save(path, **kwargs):
        Path(path).mkdir(parents=True, exist_ok=True)
        (Path(path) / "config.json").write_text("{}")

    model.save_pretrained.side_effect = save
    tokenizer.save_pretrained.side_effect = save

    model_cls = MagicMock(from_pretrained=MagicMock(return_value=model))
    tokenizer_cls = MagicMock(from_pretrained=MagicMock(return_value=tokenizer))
    return oneshot, model_cls, tokenizer_cls, MagicMock()


# The real builders import llmcompressor. Substituting them keeps the orchestration
# tests runnable without the ML stack; the recipes themselves are declarative
# one-liners covered by the integration test.
fake_recipes = patch.dict(
    "quantizer._RECIPES",
    {
        QuantScheme.W4A16: (lambda: "fake-w4a16-recipe", True),
        QuantScheme.FP8_BLOCK: (lambda: "fake-fp8-recipe", False),
    },
)


class TestCacheKey(unittest.TestCase):
    """The hash must change exactly when the output would."""

    def setUp(self):
        self.q = Quantizer(output_root="/tmp/unused")

    def test_stable_across_calls(self):
        a = self.q.cache_key("meta-llama/Llama-3-8B", QuantScheme.W4A16)
        b = self.q.cache_key("meta-llama/Llama-3-8B", QuantScheme.W4A16)
        self.assertEqual(a, b)

    def test_differs_by_scheme(self):
        self.assertNotEqual(
            self.q.cache_key("m", QuantScheme.W4A16),
            self.q.cache_key("m", QuantScheme.W8A8_INT8),
        )

    def test_differs_by_model(self):
        self.assertNotEqual(
            self.q.cache_key("model-a", QuantScheme.W4A16),
            self.q.cache_key("model-b", QuantScheme.W4A16),
        )

    def test_calibration_change_invalidates_calibrated_scheme(self):
        other = Quantizer(calibration=CalibrationConfig(num_samples=8))
        self.assertNotEqual(
            self.q.cache_key("m", QuantScheme.W4A16),
            other.cache_key("m", QuantScheme.W4A16),
        )

    def test_calibration_change_ignored_by_data_free_scheme(self):
        other = Quantizer(calibration=CalibrationConfig(num_samples=8))
        self.assertEqual(
            self.q.cache_key("m", QuantScheme.FP8_BLOCK),
            other.cache_key("m", QuantScheme.FP8_BLOCK),
        )

    def test_output_dir_keeps_readable_prefix(self):
        out = self.q.output_dir("meta-llama/Llama-3-8B", QuantScheme.W4A16)
        self.assertTrue(out.name.startswith("Llama-3-8B-W4A16-"))


@fake_recipes
class TestQuantize(unittest.TestCase):
    """Caching, dispatch, and crash-safety of the quantize() flow."""

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)
        self.q = Quantizer(output_root=self.tmpdir.name)

    @patch("quantizer._load_backend")
    def test_cache_hit_skips_work(self, mock_backend):
        mock_backend.return_value = fake_backend()
        out = self.q.output_dir("m", QuantScheme.FP8_BLOCK)
        out.mkdir(parents=True)

        self.assertEqual(self.q.quantize("m", QuantScheme.FP8_BLOCK), out)
        mock_backend.assert_not_called()  # no import, no download, no quantization

    @patch("quantizer._load_backend")
    def test_force_requantizes_over_existing(self, mock_backend):
        oneshot, *_ = backend = fake_backend()
        mock_backend.return_value = backend
        out = self.q.output_dir("m", QuantScheme.FP8_BLOCK)
        out.mkdir(parents=True)

        self.assertEqual(self.q.quantize("m", QuantScheme.FP8_BLOCK, force=True), out)
        oneshot.assert_called_once()

    def test_placeholder_scheme_raises(self):
        with self.assertRaises(NotImplementedError) as ctx:
            self.q.quantize("m", QuantScheme.W4A8)
        self.assertIn("W4A16", str(ctx.exception))  # names what IS supported

    @patch("quantizer._load_backend")
    def test_happy_path_writes_checkpoint(self, mock_backend):
        oneshot, model_cls, _, _ = backend = fake_backend()
        mock_backend.return_value = backend

        out = self.q.quantize("org/tiny", QuantScheme.FP8_BLOCK)

        self.assertTrue(out.exists())
        oneshot.assert_called_once()
        self.assertTrue(model_cls.from_pretrained.call_args[0][0] == "org/tiny")
        self.assertTrue(
            model_cls.from_pretrained.return_value.save_pretrained.call_args.kwargs[
                "save_compressed"
            ]
        )
        self.assertEqual(
            json.loads((out / "quant_key.json").read_text())["scheme"], "FP8-BLOCK"
        )
        self.assertFalse(out.parent.joinpath(out.name + ".tmp").exists())  # tmp renamed away

    @patch("quantizer._load_backend")
    def test_calibrated_scheme_passes_dataset(self, mock_backend):
        oneshot, _, tokenizer_cls, load_dataset = backend = fake_backend()
        mock_backend.return_value = backend

        self.q.quantize("org/tiny", QuantScheme.W4A16)

        kwargs = oneshot.call_args.kwargs
        self.assertIn("dataset", kwargs)
        self.assertEqual(kwargs["num_calibration_samples"], 512)
        load_dataset.assert_called_once()

    @patch("quantizer._load_backend")
    def test_data_free_scheme_skips_dataset(self, mock_backend):
        oneshot, _, _, load_dataset = backend = fake_backend()
        mock_backend.return_value = backend

        self.q.quantize("org/tiny", QuantScheme.FP8_BLOCK)

        self.assertNotIn("dataset", oneshot.call_args.kwargs)
        load_dataset.assert_not_called()

    @patch("quantizer._load_backend")
    def test_failure_leaves_no_poisoned_cache_entry(self, mock_backend):
        oneshot, *_ = backend = fake_backend()
        oneshot.side_effect = RuntimeError("CUDA OOM")
        mock_backend.return_value = backend

        with self.assertRaises(RuntimeError):
            self.q.quantize("org/tiny", QuantScheme.FP8_BLOCK)

        # A later run must re-quantize rather than serve a half-written directory.
        self.assertFalse(self.q.output_dir("org/tiny", QuantScheme.FP8_BLOCK).exists())


class TestCalibrationDataset(unittest.TestCase):
    """Base models have no chat template -- that path must not raise."""

    def _fake_dataset(self):
        ds = MagicMock()
        ds.shuffle.return_value = ds
        ds.map.return_value = ds
        ds.column_names = ["messages"]
        return ds

    def test_uses_chat_template_when_present(self):
        ds = self._fake_dataset()
        tokenizer = MagicMock(chat_template="{{ messages }}")

        Quantizer()._calibration_dataset(tokenizer, MagicMock(return_value=ds))

        formatter = ds.map.call_args_list[0][0][0]
        formatter({"messages": [{"role": "user", "content": "hi"}]})
        tokenizer.apply_chat_template.assert_called_once()

    def test_falls_back_when_no_chat_template(self):
        ds = self._fake_dataset()
        tokenizer = MagicMock(chat_template=None)

        Quantizer()._calibration_dataset(tokenizer, MagicMock(return_value=ds))

        formatter = ds.map.call_args_list[0][0][0]
        result = formatter({"messages": [{"content": "hi"}, {"content": "there"}]})
        self.assertEqual(result, {"text": "hi\nthere"})
        tokenizer.apply_chat_template.assert_not_called()


class TestIntegration(unittest.TestCase):
    """Real quantization -- requires llm-compressor and a GPU."""

    @unittest.skipUnless(
        os.environ.get("TEST_QUANT_INTEGRATION") == "1",
        "Set TEST_QUANT_INTEGRATION=1 to run live quantization",
    )
    def test_quantize_tinyllama_w4a16(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            q = Quantizer(
                output_root=tmpdir,
                calibration=CalibrationConfig(num_samples=8, max_seq_length=512),
            )
            out = q.quantize("TinyLlama/TinyLlama-1.1B-Chat-v1.0", QuantScheme.W4A16)

            self.assertTrue((out / "config.json").exists())
            self.assertEqual(
                q.quantize("TinyLlama/TinyLlama-1.1B-Chat-v1.0", QuantScheme.W4A16), out
            )


if __name__ == "__main__":
    unittest.main()
