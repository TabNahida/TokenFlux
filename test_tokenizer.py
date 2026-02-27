import contextlib
import importlib
import importlib.util
import shutil
import sys
import tempfile
import unittest
import uuid
from pathlib import Path


ROOT = Path(__file__).resolve().parent
ARTIFACTS_DIR = ROOT / "artifacts"


def _add_build_path_for_tokenflux_cpp() -> None:
    for candidate in (ROOT / "build").rglob("tokenflux_cpp.pyd"):
        build_dir = str(candidate.parent)
        if build_dir not in sys.path:
            sys.path.insert(0, build_dir)
        return


def _require_module(name: str):
    if importlib.util.find_spec(name) is None:
        raise unittest.SkipTest(f"missing optional module: {name}")
    return importlib.import_module(name)


def _require_tokenflux_cpp():
    _add_build_path_for_tokenflux_cpp()
    try:
        return importlib.import_module("tokenflux_cpp")
    except ModuleNotFoundError as exc:
        raise unittest.SkipTest("tokenflux_cpp.pyd was not built") from exc


class TokenFluxPythonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tf = _require_tokenflux_cpp()
        cls.torch = _require_module("torch")
        cls.tokenizers = _require_module("tokenizers")

    @staticmethod
    @contextlib.contextmanager
    def _tempdir():
        ARTIFACTS_DIR.mkdir(exist_ok=True)
        tmpdir = ARTIFACTS_DIR / f"pybind_test_{uuid.uuid4().hex}"
        tmpdir.mkdir(parents=True, exist_ok=False)
        try:
            yield str(tmpdir)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_default_gpt_style_special_tokens(self):
        cfg = self.tf.TrainConfig()
        args = self.tf.TokenizeArgs()

        self.assertEqual(self.tf.__version__, "0.3.2")
        self.assertEqual(cfg.unk_token, "<|endoftext|>")
        self.assertEqual(cfg.special_tokens, ["<|endoftext|>"])
        self.assertEqual(args.eos_token, "<|endoftext|>")
        self.assertEqual(args.bos_token, "")

    def test_python_file_list_train_and_tokenize(self):
        with self._tempdir() as tmp:
            tmpdir = Path(tmp)
            sample = tmpdir / "sample.jsonl"
            sample.write_text('{"text":"hello world"}\n{"text":"token flux"}\n', encoding="utf-8")

            tokenizer_path = tmpdir / "tokenizer.json"
            vocab_path = tmpdir / "vocab.json"
            merges_path = tmpdir / "merges.txt"
            chunk_dir = tmpdir / "chunks"
            out_dir = tmpdir / "tokens"

            cfg = self.tf.TrainConfig()
            cfg.trainer = self.tf.TrainerKind.byte_bpe
            cfg.vocab_size = 64
            cfg.min_freq = 1
            cfg.min_pair_freq = 1
            cfg.threads = 2
            cfg.resume = False
            cfg.chunk_dir = str(chunk_dir)
            cfg.output_json = str(tokenizer_path)
            cfg.output_vocab = str(vocab_path)
            cfg.output_merges = str(merges_path)
            self.tf.train(cfg, [str(sample)])

            self.assertTrue(tokenizer_path.exists())
            self.assertTrue(vocab_path.exists())
            self.assertTrue(merges_path.exists())

            tok = self.tf.Tokenizer(str(tokenizer_path))
            eot_id = tok.token_to_id("<|endoftext|>")
            self.assertIsNotNone(eot_id)

            ids = tok.encode("hello world")
            self.assertTrue(ids)
            self.assertTrue(all(isinstance(x, int) for x in ids))

            encoded = tok.encode_to_torch("hello world")
            self.assertEqual(tuple(encoded.shape), (len(ids),))
            self.assertEqual(str(encoded.dtype), "torch.int64")

            batch = tok.encode_batch_to_torch(["hello world", "token flux"], pad_id=0)
            self.assertIn("input_ids", batch)
            self.assertIn("lengths", batch)
            self.assertEqual(batch["input_ids"].shape[0], 2)
            self.assertEqual(batch["lengths"].tolist(), [len(ids), len(tok.encode("token flux"))])

            streamed = tok.tokenize_inputs_to_torch([str(sample)], text_field="text")
            self.assertEqual(int(streamed["num_docs"]), 2)
            self.assertEqual(streamed["doc_offsets"].tolist()[0], 0)
            self.assertEqual(streamed["doc_offsets"].tolist()[-1], streamed["token_ids"].shape[0])

            args = self.tf.TokenizeArgs()
            args.tokenizer_path = str(tokenizer_path)
            args.out_dir = str(out_dir)
            args.threads = 2
            args.resume = False
            self.tf.tokenize(args, [str(sample)])

            self.assertTrue((out_dir / "meta.json").exists())
            self.assertTrue((out_dir / "shards").exists())

    def test_hf_tokenizers_can_load_generated_tokenizer(self):
        with self._tempdir() as tmp:
            tmpdir = Path(tmp)
            sample = tmpdir / "sample.jsonl"
            sample.write_text('{"text":"hello world"}\n{"text":"hello token flux"}\n', encoding="utf-8")

            tokenizer_path = tmpdir / "tokenizer.json"
            cfg = self.tf.TrainConfig()
            cfg.trainer = self.tf.TrainerKind.byte_bpe
            cfg.vocab_size = 64
            cfg.min_freq = 1
            cfg.min_pair_freq = 1
            cfg.threads = 2
            cfg.resume = False
            cfg.chunk_dir = str(tmpdir / "chunks")
            cfg.output_json = str(tokenizer_path)
            cfg.output_vocab = str(tmpdir / "vocab.json")
            cfg.output_merges = str(tmpdir / "merges.txt")
            self.tf.train(cfg, [str(sample)])

            tokenizer_mod = self.tokenizers.Tokenizer
            decoder_mod = importlib.import_module("tokenizers.decoders")
            hf_tok = tokenizer_mod.from_file(str(tokenizer_path))
            hf_tok.decoder = decoder_mod.ByteLevel()

            enc = hf_tok.encode("hello world")
            self.assertTrue(enc.ids)
            self.assertIsNotNone(hf_tok.token_to_id("<|endoftext|>"))


if __name__ == "__main__":
    unittest.main()
