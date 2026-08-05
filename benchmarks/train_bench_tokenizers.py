"""Train realistic agent tokenizers once, cache them for the Stage 8 benchmark.

Three agents with genuinely different vocabularies, sized like production models:
  agent-A  byte-level BPE, 32k   (GPT-2 / Qwen family shape)
  agent-B  byte-level BPE, 50k   (different merges AND different size)
  agent-C  Unigram + Metaspace, 32k  (Llama-2 / Gemma family shape)
"""

import sys
from pathlib import Path

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

HERE = Path(__file__).resolve().parent
CORPUS = HERE / "corpus.txt"
OUT = HERE / "tokenizers"
OUT.mkdir(exist_ok=True)

#: A tokenizer is only realistic if trained on real text. No corpus ships with
#: this repo, so one is assembled from the local Python installation's own
#: source -- a few MB of genuine mixed code and prose, always present, no
#: download. It also contains "##", which is exactly the case that broke scheme
#: detection (see docs/AUDIT_REPORT.md, AUDIT-1).
CORPUS_ROOTS = [
    Path(sys.prefix) / "lib",
    Path("/usr/lib/python3"),
]
CORPUS_TARGET_BYTES = 14_000_000


def build_corpus() -> Path:
    if CORPUS.exists() and CORPUS.stat().st_size > 1_000_000:
        print(f"cached corpus: {CORPUS} ({CORPUS.stat().st_size / 1e6:.1f} MB)")
        return CORPUS
    chunks, total = [], 0
    for root in CORPUS_ROOTS:
        if not root.exists():
            continue
        for f in sorted(root.rglob("*.py")):
            try:
                t = f.read_text(errors="ignore")
            except OSError:
                continue
            if len(t) < 200:
                continue
            chunks.append(t)
            total += len(t)
            if total >= CORPUS_TARGET_BYTES:
                break
        if total >= CORPUS_TARGET_BYTES:
            break
    if total < 500_000:
        raise SystemExit(
            "could not assemble a corpus from the local Python installation; "
            "point CORPUS at your own text file instead"
        )
    CORPUS.write_text("\n".join(chunks))
    print(f"built corpus: {len(chunks)} files, {total / 1e6:.1f} MB -> {CORPUS}")
    return CORPUS


def lines():
    with open(CORPUS, errors="ignore") as fh:
        buf = []
        for ln in fh:
            buf.append(ln)
            if len(buf) == 200:
                yield "".join(buf)
                buf = []
        if buf:
            yield "".join(buf)


def bpe(vocab_size, seed_suffix):
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    tok.train_from_iterator(
        lines(),
        trainers.BpeTrainer(
            vocab_size=vocab_size,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            show_progress=False,
            # a different min_frequency yields genuinely different merges
            min_frequency=seed_suffix,
        ),
    )
    return tok


def unigram(vocab_size):
    tok = Tokenizer(models.Unigram())
    tok.pre_tokenizer = pre_tokenizers.Metaspace()
    tok.train_from_iterator(
        lines(),
        trainers.UnigramTrainer(
            vocab_size=vocab_size, unk_token="[UNK]", special_tokens=["[UNK]"], show_progress=False
        ),
    )
    return tok


SPECS = [
    ("agent-A-bpe32k", lambda: bpe(32000, 2)),
    ("agent-B-bpe50k", lambda: bpe(50000, 5)),
    ("agent-C-uni32k", lambda: unigram(32000)),
]

if __name__ == "__main__":
    build_corpus()
    for name, build in SPECS:
        path = OUT / f"{name}.json"
        if path.exists():
            print(f"cached  {name}")
            continue
        print(f"training {name} ...", flush=True)
        t = build()
        t.save(str(path))
        print(f"  saved {name}: {t.get_vocab_size()} tokens")
    sys.exit(0)
