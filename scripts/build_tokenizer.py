"""Build a tokenizer for life events, using HuggingFace's own `tokenizers` library.

    uv run python scripts/build_tokenizer.py

Reads   data/derived/workshop/vocabulary.csv
Writes  models/event-tokenizer/                (committed, a few hundred KB)

A text tokenizer has to *discover* its vocabulary: it looks at a corpus and works out that
"hospitalisation" is best split into pieces. We have the opposite problem. Our vocabulary is
already fixed and finite -- 949 tokens -- and every one of them is an atom. There is nothing
to split, so we use the simplest model `tokenizers` offers: WordLevel, which is a plain
lookup table from a string to an id.

That is the whole difference between tokenising English and tokenising a life. Everything
downstream -- the collator, the model, the Trainer -- is unchanged.
"""
from __future__ import annotations

import pathlib

import pandas as pd
from tokenizers import Tokenizer, models, pre_tokenizers, processors
from transformers import PreTrainedTokenizerFast

ROOT = pathlib.Path(__file__).resolve().parent.parent
VOCAB_CSV = ROOT / "data" / "derived" / "workshop" / "vocabulary.csv"
OUT = ROOT / "models" / "event-tokenizer"


def main() -> None:
    vocabulary = pd.read_csv(VOCAB_CSV)
    vocab = dict(zip(vocabulary.token, vocabulary.token_id.astype(int)))
    print(f"{len(vocab)} tokens: " + ", ".join(
        f"{k} {v}" for k, v in vocabulary.kind.value_counts().items()))

    backbone = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]"))

    # Event names contain spaces ("Sepsis (disorder)"), so we must NOT split on whitespace.
    # Sequences are joined with a tab, which no event description contains.
    backbone.pre_tokenizer = pre_tokenizers.CharDelimiterSplit("\t")

    backbone.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[("[CLS]", vocab["[CLS]"]), ("[SEP]", vocab["[SEP]"])],
    )

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backbone,
        unk_token="[UNK]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        mask_token="[MASK]",
        model_max_length=128,
    )

    OUT.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(OUT)
    print(f"saved to {OUT.relative_to(ROOT)}")

    # ---- prove it round-trips, on a background block plus two real events --------------
    demo = "\t".join(["AGE_50_54", "SEX_F", "Sepsis (disorder)", "Full-time employment (finding)"])
    encoded = tokenizer(demo)
    print("\nround-trip check")
    print(f"  input   {demo!r}")
    print(f"  ids     {encoded['input_ids']}")
    print(f"  tokens  {tokenizer.convert_ids_to_tokens(encoded['input_ids'])}")
    assert tokenizer.convert_ids_to_tokens(encoded["input_ids"])[1:-1] == demo.split("\t")

    unknown = tokenizer("\t".join(["Meconium ileus (disorder)", "SEX_F"]))
    print(f"  rare event -> {tokenizer.convert_ids_to_tokens(unknown['input_ids'])}")


if __name__ == "__main__":
    main()
