"""Pretrain a small BERT on life-event sequences, life2vec style.

    uv run python scripts/pretrain.py                  # the committed 30K subsample
    uv run python scripts/pretrain.py --epochs 10      # a longer run
    uv run python scripts/pretrain.py --sequences data/derived/sequences-full.parquet

Reads   data/derived/workshop/{sequences,vocabulary}.parquet|csv, models/event-tokenizer/
Writes  models/synthea-bert/

Almost all of this is stock HuggingFace: `BertConfig` builds the architecture,
`AutoModelForMaskedLM.from_config` initialises it randomly, `DataCollatorForLanguageModeling`
does the 15% masking with the usual 80/10/10 split, and `Trainer` runs the loop. Nothing
here is a custom training loop, and that is the point -- switching from English to life
events changes the *data*, not the machinery.

The one genuinely custom piece is how a sequence knows *when* things happened.

Each position gets three things added together before the encoder sees it:

    token embedding      what happened            (learned, 949 x 128 lookup)
  + absolute position    where in the sequence    (BERT's own, added internally)
  + Time2Vec(age)        when in the person's life

Absolute position gives order -- 1st event, 2nd event, 3rd. It cannot express that two
events were 3 days apart versus 3 years apart, because rank order discards the gaps.
Time2Vec (Kazemi et al., 2019, arXiv:1907.05321) encodes the age in years at which each
event happened, as one linear term plus a bank of learned sinusoids:

    t2v(t)[0] = w_0 * t + b_0                    trend
    t2v(t)[i] = sin(w_i * t + b_i)   for i > 0   periodicity at learned frequencies

The linear term lets the model represent "later in life"; the periodic terms let it
represent recurring rhythms (an annual check-up) without anyone specifying the period.

The model itself lives in `scripts/event_bert.py`, shared with the notebooks so that
training and inference build the input embeddings identically.

We add Time2Vec to the token embeddings and hand the result to BERT as `inputs_embeds`,
which is a supported HuggingFace input. BERT then adds its own absolute position
embeddings on top, so we get both encodings for about fifteen lines of code.
"""
from __future__ import annotations

import argparse
import math
import pathlib
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from transformers import (BertConfig, DataCollatorForLanguageModeling,
                          EarlyStoppingCallback, PreTrainedTokenizerFast, Trainer,
                          TrainingArguments)

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from event_bert import EventBertForMaskedLM, pick_device  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
WORKSHOP = ROOT / "data" / "derived" / "workshop"
TOKENIZER_DIR = ROOT / "models" / "event-tokenizer"
OUT = ROOT / "models" / "synthea-bert"

MAX_LEN = 128


# --------------------------------------------------------------------------- data
def build_dataset(sequences: pd.DataFrame, cohort: pd.DataFrame, vocab: dict) -> Dataset:
    """Compose [CLS] background [SEP] events [SEP], with an age in years per position."""
    cls_id, sep_id = vocab["[CLS]"], vocab["[SEP]"]
    age_at_index = cohort.set_index("patient_id").age_at_index

    input_ids, ages_out, days_out = [], [], []
    for row in sequences.itertuples(index=False):
        base_age = float(age_at_index.get(row.patient_id, 50.0))
        events = list(row.tokens)[-(MAX_LEN - len(row.background) - 3):]
        days = list(row.days_before_index)[-len(events):] if events else []

        ids = [cls_id, *row.background, sep_id, *events, sep_id]
        # Background and separators are timeless: pin them to the index date, i.e. 0 days
        # before it, at the person's age then. NOTE: 0.0 doubles as the padding value for
        # the days clock. That is safe only because every event is strictly BEFORE the
        # index date (days >= 1). If the window is ever widened to include the index date
        # itself, real events would collide with the padding sentinel -- use a mask then.
        pad = len(row.background) + 2
        ages = [base_age] * pad + [base_age - d / 365.25 for d in days] + [base_age]
        days_before = [0.0] * pad + [float(d) for d in days] + [0.0]
        input_ids.append(ids)
        ages_out.append(ages)
        days_out.append(days_before)
        assert len(ids) == len(ages) == len(days_before) <= MAX_LEN

    return Dataset.from_dict({
        "input_ids": input_ids,
        "attention_mask": [[1] * len(x) for x in input_ids],
        "ages": ages_out,
        "days": days_out,
    })


@dataclass
class CollatorWithClocks:
    """Wraps HuggingFace's MLM collator so the two clock columns are padded alongside."""

    base: DataCollatorForLanguageModeling

    def __call__(self, features):
        clocks = {k: [f.pop(k) for f in features] for k in ("ages", "days")}
        batch = self.base(features)
        width = batch["input_ids"].shape[1]
        for key, values in clocks.items():
            batch[key] = torch.tensor(
                [v + [0.0] * (width - len(v)) for v in values], dtype=torch.float)
        return batch


# --------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sequences", default=str(WORKSHOP / "sequences.parquet"))
    ap.add_argument("--cohort", default=None,
                    help="defaults to the cohort file matching --sequences")
    ap.add_argument("--output", default=str(OUT))
    ap.add_argument("--epochs", type=float, default=40.0,
                    help="an upper bound; early stopping decides when to stop")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--learning-rate", type=float, default=5e-4)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--mlm-probability", type=float, default=0.30,
                    help="BERT used 0.15; ModernBERT uses 0.30, which is what we "
                         "follow. More masking is a harder task and a stronger "
                         "learning signal per sequence.")
    ap.add_argument("--patience", type=int, default=3,
                    help="stop after this many epochs without an eval-loss improvement")
    ap.add_argument("--attention", choices=["sdpa", "eager", "performer"], default="sdpa",
                    help="sdpa is exact and fastest at L=128; performer (linear-time "
                         "FAVOR+, as in life2vec) only wins past ~512 tokens -- see "
                         "scripts/benchmark_attention.py")
    ap.add_argument("--rezero", action="store_true", default=True,
                    help="scalar-gated residuals instead of LayerNorm, as in life2vec")
    ap.add_argument("--no-rezero", dest="rezero", action="store_false")
    ap.add_argument("--performer-features", type=int, default=64)
    ap.add_argument("--device", default="auto", help="auto | mps | cuda | cpu")
    ap.add_argument("--seed", type=int, default=20260902)
    args = ap.parse_args()

    vocabulary = pd.read_csv(WORKSHOP / "vocabulary.csv")
    vocab = dict(zip(vocabulary.token, vocabulary.token_id.astype(int)))
    sequences = pd.read_parquet(args.sequences)
    cohort_path = args.cohort or str(args.sequences).replace("sequences", "cohort")
    cohort = pd.read_parquet(cohort_path)
    tokenizer = PreTrainedTokenizerFast.from_pretrained(TOKENIZER_DIR)
    print(f"{len(sequences):,} sequences, {len(vocab)} tokens")
    print(f"  sequences {args.sequences}")
    print(f"  cohort    {cohort_path}")

    dataset = build_dataset(sequences, cohort, vocab).train_test_split(
        test_size=0.05, seed=args.seed)
    print(f"train {len(dataset['train']):,}  eval {len(dataset['test']):,}")

    config = BertConfig(
        vocab_size=len(vocab),
        hidden_size=args.hidden,
        num_hidden_layers=args.layers,
        num_attention_heads=args.heads,
        intermediate_size=args.hidden * 4,
        max_position_embeddings=MAX_LEN,
        pad_token_id=vocab["[PAD]"],
        position_embedding_type="absolute",
    )
    model = EventBertForMaskedLM(config, attention=args.attention, rezero=args.rezero,
                                 performer_features=args.performer_features)
    device = pick_device(args.device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: {n_params:,} parameters "
          f"({args.layers} layers, hidden {args.hidden}, {args.heads} heads)")
    print(f"  attention        {args.attention}"
          + (f" ({args.performer_features} random features)" if args.attention == "performer" else ""))
    print(f"  residuals        {'ReZero' if args.rezero else 'LayerNorm'}")
    print(f"  clocks           Time2Vec(age) + Time2Vec(days before index)")
    print(f"  tied embeddings  {model.embeddings_tied()}")
    print(f"  device           {device}")
    print(f"  masking          {args.mlm_probability:.0%} of positions")
    print(f"  early stopping   patience {args.patience} epochs (max {args.epochs:.0f})")

    collator = CollatorWithClocks(DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=args.mlm_probability,
        seed=args.seed))

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(pathlib.Path(args.output) / "checkpoints"),
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            warmup_steps=200,
            weight_decay=0.01,
            eval_strategy="epoch",
            save_strategy="epoch",          # required by load_best_model_at_end
            save_total_limit=1,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            logging_steps=100,
            report_to=[],
            seed=args.seed,
            # transformers 5.x picks MPS/CUDA automatically when use_cpu is False;
            # `use_mps_device` was removed.
            use_cpu=device.type == "cpu",
        ),
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        data_collator=collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.patience)],
    )
    trainer.train()

    metrics = trainer.evaluate()
    loss = metrics["eval_loss"]
    print(f"\nfinal masked-event loss {loss:.4f}  (perplexity {math.exp(loss):.1f}, "
          f"vs {len(vocab)} for a model that has learned nothing)")

    out = pathlib.Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out)
    tokenizer.save_pretrained(out)
    shown = out.relative_to(ROOT) if out.is_relative_to(ROOT) else out
    print(f"saved to {shown}")


if __name__ == "__main__":
    main()
