"""Build the small, committable artefacts the A02 notebooks actually load.

Run scripts/build_events.py first, then:

    uv run python scripts/build_workshop_data.py

Reads   data/derived/{events,vocabulary,patients}.parquet   (58 MB, gitignored)
Writes  data/derived/workshop/                              (a few MB, committed)

    vocabulary.csv       the FULL token vocabulary: specials, background, events.
    sequences.parquet    per-patient background tokens + pre-index event tokens.
    cohort.parquet       label, eligibility, demographics, event counts.
    raw-sample.parquet   full dated timelines with readable names, RAW_SAMPLE patients.
                         Notebook b1 assembles sequences from this, so the assembly is real.
    severe-codes.csv     the outcome definition, including the exclusions and why.

Two design decisions worth knowing before reading the code, both from
planning/a02-task-design.md:

  * The index date is a fixed CALENDAR date, so every history covers the same window.
    A fixed *age* index date was tried and leaked badly.
  * Post-index events are not shipped for the subsample, so a participant cannot leak the
    label into their own features by accident. The 500-patient raw sample DOES include
    them, so notebook b1 can draw the index-date split on a real timeline.

Background tokens follow life2vec: a person's static attributes become tokens at the head
of their sequence, rather than being bolted on as separate features downstream.
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from severe_codes import EXCLUDED, GROUP_OF, SEVERE_CODES  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
DERIVED = ROOT / "data" / "derived"
OUT = DERIVED / "workshop"

EPOCH = np.datetime64("1970-01-01")
INDEX_DATE = np.datetime64("2021-08-18")   # 5 years before the simulation reference time
END_DATE = np.datetime64("2026-08-18")     # the simulation reference time
MAX_EVENTS = 119                           # + [CLS] + 6 background + 2 [SEP] = 128
MIN_EVENTS = 5                             # eligibility floor
MIN_FREQ = 30                              # event types below this collapse to [UNK]
INCOME_BINS = 25                           # equal-frequency income bands
N_PATIENTS = 50_000                        # committable subsample given to students
TEST_FRACTION = 0.20                       # held-out slice of that subsample
RAW_SAMPLE = 500                           # patients shipped as raw dated timelines
SEED = 20260902

SPECIALS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]


def as_day(d: np.datetime64) -> int:
    return int((d - EPOCH).astype("timedelta64[D]").astype(int))


def background_tokens(patients: pd.DataFrame, age_at_index: np.ndarray) -> pd.DataFrame:
    """One row per patient, six static attributes, life2vec style.

    Age is banded rather than continuous because a token vocabulary cannot hold a real
    number. Income is banded into population quintiles for the same reason.
    """
    band = np.clip((age_at_index // 5 * 5).astype(int), 30, 70)
    income = patients.INCOME.to_numpy(dtype=float)
    # 25 equal-frequency bins rather than quintiles: income is the one background attribute
    # with real spread, and coarse bins throw that away before the model ever sees it.
    cuts = np.nanquantile(income, np.arange(1, INCOME_BINS) / INCOME_BINS)
    income_bin = np.searchsorted(cuts, income, "right") + 1
    return pd.DataFrame({
        "AGE": [f"AGE_{b}_{b + 4}" for b in band],
        "SEX": "SEX_" + patients.GENDER.fillna("NA").to_numpy(),
        "RACE": "RACE_" + patients.RACE.fillna("NA").to_numpy(),
        "ETHNICITY": "ETH_" + patients.ETHNICITY.fillna("NA").to_numpy(),
        "MARITAL": "MARITAL_" + patients.MARITAL.fillna("NA").to_numpy(),
        "INCOME": [f"INCOME_B{b:02d}" for b in income_bin],
    })


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true",
                    help="write full-population sequences to data/derived/ (gitignored, "
                         "for pretraining) instead of the committed 30K subsample")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    index_day, end_day = as_day(INDEX_DATE), as_day(END_DATE)

    events = pd.read_parquet(DERIVED / "events.parquet")
    event_vocab = pd.read_parquet(DERIVED / "vocabulary.parquet")
    patients = pd.read_parquet(DERIVED / "patients.parquet")
    n_patients = len(patients)
    print(f"loaded {len(events):,} events, {n_patients:,} patients")

    pid = events.patient_id.to_numpy()
    day = events.day.to_numpy()
    raw_eid = events.event_id.to_numpy()

    # ---- the label, computed on the FULL vocabulary before anything is dropped ----------
    id_of_desc = dict(zip(event_vocab.description, event_vocab.event_id))
    missing = [c for c in SEVERE_CODES if c not in id_of_desc]
    if missing:
        raise SystemExit(f"severe codes absent from the vocabulary: {missing}")
    severe_raw_ids = np.array([id_of_desc[c] for c in SEVERE_CODES], dtype=np.int16)

    is_severe = np.isin(raw_eid, severe_raw_ids)
    severe_before = np.zeros(n_patients, bool)
    severe_before[pid[is_severe & (day < index_day)]] = True
    severe_during = np.zeros(n_patients, bool)
    severe_during[pid[is_severe & (day >= index_day) & (day <= end_day)]] = True

    pre = day < index_day
    n_pre = np.bincount(pid[pre], minlength=n_patients)

    birth = ((pd.to_datetime(patients.BIRTHDATE).to_numpy() - EPOCH)
             .astype("timedelta64[D]").astype(np.int64))
    death = pd.to_datetime(patients.DEATHDATE)
    death_day = np.where(
        death.notna(),
        (death.to_numpy() - EPOCH).astype("timedelta64[D]").astype("float64"),
        np.inf,
    )
    age_at_index = (index_day - birth) / 365.25

    eligible = (death_day > index_day) & ~severe_before & (n_pre >= MIN_EVENTS)
    print(f"eligible {eligible.sum():,}  positives {severe_during[eligible].sum():,} "
          f"({severe_during[eligible].mean():.2%})")

    # ---- the unified token vocabulary: specials, then background, then events -----------
    background = background_tokens(patients, age_at_index)
    background_types = sorted({t for col in background.columns for t in background[col].unique()})

    kept = event_vocab[event_vocab.frequency >= MIN_FREQ].sort_values(
        "frequency", ascending=False)
    dropped = event_vocab[event_vocab.frequency < MIN_FREQ]
    print(f"event types: {len(kept)} kept (>={MIN_FREQ} occurrences), {len(dropped)} -> [UNK] "
          f"({dropped.frequency.sum() / event_vocab.frequency.sum():.3%} of all events)")

    tokens = SPECIALS + background_types + kept.description.tolist()
    token_id = {t: i for i, t in enumerate(tokens)}
    print(f"total vocabulary: {len(tokens)} tokens "
          f"({len(SPECIALS)} special + {len(background_types)} background + {len(kept)} event)")

    vocabulary = pd.DataFrame({
        "token_id": np.arange(len(tokens), dtype=np.int16),
        "token": tokens,
        "kind": (["special"] * len(SPECIALS) + ["background"] * len(background_types)
                 + ["event"] * len(kept)),
        "source": ([""] * (len(SPECIALS) + len(background_types)) + kept.source.tolist()),
        "frequency": ([0] * (len(SPECIALS) + len(background_types)) + kept.frequency.tolist()),
    })
    vocabulary["is_severe_outcome"] = vocabulary.token.isin(SEVERE_CODES)
    vocabulary.to_csv(OUT / "vocabulary.csv", index=False)

    # map raw event ids -> unified token ids, with rare types folded into [UNK]
    raw_to_token = np.full(len(event_vocab), token_id["[UNK]"], np.int16)
    for desc, rid in id_of_desc.items():
        if desc in token_id:
            raw_to_token[rid] = token_id[desc]

    # ---- the workshop subsample, and its train/test split ------------------------------
    # Computed identically in BOTH modes so that --all can exclude the test slice.
    #
    #   116,232 usable patients
    #   |-- workshop 50,000
    #   |     |-- train 40,000   -> allowed into pretraining
    #   |     `-- test  10,000   -> NEVER in pretraining
    #   `-- other   66,232       -> allowed into pretraining
    #
    # Pretraining may see the workshop's TRAIN patients: it is self-supervised, never sees
    # a label, and using all available history is what you would do in practice. It must
    # never see the TEST patients, or the encoder has memorised the very sequences whose
    # embeddings are scored in segment 4 and the held-out number stops meaning anything.
    rng = np.random.default_rng(SEED)
    workshop = np.sort(rng.choice(n_patients, size=N_PATIENTS, replace=False))
    shuffled = rng.permutation(workshop)
    n_test = int(round(TEST_FRACTION * N_PATIENTS))
    workshop_test = np.sort(shuffled[:n_test])
    workshop_train = np.sort(shuffled[n_test:])
    if args.all:
        # Pretraining is unsupervised and strictly pre-index, so it can draw on patients the
        # downstream cohort excludes (died before index, already had a severe diagnosis).
        # More data, no leakage. Capped at PRETRAIN_SIZE to keep the run to ~15 minutes.
        usable = np.flatnonzero(n_pre >= MIN_EVENTS)
        chosen = np.setdiff1d(usable, workshop_test, assume_unique=False)
        out_dir, suffix = DERIVED, "-full"
        print(f"PRETRAINING CORPUS: {len(chosen):,} patients — every one of the "
              f"{len(usable):,} with >={MIN_EVENTS} pre-index events, minus the "
              f"{len(workshop_test):,} held out as the workshop test split")
        assert not np.intersect1d(chosen, workshop_test).size, "test split leaked!"
    else:
        chosen = workshop
        out_dir, suffix = OUT, ""
        print(f"WORKSHOP SUBSAMPLE {N_PATIENTS:,}: "
              f"{len(workshop_train):,} train / {len(workshop_test):,} test")
        for name, ids in [("train", workshop_train), ("test", workshop_test)]:
            ok = eligible[ids]
            print(f"    {name:5s} eligible {ok.sum():,}, "
                  f"positives {(severe_during[ids] & ok).sum():,} "
                  f"({severe_during[ids][ok].mean():.2%})")
    in_subsample = np.zeros(n_patients, bool)
    in_subsample[chosen] = True

    cohort = pd.DataFrame({
        "patient_id": chosen,
        "split": np.where(np.isin(chosen, workshop_test), "test", "train"),
        "eligible": eligible[chosen],
        "severe_within_5y": severe_during[chosen].astype(np.int8),
        "age_at_index": age_at_index[chosen].round(2),
        "sex": patients.GENDER.to_numpy()[chosen],
        "income": patients.INCOME.to_numpy()[chosen],
        "marital": patients.MARITAL.to_numpy()[chosen],
        "race": patients.RACE.to_numpy()[chosen],
        "ethnicity": patients.ETHNICITY.to_numpy()[chosen],
        "county": patients.COUNTY.to_numpy()[chosen],
        "n_events_before_index": n_pre[chosen],
        "died_before_index": np.isfinite(death_day[chosen]) & (death_day[chosen] <= index_day),
    })
    cohort.to_parquet(out_dir / f"cohort{suffix}.parquet", index=False)

    # ---- pre-index sequences, most recent MAX_EVENTS events, chronological order --------
    keep = pre & in_subsample[pid]
    order = np.lexsort((day[keep], pid[keep]))
    spid, sday = pid[keep][order], day[keep][order]
    stok = raw_to_token[raw_eid[keep][order]]

    # position of each event counted back from that patient's most recent one (0 = latest)
    boundaries = np.flatnonzero(np.diff(spid, prepend=-1))
    ends = np.append(boundaries[1:], len(spid))
    rank_from_end = (np.repeat(ends, ends - boundaries) - 1) - np.arange(len(spid))
    windowed = rank_from_end < MAX_EVENTS

    # Days are stored as a small non-negative offset BEFORE the index date rather than an
    # absolute date: it halves the column and it is the form the timing ablation wants.
    seq = (pd.DataFrame({"patient_id": spid[windowed],
                         "days_before_index": (index_day - sday[windowed]).astype(np.int16),
                         "token_id": stok[windowed]})
           .groupby("patient_id")
           .agg(tokens=("token_id", list), days_before_index=("days_before_index", list))
           .reset_index())
    seq["length"] = seq.tokens.map(len).astype(np.int16)

    bg = background.loc[seq.patient_id.to_numpy()]
    seq["background"] = [np.array([token_id[t] for t in row], np.int16)
                         for row in bg.itertuples(index=False)]

    # Written as int16 arrays; left as Python lists parquet stores int64 and the file is
    # 25% larger for no reason.
    seq["tokens"] = seq.tokens.map(lambda t: np.asarray(t, np.int16))
    seq["days_before_index"] = seq.days_before_index.map(lambda d: np.asarray(d, np.int16))
    seq = seq[["patient_id", "background", "tokens", "days_before_index", "length"]]
    seq.to_parquet(out_dir / f"sequences{suffix}.parquet", index=False,
                   compression="zstd")
    print(f"sequences: {len(seq):,} patients, median length {seq.length.median():.0f}, "
          f"{seq.length.sum():,} event tokens + {len(seq) * 6:,} background tokens")

    if args.all:
        for name in [f"sequences{suffix}.parquet", f"cohort{suffix}.parquet"]:
            print(f"  wrote {name}: {(out_dir / name).stat().st_size / 1e6:.1f} MB")
        return

    # ---- raw dated timelines for a handful of patients (notebook b1 builds from these) ---
    raw_ids = np.sort(rng.choice(chosen[eligible[chosen]], size=RAW_SAMPLE, replace=False))
    raw_mask = np.isin(pid, raw_ids)
    raw = pd.DataFrame({
        "patient_id": pid[raw_mask],
        "date": (EPOCH + day[raw_mask].astype("timedelta64[D]")),
        "description": event_vocab.set_index("event_id").description.reindex(
            raw_eid[raw_mask]).to_numpy(),
        "source": event_vocab.set_index("event_id").source.reindex(
            raw_eid[raw_mask]).to_numpy(),
    })
    raw = raw.sort_values(["patient_id", "date"]).reset_index(drop=True)
    raw["description"] = raw.description.astype("category")
    raw["source"] = raw.source.astype("category")
    raw.to_parquet(OUT / "raw-sample.parquet", index=False, compression="zstd")
    print(f"raw sample: {RAW_SAMPLE} patients, {len(raw):,} events "
          f"(full timelines, including after the index date)")

    # ---- the outcome definition, as readable CSV ---------------------------------------
    def patients_ever(code: str) -> int:
        if code not in id_of_desc:
            return 0
        return int(np.unique(pid[raw_eid == id_of_desc[code]]).size)

    rows = [{"code": c, "group": GROUP_OF[c], "in_outcome": True,
             "note": "", "patients_ever": patients_ever(c)} for c in SEVERE_CODES]
    rows += [{"code": c, "group": "excluded", "in_outcome": False, "note": why,
              "patients_ever": patients_ever(c)} for c, why in EXCLUDED.items()]
    pd.DataFrame(rows).to_csv(OUT / "severe-codes.csv", index=False)

    print(f"\nwrote {OUT.relative_to(ROOT)}/")
    total = 0
    for f in sorted(OUT.iterdir()):
        total += f.stat().st_size
        print(f"  {f.name:22s} {f.stat().st_size / 1e6:6.2f} MB")
    print(f"  {'TOTAL':22s} {total / 1e6:6.2f} MB")


if __name__ == "__main__":
    main()
