# Transformer Architectures for Computational Social Science

_A half-day workshop at ICSC 2026 on applying transformer models to longitudinal sequences of socioeconomic and health events._

* **Full details, schedule and background reading: [workshop site](https://carlomarxdk.github.io/workshop/).**
* **When:** 2 September 2026, 9:00 (4 hours), in person 
* **Where:** Nuffield College, University of Oxford 
* **Conference:** [ICSC 2026](https://icsc-conf.github.io/2026/index.html) 

We apply transformers not to text but to **sequences of life events**. The session starts hands-on with language-based transformers, then turns to life-event sequences: how to adapt transformers to them, how to obtain dense representations, how to visualise the resulting embedding space, and how to use those representations for downstream tasks.

## Schedule

TBA

## Preparation

**Assumed:** Python with `pandas`/`numpy`, familiarity with Jupyter or Colab, and basic statistics.

**Not assumed:** PyTorch `torch` or any deep-learning framework, HuggingFace `transformers`, or any prior exposure to embeddings, representation learning or NLP. There are no from-scratch coding exercises: notebook work is run-only or narrow fill-in-the-blank.

**Bring:** a laptop with a browser and a code editor (notebooks also run in [Google Colab](https://colab.research.google.com/)), plus a pen and paper.

### Environment Setup

Notebooks run in Colab with no setup at all. To run them locally, use [uv](https://docs.astral.sh/uv/) (preferred) or [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/getting-started.html).

**1. Install uv** (once, if you do not already have it):

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**2. Set up the environment** (run this in the folder containing this repo):

```bash
uv sync
```

That one command reads `pyproject.toml` and `uv.lock`, creates `.venv/` inside the project, installs the exact pinned versions, and downloads Python 3.12 first if your system does not have it. You do **not** need to create or activate a virtual environment yourself.

**3. Start Jupyter:**

```bash
uv run jupyter lab # if you work from VS Code, you do not need to do that
```

`uv run` executes a command inside the project environment. That is the whole workflow: prefix anything you want to run with `uv run` and it uses the right interpreter and packages, with no activation step:

```bash
uv run script.py     # run a script
uv run python        # open a REPL

# We will work in the Jupyter Notebooks, 
# so I do not think you would need to use this
```

To add a package (this updates `pyproject.toml` and `uv.lock`):

```bash
uv add polars        # a runtime dependency
```

> [!TIP]
> Use `uv add` rather than `pip install`. Installing with `pip` into `.venv` works until the next `uv sync`, which reconciles the environment against the lockfile and drops anything not declared there.

## Data

1. In the first hands-on session, we will use a small dataset of true and false statements: [Trilemma of Truth dataset](https://huggingface.co/datasets/carlomarxx/trilemma-of-truth).
2. The second session uses a pre-generated [Synthea](https://github.com/synthetichealth/synthea) synthetic patient population, distributed via [Harvard Dataverse](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/BWDKXS). No real patient data is used and none is required to participate.
