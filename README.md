# hpf — Hyperparameter Finder

**Stop guessing search ranges.** `hpf` analyzes your completed Optuna studies and tells you exactly how to adjust your hyperparameter bounds before the next run — with statistical reasoning and optional LLM-powered explanations.

---

## The Problem

When you set up an Optuna study, you pick search ranges somewhat arbitrarily:

```python
lr = trial.suggest_float("lr", 1e-5, 1.0)       # is 1.0 even worth searching?
depth = trial.suggest_int("max_depth", 1, 20)    # are trials wasting time on depth 1–3?
```

After 100+ trials, most of that space was wasted. `hpf` looks at where your best trials actually landed and tells you how to tighten, expand, or rescale each parameter for the next run.

---

## What it does

1. **Loads** your completed Optuna study (SQLite, PostgreSQL, or in-memory)
2. **Identifies** the top-K% best trials (default: top 20%)
3. **Analyzes** each hyperparameter:
   - Boundary detection — did optima cluster near the edge? → **expand**
   - Cluster analysis — did optima occupy only a tiny subregion? → **narrow**
   - Scale detection — do values span multiple orders of magnitude on a linear scale? → **switch to log**
4. **Outputs** a color-coded terminal report with confidence levels and reasoning
5. **Generates** a ready-to-paste Optuna search space you can drop into your objective function
6. **Optionally explains** everything in plain English via a local LLM (Ollama) or API (OpenAI / Groq / Together / any OpenAI-compatible endpoint)

---

## Installation

```bash
pip install optuna-hpf
```

Or from source:

```bash
git clone https://github.com/ParamChordiya/hpf.git
cd hpf
pip install -e .
```

---

## Quick start

```bash
# First-time setup (configure your LLM — optional but recommended)
hpf setup

# Analyze a study stored in SQLite
hpf analyze --study my_xgboost_study --file study.db

# Analyze with model type hint for better LLM tips
hpf analyze --study my_study --file study.db --model-type xgboost

# Skip LLM, just show statistical analysis
hpf analyze --study my_study --file study.db --no-llm

# Use a custom storage URL (PostgreSQL, etc.)
hpf analyze --study my_study --storage "postgresql://user:pass@host/db"
```

---

## LLM setup

`hpf` supports two modes for AI-powered explanations:

### Option 1: Local model via Ollama (free, private)

1. Install [Ollama](https://ollama.ai)
2. Pull a model: `ollama pull llama3`
3. Run `hpf setup` and select option 1

### Option 2: OpenAI-compatible API

Works with OpenAI, Groq, Together AI, Anyscale, or any provider with an OpenAI-compatible endpoint.

Run `hpf setup` and select option 2. You'll be prompted for:
- API key
- Base URL (leave blank for OpenAI, or enter e.g. `https://api.groq.com/openai/v1`)
- Model name (default: `gpt-4o-mini`)

Config is saved to `~/.hpf/config.json`.

---

## Example output

```
╭─────────────────────────────────────────╮
│   HPF — Hyperparameter Finder  v0.1.0   │
╰─────────────────────────────────────────╯

Study: my_xgboost_study  │  300 trials  │  best: 0.9412  │  direction: maximize

┌──────────────────┬───────┬──────────────────┬──────────────────┬───────────┬────┐
│ Parameter        │ Type  │ Original         │ Suggested        │ Action    │ ●  │
├──────────────────┼───────┼──────────────────┼──────────────────┼───────────┼────┤
│ learning_rate    │ FLOAT │ [0.001, 1.0]     │ [0.001, 1.0] LOG │ LOG_SCALE │ ●  │
│ max_depth        │ INT   │ [1, 20]          │ [3, 9]           │ NARROW    │ ●  │
│ n_estimators     │ INT   │ [50, 500]        │ [50, 1500]       │ EXPAND    │ ●  │
│ subsample        │ FLOAT │ [0.5, 1.0]       │ [0.5, 1.0]       │ KEEP      │ ●  │
└──────────────────┴───────┴──────────────────┴──────────────────┴───────────┴────┘

╭─── Updated Search Space ────────────────────────────────────────────╮
│ def objective(trial):                                               │
│     learning_rate = trial.suggest_float("learning_rate",           │
│         0.001, 1.0, log=True)                                       │
│     max_depth = trial.suggest_int("max_depth", 3, 9)               │
│     n_estimators = trial.suggest_int("n_estimators", 50, 1500)     │
│     subsample = trial.suggest_float("subsample", 0.5, 1.0)         │
╰─────────────────────────────────────────────────────────────────────╯
```

---

## Python API

You can also use `hpf` directly in your code.

> **Note:** The PyPI package is `optuna-hpf` (`pip install optuna-hpf`), but the Python module is `hpf`. This is intentional — same pattern as `scikit-learn` / `sklearn`.

```python
import optuna
from hpf.analyzer import StudyAnalyzer
from hpf.range_suggester import RangeSuggester
from hpf.formatters.report import Reporter
from hpf.models import HPFReport
from hpf.formatters.code_gen import generate_optuna_code

study = optuna.load_study(study_name="my_study", storage="sqlite:///study.db")

analyzer = StudyAnalyzer(top_k_percent=20)
analysis = analyzer.analyze(study, model_type="xgboost")

suggester = RangeSuggester()
suggestions = suggester.suggest(analysis)

code = generate_optuna_code(suggestions, study.study_name)
report = HPFReport(analysis=analysis, suggestions=suggestions, llm_explanation=None, optuna_code=code)

Reporter().print_report(report)
```

---

## How the analysis works

| Signal | Action | Example |
|---|---|---|
| Best trials' mean/median within 10% of a boundary | **EXPAND** | `n_estimators` best at 490–500 → expand upper bound |
| Best trials span <30% of original range | **NARROW** | `max_depth` best in 3–7 out of 1–20 → narrow to 3–9 |
| Values span >100× ratio on linear scale | **LOG SCALE** | `lr` from 0.001–0.5 on linear → switch to log |
| Well-distributed across range | **KEEP** | `subsample` best from 0.6–0.95 → keep as is |

---

## Requirements

- Python 3.10+
- Optuna 3.0+

Optional:
- `ollama` Python package + [Ollama](https://ollama.ai) running locally
- `openai` package + API key for cloud LLM

---

## License

MIT
