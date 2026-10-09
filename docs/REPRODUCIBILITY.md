# Reproduction guide

## Read or run

Use `notebooks/technical-paper.ipynb` for the complete paper and executable appendix. It contains 83 cells, including 45 code cells, and four embedded Markdown figure attachments. `technical-paper-executed.ipynb` preserves the executed notebook delivered in the original bundle. `kaggle-ready.ipynb` uses uploaded figure URLs to meet Kaggle’s source-size limit; its 45 code cells are identical to the portable notebook.

The latest Kaggle execution verified here is [Version 5](https://www.kaggle.com/code/abhishekd2512/relative-mirror-recovery-and-rookie-protection?scriptVersionId=356589018), successful in 301.8 seconds on October 9, 2026. The repository packaging does not claim a new scientific rerun.

## Official inputs

Join/access the [NFL Big Data Bowl 2027 competition](https://www.kaggle.com/competitions/nfl-big-data-bowl-2027) and obtain:

```text
players.csv
combine_results.csv
combine_tracking.csv
games.csv
player_play.csv
```

Keep these in one directory outside the repository and outside any generated run directory. The notebook uses `NFL_DATA_DIR` or discovers a single suitable Kaggle input directory. Raw data, participant-level generated tables, and private credentials are not distributed here.

## Local execution

Python 3.11+ is intended. Install the packages in `requirements.txt` in a fresh virtual environment. The requirements describe compatible package families, not an exact lock of the verified Kaggle image. The recorded Kaggle environment included NumPy 2.1.3, pandas 2.3.3, and SciPy 1.16.3; see the published run logs for the complete runtime record.

```bash
python -m venv .venv
# Activate .venv using your operating system's normal command.
python -m pip install -r requirements.txt
python scripts/verify_repository.py
python scripts/reproduce.py --data-dir /absolute/path/to/official-data
```

For Windows, quote paths containing spaces:

```powershell
python scripts/reproduce.py --data-dir "D:\NFL BDB 2027\official-data"
```

The runner validates input presence, starts a fresh Python kernel, and executes all cells. Every invocation gets a unique directory under `runs/` by default. The notebook creates its own `paper_reproduction_run/` subdirectory and confines its output cleanup there. The raw-data directory is passed as `NFL_DATA_DIR`.

On success, the runner writes `executed.ipynb` in the invocation directory. On failure it preserves the partially executed notebook as `failed.ipynb` for inspection. Generated files are ignored by Git. Do not commit a run directory wholesale: standalone modules can produce participant-level diagnostic tables before the complete notebook’s publication cleanup.

## Readable code

`src/functional/` contains six preserved modules and the fixed functional fold assignments. `src/reproduction_assets/` contains the original frozen cells, verified reference modules, readable transport appendix, figure rendering, and configuration fixtures. Keep filenames and relative fixture locations intact.

The notebook is the complete orchestration path. Individual module entry points are useful for inspection and targeted exploration; they do not independently reproduce every article result. For example:

```bash
python src/functional/run_canonical38.py --raw-dir /absolute/path/to/data --fold-map src/functional/common_folds_OL.json --output-dir runs/canonical38
```

This command is an exploratory component and may write local player-level diagnostics. Use the complete notebook’s final publication gate before sharing generated outputs.

## Verification scopes

`scripts/verify_repository.py` uses only Python’s standard library. It verifies every manifest-listed file, notebook cell counts, preserved original source hashes, the equality of portable/Kaggle code cells, and embedded figure bytes. It does not fit a model or substitute for execution with official data.

The notebook performs its scientific assertions during execution. The frozen 2025 evaluation remains a replay of a historical reserved test. Later exploratory methods use known outcomes and do not acquire confirmatory status by being rerun.
