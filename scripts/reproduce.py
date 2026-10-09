"""Execute the complete portable notebook in a fresh, isolated working directory."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = ("players.csv", "combine_results.csv", "combine_tracking.csv", "games.csv", "player_play.csv")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--runs-dir", type=Path, default=ROOT / "runs")
    parser.add_argument("--timeout", type=int, default=1800, help="Per-cell timeout in seconds")
    args = parser.parse_args()
    data = args.data_dir.expanduser().resolve()
    missing = [name for name in REQUIRED if not (data / name).is_file()]
    if missing:
        parser.error("Official input files missing: " + ", ".join(missing))
    if data.is_relative_to(ROOT):
        parser.error("Keep official data outside the repository.")
    if args.timeout <= 0:
        parser.error("Timeout must be positive.")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    work = args.runs_dir.expanduser().resolve() / f"{stamp}-{uuid4().hex[:8]}"
    if data.is_relative_to(work) or work.is_relative_to(data):
        parser.error("The run directory and official-data directory must be separate.")

    import nbformat
    from nbclient import NotebookClient

    work.mkdir(parents=True, exist_ok=False)
    notebook = nbformat.read(ROOT / "notebooks/technical-paper.ipynb", as_version=4)
    previous = os.environ.get("NFL_DATA_DIR")
    os.environ["NFL_DATA_DIR"] = str(data)
    client = NotebookClient(notebook, timeout=args.timeout, kernel_name="python3", allow_errors=False,
                            resources={"metadata": {"path": str(work)}})
    print(f"Executing 45 code cells in {work}", flush=True)
    try:
        client.execute()
    except Exception:
        nbformat.write(notebook, work / "failed.ipynb")
        print(f"Partial execution preserved: {work / 'failed.ipynb'}", flush=True)
        raise
    else:
        nbformat.write(notebook, work / "executed.ipynb")
        print(f"Completed: {work / 'executed.ipynb'}", flush=True)
    finally:
        if previous is None:
            os.environ.pop("NFL_DATA_DIR", None)
        else:
            os.environ["NFL_DATA_DIR"] = previous


if __name__ == "__main__":
    main()
