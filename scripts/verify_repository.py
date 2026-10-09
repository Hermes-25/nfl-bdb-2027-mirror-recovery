"""Verify preserved research artifacts without official data or third-party packages."""
from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def source(cell: dict) -> str:
    value = cell["source"]
    return "".join(value) if isinstance(value, list) else value


def main() -> None:
    manifest = json.loads((ROOT / "SHA256SUMS.json").read_text(encoding="utf-8"))
    for relative, expected in manifest.items():
        path = (ROOT / relative).resolve()
        require(path.is_relative_to(ROOT), f"Path outside repository: {relative}")
        require(path.is_file(), f"Missing file: {relative}")
        require(digest(path.read_bytes()) == expected, f"Hash mismatch: {relative}")

    notebook = json.loads((ROOT / "notebooks/technical-paper.ipynb").read_text(encoding="utf-8"))
    kaggle = json.loads((ROOT / "notebooks/kaggle-ready.ipynb").read_text(encoding="utf-8"))
    code = [source(c) for c in notebook["cells"] if c["cell_type"] == "code"]
    kaggle_code = [source(c) for c in kaggle["cells"] if c["cell_type"] == "code"]
    require(len(notebook["cells"]) == 83 and len(code) == 45, "Unexpected notebook cell counts")
    require(code == kaggle_code, "Kaggle code differs from portable notebook")
    actual_hashes = {digest(s.encode("utf-8")) for s in code}
    original = notebook["metadata"]["paper_revision"]["original_8_cell_sha256"]
    require(len(original) == 8, "Missing original source hashes")
    require(set(original).issubset(actual_hashes), "Original frozen code was changed")

    figures = notebook["metadata"]["paper_revision"]["attached_figure_sha256"]
    attachments = {}
    for cell in notebook["cells"]:
        for name, payload in cell.get("attachments", {}).items():
            value = payload["image/png"]
            attachments[name] = base64.b64decode("".join(value) if isinstance(value, list) else value)
    require(set(figures) == set(attachments), "Unexpected figure attachment inventory")
    for name, expected in figures.items():
        require(digest(attachments[name]) == expected, f"Embedded figure changed: {name}")
        require(digest((ROOT / "figures" / name).read_bytes()) == expected, f"Figure changed: {name}")

    executed = json.loads((ROOT / "notebooks/technical-paper-executed.ipynb").read_text(encoding="utf-8"))
    errors = [out for cell in executed["cells"] for out in cell.get("outputs", []) if out.get("output_type") == "error"]
    require(not errors, "Delivered executed notebook contains error outputs")
    print(f"PASS: {len(manifest)} file hashes; 83 cells; 45 identical code cells; 8 original sources; 4 embedded figures.")
    print("Integrity verified. Official-data scientific execution is a separate step.")


if __name__ == "__main__":
    main()
