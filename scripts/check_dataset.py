"""Produce an inventory without changing the original dataset."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from app.datasets.data import inspect_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hash-data", action="store_true", help="Hash source file contents")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    inventory = inspect_dataset(root / "waymo_trainset", hash_files=args.hash_data)
    report = inventory.model_dump()
    report["created_at"] = datetime.now(timezone.utc).isoformat()
    report["inspection_scope"] = "Parquet metadata and business keys; no geometric validation"
    output = root / "runtime" / "dataset-manifest.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"components": inventory.component_count, "files": inventory.file_count,
                      "contexts": len(inventory.contexts), "errors": inventory.errors,
                      "content_hashes": args.hash_data, "report": str(output)}, ensure_ascii=False))
    if inventory.errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
