"""``python -m swapo.run_matrix [table ...]``: run the row-group experiment, save raw results."""
import json
import sys
import time
from pathlib import Path

from .matrix import TABLES, load_table, query_sets
from .rgbench import validate

DATA = Path(__file__).resolve().parents[1] / "swapo_data"
OUT = Path(__file__).resolve().parents[1] / "results"

if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    for name in sys.argv[1:] or TABLES:
        t0 = time.time()
        tbl, meta = load_table(name, DATA)
        sets = query_sets(name, meta, DATA)
        print(name, tbl.num_rows, {k: len(v) for k, v in sets.items()},
              f"loaded {time.time() - t0:.0f}s", flush=True)
        res = validate(tbl, sets)
        res["meta"] = meta
        (OUT / f"rowgroups_{name}.json").write_text(json.dumps(res, indent=1))
        print(name, f"done {time.time() - t0:.0f}s", flush=True)
