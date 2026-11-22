"""``python -m swapo.run_formats [table ...]``: Delta and Iceberg, end to end, on real tables."""
import json
import sys
import time

from .formatbench import run_format
from .matrix import load_table, query_sets
from .report_rg import MEASURED
from .run_matrix import DATA, OUT

if __name__ == "__main__":
    for name in sys.argv[1:] or ["gnomad_window"]:
        tbl, meta = load_table(name, DATA)
        sets = query_sets(name, meta, DATA, n=40)
        for fmt in ("delta", "iceberg"):
            t0 = time.time()
            r = run_format(fmt, tbl, sets, MEASURED)
            r["table"] = name
            (OUT / f"format_{fmt}_{name}.json").write_text(json.dumps(r, indent=1))
            b, a = r["before"], r["after"]
            print(f"{name} {fmt}: {b['row_group_rows']:,} -> {r['recommended']:,} rows/rg, "
                  f"planner agree {b['planner_agreement']} / {a['planner_agreement']}, "
                  f"{b['requests']:.1f}->{a['requests']:.1f} req, "
                  f"{b['megabytes']:.2f}->{a['megabytes']:.2f} MB, "
                  f"{b['ms']:.0f}->{a['ms']:.0f} ms measured "
                  f"({r['measured_saving'] * 100:.0f}% vs predicted "
                  f"{r['predicted_saving'] * 100:.0f}%), rows ok {b['rows_correct']}/{a['rows_correct']}, "
                  f"rollback rows {r['rollback_rows']:,}/{r['rows']:,}  [{time.time() - t0:.0f}s]",
                  flush=True)
