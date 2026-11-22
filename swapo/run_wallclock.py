"""``python -m swapo.run_wallclock [table]``: elapsed time, not modelled time.

Every response from the local server is delayed by the latency and per-stream
bandwidth measured against GCS, and DuckDB's own elapsed time per query is
recorded at the writer default and at the recommended row-group size. DuckDB
overlaps some requests, so this is the check that the serial cost model does
not overstate the gain.
"""
import json
import shutil
import sys
import tempfile
from pathlib import Path

from .matrix import load_table, query_sets
from .report_rg import MEASURED
from .rgbench import _truth, measure, write_files
from .rowgroups import _contigs, choose_row_group_size, sort_genomic
from .run_matrix import DATA, OUT

if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else "gnomad_window"
    tbl, meta = load_table(name, DATA)
    qs = [q for v in query_sets(name, meta, DATA, n=15).values() for q in v]
    ordered = sort_genomic(tbl)
    truth = _truth(_contigs(ordered), qs)
    choice = choose_row_group_size(tbl, qs, MEASURED)
    out = {"table": name, "queries": len(qs), "store": MEASURED.summary(), "sizes": {}}
    base = Path(tempfile.mkdtemp())
    try:
        for r in (choice.current_rows, choice.rg_rows):
            files = write_files(ordered, base / str(r), file_rows=4_194_304, rg_rows=r)
            m = measure(base / str(r), files, qs, truth=truth, rg_rows=r,
                        latency_s=MEASURED.latency_s, bandwidth_bps=MEASURED.bandwidth_bps)
            modelled = MEASURED.seconds(m.requests, m.bytes) * 1e3
            out["sizes"][r] = {"wall_ms": m.wall_ms, "serial_model_ms": modelled,
                               "requests": m.requests, "megabytes": m.bytes / 1e6,
                               "rows_ok": m.rows_ok}
            print(f"{name} {r:>9,} rows/rg: wall {m.wall_ms:.0f} ms per query, serial "
                  f"model {modelled:.0f} ms, {m.requests:.1f} req, {m.bytes / 1e6:.2f} MB, "
                  f"rows ok {m.rows_ok}", flush=True)
    finally:
        shutil.rmtree(base, ignore_errors=True)
    a, b = (out["sizes"][k]["wall_ms"] for k in (choice.current_rows, choice.rg_rows))
    out["wall_saving"] = 1 - b / a
    print(f"elapsed saving {out['wall_saving'] * 100:.0f}% (modelled {choice.improvement * 100:.0f}%)")
    (OUT / f"wallclock_{name}.json").write_text(json.dumps(out, indent=1))
