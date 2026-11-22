"""``python -m swapo.report_rg``: score saved row-group results at several storage speeds."""
import json
import sys
from pathlib import Path

from .objstore import StoreCost
from .rgbench import score

OUT = Path(__file__).resolve().parents[1] / "results"
#: Measured from this machine against the public gnomAD buckets (objstore.measure_remote).
MEASURED = StoreCost(0.170, 5.2e6, "measured: GCS from this machine")
#: Same model, other points on the line. One request is "worth" latency*bandwidth bytes.
SWEEP = [StoreCost(0.020, 100e6, "20 ms, 100 MB/s (2 MB per request)"),
         StoreCost(0.030, 50e6, "30 ms, 50 MB/s (1.5 MB per request)"),
         MEASURED,
         StoreCost(0.010, 25e6, "10 ms, 25 MB/s (0.25 MB per request)")]

if __name__ == "__main__":
    names = sys.argv[1:] or sorted(p.stem.removeprefix("rowgroups_") for p in OUT.glob("rowgroups_*.json"))
    for name in names:
        res = json.loads((OUT / f"rowgroups_{name}.json").read_text())
        print(f"\n== {name}  {res['n_rows']:,} rows  queries {res['queries']}")
        for st in SWEEP:
            print(f"  store: {st.source}")
            for k, rows in res["sets"].items():
                s = score(rows, st, res["current_rows"])
                print(f"    {k:8} chosen {s['chosen']:>9,}  measured best {s['measured_best']:>9,}"
                      f"  loss {s['loss_vs_best']*100:5.1f}%  saving vs 1Mi {s['saving_vs_default']*100:5.1f}%"
                      f"  mean|err| {s['mean_abs_err']*100:4.1f}%  worst {s['worst_err']*100:5.1f}%"
                      f"  rows ok {s['all_rows_correct']}")
