"""``python -m swapo.run_svsplit [table ...]``: the long-record split on SV callsets, measured."""
import json
import sys

from .matrix import load_table, query_sets
from .report_rg import MEASURED
from .run_matrix import DATA, OUT
from .svsplit import split_test

if __name__ == "__main__":
    for name in sys.argv[1:] or ["dgv", "gnomad_sv"]:
        tbl, meta = load_table(name, DATA)
        qs = [q for v in query_sets(name, meta, DATA, n=60).values() for q in v]
        res = {}
        for rg in (122_880, 8_192):
            r = split_test(tbl, qs, MEASURED, rg_rows=rg)
            res[rg] = r
            print(name, "rg", rg, "sep", r["separability"]["bimodality"],
                  "unsplit ms", round(r["unsplit"]["ms"]), flush=True)
            for t in r["thresholds"]:
                print(f"   p{t['percentile']}: >{t['threshold_bp']:,} bp "
                      f"({t['long_fraction'] * 100:.1f}% rows)  {t['ms']:.0f} ms  "
                      f"saving {t['saving'] * 100:+.1f}%  rows ok {t['rows_ok']}", flush=True)
        (OUT / f"svsplit_{name}.json").write_text(json.dumps(res, indent=1))
