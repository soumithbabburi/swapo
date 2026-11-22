"""How wrong a materialized allele frequency gets when a cohort grows, measured.

The optimizer charged a staleness penalty that nobody had fitted. This module
fits the part of it that is a fact rather than a preference: the probability
that a stale allele frequency puts a variant in the wrong decision class.

The data is a real cohort expansion. 1000 Genomes released phase 1 with 1,092
people and phase 3 with 2,504. Using the phase 3 genotypes for both (so the
only thing that changes is who is in the cohort, which is exactly what a stale
materialization suffers from), we compute every chr21 frequency on the phase 1
people, add the newcomers one population at a time as they would land, and
count variants whose class moves across the thresholds variant curation uses:

=========  ==========  ===========================================
class      AF          used for
=========  ==========  ===========================================
absent     0           PM2: absent from controls
rare       < 0.1%      PM2 (supporting): rare in controls
low        0.1 to 1%   neither
BS1        1 to 5%     BS1: more common than the disorder allows
BA1        >= 5%       BA1: stand-alone benign
=========  ==========  ===========================================

What a wrong class COSTS is a judgement for the operator (a missed benign
call wastes review time, a missed pathogenic one is worse); the probability of
one is no longer a guess.
"""

from __future__ import annotations

import gzip
import os
from dataclasses import dataclass, field

import numpy as np

__all__ = ["CLASS_EDGES", "classify", "read_panel", "expansion_counts", "flip_curve",
           "calibrate", "flip_probability", "CALIBRATED_1KG_CHR21"]

CLASS_EDGES = (0.0, 0.001, 0.01, 0.05)   # absent | rare | low | BS1 | BA1
CLASS_NAMES = ("absent", "rare", "low", "BS1", "BA1")


def classify(af: np.ndarray) -> np.ndarray:
    """0 absent, 1 rare, 2 low, 3 BS1 range, 4 BA1."""
    out = np.zeros(af.shape, np.int8)
    out[af > 0] = 1
    out[af >= 0.001] = 2
    out[af >= 0.01] = 3
    out[af >= 0.05] = 4
    return out


def read_panel(path: str | os.PathLike) -> dict[str, str]:
    """sample -> population, from a 1000 Genomes .panel file."""
    out = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            f = line.split()
            if len(f) >= 2 and f[0] != "sample":
                out[f[0]] = f[1]
    return out


@dataclass
class Counts:
    """Alt-allele counts per variant for the old cohort and each arriving batch."""

    old_ac: np.ndarray
    old_an: int
    batch_ac: np.ndarray          # variants x batches
    batch_an: np.ndarray          # per batch
    batches: list[str] = field(default_factory=list)
    n_old: int = 0
    n_new: int = 0


def expansion_counts(vcf: str | os.PathLike, old: set[str], panel: dict[str, str],
                     *, max_variants: int | None = None) -> Counts:
    """Stream a phased, GT-only, biallelic-coded VCF and count alt alleles by group.

    Genotypes are read as bytes: ``a|b`` per sample, four bytes with the tab,
    so a whole line's genotypes are one numpy view. Only allele code ``1`` is
    counted as alt, which is right for the decomposed phase 3 release.
    """
    with gzip.open(vcf, "rb") as fh:
        for line in fh:
            if line.startswith(b"#CHROM"):
                samples = line.rstrip(b"\n").decode().split("\t")[9:]
                break
        is_old = np.array([s in old for s in samples])
        new_pops = sorted({panel[s] for s, o in zip(samples, is_old) if not o and s in panel})
        pop_idx = {p: i for i, p in enumerate(new_pops)}
        batch_of = np.array([-1 if o or s not in panel else pop_idx[panel[s]]
                             for s, o in zip(samples, is_old)])
        n = len(samples)
        hap_old = np.repeat(is_old, 2)
        hap_batch = np.repeat(batch_of, 2)
        old_ac, batch_ac = [], []
        for line in fh:
            # The 9 fixed columns end at the 9th tab; genotypes follow.
            k = -1
            for _ in range(9):
                k = line.index(b"\t", k + 1)
            gt = np.frombuffer(line, np.uint8, count=4 * n - 1, offset=k + 1)
            if gt.size < 4 * n - 1 or gt[1] != ord("|"):
                continue              # haploid or unphased record: skip, do not guess
            alleles = np.empty(2 * n, np.uint8)
            alleles[0::2] = gt[0::4]
            alleles[1::2] = gt[2::4]
            alt = alleles == ord("1")
            old_ac.append(int(alt[hap_old].sum()))
            batch_ac.append(np.bincount(hap_batch[(hap_batch >= 0) & alt],
                                        minlength=len(new_pops)))
            if max_variants and len(old_ac) >= max_variants:
                break
    batch_an = np.bincount(batch_of[batch_of >= 0], minlength=len(new_pops)) * 2
    return Counts(np.array(old_ac, np.int32), int(is_old.sum()) * 2,
                  np.array(batch_ac, np.int32), batch_an, new_pops,
                  int(is_old.sum()), int((batch_of >= 0).sum()))


def flip_curve(c: Counts, order: list[int] | None = None) -> list[dict]:
    """Class changes against the stale (old-cohort) AF as batches arrive in ``order``."""
    order = list(range(len(c.batches))) if order is None else order
    stale = classify(c.old_ac / c.old_an)
    ac, an = c.old_ac.astype(np.int64).copy(), c.old_an
    seen = (c.old_ac > 0) | (c.batch_ac.sum(axis=1) > 0)
    out = []
    for i in order:
        ac += c.batch_ac[:, i]
        an += int(c.batch_an[i])
        cur = classify(ac / an)
        flip = (cur != stale) & seen
        common = seen & (cur >= 2)           # AF >= 0.1% now: the variants a curator weighs
        out.append({
            "added": c.batches[i],
            "growth": round(an / c.old_an - 1, 4),
            "flip_rate_all": float(flip.sum() / seen.sum()),
            "flip_rate_af_ge_0_1pct": float((flip & common).sum() / max(common.sum(), 1)),
            "ba1_crossings": int(((cur == 4) != (stale == 4))[seen].sum()),
            "bs1_crossings": int(((cur >= 3) != (stale >= 3))[seen].sum()),
            "absent_to_present": int(((stale == 0) & (cur > 0)).sum()),
            "variants": int(seen.sum()),
        })
    return out


def calibrate(c: Counts, *, permutations: int = 50, seed: int = 11,
              grid: tuple[float, ...] = (0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.4)) -> dict:
    """P(class flip) against cohort growth, over random population arrival orders.

    Cohorts land a population at a time in real projects, and which population
    lands first changes the answer a lot, so the curve is reported as a band
    over many orders rather than as one line.
    """
    rng = np.random.default_rng(seed)
    keys = ("flip_rate_all", "flip_rate_af_ge_0_1pct")
    at = {k: [] for k in keys}
    for _ in range(permutations):
        curve = flip_curve(c, list(rng.permutation(len(c.batches))))
        g = np.array([0.0] + [r["growth"] for r in curve])
        for k in keys:
            y = np.array([0.0] + [r[k] for r in curve])
            at[k].append(np.interp(grid, g, y))
    out = {"growth": list(grid), "variants": int(((c.old_ac > 0) |
                                                  (c.batch_ac.sum(axis=1) > 0)).sum()),
           "old_samples": c.n_old, "new_samples": c.n_new, "permutations": permutations}
    for k in keys:
        a = np.array(at[k])
        out[k] = {"median": np.median(a, 0).round(4).tolist(),
                  "p10": np.percentile(a, 10, 0).round(4).tolist(),
                  "p90": np.percentile(a, 90, 0).round(4).tolist()}
    return out


def flip_probability(growth: float, curve: dict, key: str = "flip_rate_af_ge_0_1pct",
                     band: str = "median") -> float:
    """Interpolated P(wrong class) for a stale AF after the cohort grew by ``growth``."""
    g = [0.0] + list(curve["growth"])
    y = [0.0] + list(curve[key][band])
    return float(np.interp(growth, g, y))


#: Fitted by :func:`calibrate` on 1000 Genomes phase 3 chr21 (1,102,655 variants,
#: 1,033 phase 1 people + 1,471 newcomers in 26 population batches, 50 orders).
#: Source: results/staleness_1kg_chr21.json.
CALIBRATED_1KG_CHR21 = {
    "growth": [0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.4],
    "flip_rate_all": {"median": [0.0291, 0.058, 0.1012, 0.1438, 0.2081, 0.2723, 0.3407, 0.4336],
                      "p10": [0.0198, 0.0418, 0.0877, 0.1219, 0.1953, 0.2621, 0.3266, 0.4321],
                      "p90": [0.035, 0.0673, 0.1215, 0.1609, 0.2299, 0.2987, 0.3546, 0.4358]},
    "flip_rate_af_ge_0_1pct": {"median": [0.0291, 0.0561, 0.1001, 0.1344, 0.1349, 0.1686, 0.1739, 0.2135],
                               "p10": [0.0228, 0.0457, 0.0835, 0.1137, 0.1221, 0.1487, 0.1498, 0.1931],
                               "p90": [0.0493, 0.0979, 0.1307, 0.1803, 0.1808, 0.2124, 0.1967, 0.2219]},
}
