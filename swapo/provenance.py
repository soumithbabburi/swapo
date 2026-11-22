"""Lineage and freshness, so an optimization cannot quietly break reproducibility.

A layout change is harmless: the same rows, arranged differently. A *derived*
dataset is not. A materialized cohort-frequency table or a sample-oriented
projection is a scientific claim with a date on it: it was computed from these
inputs, at these versions, at this time. If new samples land and the projection
is not refreshed, queries keep answering, and they answer with numbers that no
longer match the cohort.

That is why freshness belongs in the optimizer rather than beside it. The cost
of a materialization is not only the rewrite and the storage; it is also the
obligation to refresh it every time its inputs move, and the risk of serving a
stale answer in between. :mod:`swapo.optimizer` charges for all three.

The records follow the shape of W3C PROV (entity, activity, generation, usage)
closely enough to export, without pulling in an RDF stack.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path

__all__ = ["Dataset", "Lineage", "Freshness", "DatasetKind"]


class DatasetKind(str, Enum):
    BASE = "base"            # ingested, the source of truth
    LAYOUT = "layout"        # the same rows, physically rearranged
    DERIVED = "derived"      # a projection or aggregate: can go stale


@dataclass
class Dataset:
    """One node of the lineage graph."""

    name: str
    kind: DatasetKind
    #: Names of the datasets this was computed from.
    inputs: list[str] = field(default_factory=list)
    #: Logical version of the content, bumped whenever rows change.
    version: int = 1
    #: Wall-clock time the content was produced.
    produced_at: float = field(default_factory=time.time)
    n_rows: int = 0
    n_bytes: int = 0
    #: The reference assembly and annotation release the rows were built
    #: against. A layout built on GRCh38 is not valid for GRCh37 data, and an
    #: annotation-derived table is not valid across annotation releases.
    assembly: str = "GRCh38"
    annotation_release: str | None = None
    #: How the dataset was produced, for the audit trail.
    activity: str = ""
    params: dict = field(default_factory=dict)
    #: Version of each input this content was built from. Freshness compares
    #: versions, never clocks: two events inside one clock tick (about 15 ms on
    #: Windows) have equal timestamps, and a clock comparison then reports a
    #: stale derived table as fresh.
    input_versions: dict = field(default_factory=dict)

    def to_prov(self) -> dict:
        """A PROV-shaped record: the entity, the activity, and what it used."""
        return {
            "entity": {
                "id": f"swapo:{self.name}",
                "version": self.version,
                "kind": self.kind.value,
                "assembly": self.assembly,
                "annotationRelease": self.annotation_release,
                "rows": self.n_rows,
                "bytes": self.n_bytes,
            },
            "activity": {
                "id": f"swapo:{self.name}#v{self.version}",
                "type": self.activity or "produce",
                "endedAtTime": self.produced_at,
                "parameters": self.params,
            },
            "used": [f"swapo:{i}" for i in self.inputs],
            "wasGeneratedBy": f"swapo:{self.name}#v{self.version}",
        }


@dataclass
class Freshness:
    """Whether a derived dataset still matches the inputs it was built from."""

    name: str
    stale: bool
    #: Inputs that have moved on since this was produced.
    behind: list[str] = field(default_factory=list)
    #: Seconds since the newest input changed. 0 when fresh.
    lag_seconds: float = 0.0
    reason: str = ""

    def summary(self) -> dict:
        return {"dataset": self.name, "stale": self.stale,
                "behind": self.behind, "lag_seconds": round(self.lag_seconds, 1),
                "reason": self.reason}


class Lineage:
    """The dependency graph, and the freshness questions you can ask of it."""

    def __init__(self) -> None:
        self._nodes: dict[str, Dataset] = {}

    # -- building -----------------------------------------------------------

    def register(self, ds: Dataset) -> Dataset:
        for i in ds.inputs:
            if i not in self._nodes:
                raise KeyError(f"{ds.name} declares unknown input {i!r}")
        if not ds.input_versions:
            ds.input_versions = {i: self._nodes[i].version for i in ds.inputs}
        self._nodes[ds.name] = ds
        return ds

    def bump(self, name: str, *, n_rows: int | None = None,
             activity: str = "append") -> Dataset:
        """Record that a dataset's content changed. Everything downstream is now stale."""
        ds = self._nodes[name]
        ds.version += 1
        ds.produced_at = time.time()
        ds.activity = activity
        # Rebuilding a derived dataset means it now reflects its inputs as they are.
        ds.input_versions = {i: self._nodes[i].version for i in ds.inputs}
        if n_rows is not None:
            ds.n_rows = n_rows
        return ds

    def get(self, name: str) -> Dataset:
        return self._nodes[name]

    def __contains__(self, name: object) -> bool:
        return name in self._nodes

    # -- questions ----------------------------------------------------------

    def descendants(self, name: str) -> list[str]:
        """Everything that would need refreshing if ``name`` changed."""
        out, frontier = [], [name]
        while frontier:
            cur = frontier.pop()
            for n, ds in self._nodes.items():
                if cur in ds.inputs and n not in out:
                    out.append(n)
                    frontier.append(n)
        return out

    def freshness(self, name: str) -> Freshness:
        ds = self._nodes[name]
        if ds.kind is not DatasetKind.DERIVED:
            return Freshness(name, False, reason="not a derived dataset")
        behind, newest = [], 0.0
        for i in ds.inputs:
            src = self._nodes[i]
            if src.version != ds.input_versions.get(i, src.version):
                behind.append(i)
                newest = max(newest, src.produced_at)
        if not behind:
            return Freshness(name, False, reason="up to date with every input")
        return Freshness(
            name, True, behind=behind, lag_seconds=max(0.0, newest - ds.produced_at),
            reason=f"inputs changed after this was built: {', '.join(behind)}",
        )

    def stale(self) -> list[Freshness]:
        return [f for f in (self.freshness(n) for n in self._nodes) if f.stale]

    def assembly_conflicts(self) -> list[str]:
        """Derived datasets whose assembly does not match an input's.

        Coordinates are meaningless across assemblies, so this is a correctness
        check, not a performance one. It is cheap and it catches a mistake that
        otherwise shows up as quietly wrong results.
        """
        bad = []
        for n, ds in self._nodes.items():
            for i in ds.inputs:
                if self._nodes[i].assembly != ds.assembly:
                    bad.append(f"{n} ({ds.assembly}) built from {i} "
                               f"({self._nodes[i].assembly})")
        return bad

    # -- export -------------------------------------------------------------

    def to_prov(self) -> dict:
        return {
            "@context": {"prov": "http://www.w3.org/ns/prov#",
                         "swapo": "https://example.org/swapo#"},
            "graph": [ds.to_prov() for ds in self._nodes.values()],
        }

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_prov(), indent=2), encoding="utf-8")

    def summary(self) -> list[dict]:
        return [
            {"name": n, "kind": d.kind.value, "version": d.version,
             "rows": d.n_rows, "inputs": d.inputs,
             "stale": self.freshness(n).stale}
            for n, d in self._nodes.items()
        ]
