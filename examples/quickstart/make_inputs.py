#!/usr/bin/env python
"""Write the inputs the quickstart runs on, so the quickstart needs no download.

Nothing in this repository was runnable without data a reader had to supply. That
is a real gap and this closes the smallest version of it: two TF-MoDISco-shaped
HDF5 files and a project config, generated deterministically, so the commands in
the README can be pasted and will finish.

**These are not real motifs.** The matrices are drawn from a fixed seed; they have
the shape TF-MoDISco emits and none of its biology. They exist so the pipeline can
be seen end to end -- what each stage writes, what it refuses, what a decision
record looks like. Any number the run produces is a property of this generator.
For a run on real discovery output see docs/CASE_STUDY.md.

Usage:  python examples/quickstart/make_inputs.py [OUTDIR]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import h5py
import numpy as np

MOTIF_LEN = 12
SEED = 20260731


def write_pattern(h5, group: str, name: str, rng: np.random.Generator) -> None:
    grp = h5.require_group(group).create_group(name)
    cwm = rng.normal(size=(MOTIF_LEN, 4))
    grp.create_dataset("contrib_scores", data=cwm)
    grp.create_dataset("hypothetical_contribs", data=cwm * 0.5)
    ppm = np.abs(rng.normal(size=(MOTIF_LEN, 4)))
    grp.create_dataset("sequence", data=ppm / ppm.sum(axis=1, keepdims=True))
    grp.create_group("seqlets").create_dataset(
        "n_seqlets", data=np.array(int(rng.integers(200, 900))))


def write_modisco(path: Path, seed: int, n_pos: int = 4, n_neg: int = 2) -> Path:
    """One discovery output: a few positive patterns and a few negative ones."""
    rng = np.random.default_rng(seed)
    with h5py.File(path, "w") as h5:
        for i in range(n_pos):
            write_pattern(h5, "pos_patterns", f"pattern_{i}", rng)
        for i in range(n_neg):
            write_pattern(h5, "neg_patterns", f"pattern_{i}", rng)
    return path


def main(out_dir: str = "quickstart_inputs") -> int:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Two analyses of the same peak universe by different models. That is the
    # situation the package is for: the same regions, two attribution methods,
    # and no way to tell in advance which motifs are the same motif.
    analyses = []
    for i, (analysis_id, model) in enumerate((("modelA_atac", "modelA"),
                                              ("modelB_atac", "modelB"))):
        path = write_modisco(out / f"{analysis_id}.modisco.h5", seed=SEED + i)
        analyses.append({
            "id": analysis_id,
            "model": model,
            "readout": "atac",
            # Alphanumeric: `ingest` refuses anything else, because a union id is
            # declared and never parsed out of a filename or an analysis id.
            "union_id": f"union{model[-1]}",
            "context": "promoter",
            "modisco_h5": str(path.resolve()),
        })

    config = out / "project.json"
    config.write_text(json.dumps({
        "project": "quickstart",
        "peak_universe_id": "quickstart_universe_v1",
        "analyses": analyses,
    }, indent=2) + "\n", encoding="utf-8")

    print(f"wrote {config}")
    for analysis in analyses:
        print(f"  {analysis['id']}: {analysis['modisco_h5']}")
    print("\nnext:  motifmultiverse ingest "
          f"{config} --out {out / 'registry'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(*sys.argv[1:]))
