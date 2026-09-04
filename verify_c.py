"""Measure baselines across seeds; focus on setting c in isolation."""
import os, sys
SCAFFOLD = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "nl2sql-ir-boundary", "scaffold")
BASELINES = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "nl2sql-ir-boundary", "baselines")
sys.path.insert(0, SCAFFOLD)

import torch
import torch.nn as nn
import boundary_nl2sql.train as T
from boundary_nl2sql.settings import build_setting


def load_linker(name):
    src = open(os.path.join(BASELINES, name + ".py")).read()
    ns = {"torch": torch, "nn": nn}
    exec(src, ns)
    return ns["SchemaLinker"]


def run_one(LinkCls, sname, seed):
    T.SchemaLinker = LinkCls
    ts, ss, te, se, conn, vocab = build_setting(sname, seed=seed)
    col, tab, cm, ea = T.train_and_eval(te, se, ts, ss, conn, vocab, seed=seed)
    return ea, cm


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--settings", nargs="+", default=["a", "b", "c"])
    args = p.parse_args()

    linkers = {n: load_linker(n) for n in ["weak", "middle", "strong"]}
    for sname in args.settings:
        for seed in args.seeds:
            row = []
            for bname in ["weak", "middle", "strong"]:
                ea, cm = run_one(linkers[bname], sname, seed)
                row.append(f"{bname}={ea:.4f}")
            print(f"{sname} seed={seed}: " + "  ".join(row), flush=True)
