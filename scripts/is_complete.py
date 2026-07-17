#!/usr/bin/env python3
"""Exit 0 iff an NQS run is fully complete, so sweep jobs can skip-if-complete
instead of skip-if-file-exists.

main.py writes the run .json incrementally every TDVP step, so a walltime kill
mid-run leaves a partial .json that a naive existence check would wrongly treat as
done. The .mpack is written only after the full TDVP loop finishes (main.py:151),
and the L>=6 Wilson/Renyi order params only after the end-observable block — so
those are reliable "the whole run finished" markers.

Usage: python scripts/is_complete.py <base> <Lx>
  <base> = path to the run file WITHOUT extension, e.g.
           results/nqs/G-equiv_1_L12_hx0.00_hz0.300
"""
import os
import sys
import json


def main():
    base, Lx = sys.argv[1], int(sys.argv[2])
    if not os.path.exists(base + ".mpack"):
        sys.exit(1)                         # TDVP loop didn't finish
    if Lx >= 6:
        try:
            with open(base + ".json") as f:
                op = json.load(f).get("order_params", {})
        except (OSError, json.JSONDecodeError):
            sys.exit(1)                     # missing / half-written
        if not op.get("WilsonBFFM") or not op.get("renyi2_entropy"):
            sys.exit(1)                     # end observables didn't finish
    sys.exit(0)


if __name__ == "__main__":
    main()
