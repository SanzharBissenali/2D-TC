"""
Compare an NQS run against the ED benchmark.

NQS output (`G-equiv_*.json`) stores the energy trajectory as strings under
"energy"; the converged value is the last entry. ED output (from
exact.lanczos_ed) stores the ground-state energy under "E0".

Example:
    python scripts/compare.py --nqs G-equiv_1_hz0.10.json --ed results/ed/ed_L4_hx0.00_hz0.10.json
"""

import argparse
import json


def final_nqs_energy(nqs_json):
    """Return the converged NQS energy as a float (handles complex-string entries)."""
    with open(nqs_json) as f:
        data = json.load(f)
    return complex(data["energy"][-1]).real


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--nqs", required=True, help="NQS G-equiv_*.json")
    p.add_argument("--ed", required=True, help="ED results json")
    args = p.parse_args()

    e_nqs = final_nqs_energy(args.nqs)
    with open(args.ed) as f:
        e_ed = json.load(f)["E0"]

    abs_err = abs(e_nqs - e_ed)
    rel_err = abs_err / abs(e_ed) if e_ed != 0 else float("nan")
    print(f"E_NQS = {e_nqs:.6f}")
    print(f"E_ED  = {e_ed:.6f}")
    print(f"abs error = {abs_err:.3e}")
    print(f"rel error = {rel_err:.3e}")


if __name__ == "__main__":
    main()
