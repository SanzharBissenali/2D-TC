#!/usr/bin/env python
"""Bit-identity regression harness for the QEC sign head + decoder ladder.

Phase 4c graded the EXACT functions in model/decoders.py / model/sign_head.py
against ED, so the compiled kernels that replaced their Python hot paths
(2026-09-02, docs/decoder_scaling.md) must be the SAME deterministic function
of the syndrome / configuration -- tie-breaks included. This script asserts
that, three ways:

  * kernel vs retained Python reference inside the CURRENT module
    (GreedyDecoder._one_py, UnionFindDecoder._one_py, AnchorDecoder
    .corrections_py, _VertexGraph._build_py, loop_parity01 vs
    exact.loops.count_loops);
  * CURRENT module vs the PRE-OPTIMIZATION module (``git show
    <ref-commit>:model/{decoders,sign_head}.py``, loaded side by side), for
    every decoder's corrections, TieSumDecoder.sign01, and
    QECSignHead(g, decoder=name).s01()/features() -- wherever the old code
    is feasible (its 2^F loop table needs F <= 16, its int64 packing N <= 62);
  * where the old head is infeasible (6x6+), the compiled parity is checked
    against count_loops on random sector configs and the decoders against
    their _py references.

Workloads per geometry (1x2 1x4 2x2 2x3 3x3 4x4 6x6):
  (a) exhaustive even-weight syndromes with <= 4 defects (1x2, 2x2);
  (b) random production-shaped configurations: random hexagon flips at
      p=0.5 (a random loop config) then i.i.d. link flips at densities
      0.5% / 2% / 8% / 30%, plus the full (1+N+F) single-link / hexagon-flip
      neighbourhood of a subset (what get_conn_padded feeds the head).

Runs in a numpy/scipy/numba/pymatching venv -- no netket. Prints PASS/FAIL
per (decoder, size, workload) and exits nonzero on any FAIL.

    python scripts/decoder_regress.py [--ref-commit 20bcb14] [--ref-dir DIR]
                                      [--sizes 1x2,2x2,...] [--n 750] [--quick]
"""

import argparse
import importlib.util
import itertools
import os
import subprocess
import sys
import tempfile
import time
import types

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO)

from model.honeycomb_geometry import HoneycombGeometry          # noqa: E402
from model import decoders as D                                 # noqa: E402
from model.sign_head import QECSignHead                         # noqa: E402
from exact.loops import count_loops                             # noqa: E402

DENSITIES = (0.005, 0.02, 0.08, 0.30)
SIZES = ("1x2", "1x4", "2x2", "2x3", "3x3", "4x4", "6x6")
REF_HEAD_MAX_F = 16          # old 2^F loop table
REF_PACK_MAX_V = 62          # old TieSumDecoder.sign01 int packing


# --------------------------------------------------------------------------
# reference (pre-optimization) modules
# --------------------------------------------------------------------------

def materialize_ref(commit, ref_dir):
    os.makedirs(ref_dir, exist_ok=True)
    for src, dst in (("model/decoders.py", "decoders_ref.py"),
                     ("model/sign_head.py", "sign_head_ref.py")):
        path = os.path.join(ref_dir, dst)
        if not os.path.exists(path):
            txt = subprocess.check_output(["git", "-C", REPO, "show", f"{commit}:{src}"])
            with open(path, "wb") as fh:
                fh.write(txt)
    return ref_dir


def load_ref(ref_dir):
    """ref_decoders + ref_sign_head modules; the head is rewired to import
    the REFERENCE decoders (not the current ones) so it is the old code
    end to end."""
    spec = importlib.util.spec_from_file_location(
        "ref_decoders", os.path.join(ref_dir, "decoders_ref.py"))
    ref_dec = importlib.util.module_from_spec(spec)
    sys.modules["ref_decoders"] = ref_dec
    spec.loader.exec_module(ref_dec)

    path = os.path.join(ref_dir, "sign_head_ref.py")
    with open(path) as fh:
        src = fh.read()
    assert "from model.decoders import" in src
    src = src.replace("from model.decoders import", "from ref_decoders import")
    ref_head = types.ModuleType("ref_sign_head")
    ref_head.__file__ = path
    exec(compile(src, path, "exec"), ref_head.__dict__)
    sys.modules["ref_sign_head"] = ref_head
    return ref_dec, ref_head


# --------------------------------------------------------------------------
# workloads
# --------------------------------------------------------------------------

def sector_configs(g, n, rng):
    """(n, N) uint8 zero-charge configs = random hexagon-flip subsets (p=0.5)."""
    flips = rng.random((n, g.n_plaqs)) < 0.5
    bits = np.zeros((n, g.N), dtype=np.uint8)
    for p in range(g.n_plaqs):
        mask = np.zeros(g.N, dtype=np.uint8)
        mask[g.plaq_all[p]] = 1
        bits[flips[:, p]] ^= mask
    return bits


def syndrome_ref(g, bits):
    synd = np.zeros((bits.shape[0], g.n_vertices), dtype=np.uint8)
    for v in range(g.n_vertices):
        for l in g.vertex_all[v]:
            if l != -1:
                synd[:, v] ^= bits[:, int(l)]
    return synd


def exhaustive_syndromes(g, max_defects=4):
    V = g.n_vertices
    rows = [np.zeros(V, dtype=np.uint8)]
    for k in range(2, max_defects + 1, 2):
        for comb in itertools.combinations(range(V), k):
            r = np.zeros(V, dtype=np.uint8)
            r[list(comb)] = 1
            rows.append(r)
    return np.stack(rows)


def neighbourhood(g, cfg):
    """(1+N+F, N) uint8: the config, all single-link flips, all hexagon flips."""
    out = np.tile(cfg, (1 + g.N + g.n_plaqs, 1))
    out[1:1 + g.N] ^= np.eye(g.N, dtype=np.uint8)
    for p in range(g.n_plaqs):
        out[1 + g.N + p, g.plaq_all[p]] ^= 1
    return out


def production_configs(g, n, density, rng, n_nb):
    base = sector_configs(g, n, rng)
    noise = (rng.random((n, g.N)) < density).astype(np.uint8)
    cfg = base ^ noise
    nb = [neighbourhood(g, cfg[i]) for i in range(min(n_nb, n))]
    return cfg, (np.concatenate(nb) if nb else cfg[:0])


# --------------------------------------------------------------------------
# harness
# --------------------------------------------------------------------------

class Report:
    def __init__(self):
        self.rows = []
        self.n_fail = 0

    def add(self, size, item, workload, ok, note=""):
        status = "PASS" if ok is True else ("SKIP" if ok is None else "FAIL")
        if ok is False:
            self.n_fail += 1
        self.rows.append((status, size, item, workload, note))
        print(f"[{status}] {size:>5} {item:<22} {workload:<28} {note}", flush=True)


def eq(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return a.shape == b.shape and bool(np.array_equal(a, b))


def check_size(size, rep, ref, rng, n_per_density, n_nb, quick):
    Lx, Ly = (int(t) for t in size.split("x"))
    g = HoneycombGeometry(Lx, Ly)
    ref_dec, ref_head_mod = ref
    N, V, F = g.N, g.n_vertices, g.n_plaqs
    print(f"\n=== {size}: N={N} V={V} F={F} ===")
    ref_head_ok = F <= REF_HEAD_MAX_F and N <= REF_PACK_MAX_V + 1

    # --- geometry tables: numba BFS vs python BFS vs old module -----------
    t0 = time.perf_counter()
    vg = D._VertexGraph(g)
    t_vg = time.perf_counter() - t0
    vg_py = D._VertexGraph(g, _py=True)
    vg_ref = ref_dec._VertexGraph(g)
    ok = all(eq(getattr(vg, k), getattr(o, k))
             for o in (vg_py, vg_ref)
             for k in ("dist", "_parent_vert", "_parent_link"))
    rep.add(size, "_VertexGraph", "bfs tables", ok, f"build {t_vg*1e3:.1f} ms")

    # --- loop parity kernel vs count_loops ---------------------------------
    if F <= REF_HEAD_MAX_F:
        masks = np.zeros((F, N), dtype=np.uint8)
        for p in range(F):
            masks[p, g.plaq_all[p]] = 1
        S = np.arange(1 << F, dtype=np.int64)
        sel = ((S[:, None] >> np.arange(F, dtype=np.int64)) & 1).astype(np.uint8)
        allsec = (sel.astype(np.int64) @ masks.astype(np.int64) % 2).astype(np.uint8)
        par = D.loop_parity01(allsec, g.link_endpoints, V)
        want = np.array([count_loops(1 - 2 * allsec[i].astype(int), g.link_endpoints) % 2
                         for i in range(allsec.shape[0])], dtype=np.int8)
        rep.add(size, "loop_parity01", f"all 2^{F} sector configs", eq(par, want))
        sec_cfgs = allsec
    else:
        sec_cfgs = sector_configs(g, 20000 if not quick else 3000, rng)
        par = D.loop_parity01(sec_cfgs, g.link_endpoints, V)
        want = np.array([count_loops(1 - 2 * sec_cfgs[i].astype(int), g.link_endpoints) % 2
                         for i in range(sec_cfgs.shape[0])], dtype=np.int8)
        rep.add(size, "loop_parity01", f"{sec_cfgs.shape[0]} random sector cfgs",
                eq(par, want))
    # off-sector detection == count_loops' strict assert
    noisy = sec_cfgs[:2000] ^ (rng.random((min(2000, sec_cfgs.shape[0]), N)) < 0.05
                              ).astype(np.uint8)
    par = D.loop_parity01(noisy, g.link_endpoints, V)
    ok = True
    for i in range(noisy.shape[0]):
        try:
            w = count_loops(1 - 2 * noisy[i].astype(int), g.link_endpoints) % 2
        except AssertionError:
            w = -1
        ok &= int(par[i]) == int(w)
    rep.add(size, "loop_parity01", "off-sector flag (-1)", ok,
            f"{int((par < 0).sum())}/{noisy.shape[0]} off-sector")

    head_new = QECSignHead(g)
    head_ref = ref_head_mod.QECSignHead(g) if ref_head_ok else None

    # --- workloads ----------------------------------------------------------
    workloads = []
    if size in ("1x2", "2x2"):
        synd_a = exhaustive_syndromes(g, 4)
        # configs carrying these syndromes: random loop config XOR an mwpm recovery
        corr = D.MWPMDecoder(g).corrections(synd_a)
        cfg_a = sector_configs(g, synd_a.shape[0], rng) ^ corr
        assert eq(syndrome_ref(g, cfg_a), synd_a)
        workloads.append((f"exhaustive <=4 defects ({synd_a.shape[0]})", cfg_a))
    for dens in DENSITIES:
        n = n_per_density(size, dens)
        cfg, nb = production_configs(g, n, dens, rng, n_nb)
        workloads.append((f"random p={dens:g} ({n})", cfg))
        if nb.shape[0]:
            workloads.append((f"neighbours p={dens:g} ({nb.shape[0]})", nb))

    decs = {nm: D.make_decoder(nm, g, check=True) for nm in D.DECODER_NAMES}
    decs_ref = {nm: ref_dec.make_decoder(nm, g) for nm in D.DECODER_NAMES}
    heads_new = {nm: QECSignHead(g, decoder=nm) for nm in D.DECODER_NAMES}
    heads_ref = ({nm: ref_head_mod.QECSignHead(g, decoder=nm) for nm in D.DECODER_NAMES}
                 if ref_head_ok else None)

    for wname, cfg in workloads:
        synd = head_new._syndrome(cfg)
        rep.add(size, "_syndrome", wname, eq(synd, syndrome_ref(g, cfg)))
        states = 1.0 - 2.0 * cfg.astype(np.float64)
        t_ref = ""

        for nm in ("anchor", "greedy", "unionfind", "mwpm"):
            dec = decs[nm]
            t0 = time.perf_counter()
            c_new = dec.corrections(synd)                     # check=True asserts validity
            t_new = time.perf_counter() - t0
            ok = True
            if nm == "anchor":
                ok &= eq(c_new, dec.corrections_py(synd))
            elif nm in ("greedy", "unionfind"):
                if wname.startswith("random") or wname.startswith("exhaustive") \
                        or wname.startswith("neighbours"):
                    t0 = time.perf_counter()
                    ok &= eq(c_new, dec.corrections_py(synd))
                    t_ref = f" py {(time.perf_counter()-t0)/max(synd.shape[0],1)*1e6:.0f}us"
            ok &= eq(c_new, decs_ref[nm].corrections(synd))  # old module (cached path)
            rep.add(size, nm, wname, ok,
                    f"{t_new/max(synd.shape[0],1)*1e6:.2f} us/row{t_ref}")
            t_ref = ""

        # tie_sum: sign01 vs old module (V <= 62), plus row-order / batch-split invariance
        ts = decs["tie_sum"]
        s_new = ts.sign01(cfg, synd, head_new._parity01_bits)
        perm = rng.permutation(cfg.shape[0])
        s_perm = ts.sign01(cfg[perm], synd[perm], head_new._parity01_bits)
        ok = all(eq(a[perm], b) for a, b in zip(s_new, s_perm))
        half = cfg.shape[0] // 2
        s_split = [np.concatenate([a, b]) for a, b in zip(
            ts.sign01(cfg[:half], synd[:half], head_new._parity01_bits),
            ts.sign01(cfg[half:], synd[half:], head_new._parity01_bits))]
        ok &= all(eq(a, b) for a, b in zip(s_new, s_split))
        if V <= REF_PACK_MAX_V and head_ref is not None:
            s_ref = decs_ref["tie_sum"].sign01(cfg, synd, head_ref._parity01_bits)
            ok &= all(eq(a, b) for a, b in zip(s_new, s_ref))
            rep.add(size, "tie_sum", wname, ok,
                    f"fallback {int(s_new[1].sum())} cancelled {int(s_new[3].sum())}")
        else:
            rep.add(size, "tie_sum", wname, ok, "vs old: SKIP (V>62); perm/split invariance")

        # heads: s01 per decoder, new vs old
        for nm in D.DECODER_NAMES:
            hn = heads_new[nm]
            hn.pop_head_stats()                      # reset the accounting window
            t0 = time.perf_counter()
            s_n = hn.s01(states)
            t_n = time.perf_counter() - t0
            dt, nc = hn.pop_head_stats()
            ok = (nc == cfg.shape[0]) and 0 < dt <= t_n
            if heads_ref is not None:
                s_r = heads_ref[nm].s01(states)
                ok &= eq(s_n, s_r)
                rep.add(size, f"head[{nm}].s01", wname, ok,
                        f"{t_n/cfg.shape[0]*1e6:.2f} us/cfg")
            else:
                # no old head: cross-check against count_loops on the recovered configs
                if nm == "tie_sum":
                    rep.add(size, f"head[{nm}].s01", wname, ok,
                            "vs old: SKIP (2^F table infeasible); stats ok")
                else:
                    if nm == "mwpm":
                        corr = hn._matching.decode_batch(synd).astype(np.uint8)
                    else:
                        corr = hn._alt.corrections(synd)
                    rec = cfg ^ corr
                    want = np.array([count_loops(1 - 2 * rec[i].astype(int),
                                                 g.link_endpoints) % 2
                                     for i in range(rec.shape[0])], dtype=np.float64)
                    ok &= eq(s_n, want)
                    rep.add(size, f"head[{nm}].s01", wname, ok,
                            f"{t_n/cfg.shape[0]*1e6:.2f} us/cfg (vs count_loops)")
        # features (residual arm, mwpm only)
        f_n = head_new.features(states)
        if head_ref is not None:
            rep.add(size, "head.features", wname, eq(f_n, head_ref.features(states)))
        else:
            ok = eq(f_n[:, 0], heads_new["mwpm"].s01(states)) and \
                eq(f_n[:, 1:1 + V], synd.astype(np.float64))
            rep.add(size, "head.features", wname, ok, "vs old: SKIP; s/d columns consistent")

    # sign_pm1 consistency
    s = heads_new["mwpm"].s01(states)
    rep.add(size, "head.sign_pm1", "== 1-2*s01",
            eq(heads_new["mwpm"].sign_pm1(states), 1.0 - 2.0 * s))


def scalability(rep):
    """Construction + first-call cost of the production head at 12x12."""
    for size in ("8x8", "12x12"):
        Lx, Ly = (int(t) for t in size.split("x"))
        g = HoneycombGeometry(Lx, Ly)
        t0 = time.perf_counter()
        head = QECSignHead(g)
        t_build = time.perf_counter() - t0
        rng = np.random.default_rng(7)
        cfg = sector_configs(g, 4000, rng)
        cfg ^= (rng.random(cfg.shape) < 0.01).astype(np.uint8)
        states = 1.0 - 2.0 * cfg
        t0 = time.perf_counter()
        s = head.s01(states)
        t_call = time.perf_counter() - t0
        # parity vs count_loops on the recovered configs (independent counter)
        corr = head._matching.decode_batch(head._syndrome(cfg)).astype(np.uint8)
        rec = cfg ^ corr
        want = np.array([count_loops(1 - 2 * rec[i].astype(int), g.link_endpoints) % 2
                         for i in range(rec.shape[0])], dtype=np.float64)
        rep.add(size, "head[mwpm]", "build + s01 vs count_loops", eq(s, want),
                f"build {t_build:.3f} s (N={g.N}, F={g.n_plaqs}); "
                f"s01 {t_call/cfg.shape[0]*1e6:.2f} us/cfg on {cfg.shape[0]}")
        ok = t_build < 2.0
        rep.add(size, "head[mwpm]", "construction < 2 s", ok, f"{t_build:.3f} s")
        for nm in ("anchor", "greedy", "unionfind"):
            t0 = time.perf_counter()
            h = QECSignHead(g, decoder=nm)
            t_b = time.perf_counter() - t0
            t0 = time.perf_counter()
            s2 = h.s01(states)
            t_c = time.perf_counter() - t0
            c = h._alt.corrections(head._syndrome(cfg))
            rec = cfg ^ c
            want = np.array([count_loops(1 - 2 * rec[i].astype(int), g.link_endpoints) % 2
                             for i in range(rec.shape[0])], dtype=np.float64)
            rep.add(size, f"head[{nm}]", "build + s01 vs count_loops", eq(s2, want),
                    f"build {t_b:.3f} s; s01 {t_c/cfg.shape[0]*1e6:.2f} us/cfg")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ref-commit", default="20bcb14",
                    help="git commit holding the pre-optimization modules")
    ap.add_argument("--ref-dir", default=None,
                    help="dir with decoders_ref.py + sign_head_ref.py (default: "
                         "materialize from --ref-commit into a temp dir)")
    ap.add_argument("--sizes", default=",".join(SIZES))
    ap.add_argument("--n", type=int, default=750,
                    help="random configs per density per size (4 densities => 3000/size)")
    ap.add_argument("--n-nb", type=int, default=8,
                    help="configs per density whose full (1+N+F) neighbourhood is tested")
    ap.add_argument("--quick", action="store_true", help="smaller budgets")
    ap.add_argument("--no-scale", action="store_true", help="skip the 8x8/12x12 timing block")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--allow-python", action="store_true",
                    help="do not FAIL when numba is missing (kernels then run as Python)")
    args = ap.parse_args()

    ref_dir = args.ref_dir or materialize_ref(
        args.ref_commit, os.path.join(tempfile.gettempdir(), f"decoder_regress_ref_{args.ref_commit}"))
    ref = load_ref(ref_dir)
    print(f"reference modules: {ref_dir} (commit {args.ref_commit}); "
          f"numba={'on' if D.HAVE_NUMBA else 'OFF (python fallback)'}")

    def n_per_density(size, dens):
        n = args.n if not args.quick else max(args.n // 5, 50)
        if size == "6x6":                       # python references are O(d^3) / slow here
            n = {0.005: n, 0.02: n, 0.08: max(n // 3, 50), 0.30: max(n // 8, 30)}[dens]
        elif size == "4x4" and dens >= 0.30:
            n = max(n // 2, 50)
        return n

    rep = Report()
    rep.add("*", "numba kernels", "compiled (not python fallback)",
            True if D.HAVE_NUMBA else (None if args.allow_python else False))
    rng = np.random.default_rng(args.seed)
    t0 = time.perf_counter()
    for size in args.sizes.split(","):
        check_size(size.strip(), rep, ref, rng, n_per_density, args.n_nb, args.quick)
    if not args.no_scale:
        print("\n=== scalability (no reference; timing + count_loops cross-check) ===")
        scalability(rep)

    n_pass = sum(r[0] == "PASS" for r in rep.rows)
    n_skip = sum(r[0] == "SKIP" for r in rep.rows)
    print(f"\nSUMMARY: {n_pass} PASS, {rep.n_fail} FAIL, {n_skip} SKIP "
          f"({time.perf_counter()-t0:.0f} s)")
    if rep.n_fail:
        for r in rep.rows:
            if r[0] == "FAIL":
                print("  FAIL:", *r[1:])
    sys.exit(1 if rep.n_fail else 0)


if __name__ == "__main__":
    main()
