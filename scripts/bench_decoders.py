#!/usr/bin/env python
"""Microbenchmark: QEC sign head x decoder ladder vs system size (Phase 4c scaling).

Times ``QECSignHead.s01`` (model/sign_head.py) for each decoder in the ladder
(mwpm, anchor, greedy, unionfind, tie_sum) on honeycomb patches of growing size,
producing the data for "time per head evaluation (and per VMC step) vs number of
qubits N", one curve per decoder. Host-only: numpy/scipy/pymatching(/numba via the
head) -- NO netket/jax import, so it runs on the laptop and on a NERSC node alike.
Built against the public API only:

    g = HoneycombGeometry(Lx, Ly); head = QECSignHead(g, decoder=name)
    s = head.s01(states)          # states (B, N) +-1 floats, bit 1 = spin down

Workloads per size (both evaluated in chunks of --chunk rows, netket-style):
  neighborhood  production-shaped: for each of --n_base base configs (random hexagon
                flips w.p. 1/2 => random closed-loop config, then i.i.d. link flips at
                density p), the base config AND its 1+N+F connected configs (identity,
                N single-link flips, F hexagon flips) -- exactly the rows the framed
                Hamiltonian's get_conn_padded hands the head at hx != 0. One cell per
                density in --densities.
  uniform       fully random +-1 configs (stress column, dense defects), same row
                count as a neighborhood pass; density recorded as null.

Timing protocol per (size, decoder, workload, density) "cell": one untimed warm-up
call on a separate small set (numba JIT / caches), then --repeats passes over
freshly generated workloads, each timed chunk by chunk with time.perf_counter.
The per-cell --budget (seconds of timed work) stops a pass when the next chunk is
predicted to exceed it (``truncated: true``, rate from completed chunks) and skips
further repeats when a full extra pass would not fit (``repeats_done`` < repeats).
Safety nets so the whole run always finishes: the chunk is shrunk (powers of two,
>= 16 rows, recorded as ``chunk_used``) when the warm-up rate predicts a full
chunk > budget/4; the whole cell (and head construction) runs under a SIGALRM
hard cap of --hard_cap x budget + 5 s (``hard_timeout: true`` / ``construct_timeout``).
Pure-Python/numpy work is interruptible; a single very long numba/C++ call is not.

Sanity (per size, per decoder, before timing): s01 must return shape (B,) with
values in {0., 1.} on the zero-syndrome base configs, and must agree with the
count_loops parity AND with the mwpm head there (all decoders are exact on the
zero-defect sector). A failure is recorded (``sanity.ok=false``), printed loudly,
skips that decoder's cells at that size, and makes the script exit 1 at the end.

Thread pinning: OMP/MKL/OPENBLAS/NUMBA_NUM_THREADS (and VECLIB/NUMEXPR) are set to
--threads (default 1) BEFORE numpy is imported, so timings are single-thread
reproducible; the values used are recorded in meta.threads_env.

JSON schema (--out; rewritten atomically after every cell => a killed run leaves a
valid partial file):
  meta:     tag, timestamp (UTC ISO), hostname, platform, cpu (platform.processor()
            + /proc/cpuinfo "model name" on Linux / sysctl brand on macOS),
            python, numpy, scipy, numba, pymatching versions, threads_env {var: val},
            code_path (repo root the model/ package was imported from), git_hash
            (`git rev-parse HEAD` there, or --git_hash), git_dirty (list of modified
            paths under model/ exact/, or null if unknown), args (all CLI values),
            step_formula, total_wall_s (final).
  sizes:    {"LxxLy": {Lx, Ly, N, F, V, rows_per_pass = n_base*(1+N+F)}}
  heads:    one entry per (size, decoder): size, decoder, N, F, V, construct_s,
            construct_timeout, error (str|null), sanity {ok, n_checked, detail,
            agree_with_mwpm (bool|null)}, cache_stats_after_sanity (dict|null).
  results:  one entry per cell: size, N, F, V, decoder, workload
            ("neighborhood"|"uniform"), density (float|null), n_base, rows_per_pass,
            chunk (requested), chunk_used, warmup_rows, warmup_s, repeats
            (requested), repeats_done, n_chunks_timed, n_configs_timed,
            timed_s (sum of timed chunk seconds), us_per_config_median /
            us_per_config_min (over per-repeat rates), us_per_config_per_repeat
            [..], us_per_config_pooled (timed_s/n_configs_timed),
            s_per_step_equiv (= us_median*8192*(2+N+F)/1e6), s_per_step_equiv_min,
            mean_defects (mean syndrome weight over the workload rows),
            zero_syndrome_frac, truncated, hard_timeout, error (str|null),
            cache_stats_before / cache_stats_after (dict|null -- head.cache_stats()
            if the head exposes it, else lengths of any *cache* attributes on the
            head / its decoder), wall_s (cell wall incl. generation + warm-up).
  sanity_failures: int (script exit code is 1 if > 0).

Example:
  PYTHONPATH=. python scripts/bench_decoders.py --sizes 1x2,2x2,2x3 \
      --decoders mwpm,greedy --budget 30 --n_base 16 --repeats 1 \
      --tag laptop --out results/diagnostics/bench_decoders_laptop.json
"""

import argparse
import os
import sys


def _pre_parse_threads(argv):
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--threads", type=int, default=1)
    a, _ = p.parse_known_args(argv)
    return max(1, int(a.threads))


_THREADS = _pre_parse_threads(sys.argv[1:])
_THREAD_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "NUMBA_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")
for _k in _THREAD_VARS:
    os.environ[_k] = str(_THREADS)

import json          # noqa: E402
import platform      # noqa: E402
import signal        # noqa: E402
import socket        # noqa: E402
import statistics    # noqa: E402
import subprocess    # noqa: E402
import time          # noqa: E402
from datetime import datetime, timezone  # noqa: E402

import numpy as np   # noqa: E402

_SCRIPT_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
try:
    import model  # noqa: F401  (resolved via PYTHONPATH -- the code under test)
except ImportError:
    sys.path.insert(0, _SCRIPT_REPO)
    import model  # noqa: F401
from model.honeycomb_geometry import HoneycombGeometry  # noqa: E402
from model.sign_head import QECSignHead                  # noqa: E402
from exact.loops import count_loops                      # noqa: E402

CODE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(model.__file__)))
N_SAMPLES_PER_STEP = 8192
STEP_FORMULA = ("s_per_step_equiv = us_per_config * 8192 * (2 + N + F) / 1e6  "
                "(head on the 8192 samples + their 8192*(1+N+F) connected rows)")


# ----------------------------------------------------------------------------
# hard-cap machinery
# ----------------------------------------------------------------------------
class CellTimeout(Exception):
    pass


class _Alarm:
    """SIGALRM-based wall cap for one block (Unix; main thread)."""

    def __init__(self, seconds):
        self.seconds = float(seconds)
        self._prev = None

    def _handler(self, signum, frame):
        raise CellTimeout(f"hard cap {self.seconds:.1f}s exceeded")

    def __enter__(self):
        self._prev = signal.signal(signal.SIGALRM, self._handler)
        signal.setitimer(signal.ITIMER_REAL, self.seconds)
        return self

    def __exit__(self, *exc):
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, self._prev)
        return False


# ----------------------------------------------------------------------------
# environment / metadata
# ----------------------------------------------------------------------------
def _run(cmd, cwd=None):
    try:
        return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True,
                              timeout=20).stdout.strip()
    except Exception:
        return ""


def _cpu_model():
    info = {"processor": platform.processor(), "machine": platform.machine()}
    if sys.platform.startswith("linux"):
        try:
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.lower().startswith("model name"):
                        info["model_name"] = line.split(":", 1)[1].strip()
                        break
        except OSError:
            pass
    elif sys.platform == "darwin":
        brand = _run(["sysctl", "-n", "machdep.cpu.brand_string"])
        if brand:
            info["model_name"] = brand
    return info


def _versions():
    out = {"python": platform.python_version(), "numpy": np.__version__}
    for name in ("scipy", "numba", "pymatching"):
        try:
            out[name] = __import__(name).__version__
        except Exception:
            out[name] = None
    return out


def _git_info(code_path, override):
    h = override or _run(["git", "-C", code_path, "rev-parse", "HEAD"]) or None
    dirty = None
    if not override and h:
        st = _run(["git", "-C", code_path, "status", "--porcelain", "--",
                   "model", "exact"])
        # porcelain rows are "XY path"; _run() strips stdout, so split instead of slicing
        dirty = [ln.strip().split(maxsplit=1)[-1] for ln in st.splitlines() if ln.strip()]
    return h, dirty


def _cache_stats(head):
    fn = getattr(head, "cache_stats", None)
    if callable(fn):
        try:
            return fn()
        except Exception:
            pass
    out = {}
    for owner_name, owner in (("head", head), ("alt", getattr(head, "_alt", None))):
        if owner is None:
            continue
        try:
            items = vars(owner).items()
        except TypeError:
            continue
        for k, v in items:
            if "cache" in k.lower() and hasattr(v, "__len__"):
                try:
                    out[f"{owner_name}.{k}"] = len(v)
                except Exception:
                    pass
    return out or None


# ----------------------------------------------------------------------------
# workloads
# ----------------------------------------------------------------------------
def _plaq_incidence(g):
    P = np.zeros((int(g.n_plaqs), int(g.N)), dtype=np.uint8)
    for p in range(int(g.n_plaqs)):
        P[p, np.asarray(g.plaq_all[p], dtype=int)] ^= 1
    return P


def _vertex_incidence_T(g):
    incT = np.zeros((int(g.N), int(g.n_vertices)), dtype=np.float32)
    for l, (u, v) in enumerate(np.asarray(g.link_endpoints, dtype=int)):
        incT[l, u] = 1.0
        incT[l, v] = 1.0
    return incT


def make_zero_syndrome(rng, n_base, P):
    """Random closed-loop configs: XOR of a random hexagon subset (w.p. 1/2)."""
    S = (rng.random((n_base, P.shape[0])) < 0.5).astype(np.int32)
    return ((S @ P.astype(np.int32)) % 2).astype(np.uint8)


def make_neighborhood(rng, n_base, density, P, N):
    """(n_base*(1+N+F), N) uint8 bit rows, base-major: [base, N flips, F hex flips]."""
    base = make_zero_syndrome(rng, n_base, P)
    if density > 0:
        base ^= (rng.random(base.shape) < density).astype(np.uint8)
    nb = np.concatenate([np.zeros((1, N), dtype=np.uint8),
                         np.eye(N, dtype=np.uint8), P], axis=0)
    return (base[:, None, :] ^ nb[None, :, :]).reshape(-1, N)


def make_uniform(rng, n_rows, N):
    return rng.integers(0, 2, size=(n_rows, N), dtype=np.uint8)


def to_states(bits):
    """bits (bit 1 = spin down) -> +-1 float64 spins (netket sample convention)."""
    return (1.0 - 2.0 * bits.astype(np.float64))


def make_workload(kind, rng, n_base, density, P, N):
    if kind == "neighborhood":
        return make_neighborhood(rng, n_base, density, P, N)
    if kind == "uniform":
        return make_uniform(rng, n_base * (1 + N + P.shape[0]), N)
    raise ValueError(kind)


def syndrome_stats(bits, incT):
    """Mean defect count + zero-syndrome fraction via an exact float32 GEMM mod 2."""
    synd = np.rint(bits.astype(np.float32) @ incT) % 2
    w = synd.sum(axis=1)
    return float(w.mean()), float((w == 0).mean())


# ----------------------------------------------------------------------------
# checks
# ----------------------------------------------------------------------------
def _check_output(s, n_rows):
    s = np.asarray(s)
    assert s.shape == (n_rows,), f"s01 shape {s.shape} != {(n_rows,)}"
    assert np.isin(s, (0.0, 1.0)).all(), "s01 values not in {0., 1.}"
    return s


def run_sanity(head, zero_bits, ref, mwpm_ref):
    states = to_states(zero_bits)
    s = _check_output(head.s01(states), states.shape[0])
    n_bad = int((s != ref).sum())
    assert n_bad == 0, (f"{n_bad}/{len(ref)} zero-syndrome configs disagree "
                        f"with the count_loops parity")
    agree = None
    if mwpm_ref is not None:
        agree = bool(np.array_equal(s, mwpm_ref))
        assert agree, "disagrees with the mwpm head on zero-syndrome configs"
    return {"ok": True, "n_checked": int(len(ref)), "detail": None,
            "agree_with_mwpm": agree}, s


# ----------------------------------------------------------------------------
# one timing cell
# ----------------------------------------------------------------------------
def run_cell(head, kind, density, size_info, args, P, incT, seed_tuple):
    N, F = size_info["N"], size_info["F"]
    rows_per_pass = args.n_base * (1 + N + F)
    rec = {
        "size": size_info["label"], "N": N, "F": F, "V": size_info["V"],
        "decoder": head.decoder_name if hasattr(head, "decoder_name") else None,
        "workload": kind, "density": density, "n_base": args.n_base,
        "rows_per_pass": rows_per_pass, "chunk": args.chunk, "chunk_used": None,
        "warmup_rows": None, "warmup_s": None, "repeats": args.repeats,
        "repeats_done": 0, "n_chunks_timed": 0, "n_configs_timed": 0,
        "timed_s": 0.0, "us_per_config_median": None, "us_per_config_min": None,
        "us_per_config_per_repeat": [], "us_per_config_pooled": None,
        "s_per_step_equiv": None, "s_per_step_equiv_min": None,
        "mean_defects": None, "zero_syndrome_frac": None,
        "truncated": False, "hard_timeout": False, "error": None,
        "cache_stats_before": _cache_stats(head), "cache_stats_after": None,
        "wall_s": None,
    }
    t_cell = time.perf_counter()
    budget = float(args.budget)
    per_repeat = []
    try:
        with _Alarm(args.hard_cap * budget + 5.0):
            # -- warm-up (separate configs; untimed; triggers numba JIT) ------
            rng_w = np.random.default_rng(seed_tuple + (7919,))
            Xw = make_workload(kind, rng_w, max(1, args.warmup // (1 + N + F) + 1),
                               density, P, N)[: args.warmup]
            Sw = to_states(Xw)
            t0 = time.perf_counter()
            _check_output(head.s01(Sw), Sw.shape[0])
            tw = time.perf_counter() - t0
            rec["warmup_rows"], rec["warmup_s"] = int(Sw.shape[0]), tw
            rate = tw / Sw.shape[0]                    # s per row (running estimate)

            chunk = int(args.chunk)
            while chunk > 16 and rate * chunk > budget / 4:
                chunk //= 2
            rec["chunk_used"] = chunk

            # -- timed passes -----------------------------------------------
            elapsed = 0.0
            rows_total = 0
            for r in range(args.repeats):
                rng = np.random.default_rng(seed_tuple + (r,))
                X = make_workload(kind, rng, args.n_base, density, P, N)
                if r == 0:
                    rec["mean_defects"], rec["zero_syndrome_frac"] = \
                        syndrome_stats(X, incT)
                S = to_states(X)
                n_rows = S.shape[0]
                t_pass, rows_pass = 0.0, 0
                for start in range(0, n_rows, chunk):
                    n = min(chunk, n_rows - start)
                    first = rows_total + rows_pass == 0
                    if not first and elapsed + t_pass + rate * n > budget:
                        rec["truncated"] = True
                        break
                    t0 = time.perf_counter()
                    s = head.s01(S[start:start + n])
                    dt = time.perf_counter() - t0
                    _check_output(s, n)
                    t_pass += dt
                    rows_pass += n
                    rec["n_chunks_timed"] += 1
                    rate = (elapsed + t_pass) / (rows_total + rows_pass)
                if rows_pass:
                    per_repeat.append(1e6 * t_pass / rows_pass)
                    rec["repeats_done"] += 1
                elapsed += t_pass
                rows_total += rows_pass
                if rec["truncated"]:
                    break
                if r + 1 < args.repeats and elapsed + rate * n_rows > budget:
                    break                              # no room for another full pass
            rec["timed_s"], rec["n_configs_timed"] = elapsed, int(rows_total)
    except CellTimeout as e:
        rec["truncated"] = True
        rec["hard_timeout"] = True
        rec["error"] = f"CellTimeout: {e}"
    except Exception as e:                            # noqa: BLE001
        rec["error"] = f"{type(e).__name__}: {e}"

    if per_repeat:
        rec["us_per_config_per_repeat"] = per_repeat
        med, mn = statistics.median(per_repeat), min(per_repeat)
        rec["us_per_config_median"], rec["us_per_config_min"] = med, mn
        if rec["n_configs_timed"]:
            rec["us_per_config_pooled"] = 1e6 * rec["timed_s"] / rec["n_configs_timed"]
        fac = N_SAMPLES_PER_STEP * (2 + N + F) / 1e6
        rec["s_per_step_equiv"], rec["s_per_step_equiv_min"] = med * fac, mn * fac
    rec["cache_stats_after"] = _cache_stats(head)
    rec["wall_s"] = time.perf_counter() - t_cell
    return rec


# ----------------------------------------------------------------------------
# reporting
# ----------------------------------------------------------------------------
def _fmt(x, w, prec=2, unit=""):
    if x is None:
        return f"{'--':>{w}}"
    s = f"{x:.{prec}f}{unit}" if abs(x) < 1e5 else f"{x:.2e}{unit}"
    return f"{s:>{w}}"


def _row(rec):
    wl = "uniform" if rec["density"] is None else f"nbhd p={rec['density']:<6g}"
    flags = ("T" if rec["truncated"] else "") + ("H" if rec["hard_timeout"] else "")
    err = f"  ERR {rec['error']}" if rec["error"] else ""
    return (f"{rec['size']:>6} N={rec['N']:<4d}{rec['decoder']:<10}"
            f"{wl:<15}rows={rec['n_configs_timed']:<7d}x{rec['repeats_done']} "
            f"{_fmt(rec['us_per_config_median'], 10)} us/cfg "
            f"(min {_fmt(rec['us_per_config_min'], 9)}) "
            f"step {_fmt(rec['s_per_step_equiv'], 9, 3)} s "
            f"def {_fmt(rec['mean_defects'], 6, 1)} {flags:<2}{err}")


def _dump(out, path):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(out, f, indent=1)
    os.replace(tmp, path)


def _parse_sizes(s):
    out = []
    for tok in s.split(","):
        tok = tok.strip().lower()
        if tok:
            a, b = tok.split("x")
            out.append((int(a), int(b)))
    return out


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--sizes", default="1x2,2x2,2x3,3x3,4x4,5x5,6x6,8x8,10x10,12x12")
    ap.add_argument("--decoders", default="mwpm,anchor,greedy,unionfind,tie_sum")
    ap.add_argument("--densities", default="0.005,0.02,0.08",
                    help="link-flip densities for the neighborhood workload")
    ap.add_argument("--no_uniform", action="store_true", help="skip the uniform column")
    ap.add_argument("--n_base", type=int, default=64)
    ap.add_argument("--budget", type=float, default=120.0,
                    help="timed seconds per (size, decoder, workload, density) cell")
    ap.add_argument("--construct_budget", type=float, default=None,
                    help="hard cap on head construction (default: --budget)")
    ap.add_argument("--hard_cap", type=float, default=1.5,
                    help="SIGALRM wall cap per cell = hard_cap*budget + 5 s")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--chunk", type=int, default=4096)
    ap.add_argument("--warmup", type=int, default=64, help="untimed warm-up rows")
    ap.add_argument("--threads", type=int, default=1,
                    help="BLAS/numba thread count pinned before numpy import")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default=socket.gethostname())
    ap.add_argument("--git_hash", default=None,
                    help="override the recorded code hash (e.g. for a git-archive export)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.construct_budget is None:
        args.construct_budget = args.budget
    out_path = args.out or os.path.join(
        _SCRIPT_REPO, "results", "diagnostics", f"bench_decoders_{args.tag}.json")
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

    sizes = _parse_sizes(args.sizes)
    decoders = [d.strip() for d in args.decoders.split(",") if d.strip()]
    if "mwpm" in decoders:                   # mwpm first: it is the sanity reference
        decoders.remove("mwpm")
        decoders.insert(0, "mwpm")
    densities = [float(x) for x in args.densities.split(",") if x.strip()]
    cells = [("neighborhood", p) for p in densities]
    if not args.no_uniform:
        cells.append(("uniform", None))

    git_hash, git_dirty = _git_info(CODE_PATH, args.git_hash)
    out = {
        "meta": {
            "tag": args.tag,
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "hostname": socket.gethostname(), "platform": platform.platform(),
            "cpu": _cpu_model(), **_versions(),
            "threads_env": {k: os.environ.get(k) for k in _THREAD_VARS},
            "code_path": CODE_PATH, "git_hash": git_hash, "git_dirty": git_dirty,
            "args": vars(args), "step_formula": STEP_FORMULA, "total_wall_s": None,
        },
        "sizes": {}, "heads": [], "results": [], "sanity_failures": 0,
    }
    print(f"# bench_decoders  tag={args.tag}  code={CODE_PATH}  git={git_hash}"
          f"{' DIRTY:' + ','.join(git_dirty) if git_dirty else ''}", flush=True)
    print(f"# cpu={out['meta']['cpu'].get('model_name', out['meta']['cpu']['processor'])}"
          f"  threads={_THREADS}  numpy={np.__version__}"
          f"  pymatching={out['meta']['pymatching']}  numba={out['meta']['numba']}",
          flush=True)
    print(f"# sizes={args.sizes}  decoders={','.join(decoders)}  densities={densities}"
          f"  uniform={not args.no_uniform}  n_base={args.n_base}  budget={args.budget}s"
          f"  repeats={args.repeats}  chunk={args.chunk}  -> {out_path}", flush=True)
    print("# flags: T = truncated by budget, H = SIGALRM hard cap", flush=True)

    t_all = time.perf_counter()
    for Lx, Ly in sizes:
        label = f"{Lx}x{Ly}"
        g = HoneycombGeometry(Lx, Ly)
        N, F, V = int(g.N), int(g.n_plaqs), int(g.n_vertices)
        P, incT = _plaq_incidence(g), _vertex_incidence_T(g)
        size_info = {"label": label, "Lx": Lx, "Ly": Ly, "N": N, "F": F, "V": V,
                     "rows_per_pass": args.n_base * (1 + N + F)}
        out["sizes"][label] = {k: v for k, v in size_info.items() if k != "label"}
        print(f"\n== {label}: N={N} F={F} V={V}  rows/pass={size_info['rows_per_pass']}"
              f"  step factor (2+N+F)={2 + N + F}", flush=True)

        # zero-syndrome sanity set + count_loops reference parity
        rng_s = np.random.default_rng((args.seed, Lx, Ly, 999))
        zero_bits = np.concatenate([np.zeros((1, N), dtype=np.uint8),
                                    make_zero_syndrome(rng_s, args.n_base, P)])
        ref = np.array([count_loops(1 - 2 * row.astype(int), g.link_endpoints) % 2
                        for row in zero_bits], dtype=np.float64)
        mwpm_ref = None

        for dec in decoders:
            hrec = {"size": label, "decoder": dec, "N": N, "F": F, "V": V,
                    "construct_s": None, "construct_timeout": False, "error": None,
                    "sanity": None, "cache_stats_after_sanity": None}
            head = None
            t0 = time.perf_counter()
            try:
                with _Alarm(args.construct_budget):
                    head = QECSignHead(g, decoder=dec)
                hrec["construct_s"] = time.perf_counter() - t0
            except CellTimeout as e:
                hrec["construct_timeout"] = True
                hrec["construct_s"] = time.perf_counter() - t0
                hrec["error"] = f"CellTimeout (construction): {e}"
            except Exception as e:                    # noqa: BLE001
                hrec["error"] = f"{type(e).__name__} (construction): {e}"
            if head is not None:
                try:
                    with _Alarm(args.construct_budget):
                        hrec["sanity"], s_zero = run_sanity(head, zero_bits, ref, mwpm_ref)
                    if dec == "mwpm":
                        mwpm_ref = s_zero
                except AssertionError as e:
                    hrec["sanity"] = {"ok": False, "n_checked": int(len(ref)),
                                      "detail": str(e), "agree_with_mwpm": None}
                    out["sanity_failures"] += 1
                    print(f"!!! SANITY FAIL {label} {dec}: {e}", flush=True)
                    head = None
                except CellTimeout as e:
                    hrec["error"] = f"CellTimeout (sanity): {e}"
                    head = None
                except Exception as e:                # noqa: BLE001
                    hrec["error"] = f"{type(e).__name__} (sanity): {e}"
                    head = None
                hrec["cache_stats_after_sanity"] = _cache_stats(head) if head else None
            out["heads"].append(hrec)
            status = ("ok" if head is not None else
                      ("TIMEOUT" if hrec["construct_timeout"] else "FAILED"))
            print(f"  head {dec:<10} construct {_fmt(hrec['construct_s'], 8, 2)} s  {status}"
                  f"{'  ' + hrec['error'] if hrec['error'] else ''}", flush=True)
            _dump(out, out_path)
            if head is None:
                continue

            for ci, (kind, dens) in enumerate(cells):
                seed_tuple = (args.seed, Lx, Ly, decoders.index(dec), ci)
                rec = run_cell(head, kind, dens, size_info, args, P, incT, seed_tuple)
                rec["decoder"] = dec
                out["results"].append(rec)
                print("  " + _row(rec), flush=True)
                _dump(out, out_path)
            del head

    out["meta"]["total_wall_s"] = time.perf_counter() - t_all
    _dump(out, out_path)
    print(f"\n# done in {out['meta']['total_wall_s']:.1f} s -> {out_path}"
          f"  (sanity failures: {out['sanity_failures']})", flush=True)
    return 1 if out["sanity_failures"] else 0


if __name__ == "__main__":
    sys.exit(main())
