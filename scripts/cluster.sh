#!/bin/bash
# Claude-driven NERSC (Perlmutter) cluster control for 2D-TC.
#
# Access model: sshproxy mints a 24h SSH cert at $NERSC_KEY (default ~/.ssh/nersc);
# this script drives everything over `ssh -i $NERSC_KEY`. No ~/.ssh/config edits,
# no committed secrets (connection settings live in the gitignored config.local.sh).
#
# Safeguard: read-only subcommands (status/sync/logs/fetch) are allowlisted in
# .claude/settings.local.json so Claude can monitor without prompting. The
# state-changing ones (submit/cancel) are deliberately NOT allowlisted, so they
# always prompt the user, and `submit` first prints a job spec (see preflight()).
# Per CLAUDE.md, Claude must also consult the user (experiment / resources /
# walltime / why) before ever calling `submit`.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/.." && pwd)"
CFG="$HERE/cluster/config.local.sh"

if [ ! -f "$CFG" ]; then
    echo "ERROR: missing $CFG"
    echo "  cp scripts/cluster/config.local.sh.example scripts/cluster/config.local.sh"
    echo "  then set NERSC_USER."
    exit 1
fi
# shellcheck disable=SC1090
source "$CFG"

: "${NERSC_USER:?set NERSC_USER in scripts/cluster/config.local.sh}"
NERSC_HOST="${NERSC_HOST:-perlmutter.nersc.gov}"
NERSC_REPO="${NERSC_REPO:-2D-TC}"
NERSC_KEY="${NERSC_KEY:-$HOME/.ssh/nersc}"

SSH=(ssh -i "$NERSC_KEY" -o IdentitiesOnly=yes -o BatchMode=yes "${NERSC_USER}@${NERSC_HOST}")

check_key() {
    if [ ! -f "$NERSC_KEY" ]; then
        echo "ERROR: no NERSC key at $NERSC_KEY — the sshproxy cert is missing or was purged."
        echo "Mint a fresh 24h cert (prompts for password + MFA OTP):"
        echo "    sshproxy -u $NERSC_USER"
        exit 1
    fi
}

# Run a command inside the cluster repo checkout.
remote() { check_key; "${SSH[@]}" "cd '$NERSC_REPO' && $*"; }

# preflight <jobfile> [sbatch args...]: show exactly what is being submitted BEFORE
# sbatch, so the approval prompt is informed (the #SBATCH lines baked into the script
# plus the per-submit overrides like -t / --array / --export).
preflight() {
    local job="$1"; shift
    echo "──────── job spec ────────"
    echo "  script          : $job"
    echo "  sbatch overrides: ${*:-(none; using script #SBATCH defaults)}"
    echo "  #SBATCH in script:"
    grep -E '^#SBATCH' "$job" | sed 's/^/    /'
    echo "──────────────────────────"
}

usage() {
    cat <<'EOF'
usage: bash scripts/cluster.sh <command>

  status              squeue for your jobs
  sync                git pull --ff-only on the cluster (fetch latest code)
  checkout <branch>   fetch + checkout a branch on the cluster
  logs [pattern]      tail the newest logs/*<pattern>*.out
  fetch               commit+push results/ on the cluster, then git pull locally
  wandb-sync          push offline W&B runs (wandb/) to the dashboard from the login node
  submit <jobfile>    sbatch a job  (PROMPTS; consult first — not allowlisted)
  cancel <jobid>      scancel a job (PROMPTS — not allowlisted)
EOF
}

cmd="${1:-}"
case "$cmd" in
    status)
        check_key
        "${SSH[@]}" 'squeue --me --format="%.12i %.12j %.9T %.10M %.10l %.5D %R"'
        ;;
    sync)
        remote 'git pull --ff-only'
        ;;
    checkout)
        shift; br="${1:?usage: bash scripts/cluster.sh checkout <branch>}"
        remote "git fetch origin && git checkout '$br' && git pull --ff-only"
        ;;
    logs)
        shift; pat="${1:-}"
        remote "ls -t logs/*${pat}*.out 2>/dev/null | head -1 | xargs -r tail -n 60"
        ;;
    get)
        # Read-only: cat a file from the cluster repo to a local path (for
        # gitignored outputs like logs/smoke_*/ that fetch doesn't carry).
        shift; rpath="${1:?usage: bash scripts/cluster.sh get <remote-repo-path> <local-path>}"
        lpath="${2:?usage: bash scripts/cluster.sh get <remote-repo-path> <local-path>}"
        # write via a temp file so a failed remote cat leaves no stray empty file
        remote "cat '$rpath'" > "$lpath.part" && mv "$lpath.part" "$lpath" \
            || { rm -f "$lpath.part"; echo "get failed: $rpath"; exit 1; }
        ;;
    sacct)
        # Read-only SLURM accounting for a finished job (state/exit/memory) --
        # the only way to diagnose jobs that left the queue without a log file.
        shift; jid="${1:?usage: bash scripts/cluster.sh sacct <jobid>}"
        remote "sacct -j '$jid' --format=JobID,JobName%12,State%22,Elapsed,Start,ExitCode,MaxRSS,NodeList -P"
        ;;
    fetch)
        # Commit any new results, then ALWAYS push — covers the case where a prior
        # fetch committed but the push was skipped (git commit exits non-zero when
        # there is nothing new, which short-circuits an && chain), leaving the
        # cluster ahead-by-N and results stranded.
        remote 'git add -A results/ && { git commit -q -m "cluster: sweep results" || true; } && git pull --rebase --autostash --no-edit && git push' \
            || echo "(cluster: push failed or nothing to push)"
        echo "--- pulling results locally ---"
        git -C "$REPO_ROOT" pull --ff-only
        ;;
    wandb-sync)
        # Login nodes DO have internet (compute nodes don't) — push the offline run
        # dirs to the dashboard. Safe to run repeatedly (mid-run for near-live views,
        # or once after a job); wandb dedupes by run id. Needs a one-time `wandb login`.
        remote 'module load conda >/dev/null 2>&1 && conda activate 2dtc && wandb sync --sync-all' \
            || echo "(wandb sync failed — run \`wandb login\` on the login node once, or no offline runs yet)"
        ;;
    submit)
        shift; job="${1:?usage: bash scripts/cluster.sh submit <jobfile> [sbatch args...]}"
        shift; sbatch_args=("$@")
        # bash 3.2 (macOS) errors on empty-array expansion under `set -u`; the
        # ${arr[@]+...} guard expands to nothing when no overrides were passed.
        preflight "$job" ${sbatch_args[@]+"${sbatch_args[@]}"}
        remote "mkdir -p logs && sbatch ${sbatch_args[@]+${sbatch_args[*]}} '$job'"
        ;;
    cancel)
        shift; jid="${1:?usage: bash scripts/cluster.sh cancel <jobid>}"
        remote "scancel '$jid'"
        ;;
    ""|-h|--help|help)
        usage ;;
    *)
        echo "unknown command: $cmd"; echo; usage; exit 1 ;;
esac
