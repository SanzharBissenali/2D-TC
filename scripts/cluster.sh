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
