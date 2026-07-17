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

# preflight <jobfile>: show the user exactly what they are approving BEFORE sbatch.
# TODO(you): implement. See the note printed by Claude for the design question.
preflight() {
    local job="$1"
    echo "TODO: preflight display not implemented — see scripts/cluster.sh preflight()"
}

usage() {
    cat <<'EOF'
usage: bash scripts/cluster.sh <command>

  status              squeue for your jobs
  sync                git pull --ff-only on the cluster (fetch latest code)
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
    logs)
        shift; pat="${1:-}"
        remote "ls -t logs/*${pat}*.out 2>/dev/null | head -1 | xargs -r tail -n 60"
        ;;
    fetch)
        remote 'git add results/ && git commit -q -m "cluster: sweep results" && git push' \
            || echo "(cluster: nothing new to commit)"
        echo "--- pulling results locally ---"
        git -C "$REPO_ROOT" pull --ff-only
        ;;
    submit)
        shift; job="${1:?usage: bash scripts/cluster.sh submit jobs/nersc_nqs.sh}"
        preflight "$job"
        remote "mkdir -p logs && sbatch '$job'"
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
