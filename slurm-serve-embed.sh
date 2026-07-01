#!/usr/bin/env bash
# slurm-serve-embed.sh [--port N] [--partition P] [--time HH:MM:SS]
# Submits a SLURM job running the sentence-transformers GPU embed server.
# Writes URL to ~/.chitta-bridge/endpoints/embed-server.url for chitta discovery.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PORT="${EMBED_PORT:-11436}"
PARTITION="${SLURM_PARTITION:-compregular}"
GRES="${SLURM_GRES:-gpu:a100:1}"
TIME="${SLURM_TIME:-12:00:00}"
LOG_DIR="${CHITTA_BRIDGE_LOG_DIR:-$HOME/.chitta-bridge/logs}"
URL_DIR="${CHITTA_BRIDGE_URL_DIR:-$HOME/.chitta-bridge/endpoints}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --port)      PORT="$2";      shift 2 ;;
        --partition) PARTITION="$2"; shift 2 ;;
        --time)      TIME="$2";      shift 2 ;;
        --gres)      GRES="$2";      shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

mkdir -p "$LOG_DIR" "$URL_DIR"

EXISTING=$(squeue --me --name chitta-embed --noheader --format="%i %T" 2>/dev/null | head -1)
if [[ -n "$EXISTING" ]]; then
    echo "chitta-embed already queued: $EXISTING (skipping submit)"
    exit 0
fi

JOBID=$(sbatch --parsable \
    --job-name="chitta-embed" \
    --partition="$PARTITION" \
    --gres="$GRES" \
    --mem=24G \
    --cpus-per-task=4 \
    --time="$TIME" \
    --output="$LOG_DIR/embed-server-%j.log" \
    --wrap="
set -euo pipefail
NODE=\$(hostname)
URL=\"http://\${NODE}:${PORT}\"
echo \"\$URL\" > '${URL_DIR}/embed-server.url'
echo \"\$(date): starting embed server on \$URL\"
EMBED_PORT=${PORT} python3 '${SCRIPT_DIR}/embed-server.py'
")

echo "Submitted job $JOBID"
echo "Log: $LOG_DIR/embed-server-${JOBID}.log"
echo "URL will appear at: $URL_DIR/embed-server.url"
echo ""
echo "Once running, wire into chitta daemon:"
echo "  Environment=CHITTA_EMBED_URL=http://<node>:${PORT}"
echo "  systemctl --user restart chittad"
