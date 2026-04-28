#!/usr/bin/env bash
# slurm-serve-ollama.sh <model> [--gpus N] [--partition P] [--time HH:MM:SS]
# Submits a SLURM job that starts ollama serve on a GPU node and writes the
# endpoint URL to $CHITTA_BRIDGE_URL_DIR/ollama-server-<model>.url for
# chitta-bridge discovery (default: ~/.chitta-bridge/endpoints).
set -euo pipefail

OLLAMA_BIN="${OLLAMA_BIN:-$(command -v ollama || echo ollama)}"
OLLAMA_MODELS="${OLLAMA_MODELS:-$HOME/.ollama/models}"
OLLAMA_PORT="${OLLAMA_PORT:-11434}"
PARTITION="${SLURM_PARTITION:-compregular}"
GRES="${SLURM_GRES:-gpu:a100:1}"
TIME="${SLURM_TIME:-04:00:00}"
LOG_DIR="${CHITTA_BRIDGE_LOG_DIR:-$HOME/.chitta-bridge/logs}"

usage() { echo "Usage: $0 <model> [--gpus N] [--partition P] [--time HH:MM:SS]"; exit 1; }
[ $# -lt 1 ] && usage

MODEL="$1"; shift
while [[ $# -gt 0 ]]; do
    case "$1" in
        --gres)      GRES="$2";      shift 2 ;;
        --partition) PARTITION="$2"; shift 2 ;;
        --time)      TIME="$2";      shift 2 ;;
        *) echo "Unknown arg: $1"; usage ;;
    esac
done

# Sanitize model name for filename (replace / and : with -)
MODEL_SAFE="${MODEL//[\/:]/-}"
URL_DIR="${CHITTA_BRIDGE_URL_DIR:-$HOME/.chitta-bridge/endpoints}"
mkdir -p "$URL_DIR"
URL_FILE="${URL_DIR}/ollama-server-${MODEL_SAFE}.url"

# Abort if a job for this model is already queued or running (match on prefix, tolerates : vs - in name)
EXISTING=$(squeue -u "$USER" -h -o "%i %j %T" 2>/dev/null | grep -i "ollama-${MODEL_SAFE}\|ollama-${MODEL}" | head -1)
if [[ -n "$EXISTING" ]]; then
    echo "Job already active for model '${MODEL}': ${EXISTING}"
    echo "Cancel it first with: scancel <job_id>"
    exit 1
fi

# Remove stale URL file if it exists
rm -f "$URL_FILE"

mkdir -p "$LOG_DIR"

JOB_SCRIPT=$(mktemp "${TMPDIR:-/tmp}/slurm-ollama-XXXXXX.sh")
cat > "$JOB_SCRIPT" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=ollama-${MODEL_SAFE}
#SBATCH --partition=${PARTITION}
#SBATCH --gres=${GRES}
#SBATCH --time=${TIME}
#SBATCH --output=${LOG_DIR}/slurm-ollama-${MODEL_SAFE}-%j.out
#SBATCH --signal=B:TERM@60

NODE=\$(hostname -s)
URL="http://\${NODE}:${OLLAMA_PORT}"
URL_FILE="${URL_FILE}"

cleanup() {
    echo "[\$(date)] SLURM job ending — cleaning up"
    rm -f "\$URL_FILE"
    kill \$SERVE_PID 2>/dev/null || true
}
trap cleanup EXIT TERM INT

export OLLAMA_MODELS="${OLLAMA_MODELS}"
export OLLAMA_HOST="0.0.0.0:${OLLAMA_PORT}"
export OLLAMA_KEEP_ALIVE="-1"

echo "[\$(date)] Starting ollama serve on \${NODE}:${OLLAMA_PORT} for model ${MODEL}"
"${OLLAMA_BIN}" serve &> "${LOG_DIR}/ollama-serve-${MODEL_SAFE}-\${SLURM_JOB_ID}.log" &
SERVE_PID=\$!

# Wait for ollama to be ready
for i in \$(seq 1 30); do
    if curl -sf "http://localhost:${OLLAMA_PORT}/api/tags" > /dev/null 2>&1; then
        echo "[\$(date)] ollama ready after \${i}s"
        break
    fi
    sleep 1
done

# Write URL file so chitta-bridge can discover this node
echo "\${URL}/v1" > "\$URL_FILE"
echo "[\$(date)] Endpoint written: \${URL}/v1 -> \${URL_FILE}"

# Pull model if not already present
echo "[\$(date)] Ensuring model ${MODEL} is available"
"${OLLAMA_BIN}" pull "${MODEL}" 2>&1 || echo "[\$(date)] Pull failed or model already present"

echo "[\$(date)] Ready. Waiting for SLURM job to end."
wait \$SERVE_PID
EOF

chmod +x "$JOB_SCRIPT"
JOB_ID=$(sbatch --parsable "$JOB_SCRIPT")
rm -f "$JOB_SCRIPT"

echo "Submitted SLURM job $JOB_ID for model '${MODEL}'"
echo "  Partition : ${PARTITION} | GRES: ${GRES} | Time: ${TIME}"
echo "  URL file  : ${URL_FILE}  (shared path — written once ollama is ready)"
echo ""
echo "Monitor: squeue -j $JOB_ID"
echo "Logs   : ${LOG_DIR}/slurm-ollama-${MODEL_SAFE}-${JOB_ID}.out"
echo ""
echo "Once running, chitta-bridge will auto-discover the endpoint via local_discover."
