#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────
# AlienBody vLLM Server Launcher
#
# Start Qwen VLMs on A800 servers for AlienBody evaluation.
# Run this ON the A800 server, not locally.
#
# Usage:
#   bash scripts/start_vllm_servers.sh           # start all 4 models
#   bash scripts/start_vllm_servers.sh --model 4b  # start only 4B
#   bash scripts/start_vllm_servers.sh --stop      # stop all vLLM servers
# ──────────────────────────────────────────────────────────────
set -euo pipefail

# ── Model Config ────────────────────────────────────────────

declare -A MODEL_PATHS=(
    ["4b"]="models/Qwen3.5-VL-4B-Instruct"
    ["9b"]="models/Qwen3.5-VL-9B-Instruct"
    ["35b"]="models/Qwen3.5-VL-35B-Instruct"
    ["397b"]="models/Qwen3.5-VL-397B-Instruct"
)

declare -A PORTS=(
    ["4b"]="8000"
    ["9b"]="8001"
    ["35b"]="8002"
    ["397b"]="8003"
)

declare -A GPU_COUNTS=(
    ["4b"]="1"
    ["9b"]="1"
    ["35b"]="2"
    ["397b"]="8"
)

# ── Parse Args ──────────────────────────────────────────────

MODEL_FILTER=""
ACTION="start"

while [[ $# -gt 0 ]]; do
    case $1 in
        --model) MODEL_FILTER="$2"; shift 2 ;;
        --stop)  ACTION="stop"; shift ;;
        --status) ACTION="status"; shift ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done

# ── Stop ─────────────────────────────────────────────────────

if [ "$ACTION" = "stop" ]; then
    echo "=== Stopping vLLM servers ==="
    for model in "${!PORTS[@]}"; do
        if [ -n "$MODEL_FILTER" ] && [ "$model" != "$MODEL_FILTER" ]; then
            continue
        fi
        port="${PORTS[$model]}"
        pid=$(lsof -ti :"$port" 2>/dev/null || true)
        if [ -n "$pid" ]; then
            echo "  Killing vLLM $model on port $port (PID $pid)"
            kill -TERM $pid 2>/dev/null || true
            sleep 2
            kill -KILL $pid 2>/dev/null || true
        else
            echo "  vLLM $model (port $port): not running"
        fi
    done
    echo "Done."
    exit 0
fi

# ── Status ───────────────────────────────────────────────────

if [ "$ACTION" = "status" ]; then
    echo "=== vLLM Server Status ==="
    for model in "${!PORTS[@]}"; do
        port="${PORTS[$model]}"
        pid=$(lsof -ti :"$port" 2>/dev/null || true)
        if [ -n "$pid" ]; then
            echo "  $model (port $port): RUNNING (PID $pid)"
        else
            echo "  $model (port $port): stopped"
        fi
    done
    echo ""
    echo "=== GPU Status ==="
    nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv 2>/dev/null || echo "  nvidia-smi not available"
    exit 0
fi

# ── Start ────────────────────────────────────────────────────

echo "=== Starting vLLM servers for AlienBody ==="
echo ""

GPU_OFFSET=0
for model in "4b" "9b" "35b" "397b"; do
    if [ -n "$MODEL_FILTER" ] && [ "$model" != "$MODEL_FILTER" ]; then
        continue
    fi

    model_path="${MODEL_PATHS[$model]}"
    port="${PORTS[$model]}"
    ngpus="${GPU_COUNTS[$model]}"
    gpu_ids=$(seq -s ',' $GPU_OFFSET $((GPU_OFFSET + ngpus - 1)))

    # Check if already running
    if lsof -ti :"$port" >/dev/null 2>&1; then
        echo "  [$model] Port $port already in use, skipping"
        GPU_OFFSET=$((GPU_OFFSET + ngpus))
        continue
    fi

    # Check model path
    if [ ! -d "$model_path" ]; then
        echo "  [$model] Model not found: $model_path, SKIPPING"
        GPU_OFFSET=$((GPU_OFFSET + ngpus))
        continue
    fi

    echo "  [$model] Starting on GPUs $gpu_ids, port $port..."
    echo "          Model: $model_path"

    nohup env CUDA_VISIBLE_DEVICES="$gpu_ids" \
        vllm serve "$model_path" \
            --port "$port" \
            --max-model-len 8192 \
            --gpu-memory-utilization 0.85 \
            --max-num-seqs 8 \
            --enable-prefix-caching \
            --async-scheduling \
            --mm-processor-cache-type shm \
            > "/tmp/vllm_${model}.log" 2>&1 &

    echo "          PID: $! (log: /tmp/vllm_${model}.log)"

    GPU_OFFSET=$((GPU_OFFSET + ngpus))
done

echo ""
echo "Waiting for servers to be ready..."
for model in "4b" "9b" "35b" "397b"; do
    if [ -n "$MODEL_FILTER" ] && [ "$model" != "$MODEL_FILTER" ]; then
        continue
    fi
    port="${PORTS[$model]}"
    echo -n "  $model (port $port): "
    for i in $(seq 1 60); do
        if curl -s "http://localhost:$port/v1/models" >/dev/null 2>&1; then
            echo "READY (${i}s)"
            break
        fi
        sleep 2
    done
    if [ $i -eq 60 ]; then
        echo "TIMEOUT — check /tmp/vllm_${model}.log"
    fi
done

echo ""
echo "=== All vLLM servers started ==="
echo ""
echo "Test with:"
echo "  curl http://localhost:8000/v1/models"
echo ""
echo "Run evaluation:"
echo "  cd ~/workspace/CLAUDE_HOME/research-bot/papers/alienbody/code"
echo "  python scripts/run_eval.py --agent vllm:qwen3.5-4b --family all --split test --modality text --output results/text_qwen4b --resume --vllm-host localhost --vllm-port 8000"
