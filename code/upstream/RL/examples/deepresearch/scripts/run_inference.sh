#!/bin/bash
#
# Deploy DeepResearch retrieval server (GPU0) + vLLM model server (GPU1),
# then run BrowseComp-Plus inference.
#
# Usage:
#   bash examples/deepresearch/scripts/run_inference.sh \
#     [MODEL_PATH] [SERVER_PORT] [VLLM_PORT] [SAVE_PREFIX]
#
# Example:
#   bash examples/deepresearch/scripts/run_inference.sh \
#     /path/to/model 8000 6001 MACRO_7B_Qwen3Emb8B
#

# Configuration
MODEL_PATH=${1:-"/PATH/TO/MODEL"}
SERVER_PORT=${2:-8000}
VLLM_PORT=${3:-6001}
SAVE_PREFIX=${4:-"TravelPlanner_SFT_7B"}

# Retrieval server config
QWEN_VARIANT=${QWEN_VARIANT:-"qwen3-embedding-8b"}

# Timeout and sleep interval for port checking
timeout=${TIMEOUT:-1200}  # Default 20 minutes
sleep_interval=${SLEEP_INTERVAL:-60}  # Default 60 seconds

echo "Using model path: $MODEL_PATH"
echo "Using retrieval index: $QWEN_VARIANT"
echo ""

check_port() {
    local port=$1
    local model_name=$2
    local start_time=$(date +%s)
    
    echo "Wait for $port ($model_name) to start..."
    while true; do
        # Check if port is listening
        if (netstat -tuln 2>/dev/null | grep -q ":$port ") || (ss -tuln 2>/dev/null | grep -q ":$port "); then
            # Try to connect via HTTP
            if curl -s --connect-timeout 5 --max-time 5 "http://localhost:$port/" > /dev/null 2>&1 || \
               curl -s --connect-timeout 5 --max-time 5 "http://localhost:$port/health" > /dev/null 2>&1 || \
               curl -s --connect-timeout 5 --max-time 5 "http://localhost:$port/v1" > /dev/null 2>&1; then
                echo "Port $port ($model_name) is ready."
                return 0
            fi
        fi
    
        current_time=$(date +%s)
        elapsed=$((current_time - start_time))
        if [ $elapsed -gt $timeout ]; then
            echo "Error: start port $port ($model_name) timeout ($timeout seconds)!"
            return 1
        fi
        echo "Port $port ($model_name) is not started, waiting $sleep_interval seconds to retry... (elapsed ${elapsed} seconds)"
        sleep $sleep_interval
    done
}


# Step 1: Start retrieval server in background (GPU0)
echo "=========================================="
echo "Step 1: Starting retrieval server..."
echo "=========================================="

# Activate the retrieval environment
# conda activate llamafactory

cd /PATH/TO/RL/examples/deepresearch
CUDA_VISIBLE_DEVICES=0 python launch_server.py \
    --qwen_variant "$QWEN_VARIANT" \
    --port "$SERVER_PORT" \
    > /tmp/deepresearch_server_${SERVER_PORT}.log 2>&1 &
SERVER_PID=$!
echo "Retrieval server started with PID: $SERVER_PID"

export LOCAL_RETRIEVAL_SERVER_URL="http://localhost:$SERVER_PORT"


# Step 2: Start vLLM server in background (GPU1)
echo ""
echo "=========================================="
echo "Step 2: Starting vLLM server..."
echo "=========================================="

# Activate the model deployment environment
# conda activate base

CUDA_VISIBLE_DEVICES=1 vllm serve "$MODEL_PATH" \
    --host 0.0.0.0 \
    --port "$VLLM_PORT" \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.7 \
    --max-model-len 32768 > /tmp/vllm_server_${VLLM_PORT}.log 2>&1 &
VLLM_PID=$!
echo "vLLM server started with PID: $VLLM_PID"


# Step 3: Wait for both servers to be ready
echo ""
echo "=========================================="
echo "Step 3: Waiting for servers to be ready..."
echo "=========================================="

check_port $SERVER_PORT "DeepResearch retrieval server" || exit 1
check_port $VLLM_PORT "vLLM server" || exit 1

echo ""
echo "Both servers are ready!"
echo "  - Retrieval server: http://localhost:$SERVER_PORT"
echo "  - vLLM server: http://localhost:$VLLM_PORT"


# Step 4: Run BrowseComp-Plus inference
echo ""
echo "=========================================="
echo "Step 4: Running BrowseComp-Plus inference..."
echo "=========================================="

# TODO: activate the RL environment
# conda activate rllm2
cd /PATH/TO/RL

mkdir -p examples/deepresearch/trajectory/${SAVE_PREFIX} examples/deepresearch/logs

echo "Running benchmark: BrowseComp-Plus"
python3 -u -m examples.deepresearch.my_run_browsecomp \
        --model_name "$MODEL_PATH" \
        --parallel_num 40 \
        --n_repeat 1 \
        --save_traj \
        --server_url "http://localhost:$VLLM_PORT/v1" \
        --save_traj_path "examples/deepresearch/trajectory/${SAVE_PREFIX}/Top5Max1K_Qwen3Emb8B.jsonl" \
        >> "examples/deepresearch/logs/${SAVE_PREFIX}_Top5Max1K_Qwen3Emb8B.log" 2>&1
echo "Completed: Top5Max1K"

echo ""
echo "Cleaning up..."
kill -9 $SERVER_PID || true
kill -9 $VLLM_PID || true
