#!/bin/bash
#
# Complete script for running Search Benchmarks evaluation.
# This script:
# 1. Starts the retrieval server in the background
# 2. Starts the vLLM server in the background
# 3. Waits for both servers to be ready
# 4. Runs the benchmark tests
# 5. Optionally cleans up background processes
#
# Usage:
#     bash examples/searchqa/scripts/run_inference.sh [MODEL_PATH] [RETRIEVAL_PORT] [VLLM_PORT] [DATA_DIR]
#
# Example:
#     bash examples/searchqa/scripts/run_inference.sh /path/to/model 3000 6001 ./search_data/prebuilt_indices

# Configuration
MODEL_PATH=${1:-"/PATH/TO/MODEL"}
RETRIEVAL_PORT=${2:-8000}
VLLM_PORT=${3:-6001}
SAVE_PREFIX=${4:-"TravelPlanner_Instruct_3B_1128_1K_4Epoch"}

# Timeout and sleep interval for port checking
timeout=${TIMEOUT:-1200}  # Default 20 minutes
sleep_interval=${SLEEP_INTERVAL:-60}  # Default 60 seconds


echo "Using model path: $MODEL_PATH"
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

# Step 1: Start retrieval server in background
echo "=========================================="
echo "Step 1: Starting retrieval server..."
echo "=========================================="

# Activate the retrieval environment
# conda activate llamafactory 

cd /PATH/TO/RL/examples/searchqa 
python3 launch_server.py --data_dir ./search_data/prebuilt_indices --port $RETRIEVAL_PORT > /tmp/retrieval_server_${RETRIEVAL_PORT}.log 2>&1 &
RETRIEVAL_PID=$!
echo "Retrieval server started with PID: $RETRIEVAL_PID" 

export LOCAL_RETRIEVAL_SERVER_URL="http://localhost:$RETRIEVAL_PORT"


# Step 2: Start vLLM server in background
echo ""
echo "=========================================="
echo "Step 2: Starting vLLM server..."
echo "=========================================="

# Activate the model deployment environment
# conda activate rllm2 

CUDA_VISIBLE_DEVICES=0,1 vllm serve "$MODEL_PATH" \
    --host 0.0.0.0 \
    --port "$VLLM_PORT" \
    --tensor-parallel-size 2 \
    --gpu-memory-utilization 0.6 \
    --max-model-len 32768 > /tmp/vllm_server_${VLLM_PORT}.log 2>&1 &
VLLM_PID=$!
echo "vLLM server started with PID: $VLLM_PID" 


# Step 3: Wait for both servers to be ready
echo ""
echo "=========================================="
echo "Step 3: Waiting for servers to be ready..."
echo "=========================================="

check_port $RETRIEVAL_PORT "Retrieval server" || exit 1 
check_port $VLLM_PORT "vLLM server" || exit 1 

echo ""
echo "Both servers are ready!"
echo "  - Retrieval server: http://localhost:$RETRIEVAL_PORT"
echo "  - vLLM server: http://localhost:$VLLM_PORT"


# Step 4: Run benchmark tests
echo ""
echo "=========================================="
echo "Step 4: Running benchmark tests..."
echo "=========================================="

# Activate the RL environment
# conda activate rllm2 

cd /PATH/TO/RL 

# Create output directories if they don't exist
mkdir -p examples/searchqa/output_traj/${SAVE_PREFIX} examples/searchqa/logs/${SAVE_PREFIX}

for DATASET in nq triviaqa popqa hotpotqa 2wikimultihopqa musique bamboogle ; do
    echo ""
    echo "Running benchmark: $DATASET"
    
    # 7B: parallel_num = 25
    # 3B: parallel_num = 100 
    python3 -u -m examples.searchqa.my_run_search_agent \
        --model_name "$MODEL_PATH" \
        --test_data_name "$DATASET" \
        --parallel_num 60 \
        --n_repeat 1 \
        --save_traj \
        --server_url "http://localhost:$VLLM_PORT/v1" \
        --save_traj_path "examples/searchqa/output_traj/${SAVE_PREFIX}/${DATASET}.jsonl" \
        >> "examples/searchqa/logs/${SAVE_PREFIX}/${DATASET}.log" 2>&1
    echo "Completed: $DATASET"
done

echo ""
echo "=========================================="
echo "All benchmarks completed successfully!"
echo "=========================================="

kill -9 $RETRIEVAL_PID
kill -9 $VLLM_PID
