#! /bin/bash
# Launch the local retrieval server at port 8000

cd /PATH/TO/RL/examples/deepresearch
python launch_server.py --qwen_variant qwen3-embedding-8b --host 0.0.0.0 --port 8000 
