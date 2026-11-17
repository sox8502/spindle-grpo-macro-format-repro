set -x

# change directory to rllm root
cd /PATH/TO/RL
mkdir -p logs

# TODO: activate the RL environment
#conda activate rllm2

export PYTHONUNBUFFERED=1
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:False"
export VLLM_USE_V1=1
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_ENGINE_ITERATION_TIMEOUT_S=100000000000

ray stop || true
unset RAY_ADDRESS
export RAY_ADDRESS=""

# Batch size and group size of GRPO
export BATCH_SIZE=32
export MINI_BATCH_SIZE=32
export N=8

# Model and parallel parameters
export TRAINING_MODEL="/PATH/TO/MODEL" # RL base model
export TRAINING_MODEL_NAME=$(basename "$TRAINING_MODEL")
export MAX_NEW_TOKEN=$((28*1024))
export MAX_PPO_TOKEN=$((30*1024))
export TP_SIZE=1
export SP_SIZE=1
export N_NODES=1

export OVERLONG_FILTER=True
export TRAJECTORY_TIMEOUT=1200 # Timeout for trajectory generation 

export TRAIN_DATA="rllm/data/travel_1107_1K_E4M3H3_1212.jsonl"
export TEST_DATA="rllm/data/travel_val180.jsonl"
export TRAIN_DATA_NAME="travel_1107_1K_E4M3H3_1212"
export TEST_DATA_NAME="travel_val180"

export AGENT_TOOLS="SearchFlight,SearchCity,SearchAttraction,SearchRestaurant,SearchAccommodation,GoogleDistanceMatrix"
export ENV_TOOLS="SearchFlight,SearchCity,SearchAttraction,SearchRestaurant,SearchAccommodation,GoogleDistanceMatrix"
export PARSER_NAME="qwen"
export ENV_MAX_STEPS=60
export AGENT_MAX_STEPS=60 
export REWARD_TYPE="MACRO"


# TODO: update the project and experiment names
export SWANLAB_API_KEY="YOUR_SWANLAB_API_KEY"
export PROJECT_NAME="TravelPlanner_3B_Algorithm"
export EXPERIMENT_NAME="NEW_EXPERIMENT_NAME"
export SAVE_PATH="/PATH/TO/SAVE_PATH/${EXPERIMENT_NAME}"


# For DAPO, set rllm.rejection_sample.enable to True
TIMESTAMP=$(date +"%Y-%m-%d-%H:%M:%S")
python3 -u -m examples.travel.train_travel_agent_format \
    algorithm.adv_estimator=grpo \
    rllm.rejection_sample.enable=True \
    rllm.rejection_sample.multiplier=1 \
    data.train_batch_size=${BATCH_SIZE} \
    data.val_batch_size=200 \
    data.max_prompt_length=2048 \
    data.max_response_length=${MAX_NEW_TOKEN} \
    actor_rollout_ref.model.path=${TRAINING_MODEL} \
    actor_rollout_ref.hybrid_engine=True \
    actor_rollout_ref.actor.optim.lr=2e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.loss_agg_mode=token-mean \
    actor_rollout_ref.actor.ppo_mini_batch_size=${MINI_BATCH_SIZE} \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${MAX_PPO_TOKEN} \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=${SP_SIZE} \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${TP_SIZE} \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode="async" \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.n=${N} \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.entropy_coeff=0 \
    algorithm.kl_ctrl.kl_coef=0.001 \
    rllm.mask_truncated_samples=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console','swanlab'] \
    trainer.project_name=${PROJECT_NAME} \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.val_before_train=True \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=${N_NODES} \
    trainer.default_local_dir=${SAVE_PATH} \
    trainer.max_actor_ckpt_to_keep=20 \
    trainer.max_critic_ckpt_to_keep=20 \
    trainer.save_freq=20 \
    trainer.test_freq=10 \
    trainer.default_hdfs_dir=null \
    rllm.agent.max_steps=${AGENT_MAX_STEPS} \
    rllm.stepwise_advantage.enable=False \
    rllm.stepwise_advantage.mode="mc_return" \
    rllm.agent.overlong_filter=${OVERLONG_FILTER} \
    rllm.agent.trajectory_timeout=${TRAJECTORY_TIMEOUT} \
    trainer.total_epochs=5 >logs/${EXPERIMENT_NAME}_${TIMESTAMP}.log 2>&1 
