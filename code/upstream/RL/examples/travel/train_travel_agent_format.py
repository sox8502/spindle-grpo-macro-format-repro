import os 
import json 
import hydra 

import copy 
from datasets import Dataset
from rllm.agents import TravelAgentFormat 
from rllm.environments.tools.travel_tool_env import TravelToolEnvironment  
from rllm.data.dataset import DatasetRegistry  
from rllm.rewards.travel_reward import sum_travel_reward_fn, success_travel_reward_fn, macro_travel_reward_fn, planner_r1_reward_fn
from rllm.trainer.agent_trainer import AgentTrainer  
from rllm.agents.system_prompts import TRAVEL_SYSTEM_PROMPT 


def parse_tools_from_env(tools_str: str) -> list[str]: 
    if not tools_str:
        return ["SearchFlight", "SearchCity", "SearchAttraction", "SearchRestaurant", "SearchAccommodation", "GoogleDistanceMatrix"]
    
    return [tool.strip() for tool in tools_str.split(",")] 


def get_reward_fn(reward_type: str):
    assert reward_type in ["SUM", "SUCCESS", "MACRO", "PLANNER_R1"]
    return {
        "SUM": sum_travel_reward_fn,
        "SUCCESS": success_travel_reward_fn,
        "MACRO": macro_travel_reward_fn,
        "PLANNER_R1": planner_r1_reward_fn,
    }[reward_type]


def prepare_dataset(data_path: str, data_name: str, is_train: bool = True):
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset file not found at {data_path}")

    dataset = DatasetRegistry.load_dataset(data_name, split="train" if is_train else "test")
    if dataset is not None:
        print(f"Dataset {data_name} loaded from registry !")
        return dataset 
    
    with open(data_path, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]

    processed_data = [] 
    for item in data: 
        current_item = copy.deepcopy(item) 
        current_item["ground_truth"] = "" 
        current_item["data_source"] = data_path 
        current_item["solution"] = ""
        current_item["question"] = current_item["query"] 
        processed_data.append(current_item)  
    
    dataset = Dataset.from_list(processed_data) 
    split_type = "train" if is_train else "test"  
    dataset = DatasetRegistry.register_dataset(data_name, dataset, split_type) 
    print(f"Dataset {data_name} prepared with {len(processed_data)} examples") 
    return dataset 
 

def get_env_config():
    return {
        "train_data": os.getenv("TRAIN_DATA", ""),
        "test_data": os.getenv("TEST_DATA", ""),
        "train_data_name": os.getenv("TRAIN_DATA_NAME", ""), 
        "test_data_name": os.getenv("TEST_DATA_NAME", ""), 

        "agent_tools": os.getenv("AGENT_TOOLS", "SearchFlight,SearchCity,SearchAttraction,SearchRestaurant,SearchAccommodation,GoogleDistanceMatrix"),
        "env_tools": os.getenv("ENV_TOOLS", "SearchFlight,SearchCity,SearchAttraction,SearchRestaurant,SearchAccommodation,GoogleDistanceMatrix"),
        "parser_name": os.getenv("PARSER_NAME", "qwen"), 
        "env_max_steps": os.getenv("ENV_MAX_STEPS", 60), 

        "reward_type": os.getenv("REWARD_TYPE", "SUM")
    }


@hydra.main(config_path="pkg://rllm.trainer.config", config_name="agent_ppo_trainer", version_base=None)
def main(config): 
    env_config = get_env_config() 

    print("=========ENV CONFIG=========") 
    print(json.dumps(env_config, indent=2, ensure_ascii=False))
    print("============================")

    try:
        train_dataset = prepare_dataset(env_config["train_data"], env_config["train_data_name"], True) 
        test_dataset = prepare_dataset(env_config["test_data"], env_config["test_data_name"], False)  
    except Exception as e: 
        print(f"Error preparing dataset: {e}")
        raise 

    agent_tools_list = parse_tools_from_env(env_config["agent_tools"])
    env_tools_list = parse_tools_from_env(env_config["env_tools"])
    print(f"Agent tools: {agent_tools_list}")
    print(f"Env tools: {env_tools_list}")

    agent_args = {
        "tools": agent_tools_list,
        "parser_name": env_config["parser_name"], 
        "system_prompt": TRAVEL_SYSTEM_PROMPT, 
    }

    env_args = {
        "tools": env_tools_list, 
        "reward_fn": get_reward_fn(env_config["reward_type"]), 
        "max_steps": env_config["env_max_steps"], 
    }

    print("=========START TRAINING=========") 
    trainer = AgentTrainer(
        agent_class=TravelAgentFormat,
        env_class=TravelToolEnvironment, 
        agent_args=agent_args, 
        env_args=env_args, 
        config=config, 
        train_dataset=train_dataset, 
        val_dataset=test_dataset
    ) 
    trainer.train() 


if __name__ == "__main__": 
    main()
