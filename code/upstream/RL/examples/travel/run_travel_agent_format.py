import asyncio 
import json 
import time 
import copy 
from transformers import AutoTokenizer   

from datasets import Dataset
from rllm.agents.travel_agent import TravelAgentFormat  
from rllm.agents.system_prompts import TRAVEL_SYSTEM_PROMPT  
from rllm.data.dataset import DatasetRegistry   
from rllm.engine.agent_execution_engine import AgentExecutionEngine   
from rllm.environments.tools.travel_tool_env import TravelToolEnvironment    
from rllm.rewards.travel_reward import sum_travel_reward_fn, success_travel_reward_fn, macro_travel_reward_fn  
from rllm.utils import compute_pass_at_k, convert_numpy_types


def parse_tools_from_env(tools_str: str) -> list[str]: 
    if not tools_str:
        return ["SearchFlight", "SearchCity", "SearchAttraction", "SearchRestaurant", "SearchAccommodation", "GoogleDistanceMatrix"]
    
    return [tool.strip() for tool in tools_str.split(",")] 


def prepare_dataset(data_path: str, data_name: str, is_train: bool = True):
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset file not found at {data_path}")
    
    with open(data_path, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]
    
    processed_data = [] 
    for item in data: 
        current_item = copy.deepcopy(item) 
        current_item["ground_truth"] = "" 
        current_item["data_source"] = data_path 
        current_item["solution"] = ""
        processed_data.append(current_item)  
    
    dataset = Dataset.from_list(processed_data) 
    split_type = "train" if is_train else "test"  
    dataset = DatasetRegistry.register_dataset(data_name, dataset, split_type) 
    print(f"Dataset {data_name} prepared with {len(processed_data)} examples") 
    return dataset 
 

if __name__ == "__main__":
    import os 
    import argparse 

    parser = argparse.ArgumentParser(description="🚀 Run travel agent ...") 
    parser.add_argument("--model_name", type=str, default="xxwu/Agent-STAR-RL-7B", help="Model name") 
    parser.add_argument("--tools", type=str, default="SearchFlight,SearchCity,SearchAttraction,SearchRestaurant,SearchAccommodation,GoogleDistanceMatrix", help="Available tools")
    parser.add_argument("--server_url", type=str, default="http://localhost:30001/v1", help="Server URL") 
    parser.add_argument("--parallel_num", type=int, default=50, help="Parallel number")
    parser.add_argument("--n_repeat", type=int, default=1, help="Repeat number")
    parser.add_argument("--max_steps", type=int, default=60, help="Max steps")

    parser.add_argument("--test_data", type=str, default="rllm/data/travel_val180.jsonl")
    parser.add_argument("--test_data_name", type=str, default="travel_val180")
    parser.add_argument("--max_prompt_length", type=int, default=2*1024)
    parser.add_argument("--max_response_length", type=int, default=30*1024)
    parser.add_argument("--save_traj", action="store_true")
    parser.add_argument("--save_traj_path", type=str, default="")

    args = parser.parse_args()  

    os.environ["TOKENIZERS_PARALLELISM"] = "true"  
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)   

    available_tools = parse_tools_from_env(args.tools)
    
    agent_args = {
        "tools": available_tools,
        "parser_name": "qwen",
        "system_prompt": TRAVEL_SYSTEM_PROMPT,
    }
    env_args = {
        "tools": available_tools,
        "reward_fn": success_travel_reward_fn,
        "max_steps": args.max_steps,
    }
    
    sampling_params = {"temperature": 0.6, "top_p": 0.95, "model": args.model_name} 

    engine = AgentExecutionEngine(
        agent_class=TravelAgentFormat,
        agent_args=agent_args,
        env_class=TravelToolEnvironment,
        env_args=env_args,
        engine_name="openai",
        tokenizer=tokenizer,
        sampling_params=sampling_params,
        rollout_engine_args={"base_url": args.server_url, "api_key": "None"},
        n_parallel_agents=args.parallel_num,
        max_steps=args.max_steps,
        max_response_length=args.max_response_length,
        max_prompt_length=args.max_prompt_length,
    ) 

    try:
        test_dataset = prepare_dataset(args.test_data, args.test_data_name, False) 
    except Exception as e: 
        print(f"Error preparing dataset: {e}")
        raise 

    tasks = test_dataset.repeat(n=args.n_repeat)
    results = asyncio.run(engine.execute_tasks(tasks))  

    for traj in results: 
        traj.task = convert_numpy_types(traj.task)
    compute_pass_at_k(results) 

    if not args.save_traj: 
        exit() 

    with open(args.save_traj_path, "w") as f:
        for traj in results: 
            try:
                query = traj.task.get("query", "") 
            except:
                query = ""

            try:
                prediction = traj.steps[-1].chat_completions[-1].get("content", "").split("<answer>")[1].split("</answer>")[0].strip()
            except:
                prediction = "" 
            
            traj_data = {
                "query": query, 
                "plan": prediction, 
                "reward": int(traj.reward), 
                "messages": traj.steps[-1].chat_completions,  
                "item": convert_numpy_types(traj.task),
                "judgement": traj.info.get("judgement", {}),
                "formatted_plan": traj.info.get("formatted_plan", []),
            }
            # print(traj_data)
            try:
                f.write(json.dumps(traj_data, ensure_ascii=False) + "\n") 
            except Exception as e: 
                import traceback 
                traceback.print_exc()
                print(f"Error writing task to file: {e}")

    print(f"Results saved to {args.save_traj_path}") 
