import asyncio 
import json 
from .local_retrieval_tool import LocalRetrievalTool
from transformers import AutoTokenizer   
from rllm.agents.travel_ood_agent import OODTestAgentFormat, SEARCH_SYSTEM_PROMPT
from rllm.data.dataset import DatasetRegistry   
from rllm.engine.agent_execution_engine import AgentExecutionEngine   
from rllm.environments.tools.travel_tool_env import TravelToolEnvironment    
from rllm.rewards.travel_ood_reward import my_search_reward_fn
from rllm.utils import compute_pass_at_k, convert_numpy_types
import time 
from datasets import Dataset 
from rllm.data.dataset import DatasetRegistry   


def load_test_data(dataset_name: str): 
    # Search testing dataset
    assert dataset_name in ["nq", "triviaqa", "popqa", "hotpotqa", "2wikimultihopqa", "musique", "bamboogle"], f"Dataset {dataset_name} not found" 

    test_dataset = DatasetRegistry.load_dataset(dataset_name, "test")

    if not test_dataset: 
        print("Dataset not found, preparing dataset...")  
        with open(f"examples/searchqa/test_data/{dataset_name}.jsonl", "r") as f: 
            test_samples = []
            for line in f:
                data = json.loads(line) 
                test_samples.append(data)
        
        test_dataset = Dataset.from_list(test_samples)
        test_dataset = DatasetRegistry.register_dataset(dataset_name, test_dataset, "test")
        print(f"Dataset {dataset_name} prepared with {len(test_samples)} examples")
    return test_dataset


if __name__ == "__main__":
    import os 
    import argparse 

    parser = argparse.ArgumentParser(description="🚀 Run travel agent on OOD test data ") 
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-3B-Instruct", help="Model name") 
    parser.add_argument("--server_url", type=str, default="http://localhost:6001/v1", help="Server URL") 
    parser.add_argument("--parallel_num", type=int, default=64, help="Parallel number")
    parser.add_argument("--n_repeat", type=int, default=1, help="Repeat number")
    parser.add_argument("--max_steps", type=int, default=60, help="Max steps")
    
    parser.add_argument("--test_data_name", type=str, default="bamboogle")
    parser.add_argument("--max_prompt_length", type=int, default=2*1024)
    parser.add_argument("--max_response_length", type=int, default=30*1024)
    parser.add_argument("--save_traj", action="store_true")
    parser.add_argument("--save_traj_path", type=str, default="")

    args = parser.parse_args()  
    start_time = time.time()  
    # TODO: specifically adjust the parallel number for 1.5B and 3B model
    # args.parallel_num = 100 if not ("7B" in args.model_name) else 25
    print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')} ")
    print(f"Args: {args} \n") 
    
    # Model related
    os.environ["TOKENIZERS_PARALLELISM"] = "true"  
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)  
    sampling_params = {"temperature": 0.6, "top_p": 0.95, "model": args.model_name} 
    
    # Tools 
    tool_map = {"Search": LocalRetrievalTool}

    engine = AgentExecutionEngine(
        agent_class=OODTestAgentFormat,
        agent_args={"tool_map": tool_map, "system_prompt": SEARCH_SYSTEM_PROMPT, "parser_name": "qwen"},
        env_class=TravelToolEnvironment,
        env_args={"tool_map": tool_map, "reward_fn": my_search_reward_fn, "max_steps": args.max_steps},
        rollout_engine=None,
        engine_name="openai",
        tokenizer=tokenizer,
        sampling_params=sampling_params,
        rollout_engine_args={"base_url": args.server_url, "api_key": "None"},
        max_response_length=args.max_response_length,
        max_prompt_length=args.max_prompt_length,
        n_parallel_agents=args.parallel_num,
        max_steps=args.max_steps,
    )

    test_dataset = load_test_data(args.test_data_name) 
 
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
                query = traj.task.get("question", "") 
            except:
                query = ""

            try:
                prediction = traj.steps[-1].chat_completions[-1].get("content", "").split("<answer>")[1].split("</answer>")[0].strip()
            except:
                prediction = "" 
            
            try:
                answer = traj.task.get("answer", "")
            except:
                answer = ""
            
            traj_data = {
                "question": query, 
                "prediction": prediction, 
                "answer": answer,
                "reward": int(traj.reward), 
                "messages": traj.steps[-1].chat_completions,  
                "item": convert_numpy_types(traj.task),
            }
            # print(traj_data)
            try:
                f.write(json.dumps(traj_data, ensure_ascii=False) + "\n") 
            except Exception as e: 
                import traceback 
                traceback.print_exc()
                print(f"Error writing task to file: {e}")

    print(f"Results saved to {args.save_traj_path}") 
    cost_time = round((time.time() - start_time) / 60, 2)
    print(f"Cost time: {cost_time} minutes !")
