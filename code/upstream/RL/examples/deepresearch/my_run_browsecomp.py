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
import re 
import numpy as np 


def load_test_data(): 
    try:
        test_dataset = DatasetRegistry.load_dataset("browsecomp_plus", "test")
    except:
        print("Dataset not found, preparing dataset...") 
    
    if not test_dataset:
        records = []
        with open("examples/deepresearch/data/browsecomp_plus_full.jsonl", "r") as f:
            for line in f:
                data = json.loads(line) 
                record = {
                    "question": data["question"],
                    "answer": data["answer"], 
                    "data_source": "browsecomp_plus", 
                    "gold_docs": [doc["docid"] for doc in data["gold_docs"]], 
                    "evidence_docs": [doc["docid"] for doc in data["evidence_docs"]], 
                }
                records.append(record) 
        test_dataset = Dataset.from_list(records)
        test_dataset = DatasetRegistry.register_dataset("browsecomp_plus", test_dataset, "test") 

    return test_dataset


if __name__ == "__main__":
    import os 
    import argparse 

    parser = argparse.ArgumentParser(description="🚀 Run travel agent on OOD test data ") 
    parser.add_argument("--model_name", type=str, default="xxwu/Agent-STAR-RL-3B")
    parser.add_argument("--server_url", type=str, default="http://localhost:6001/v1", help="Server URL") 
    parser.add_argument("--parallel_num", type=int, default=24, help="Parallel number")
    parser.add_argument("--n_repeat", type=int, default=1, help="Repeat number")
    parser.add_argument("--max_steps", type=int, default=60, help="Max steps")
    
    parser.add_argument("--max_prompt_length", type=int, default=2*1024)
    parser.add_argument("--max_response_length", type=int, default=30*1024)
    parser.add_argument("--save_traj", action="store_true")
    parser.add_argument("--save_traj_path", type=str, default="")

    args = parser.parse_args()  
    start_time = time.time()  
    
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

    test_dataset = load_test_data() 
 
    tasks = test_dataset.repeat(n=args.n_repeat)
    results = asyncio.run(engine.execute_tasks(tasks))  

    for traj in results: 
        traj.task = convert_numpy_types(traj.task)
    compute_pass_at_k(results) 

    if not args.save_traj: 
        exit() 
    
    recall_scores = []
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
            
            try:
                evidence_docs = traj.task.get("evidence_docs", []) 
            except:
                evidence_docs = [] 
            
            retrieved_docs = []
            for msg in traj.steps[-1].chat_completions: 
                if msg.get("role", "") == "user" and "<tool_response>" in msg.get("content") and "</tool_response>" in msg.get("content"): 
                    doc_ids = re.findall(r"\(ID:\s*(\d+)\s*,", msg.get("content", ""))
                    doc_ids = list(set(doc_ids)) 
                    retrieved_docs.extend(doc_ids)  
            if evidence_docs and retrieved_docs: 
                recall = len(set(evidence_docs) & set(retrieved_docs)) / len(evidence_docs)
            else:
                recall = 0.0 
            recall_scores.append(recall)
         
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
    print(f"Avg. recall scores on evidence docs: {np.mean(recall_scores)*100:.2f}%")
    cost_time = round((time.time() - start_time) / 60, 2)
    print(f"Cost time: {cost_time} minutes !")
