import os 
import requests 
import json 
import time 
from rllm.rewards.travel_utils.eval_commonsense import check_commonsense
from rllm.rewards.travel_utils.eval_hardconstraint import check_hardconstraint 
from rllm.rewards.reward_types import RewardOutput
from rllm.utils import convert_numpy_types 


DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "YOUR_DEEPSEEK_API_KEY") 


FORMAT_PLANNING = """Please assist me in extracting valid information from a given natural language text and reconstructing it in JSON format, as demonstrated in the following example. 

**IMPORTANT: You must output ONLY valid JSON format. Do not include any explanatory text, markdown code blocks, or additional formatting.**

**GUIDELINES**
1. each item should include ['day', 'current_city', 'transportation', 'breakfast', 'attraction', 'lunch', 'dinner', 'accommodation']. Replace non-specific information like 'eat at home/on the road' with '-'. Additionally, delete any '$' symbols.
2. transportation: If transportation details indicate a journey from one city to another (e.g., from A to B), the 'current_city' should be updated to the destination city (in this case, B). If there's information about transportation, ensure that the 'current_city' aligns with the destination mentioned in the transportation details (i.e., the current city should follow the format 'from A to B'). Also, ensure that all flight numbers and costs are followed by a colon (i.e., 'Flight Number:' and 'Cost:'), consistent with the provided example. For the 'driving' or 'drive' mode, unless otherwise specified, always use 'Self-driving' consistently.
3. attraction: Use a ';' to separate different attractions, with each attraction strictly formatted as 'Name, City'. 
4. day: start from 1.

**OUTPUT FORMAT: Return ONLY the JSON array, nothing else.**
 
**EXAMPLES**
[{
    "day": 1,
    "current_city": "from Dallas to Peoria",
    "transportation": "Flight Number: 4044830, from Dallas to Peoria, Departure Time: 13:10, Arrival Time: 15:01",
    "breakfast": "-",
    "attraction": "Peoria Historical Society, Peoria;Peoria Holocaust Memorial, Peoria;",
    "lunch": "-",
    "dinner": "Tandoor Ka Zaika, Peoria",
    "accommodation": "Bushwick Music Mansion, Peoria"
},
{
    "day": 2,
    "current_city": "Peoria",
    "transportation": "-",
    "breakfast": "Tandoor Ka Zaika, Peoria",
    "attraction": "Peoria Riverfront Park, Peoria;The Peoria PlayHouse, Peoria;Glen Oak Park, Peoria;",
    "lunch": "Cafe Hashtag LoL, Peoria",
    "dinner": "The Curzon Room - Maidens Hotel, Peoria",
    "accommodation": "Bushwick Music Mansion, Peoria"
},
{
    "day": 3,
    "current_city": "from Peoria to Dallas",
    "transportation": "Flight Number: 4045904, from Peoria to Dallas, Departure Time: 07:09, Arrival Time: 09:20",
    "breakfast": "-",
    "attraction": "-",
    "lunch": "-",
    "dinner": "-",
    "accommodation": "-"
}]


Now please help me extract valid information from the following text and reconstruct it in a strict JSON format. 

**CRITICAL: Your response must be ONLY the JSON array. Do not include any markdown code blocks (```json), explanatory text, or other formatting. Just the raw JSON array starting with [ and ending with ].**

Original user query: {{{user_query}}}

Planning text: {{{planning_text}}}

"""


def extract_answer(response: str) -> str: 
    if "<answer>" in response and "</answer>" in response: 
        lpos = response.find("<answer>")
        rpos = response.rfind("</answer>")
        return response[lpos+len("<answer>"):rpos].strip()
    
    return response


def call_format_model(messages: list[dict], format_model="deepseek-chat", max_retries=20):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}"
    }
    base_url = "https://api.deepseek.com/chat/completions"
    payload = {
        "model": format_model,
        "messages": messages,
        "max_tokens": 8192, 
        "temperature": 0.6,
        "top_p": 0.95,
    }
    
    while max_retries:
        try:
            response = requests.post(base_url, headers=headers, json=payload, timeout=180)
            content = response.json()["choices"][0]["message"]["content"] 

            if content:
                content = content.replace("```json", "").replace("```", "").strip()
                content = json.loads(content)
                return content 
        except Exception as e:
            max_retries -= 1
            time.sleep(0.2)

            if max_retries == 0:
                print(f"Failed to call format model after {max_retries} attempts. Error: {e}")
                raise e


def invoke_format_model_post_process(query: str, action: str, format_model="deepseek-chat"):
    action = extract_answer(action)
    format_prompt = FORMAT_PLANNING.replace("{{{user_query}}}", query).replace("{{{planning_text}}}", action)
    messages = [
        {"role": "system", "content": "You are a helpful assistant that formats text into a strict JSON format."},
        {"role": "user", "content": format_prompt}
    ]

    response = call_format_model(messages, format_model, max_retries=2) 
    
    return response


def eval_one_sample(question: dict, tested_data: list[dict]) -> dict: 
    # TODO: important! convert numpy types to python types
    question = convert_numpy_types(question)
    if not tested_data or len(tested_data) == 0:
        return {
            "commonsense_macro_score": 0,
            "commonsense_micro_score": 0,
            "hard_macro_score": 0,
            "hard_micro_score": 0,
            "success_pass": 0,
        }
    
    commonsense_results = check_commonsense(question, tested_data) 

    hard_results = None 
    try:
        question["local_constraint"] = eval(question.get("local_constraint", "{}"))
    except:
        pass 

    if commonsense_results and commonsense_results["is_not_absent"][0] and commonsense_results["is_valid_information_in_sandbox"][0]:
        hard_results = check_hardconstraint(question, tested_data)
    
    commonsense_all_pass, micro_pass = True, 0
    for constraint, (is_valid, message) in commonsense_results.items():  
        if is_valid is not None:
            if not is_valid: 
                commonsense_all_pass = False 
            else: # Pass the commonsense constraint
                micro_pass += 1 

    commonsense_macro_score, commonsense_micro_score = int(commonsense_all_pass), micro_pass * 1.0 / 8
    
    hard_macro_score, hard_micro_score = None, None
    if hard_results:
        # Check if all hard constraints are None (not applicable)
        all_none = all(is_valid is None for is_valid, _ in hard_results.values())
        
        if not all_none:
            hard_all_pass, micro_pass, micro_total = True, 0, 0
            for constraint, (is_valid, message) in hard_results.items():  
                if is_valid is not None and not is_valid:
                    hard_all_pass = False
                if is_valid is not None:
                    micro_pass += 1 if is_valid else 0
                    micro_total += 1
            hard_macro_score, hard_micro_score = int(hard_all_pass), micro_pass * 1.0 / micro_total if micro_total > 0 else 0

    # Final pass rate: commonsense must pass, and if hard constraints exist and are applicable, they must also pass
    if hard_macro_score is None:
        # All hard constraints are not applicable, only consider commonsense
        success_pass = int(commonsense_all_pass)
        hard_macro_score, hard_micro_score = 0., 0.
    else:
        success_pass = int(commonsense_all_pass and hard_macro_score)

    return {
        "commonsense_macro_score": commonsense_macro_score,
        "commonsense_micro_score": commonsense_micro_score,
        "hard_macro_score": hard_macro_score,
        "hard_micro_score": hard_micro_score,
        "success_pass": success_pass,
        "details": {
            "commonsense_results": commonsense_results,
            "hard_results": hard_results
        }
    }
    

def fine_grained_travel_reward_fn(task_info: dict, action: str, format_model="deepseek-chat"):
    # 0 - Format error with 0 reward
    if "Format error occurs in the previous turn!" in action: 
      return False, "Format error occurs in the previous turn!", [] 

    # 1 - format planning 
    try:
        formatted_plan = invoke_format_model_post_process(task_info["query"], action, format_model)
    except Exception as e:
        return False, f"Error formatting planning: {e}", []

    # 2 - calculate reward  
    try:
        reward_dict = eval_one_sample(task_info, formatted_plan)
        return True, reward_dict, formatted_plan
    except Exception as e:
        return False, f"Error evaluating one sample: {e}", []


def sum_travel_reward_fn(task_info: dict, action: str, format_model="deepseek-chat") -> RewardOutput: 
    flg, res, formatted_plan = fine_grained_travel_reward_fn(task_info, action, format_model) 
    if not flg:
        return RewardOutput(reward=0.0, metadata={"error": res})
    
    reward_value = res.get("commonsense_macro_score", 0) + res.get("commonsense_micro_score", 0) + res.get("hard_macro_score", 0) + res.get("hard_micro_score", 0) + res.get("success_pass", 0)
    return RewardOutput(reward=reward_value, metadata={"judgement": res, "formatted_plan": formatted_plan})
 

def success_travel_reward_fn(task_info: dict, action: str, format_model="deepseek-chat") -> RewardOutput: 
    flg, res, formatted_plan = fine_grained_travel_reward_fn(task_info, action, format_model) 
    if not flg:
        return RewardOutput(reward=0.0, metadata={"error": res})
    
    return RewardOutput(reward=res.get("success_pass", 0), metadata={"judgement": res, "formatted_plan": formatted_plan})


def macro_travel_reward_fn(task_info: dict, action: str, format_model="deepseek-chat") -> RewardOutput: 
    flg, res, formatted_plan = fine_grained_travel_reward_fn(task_info, action, format_model) 
    if not flg:
        return RewardOutput(reward=0.0, metadata={"error": res})
    
    reward_value = res.get("commonsense_macro_score", 0) + res.get("hard_macro_score", 0) + res.get("success_pass", 0)
    return RewardOutput(reward=reward_value, metadata={"judgement": res, "formatted_plan": formatted_plan})


def planner_r1_reward_fn(task_info: dict, action: str, current_step: int, total_steps: int, format_model="deepseek-chat") -> RewardOutput: 
    flg, res, formatted_plan = fine_grained_travel_reward_fn(task_info, action, format_model) 
    if not flg:
        return RewardOutput(reward=0.0, metadata={"error": res})
    
    if total_steps and current_step / total_steps <= 0.33:
        reward_value = res.get("commonsense_macro_score", 0) + res.get("commonsense_micro_score", 0) + res.get("hard_macro_score", 0) + res.get("hard_micro_score", 0) + res.get("success_pass", 0) 
    elif total_steps and current_step / total_steps <= 0.67:  
        reward_value = res.get("commonsense_macro_score", 0) + res.get("hard_macro_score", 0) + res.get("success_pass", 0) 
    else:
        reward_value = res.get("success_pass", 0) 
    
    return RewardOutput(reward=reward_value, metadata={"judgement": res, "formatted_plan": formatted_plan})


if __name__ == "__main__":
    task_info = {
        "query": "Can you design a 3-day travel itinerary for 2 people, departing from Ithaca and heading to Newark from March 18th to March 20th, 2022? Our budget is set at $1,200, and we require our accommodations to be entire rooms and visitor-friendly. Please note that we prefer not to drive ourselves during this trip.",
        "org": "Ithaca", 
        "dest": "Newark", 
        "days": 3, 
        "date": ["2022-03-18", "2022-03-19", "2022-03-20"], 
        "visiting_city_number": 1, 
        "people_number": 2, 
        "local_constraint": {
            "house_rule": "visitors",
            "cuisine": None, 
            "room type": "entire room",  
            "transportation": "no self-driving", 
        },
        "budget": 1000, 
        "level": "hard",
        "id": "hard_jsjsy2", 
    } 

    action = """ "# 3-Day Newark Travel Itinerary from Ithaca
**Travel Dates: March 18-20, 2022 | For 2 People | Budget: $1,200**

## Day 1: March 18, 2022 (Travel Day)
- **Morning**: Depart from Ithaca by car.
- **Transportation**: Drive from Ithaca to Newark (3 hours 38 minutes, 343 km, cost: $17 one way).
- **Afternoon**: Arrive in Newark and check into accommodation: **"1 Bedroom in UWS Manhattan"** ($117/night, 2 nights total $234).
- **Activity**: Visit **Military Park** (free, contact: (973) 900-5800).
- **Dinner**: At **Artistry** (avg cost $79/person, 2 meals $158 total, cuisines: Cafe, Pizza, Desserts, Seafood).
- **Accommodation**: "1 Bedroom in UWS Manhattan" ($117/night).

## Day 2: March 19, 2022 (Full Day in Newark)
- **Breakfast**: At **Harmann Restaurant** (avg cost $57/person, 2 meals $114 total, cuisines: Tea, Cafe, American, Fast Food).
- **Morning**: Visit **The Newark Museum of Art** (contact: (973) 596-6550).
- **Lunch**: At **Jaguar** (avg cost $38/person, 2 meals $76 total, cuisines: French, Bakery, Seafood).
- **Afternoon**: Visit **Branch Brook Park** (contact: (973) 268-3500).
- **Dinner**: At **Escape Terrace Bar Kitchen** (avg cost $100/person, 2 meals $200 total, cuisines: Tea, Cafe, Bakery, Chinese, Seafood).
- **Accommodation**: "1 Bedroom in UWS Manhattan" ($117/night).

## Day 3: March 20, 2022 (Travel Day)
- **Morning**: Check out from accommodation.
- **Transportation**: Drive from Newark to Ithaca (3 hours 38 minutes, 343 km, cost: $17 one way).
- **Afternoon**: Arrive back in Ithaca.

## Budget Breakdown
- Transportation (round trip): $17 + $17 = $34
- Accommodation (2 nights): $117 × 2 = $234
- Meals (Day 1 dinner, Day 2 breakfast, lunch, dinner): $158 + $114 + $76 + $200 = $548
- Attractions (assumed minimal or free): $0 (based on tool data)
- **Total cost**: $34 + $234 + $548 + $0 = $786

## Additional Notes
- All accommodations are entire rooms and visitor-friendly.
- Meal costs are averages based on restaurant data.
- Attractions may have entry fees; check individual websites for details.
- Total cost is within the $1,200 budget.

Enjoy your trip to Newark!
    """
    start_time = time.time()
    flg, res, plan = fine_grained_travel_reward_fn(task_info, action)
    print(flg, res, plan)
    print(f"Time taken: {round(time.time() - start_time, 2)} seconds")
