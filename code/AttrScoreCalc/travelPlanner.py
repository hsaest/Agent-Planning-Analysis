import time
from concurrent.futures import ThreadPoolExecutor
from tqdm.contrib.concurrent import thread_map
import yaml
import json
import os
from datasets import load_dataset
import re
import random
import argparse
import bitsandbytes as bnb
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from captum.attr import (
    FeatureAblation, 
    LLMAttribution, 
    TextTemplateInput
)

# Load line-delimited JSON data from a file
def load_line_json_data(filename):
    data = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f.read().strip().split('\n'):
            unit = json.loads(line)
            data.append(unit)
    return data

# Load a quantized model and its tokenizer
def load_model(model_name, bnb_config):
    n_gpus = torch.cuda.device_count()
    max_memory = ["40000MB"] * n_gpus

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        max_memory={i: max_memory[i] for i in range(n_gpus)}
    )
    # Load tokenizer, set pad_token for LLaMA
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=True)
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer

# Create BitsAndBytes quantization config
def create_bnb_config():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    return bnb_config

# Replace placeholders in the input string and collect (selected_xxx, value) pairs
def replace_placeholders_and_collect_pairs(input_string):
    # pattern matches: [PLACEHOLDER n [selected_xxx] value]
    pattern = r'\[PLACEHOLDER \d+ \[(selected_[\w\d_]+)\] ([\d\w.]+)\]'
    pairs = []
    def replace_placeholders(match):
        selected_item = match.group(1)
        value = match.group(2)
        pairs.append((selected_item, value))
        return "{}"
    
    output_string = re.sub(pattern, replace_placeholders, input_string)
    return output_string, pairs

# Generate the input prompt for the LLM attribution
def generate_input(instance, reference_information):
    question = instance['query']
    processed_ref_inf, pairs = replace_placeholders_and_collect_pairs(reference_information)
    processed_ref_inf = processed_ref_inf.replace("{","{{").replace("}","}}")
    pair_value = [x[1] for x in pairs]
    inp = TextTemplateInput(
        template=(
            """You are a proficient planner. Based on the provided information and query, please give me a detailed plan, including specifics such as flight numbers (e.g., F0123456), restaurant names, and accommodation names. Note that all the information in your plan should be derived from the provided data. You must adhere to the format given in the example. Additionally, all details should align with commonsense. The symbol '-' indicates that information is unnecessary. For example, in the provided sample, you do not need to plan after returning to the departure city. When you travel to two cities in one day, you should note it in the 'Current City' section as in the example (i.e., from A to B).
Background Information:
Attractions\nName: The name of the attraction.\nLatitude: The geographical latitude where the attraction is located.\nLongitude: The geographical longitude where the attraction is located.\nAddress: The physical address of the attraction.\nPhone: Contact phone number for the attraction.\nWebsite: URL of the official website for the attraction.\nCity: The city where the attraction is located, which is Rockford in this case.\nRestaurants\nName: The name of the restaurant.\nAverage Cost: The average cost for a meal at the restaurant, in dollars.\nCuisines: Types of cuisine offered by the restaurant.\nAggregate Rating: The overall rating of the restaurant on a scale.\nCity: The city where the restaurant is located.\nAccommodations\nNAME: The name of the accommodation.\nprice: The price per night for staying at the accommodation.\nroom type: The type of room offered.\nhouse_rules: Specific rules that guests must follow during their stay.\nminimum nights: The minimum number of nights required for booking.\nmaximum occupancy: The maximum number of people allowed to stay. If the group size exceeds the maximum occupancy, multiple rooms are required to accommodate everyone.\nreview rate number: A rating or review score, typically out of 5.\ncity: The city where the accommodation is located, which is Rockford.\nFlights\nFlight Number: The specific number assigned to the flight.\nPrice: The cost of the flight ticket.\nDepTime: The departure time of the flight.\nArrTime: The arrival time of the flight.\nActualElapsedTime: The duration of the flight.\nFlightDate: The date of the flight.\nOriginCityName: The city from where the flight departs.\nDestCityName: The destination city of the flight.\nDistance: The distance flown in miles.\nSelf-driving\nSelf-driving from {{origin}} to {{destination}}, duration: {{duration}}, distance: {{distance}}, cost: {{cost}}. If no valid information is available, it will be presented as \"Self-driving, from {{origin}} to {{destination}}, no valid information.\" The default occupancy of a self-driving car is 5 people. If the group size exceeds the maximum occupancy, multiple cars are required to accommodate everyone.\nTaxi\nTaxi from {{origin}} to {{destination}}, duration: {{duration}}, distance: {{distance}}, cost: {{cost}}. If no valid information is available, it will be presented as \"Taxi, from {{origin}} to {{destination}}, no valid information.\" If the group size exceeds the maximum occupancy, multiple cars are required to accommodate everyone.
***** Example *****
Query: Please help me plan a trip from St. Petersburg to Rockford spanning 3 days from March 16th to March 18th, 2022. The travel should be planned for a single person with a budget of $1,700.

Plan:
[
{{
\"days\": 1,
\"current_city\": \"from St. Petersburg to Rockford\",
\"transportation\": \"Flight Number: F3573659, from St. Petersburg to Rockford, Departure Time: 15:40, Arrival Time: 17:04\",
\"breakfast\": \"-\",
\"attraction\": \"-\",
\"lunch\": \"-\",
\"dinner\": \"Coco Bambu, Rockford\",
\"accommodation\": \"Pure luxury one bdrm + sofa bed on Central Park, Rockford\"
}},
{{
\"days\": 2,
\"current_city\": \"Rockford\",
\"transportation\": \"-\",
\"breakfast\": \"Flying Mango, Rockford\",
\"attraction\": \"Burpee Museum of Natural History, Rockford; Midway Village Museum, Rockford; Discovery Center Museum, Rockford\",
\"lunch\": \"Grappa - Shangri-La's - Eros Hotel, Rockford\",
\"dinner\": \"Dunkin' Donuts, Rockford\",
\"accommodation\": \"Pure luxury one bdrm + sofa bed on Central Park, Rockford\"
}},
{{
\"days\": 3,
\"current_city\": \"from Rockford to St. Petersburg\",
\"transportation\": \"Flight Number: F3573120, from Rockford to St. Petersburg, Departure Time: 19:00, Arrival Time: 22:43\",
\"breakfast\": \"Subway, Rockford\",
\"attraction\": \"Klehm Arboretum & Botanic Garden, Rockford; Sinnissippi Park, Rockford\",
\"lunch\": \"Cafe Coffee Day, Rockford\",
\"dinner\": \"Dial A Cake, Rockford\",
\"accommodation\": \"-\"
}}
]
***** Example Ends *****

Given information: """ + processed_ref_inf + """
Query: {}\nTravel Plan:\n"""),
    values=pair_value + [question],
    )
    return inp

# Run attribution for a single instance and save the result
def process_instance(index, inp, target, llm_attr, task, model_name, save_path):
    time.sleep(random.uniform(30, 120))  # Random sleep to avoid resource conflicts
    attr_res = llm_attr.attribute(inp, target=target, show_progess=True)
    # Save the result to a separate file to avoid conflicts
    result = {
        'index': index,
        'input_tokens': attr_res.input_tokens,
        'output_tokens': attr_res.output_tokens,
        'attr': attr_res.token_attr
    }
    torch.save(result, f'{save_path}/{model_name}_task_{task}_index_{index}.pt')

# Run attribution in parallel for multiple instances
def run_parallel_computation(instances, data, plans, ref_info, llm_attr, task, model_name, save_path):
    # Use thread_map for multithreading and progress bar
    def process_wrapper(index):
        inp = generate_input(data[index], ref_info[index])
        target = str(plans[index]['plan'])
        process_instance(index, inp, target, llm_attr, task, model_name, save_path)
    thread_map(process_wrapper, instances, max_workers=8, chunksize=1, desc=f"Processing {task}")



# Main entry point for running the attribution pipeline
def main():
    parser = argparse.ArgumentParser(description="Run LLM Attribution Score Calculation.")
    parser.add_argument('--save_path', type=str, default='../../data/TravelPlanner/attr_score', help='Directory to save results')
    parser.add_argument('--model_name', type=str, default='llama3.1-8B', help='Model name')
    parser.add_argument('--model_path', type=str, default='meta-llama/Llama-3.1-8B-Instruct', help='Model path')
    parser.add_argument('--data_path', type=str, default='../../data/TravelPlanner', help='Path to data/config files')
    args = parser.parse_args()

    save_path = args.save_path
    model_name = args.model_name
    model_path = args.model_path
    data_path = args.data_path
    
    bnb_config = create_bnb_config()
    model, tokenizer = load_model(model_path, bnb_config)
    model.eval()
    fa = FeatureAblation(model)
    llm_attr = LLMAttribution(fa, tokenizer)

    for task in ['travelplanner']:
        instances = [i for i in range(180)]
        data = load_dataset('osunlp/TravelPlanner','validation')['validation']
        plans = load_line_json_data(f"{data_path}/results/validation_{model_name}_direct_sole-planning_submission.jsonl")
        ref_info = load_line_json_data(f"{data_path}/TravelPlanner_Items/{model_name}_attr_ref_info.jsonl")
        run_parallel_computation(instances, data, plans, ref_info, llm_attr, task, model_name, save_path)

if __name__ == "__main__":
    main()