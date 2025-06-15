import time
from concurrent.futures import ThreadPoolExecutor
from tqdm.contrib.concurrent import thread_map
import yaml
import json
import os
import bitsandbytes as bnb
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import argparse

from captum.attr import (
    FeatureAblation, 
    LLMAttribution, 
    TextTemplateInput
)


def load_model(model_name, bnb_config):
    n_gpus = torch.cuda.device_count()
    max_memory = "48000MB"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        max_memory={i: max_memory for i in range(n_gpus)}
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=True)

    # Needed for LLaMA tokenizer
    tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer

def create_bnb_config():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    return bnb_config


def generate_input(question):
    inp = TextTemplateInput(
    template="""I am playing with a set of blocks where I need to arrange the blocks into stacks. Here are the actions I can do
{}
{}
{}
{}

I have the following restrictions on my actions:
 
{}
{}
{}
{}
{}
{}
{}
{}
{}
{}
{}
{}
""", 
    values=["Pick up a block",
"Unstack a block from on top of another block",
"Put down a block",
"Stack a block on top of another block",
"I can only pick up or unstack one block at a time.", 
"I can only pick up or unstack a block if my hand is empty.", 
"I can only pick up a block if the block is on the table and the block is clear. A block is clear if the block has no other blocks on top of it and if the block is not picked up.",
"I can only unstack a block from on top of another block if the block I am unstacking was really on top of the other block.",
"I can only unstack a block from on top of another block if the block I am unstacking is clear.", 
"Once I pick up or unstack a block, I am holding the block.",
"I can only put down a block that I am holding.",
"I can only stack a block on top of another block if I am holding the block being stacked.",
"I can only stack a block on top of another block if the block onto which I am stacking the block is clear.",
"Once I put down or stack a block, my hand becomes empty.",
"Once you stack a block on top of a second block, the second block is no longer clear.",
f"""[STATEMENT]
{question}"""],
)
    return inp


def process_instance(index, struct_output, llm_attr, task, model_name, save_path):
    question = struct_output["query"].split('[STATEMENT]')[-1].strip()
    response = struct_output["llm_raw_response"]
    if len(response.split(' ')) > 256 or len(response.strip())==0:
        return None  # Skip responses that are too long or empty

    inp = generate_input(question)
    target = response
    attr_res = llm_attr.attribute(inp, target=target, show_progess=True)

    # Save the result to a separate file to avoid conflicts
    result = {
        'index': index,
        'is_correct': struct_output["llm_raw_correct"],
        'input_tokens': attr_res.input_tokens,
        'output_tokens': attr_res.output_tokens,
        'attr': attr_res.token_attr
    }
    
    torch.save(result, f'{save_path}/{model_name}_task_{task}_index_{index}.pt')


def run_parallel_computation(instances, response_res, llm_attr, task, model_name, save_path):
    # Use thread_map for multithreading and progress bar
    def process_wrapper(index):
        struct_output = response_res[index - 1]
        process_instance(index, struct_output, llm_attr, task, model_name, save_path)
    thread_map(process_wrapper, instances, max_workers=16, chunksize=1, desc=f"Processing {task}")



def main():
    parser = argparse.ArgumentParser(description="Run LLM Attribution Score Calculation.")
    parser.add_argument('--save_path', type=str, default='../../data/BlocksWorld/attr_score', help='Directory to save results')
    parser.add_argument('--model_name', type=str, default='qwen2-7B', help='Model name')
    parser.add_argument('--model_path', type=str, default='Qwen/Qwen2-7B-Instruct', help='Model path')
    parser.add_argument('--data_path', type=str, default='../../data/BlocksWorld', help='Path to data/config files')
    
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


    for task in ['blocksworld', 'blocksworld_3']:
        with open(f'{data_path}/config/{task}_attr_sample.yaml', 'r') as file:
            instances = yaml.safe_load(file)
        with open(f'{data_path}/results/{task}/{model_name}/task_1_plan_generation_zero_shot.json', 'r') as file:
            response_res = json.load(file)['instances']
        run_parallel_computation(instances, response_res, llm_attr, task, model_name, save_path)


if __name__ == "__main__":
    main()