from transformers import AutoModelForSequenceClassification, AutoTokenizer
import json
from tqdm import tqdm
import numpy as np
import torch
import argparse
import os
def parse_args():
    parser = argparse.ArgumentParser(
        description=
        "Finetune a transformers model on a causal language modeling task")
    parser.add_argument('--input_file',
                        type=str,
                        required = True,
                        help='Input file path')
    parser.add_argument('--output_file',
                        type=str,
                        default=None,
                        help='Output file path')
    args = parser.parse_args()
    return args
reward_name = "OpenAssistant/reward-model-deberta-v3-large-v2"

reward_model = AutoModelForSequenceClassification.from_pretrained(reward_name, device_map="cuda").eval()
tokenizer = AutoTokenizer.from_pretrained(reward_name)
# Read from the file and calculate the score from the reward model
if __name__ == "__main__":
    args = parse_args()
    file_path = args.input_file
    if args.output_file:
        output_file = args.output_file
    else:
        output_file = f"./eval_result/ourmetric_{os.path.basename(file_path)}"
    ref_results = []
    baseline_results = []
    our_results = []
    wts_results = []
    prompts = []
    with open(file_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            our_result = data['completion']
            our_results.append(our_result)
            prompts.append(data['prompt'])

    ref_scores = []
    baseline_scores = []
    our_scores = []
    wts_scores = []

    for i in tqdm(range(len(prompts))):
        our_inputs = tokenizer(prompts[i], our_results[i], return_tensors='pt').to(reward_model.device)
        our_score = reward_model(**our_inputs).logits[0].cpu().detach()
        our_scores.append(our_score.item())

    our_scores = torch.tensor(our_scores)


    output_path = file_path.replace(".jsonl", "_scores.json")
    print(file_path)
    print(f"Bert score: {our_scores.mean():.4f}")
