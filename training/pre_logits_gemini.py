import os
import torch
import hashlib
from tqdm import tqdm
from transformers import AutoTokenizer, AutoConfig
from datasets import load_dataset
from google import genai
from google.genai.types import GenerateContentConfig 
from google.genai import types   
import time
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="gemini-2.5-pro", help="Path to the pre-trained model or model identifier from huggingface.co/models.")
    parser.add_argument("--output_dir", type=str, default="./cached_logits/gemini_pro", help="Directory to save the processed data.")
    parser.add_argument("--max_samples", type=int, default=100, help="Maximum number of samples to process from the dataset.")
    args = parser.parse_args()
    return args

def format_gemini_prompt(question, answer=None):
    if answer is None:
        prompt = [
            {"role": "user", "parts": [{"text": question}]}
        ]
    else:
        prompt = [
            {"role": "user", "parts": [{"text": question}]},
            {"role": "model", "parts": [{"text": answer}]}
        ]
    return prompt

def get_logprobs_gemini(client, tokenizer, question:str, answer:str, args):
    answer_ids = tokenizer.encode(answer, add_special_tokens=False)
    config = AutoConfig.from_pretrained("google/gemma-3-1b-pt")
    vocab_size = config.vocab_size
    results = []
    for i in range(len(answer_ids)):
        if i == 0:
            prompt = format_gemini_prompt(question)
        else:
            prompt = format_gemini_prompt(question, tokenizer.decode(answer_ids[:i]))
        # print(prompt)
        completion = client.models.generate_content(
            model=args.model_name_or_path,
            contents=prompt,
            config=GenerateContentConfig(response_logprobs=True, logprobs=5, max_output_tokens=5,)
        )
        if not completion or not completion.candidates:
            print("No completion returned")
            print("Skipping this example:", question)
            return None
        logprobs_result = completion.candidates[0].logprobs_result
        if logprobs_result is None or not logprobs_result.top_candidates :
            print("No logprobs returned")
            print("Skipping this example:", question)
            return None
        first_token_logprobs = logprobs_result.top_candidates[0].candidates
        all_logprobs = [logprob_item.log_probability for logprob_item in first_token_logprobs]
        if all_logprobs is None:
            print("No logprobs returned")
            print("Skipping this example:", question)
            return None
        min_logprob = min(all_logprobs)
        fill_value = min_logprob - 10.0
        logprob_tensor = torch.full((vocab_size,), fill_value, dtype=torch.float32)
        for logprob_item in first_token_logprobs:
            token_id = logprob_item.token_id
            log_probability = logprob_item.log_probability
            logprob_tensor[token_id] = log_probability
        results.append(logprob_tensor.unsqueeze(0))
        time.sleep(1)  # To avoid rate limiting
    log_probs = torch.cat(results, dim=0).unsqueeze(0)  # Shape: [1, seq_len, vocab_size]
    answer_ids_tensor = torch.tensor(answer_ids).unsqueeze(0)  # Shape: [1, seq_len]
    print("log_probs shape:", log_probs.shape)
    print("answer_ids_tensor shape:", answer_ids_tensor.shape)
    return {"log_probs": log_probs.cpu(), "labels": answer_ids_tensor.cpu()}

def main():
    args = parse_args()
    model_name_or_path = args.model_name_or_path
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-1b-pt", use_fast=False)
    client = genai.Client(
      vertexai=True,
      project=os.environ["GOOGLE_CLOUD_PROJECT"],
      location=os.environ.get("GOOGLE_CLOUD_LOCATION", "global"),
    )
    dataset = load_dataset('LLM-LAT/harmful-dataset')
    traindata = dataset['train']
    questions = [item['prompt'] for item in traindata]
    answers = [item['rejected'] for item in traindata]
    for question, answer in tqdm(zip(questions, answers), total=args.max_samples):
        data_hash = hashlib.md5((question + answer).encode('utf-8')).hexdigest()
        output_path = os.path.join(output_dir, f"{data_hash}.pt")
        if os.path.exists(output_path):
            continue
        result = get_logprobs_gemini(client, tokenizer, question, answer, args)
        if result is None:
            continue
        torch.save(result, output_path)
        if len(os.listdir(output_dir)) >= args.max_samples:
            break
    print(f"Processed {len(os.listdir(output_dir))} samples and saved to {output_dir}")
if __name__ == "__main__":
    main()
