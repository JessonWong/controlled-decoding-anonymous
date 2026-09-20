import os
import torch
import hashlib
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import argparse
import os
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="meta-llama/Llama-3.1-8B-Instruct", help="Path to the pre-trained model or model identifier from huggingface.co/models.")
    parser.add_argument("--output_dir", type=str, default="./cached_logits", help="Directory to save the processed data.")
    parser.add_argument("--max_samples", type=int, default=100, help="Maximum number of samples to process from the dataset.")
    args = parser.parse_args()
    return args

def get_logprobs(model, tokenizer, question:str, answer:str):
    question_template = [
        {"role": "user", "content": question},
        {"role": "assistant", "content": ""}
    ]
    whole_template = [
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer}
    ]
    question_str = tokenizer.apply_chat_template(question_template, tokenize = False)
    whole_str = tokenizer.apply_chat_template(whole_template, tokenize = False)
    input_ids = tokenizer(whole_str, return_tensors="pt").input_ids.to(model.device)
    question_ids = tokenizer(question_str, return_tensors="pt").input_ids.to(model.device)
    special_token_ids = set(tokenizer.all_special_ids or [])
    trimmed_question_ids = question_ids
    while trimmed_question_ids.shape[1] > 0 and trimmed_question_ids[0, -1].item() in special_token_ids:
        trimmed_question_ids = trimmed_question_ids[:, :-1]

    compare_len = min(trimmed_question_ids.shape[1], input_ids.shape[1])
    divergence_idx = compare_len
    for idx in range(compare_len):
        if trimmed_question_ids[0, idx].item() != input_ids[0, idx].item():
            divergence_idx = idx
            break

    if divergence_idx == 0:
        raise ValueError("Question tokens do not align with the beginning of the input_ids.")

    if divergence_idx < trimmed_question_ids.shape[1]:
        trimmed_question_ids = trimmed_question_ids[:, :divergence_idx]

    if trimmed_question_ids.shape[1] == 0:
        raise ValueError("Unable to align question tokens with the input_ids.")

    if trimmed_question_ids.shape[1] > input_ids.shape[1]:
        trimmed_question_ids = trimmed_question_ids[:, :input_ids.shape[1]]

    if not torch.equal(trimmed_question_ids, input_ids[:, :trimmed_question_ids.shape[1]]):
        raise ValueError("Question tokens do not align with the beginning of the input_ids.")

    question_ids = trimmed_question_ids
    with torch.no_grad():
        outputs = model(input_ids=input_ids)
        logits = outputs.logits
        log_probs = torch.log_softmax(logits, dim=-1)
    answer_ids = input_ids[:, question_ids.shape[1]:]
    answer_log_probs = log_probs[:, question_ids.shape[1]-1:input_ids.shape[1]-1, :]
    assert answer_ids.shape[1] == answer_log_probs.shape[1]
    return {"log_probs": answer_log_probs.cpu(), "labels": answer_ids.cpu()}

def main():
    args = parse_args()
    model_name_or_path = args.model_name_or_path
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True)
    model.eval()
    dataset = load_dataset('LLM-LAT/harmful-dataset')
    questions = []
    answers = []
    for i in range(100, 1000):
        item = dataset['train'][i]
        questions.append(item['prompt'])
        answers.append(item['rejected'])
    for question, answer in tqdm(zip(questions, answers), total=args.max_samples):
        data_hash = hashlib.md5((question + answer).encode('utf-8')).hexdigest()
        output_path = os.path.join(output_dir, f"{data_hash}.pt")
        if os.path.exists(output_path):
            continue
        result = get_logprobs(model, tokenizer, question, answer)
        torch.save(result, output_path)
        if len(os.listdir(output_dir)) >= args.max_samples:
            break
    print(f"Processed {len(os.listdir(output_dir))} samples. Logits saved in {output_dir}.")

if __name__ == "__main__":
    main()
