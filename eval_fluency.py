import os
import numpy as np
import math
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json

PROMPTS_DATASET = "" # The path of the prompts file
GEN_FILE        = "" # The path of the "generations.jsonl" produced by a NLG system like UDDIA: one generation a line; k*25~(k+1)*25 lines correspond with the k-th prompt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModelForCausalLM.from_pretrained('gpt2-xl').to(device)
tokenizer = AutoTokenizer.from_pretrained('gpt2-xl')

prompt_list  = open(PROMPTS_DATASET, "r").read().splitlines()
gen_list     = open(GEN_FILE, "r").read().splitlines()

perplexities = []
with torch.no_grad():
    for line in prompt_list:
        prompt = json.loads(line)["prompt"]["text"]
        prompt_input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
        prompt_loss = model(prompt_input_ids, labels=prompt_input_ids)[0] * (prompt_input_ids.shape[1]-1)

        for jj in range(cnt*25, cnt*25+25):
            now_sent = json.loads(gen_list[jj])
            full_input_ids = tokenizer.encode(prompt+now_sent, return_tensors='pt').to(device)
            full_loss = model(full_input_ids, labels=full_input_ids)[0] * (full_input_ids.shape[1]-1)
            loss = (full_loss - prompt_loss) / (full_input_ids.shape[1] - prompt_input_ids.shape[1])
            ppl = math.exp(loss.item())
            perplexities.append(ppl)

print("PPL = {}".format(round(np.nanmean(perplexities), 2)))