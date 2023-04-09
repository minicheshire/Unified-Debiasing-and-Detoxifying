#! /usr/bin/env python3
# coding=utf-8

# Copyright (c) 2019 Uber Technologies, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from operator import add
from typing import Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from tqdm import trange
from transformers import GPT2Tokenizer, Pipeline
from transformers.file_utils import cached_path
from transformers.modeling_gpt2 import GPT2LMHeadModel

from modeling.pplm_classification_head import ClassificationHead

import math
import json

SMALL_CONST = 1e-15
BIG_CONST = 1e10
VERBOSE = False

DISCRIMINATOR_MODELS_PARAMS = {
    "sentiment-large": {
        "path": "models/pplm_classifiers/sentiment_classifierhead_1280/SST_classifier_head_epoch_10.pt",
        "class_size": 5,
        "embed_size": 1280,
        "class_vocab": {"very_positive": 2, "very_negative": 3},
        "default_class": 3,
        "pretrained_model": "gpt2-large",
    },
    "toxicity-large": {
        "path": "models/pplm_classifiers/toxicity_classifierhead_1280/toxic_classifier_head_epoch_10.pt",
        "class_size": 2,
        "embed_size": 1280,
        "class_vocab": {"non_toxic": 0, "toxic": 1},
        "default_class": 0,
        "pretrained_model": "gpt2-large",
    },
}


def check(n, l, isTop):
    nnn = n.split(".")
    ret = True
    if isTop:
        if (nnn[1] == "h") and (int(nnn[2])<36-l): ret=False
    else:
        if (nnn[1] == "h") and (int(nnn[2])>=l): ret=False
    return ret

def to_var(x, requires_grad=False, volatile=False, device="cuda"):
    if torch.cuda.is_available() and device == "cuda":
        x = x.cuda()
    elif device != "cuda":
        x = x.to(device)
    return Variable(x, requires_grad=requires_grad, volatile=volatile)


def top_k_top_p_filtering(probs_or_logits, top_k: int=0, top_p: float=1.0, is_probs=False):
    """
    Top k or top p sampling
    """
    if top_k == 0 and top_p == 1.0:
        return probs_or_logits
    elif top_k > 0:
        values = torch.topk(probs_or_logits, top_k)[0]
        batch_mins = values[:, -1].view(-1, 1).expand_as(probs_or_logits)
        filter_value = 0.0 if is_probs else -BIG_CONST
        return torch.where(probs_or_logits < batch_mins, torch.ones_like(probs_or_logits) * filter_value, probs_or_logits)
    elif top_p < 1.0:
        sorted, indices = torch.sort(probs_or_logits, descending=True)
        sorted_probs = sorted if is_probs else F.softmax(sorted, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        # Remove tokens with cumulative probability above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        
        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(1, indices, sorted_indices_to_remove)
        filter_value = 0.0 if is_probs else -BIG_CONST
        probs_or_logits[indices_to_remove] = filter_value
        
        return probs_or_logits


def get_classifier(name: str, device: str) -> ClassificationHead:
    params = DISCRIMINATOR_MODELS_PARAMS[name]
    classifier = ClassificationHead(class_size=params["class_size"], embed_size=params["embed_size"]).to(device)
    if "url" in params:
        resolved_archive_file = cached_path(params["url"])
    elif "path" in params:
        resolved_archive_file = params["path"]
    else:
        raise ValueError("Either url or path have to be specified " "in the discriminator model parameters")
    classifier.load_state_dict(torch.load(resolved_archive_file, map_location=device))
    classifier.eval()
    return classifier


def get_class_id(name: str, class_label: Union[str, int]) -> Optional[int]:
    params = DISCRIMINATOR_MODELS_PARAMS[name]
    if isinstance(class_label, str):
        if class_label in params["class_vocab"]:
            label_id = params["class_vocab"][class_label]
        else:
            label_id = params["default_class"]
            print("class_label {} not in class_vocab".format(class_label))
            print("available values are: {}".format(params["class_vocab"]))
            print("using default class {}".format(label_id))
    elif isinstance(class_label, int):
        if class_label in set(params["class_vocab"].values()):
            label_id = class_label
        else:
            label_id = params["default_class"]
            print("class_label {} not in class_vocab".format(class_label))
            print("available values are: {}".format(params["class_vocab"]))
            print("using default class {}".format(label_id))
    else:
        label_id = params["default_class"]
    return label_id

def generate_text_uddia(
        model,
        biased_distri,
        bias_weight,
        gender_matrix,
        gender_direction,
        gender_direction_norm,        
        original_bias=[],
        context=None,
        device="cuda",
        classifier=None,
        class_label=None,
        length=100,
        temperature=1.0,
        top_k=0,
        top_p=1.0,
        sample=False,
        dt_lr=0.01,
        dt_iter=5,
        horizon_length=1,
        gm_scale=0.9,
        kl_scale=0.01,
        repetition_penalty=1.0,
        isTop=True,
        isMLP=True,
        layer_tune_num=36, # The upper `layer_tune_num` layers to be tuned (T_0)
):
    output_so_far = None
    if context:
        context_t = torch.tensor(context, device=device, dtype=torch.long)     
        while len(context_t.shape) < 2:
            context_t = context_t.unsqueeze(0)
        output_so_far = context_t
        with torch.no_grad():
            prompt_loss = model(output_so_far, labels=output_so_far)[0] * (output_so_far.shape[1]-1)

    last = None
    past_no_last = None
    accumulated_hidden = None
    bias_intervene_times = 0
    for i in trange(length, desc='Generating with UDDIA', disable=True):
        optimizer = torch.optim.Adam(model.parameters(), dt_lr)

        # Get past/probs for current output, except for last 1 word
        # Note that GPT takes 2 inputs: past + current_token

        # run model forward to obtain unperturbed
        if past_no_last is None and output_so_far is not None:
            last = output_so_far[:, -1:]
            if output_so_far.shape[1] > 1:
                _, past_no_last, _ = model(output_so_far[:, :-1])

        if accumulated_hidden is None:
            logits_all, past_all, hidden_all = model(output_so_far)
            accumulated_hidden = hidden_all[-1][:, :-1, :] # top layer, all but last tokens: bsz, len, dim
            accumulated_hidden = torch.sum(accumulated_hidden, dim=1) # bsz, dim

        # past_no_last keeps unchanged
        # updated dt_params leads to changed hidden activations with the ``last" as input

        # When to intervene for detoxifying: every time step
        for j in range(dt_iter):
            now_logits_last, _, now_hidden_last = model(last, past=past_no_last)
            new_accumulated_hidden = accumulated_hidden + torch.sum(now_hidden_last[-1], dim=1).detach()
            logits = now_logits_last[:, -1, :]
            probs = F.softmax(logits, dim=-1)

            loss = 0.0
            ce_loss = torch.nn.CrossEntropyLoss()
            curr_probs = torch.unsqueeze(probs, dim=1)
            wte = model.resize_token_embeddings()
            for _ in range(horizon_length):
                inputs_embeds = torch.matmul(curr_probs, wte.weight.data)
                _, _, now_hidden_next = model(past=past_all, inputs_embeds=inputs_embeds)
                new_accumulated_hidden = new_accumulated_hidden + torch.sum(now_hidden_next[-1], dim=1)

            _, _, _, curr_length, _ = past_no_last[0].shape
            prediction = classifier(new_accumulated_hidden / (curr_length + 1 + horizon_length))
            label = torch.tensor(prediction.shape[0] * [class_label], device=device, dtype=torch.long)
            discrim_loss = ce_loss(prediction, label)
            loss += discrim_loss


            # Compute the Hellinger distance (`delta`)
            hell0 = torch.sqrt(
                1-torch.sqrt(probs*biased_distri[:, 0].unsqueeze(0)).sum()
            )
            hell1 = torch.sqrt(
                1-torch.sqrt(probs*biased_distri[:, 1].unsqueeze(0)).sum()
            )
            delta = np.abs(hell0.item()-hell1.item()) 

            # The threshold for delta is 0.1
            # if delta > 0.1, then also need to debias
            if delta > 0.1:
                # probs: (B, V)
                loss1 = torch.sum(probs * bias_weight[:, 0].unsqueeze(0) \
                    + probs * bias_weight[:, 1].unsqueeze(0), dim=-1) / 2

                B, L, D = output_so_far.size(0), output_so_far.size(1), 1280
                c_rep = model.transformer.wte(output_so_far).mean(dim=1, keepdim=True) # (B, L, D) - > (B, 1, D)
                # (B, V, D) + (1, V, D) -> (B, V, D)
                xc_rep = (c_rep.expand(B, 50257, model.transformer.wte.weight.shape[1]) * L \
                    + model.transformer.wte.weight.unsqueeze(0)) / (L+1.0)

                w = xc_rep.view(-1, D)
                # w: (M, D)
                w_norm = torch.linalg.norm(w, dim=-1, keepdim=True) # (M, 1)
                #print(w_norm.size())
                # (M, D) * (D, 1) -> (M, 1)
#                print(w.shape, gender_direction.shape, gender_direction_norm.shape, w_norm.shape)
                theta = w.matmul(gender_direction) / (gender_direction_norm * w_norm)
                #print(theta.size())
                
                ppp = (1.0 + theta)/2
                p_xc = torch.cat([1-ppp, ppp], dim=-1)
                p_xc = F.softmax(p_xc/0.1, dim=-1) # tau=0.1

                p_xc = p_xc.view(B, 50257, 2) # (B, V, K), p(a|xc)
                
                # (B, V) * (B, V, 2)
                
                kl = p_xc * torch.log(p_xc / (gender_matrix.unsqueeze(0) + 1e-10) + 1e-10)
                kl = kl.sum(dim=-1) # (B, V)
                
                #print(kl.size())
                #input(">")
                
                loss_b = torch.sum(probs * kl, dim=-1) + loss1
                loss_b = loss.mean()                

                loss = loss + 0.05 * loss_b #bias loss weight = 0.05
                bias_intervene_times += 1

            if VERBOSE: print("Decoding time step {}, optimization step {}, discrim_loss {}, kl_loss {}".format(i, j, discrim_loss.data, kl_loss.data))
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()

        # reset part of the bias terms immediately after optimizing
        cnt = 0
        for n,p in model.named_parameters():
            if (layer_tune_num != 18) and ("bias" in n) and check(n,18,isTop) and (not check(n,layer_tune_num,isTop)):
                p.data = original_bias[cnt].to(device)
                cnt += 1            

        tuned_logits, tuned_past, tuned_all_hidden = model(last, past=past_no_last)
        tuned_logits = tuned_logits[:, -1, :] / temperature  # + SMALL_CONST

        for token_idx in set(output_so_far[0].tolist()):
            if tuned_logits[0, token_idx] < 0:
                tuned_logits[0, token_idx] *= repetition_penalty
            else:
                tuned_logits[0, token_idx] /= repetition_penalty

        tuned_probs = F.softmax(tuned_logits, dim=-1)

        # Fuse the modified model and original model
        untuned_probs = F.softmax(logits_all[:, -1, :], dim=-1)
        tuned_probs = (tuned_probs ** gm_scale) * (untuned_probs ** (1 - gm_scale))  # + SMALL_CONST
        tuned_probs = top_k_top_p_filtering(tuned_probs, top_k=top_k, top_p=top_p, is_probs=True)  # + SMALL_CONST

        # rescale
        if torch.sum(tuned_probs) <= 1:
            tuned_probs = tuned_probs / torch.sum(tuned_probs)

        # reset the bias
        for n,p in model.named_parameters():
            if ("bias" in n) and check(n,layer_tune_num,isTop):
                p.data = original_bias[cnt].to(device)
                cnt += 1

        _, _, hidden = model(last, past=past_no_last)
        accumulated_hidden = accumulated_hidden + torch.sum(hidden_all[-1], dim=1) # update the accumulated hidden with the token before the ``last" is updated

        # sample or greedy
        if sample:
            last = torch.multinomial(tuned_probs, num_samples=1)
        else:
            _, last = torch.topk(tuned_probs, k=1, dim=-1)

        past_no_last = past_all # now past_no_last is the past_all of the last step i
        logits_all, past_all, hidden_all = model(last, past=past_no_last) # now past_all is based on the updated past_no_last

        # update context/output_so_far appending the new token
        output_so_far = last if output_so_far is None else torch.cat((output_so_far, last), dim=1)
    
    with torch.no_grad():
        full_loss = model(output_so_far, labels=output_so_far)[0] * (output_so_far.shape[1]-1)
    ppl = math.exp((full_loss - prompt_loss).item() / length)

    return output_so_far, ppl, bias_intervene_times

class UDDIAGeneration(Pipeline):
        
    def __init__(self,
                 discrim: str,
                 seed=0,
                 isTop=True,
                 isMLP=False,
                 layer_tune_num=36,
                 **kwargs):
        # Set random seed
        torch.manual_seed(seed)
        np.random.seed(seed)

        pretrained_model = DISCRIMINATOR_MODELS_PARAMS[discrim]["pretrained_model"]
        print("discrim = {}, pretrained_model set " "to discriminator's = {}".format(discrim, pretrained_model))

        # load pretrained model
        model = GPT2LMHeadModel.from_pretrained(pretrained_model, output_hidden_states=True)
        model.eval()

        # Freeze GPT-2 weights
        for param in model.parameters():
            param.requires_grad = False

        # But enable the grad of the bias terms
        original_bias = []
        for n,p in model.named_parameters():
            if ("bias" in n) and check(n,layer_tune_num,isTop):
                original_bias.append(p.data.clone().detach())
                p.requires_grad = True

        # load tokenizer
        tokenizer = GPT2Tokenizer.from_pretrained(pretrained_model)

        super().__init__(model=model, tokenizer=tokenizer, **kwargs)

        # Additional setup after creating model and tokenizer
        self.discrim = discrim
        classifier = get_classifier(self.discrim, self.device)
        self.classifier = classifier
        self.original_bias = original_bias

        # Load the classifier for debiasing (different from that for detoxifying)
        self._load_classifier("models/debias-classifiers/gender_matrixlarge.npy",
                              "models/debias-classifiers/gender_components_spcalarge.npy",
                              "models/debias-classifiers/openwebtext_freq.json")

    def _load_classifier(self, matrix_path, direction_path, freq_path):
        print("load classifier...")
        self.gender_direction = torch.tensor(np.load(direction_path)[0],
            dtype=torch.float, device=self.device).view(-1, 1)
        self.gender_direction_norm = torch.linalg.norm(self.gender_direction)
        
        # gender matrix: p(a|x), x represents a token
        gender_matrix = torch.tensor(np.load(matrix_path), device=self.device)
        self.gender_matrix = F.softmax(gender_matrix/0.1, dim=-1) #  tau=0.1
        
        # p(a|x)* log[p(a|x)*2.0]
        self.bias_weight = self.gender_matrix * torch.log(self.gender_matrix*2.0+1e-10)
        
        # build biased token distribution
        # p(x|a) = p(x) * p(a|x) / p (a)
        freqs_count = []
        with open(freq_path, 'r') as fin:
            for line in fin:
                dic = json.loads(line.strip())
                freqs_count.append(dic['freq'])

        freqs_count = freqs_count[0:-1]    
        freqs_count = torch.tensor(np.array(freqs_count) / np.sum(freqs_count),
            dtype=torch.float, device=self.device)

        biased_distri = torch.zeros_like(self.gender_matrix)
        
        self.biased_mask = self.gender_matrix.gt(0.75).to(torch.float) #thres=0.75
        biased_distri[:, 0] = freqs_count * self.gender_matrix[:, 0] * self.biased_mask[:, 0]
        biased_distri[:, 1] = freqs_count * self.gender_matrix[:, 1] * self.biased_mask[:, 1]
        
        self.biased_distri = biased_distri
        self.biased_distri[:, 0] = biased_distri[:, 0] / torch.sum(biased_distri[:, 0])
        self.biased_distri[:, 1] = biased_distri[:, 1] / torch.sum(biased_distri[:, 1])

    # Default parameters correspond to those in the PPLM paper for the toxicity discriminative model
    # Others (such as sampling) taken from https://github.com/huggingface/transformers/tree/master/examples/pplm
    def __call__(self,
                 cond_text='',
                 num_samples=1,
                 class_label=-1,
                 length=20,
                 dt_lr=0.01,
                 dt_iter=5,                 
                 temperature=1.0,
                 top_k=0,
                 top_p=1.0,
                 sample=True,
                 horizon_length=1,
                 gm_scale=0.9,
                 kl_scale=0.01,
                 repetition_penalty=1.0,
                 clean_up_tokenization_spaces=True,
                 include_context_in_generation=False,
                 isTop=True,
                 isMLP=False,
                 layer_tune_num=36,
                 ppl_thres=40,
                 layer_tune_freq=6):
        # Tokenize text
        tokenized_cond_text = self.tokenizer.encode(self.tokenizer.bos_token + cond_text)

        class_id = get_class_id(self.discrim, class_label)

        pert_gen_tok_texts = []
        records = []
        for i in range(num_samples):
            ppl = 10000
            min_ppl = 10000
            min_pert_gen_tok_text = None
            now_layer_tune_num = layer_tune_num
            ret = []

            # Redo mechanism
            while ppl > ppl_thres:
                pert_gen_tok_text, ppl, bias_intervene_times = generate_text_uddia(
                    dt_lr=dt_lr,
                    dt_iter=dt_iter,
                    original_bias=self.original_bias,
                    model=self.model,
                    context=tokenized_cond_text,
                    device=self.device,
                    classifier=self.classifier,
                    class_label=class_id,
                    length=length,                
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    sample=sample,
                    horizon_length=horizon_length,
                    gm_scale=gm_scale,
                    kl_scale=kl_scale,
                    repetition_penalty=repetition_penalty,
                    isTop=isTop,
                    isMLP=isMLP,
                    layer_tune_num=now_layer_tune_num,
                    biased_distri=self.biased_distri,
                    gender_matrix=self.gender_matrix,
                    gender_direction=self.gender_direction,
                    gender_direction_norm=self.gender_direction_norm,
                    bias_weight=self.bias_weight
                )
                ret.append([now_layer_tune_num, ppl, bias_intervene_times])#, pert_gen_tok_text])
                
                if ppl < min_ppl:
                    min_ppl = ppl
                    min_pert_gen_tok_text = pert_gen_tok_text
                
                # `layer_tune_freq` is the \Delta T
                now_layer_tune_num -= layer_tune_freq
                if now_layer_tune_num <= 0:
                    pert_gen_tok_text = min_pert_gen_tok_text
                    break

            pert_gen_tok_texts.append(pert_gen_tok_text)
            records.append(ret)

        decode_start_idx = 0 if include_context_in_generation else len(tokenized_cond_text)
        pert_gen_tok_texts = [x[0, decode_start_idx:].tolist() for x in pert_gen_tok_texts]
        pert_gen_texts = [
            self.tokenizer.decode(pert_gen_tok_text, clean_up_tokenization_spaces=clean_up_tokenization_spaces)
            for pert_gen_tok_text in pert_gen_tok_texts
        ]

        return pert_gen_texts, records
