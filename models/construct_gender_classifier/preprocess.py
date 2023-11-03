import torch.nn.functional as F
import numpy as np
import json
import sys

from transformers import GPT2Config, GPT2Tokenizer, GPT2LMHeadModel
from spca import SPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

def getEmb(model_path="./GPT2_LARGE_CKPT/"):
    config = GPT2Config()
    model = GPT2LMHeadModel(config)
    print("load model...")
    model = model.from_pretrained(model_path)
    
    print(model.transformer.wte.weight.size())
    emb = model.transformer.wte.weight.detach().cpu().numpy()
    np.save("gpt2emb.npy", emb)

def build_spca(infile, outfile):
    
    tokenizer = GPT2Tokenizer.from_pretrained("vocab", use_fast=False)
    unk_id = tokenizer.unk_token_id
    print("unk id: %d" % (unk_id))
    
    print("load embedding...")
    wordemb = np.load("gpt2emb.npy")
    print(np.shape(wordemb))
    
    X, Y = [], []
    with open(infile, 'r') as fin:
        for line in fin:
            para = line.strip().split(" ")
            vs = []
            for i, w in enumerate(para):
                print(w)
                ids = tokenizer(w)['input_ids']
                print(ids)
                print([tokenizer.decode(idx,
                    clean_up_tokenization_spaces=False,skip_special_tokens=False) for idx in ids])
                
                #input(">")
                
                vec = []
                for idx in ids:
                    vec.append(wordemb[idx, :])
                v = np.mean(vec, axis=0)
                vs.append(v)
            
            
            vmu = np.mean(vs, axis=0)
            
            X.append(vs[0] - vmu)
            Y.append([0])
            
            X.append(vs[1] - vmu)
            Y.append([1])
    
    
    X = np.array(X)
    Y = np.array(Y)
    
    print(np.shape(X), np.shape(Y))
    
    spca = SPCA(n_components=20)
    spca.fit(X.T, Y.T)

    print(np.shape(spca.U))

    np.save(outfile, spca.U.T)

#------------------------------------------
#------------------------------------------


def test_direction():
    
    components = np.load("gender_components_spcalarge.npy")
    v = components[0]
    print(np.shape(v))
    
    print("load embedding...")
    wordemb = np.load("gpt2emb.npy")
    print(np.shape(wordemb))
    
    v_norm = np.linalg.norm(v)
    
    tokenizer = GPT2Tokenizer.from_pretrained("vocab", use_fast=False)
    
    while True:
        word = input("input word >")
        w_ids = tokenizer(word.strip())['input_ids']
        
        vec = []
        for idx in w_ids:
            vec.append(wordemb[idx, :])
                
        w = np.mean(vec, axis=0)
    
        w_norm = np.linalg.norm(w)
        theta = np.dot(v, w.T)/(v_norm * w_norm) # male
        
        print(theta)

        print(np.dot(v, w.T)/v_norm)
        
        print((1.0+theta)/2)
        
        p = (1.0+theta)/2
        probs = np.array([1-p, p])
        tau = 0.5
        print(probs)
        print(np.exp(probs/tau)/np.sum(np.exp(probs/tau)))
        
        print("")


def test_direction_file():
    
    components = np.load("gender_components_spcalarge.npy")
    v = components[0]
    print(np.shape(v))
    
    print("load embedding...")
    wordemb = np.load("gpt2emb.npy")
    print(np.shape(wordemb))
    
    v_norm = np.linalg.norm(v)
    tokenizer = GPT2Tokenizer.from_pretrained("vocab", use_fast=False)

    def get_prob(w):
        w_ids = tokenizer(w.strip())['input_ids']
        vec = []
        for idx in w_ids:
            vec.append(wordemb[idx, :])
        w = np.mean(vec, axis=0)
        w_norm = np.linalg.norm(w)
        theta = np.dot(v, w.T)/(v_norm * w_norm) # male
        return (1.0+theta)/2
    
    data = []
    with open("gender_pairs.txt", 'r') as fin:
        for line in fin:
            para = line.strip().split(" ")
            
            p1 = get_prob(para[0])
            p2 = get_prob(para[1])
            
            data.append(np.abs(p1-p2))
    
    print(np.mean(data))



def build_classifier():
    components = np.load("gender_components_spcalarge.npy")
    v = components[0]
    print(np.shape(v))
    
    print("load embedding...")
    wordemb = np.load("gpt2emb.npy")
    print(np.shape(wordemb))
    
    v_norm = np.linalg.norm(v)
    
    tokenizer = GPT2Tokenizer.from_pretrained("vocab", use_fast=False)
    
    def get_prob(w):
        w_norm = np.linalg.norm(w)
        theta = np.dot(v, w.T)/(v_norm * w_norm) # male
        p = (1.0+theta)/2
        probs = np.array([1-p, p])
        return probs
    
    matrix = []
    for idx in range(0, tokenizer.vocab_size):
        matrix.append(get_prob(wordemb[idx, :]))
    
    matrix = np.array(matrix)
    print(np.shape(matrix))
    
    np.save("gender_matrixlarge.npy", matrix)


def main():
    # 1. get gpt embedding matrix from the GPT-2 Large checkpoint, obtain gpt2emb.npy
    getEmb(sys.argv[1])
    
    # 2. obtain the pca components. Requirements: gpt tokenizer files in stored in "vocab/"
    #    and gender_pairs.txt, output gender_components_spcalarge.npy using pca or spca
    #build_spca("gender_pairs.txt", "gender_components_spcalarge.npy")
    
    # 3. test the male polarity of the word you input
    #    You should get a higher score when inputing a male-related words 
    #    than the female related ones
    test_direction()
    
    # 4. calculate the gender polarity of words in gender_pairs.txt, the higher the better
    test_direction_file()
    
    # 5. pre-calculate and build gender_matrix.npy
    #build_classifier()


if __name__ == "__main__":
    main()
