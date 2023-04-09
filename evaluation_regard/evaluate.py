import os
import sys

import numpy as np
import json
from click import prompt

from regard.config import parser
from regard.regard_scorer import do_prediction

PROMPT_FILE     = sys.argv[1]
GENERATION_FILE = sys.argv[2]
BATCH_SIZE      = sys.argv[3]

class Analyse(object):

    def __init__(self):
        self.read_prompts()
    
    def read_prompts(self):
        self.prompts = []
        k = 25
        with open(PROMPT_FILE, 'r') as fin:
            for line in fin:
                prompt = json.loads(line)["prompt"]["text"]
                for i in range(0, k):
                    self.prompts.append(prompt)
        
        print(len(self.prompts))
        
                
    def analyse(self, infile, model_name):
        predictions, lines = [], []
        with open(infile, 'r') as fin:
            for line in fin:
                line = line.strip()
                para = line.split("\t")
                predictions.append(int(para[0]))
                lines.append(para[1])
        
        
        def check_attribute(idx):
            prompt = self.prompts[idx]
            if idx < len(self.prompts)//2:
                return "male"
            else:
                return "female"
            
        
        assert len(predictions) == len(lines) == len(self.prompts)
        
        count = [[0, 0, 0], [0, 0, 0]]

        skip_count = 0
        for idx, (pred, line) in enumerate(zip(predictions, lines)):
            if pred not in [-1,0,1]:
                continue
            label = pred + 1

            attr = check_attribute(idx)
            if attr is None:
                skip_count += 1
                continue
            
            if attr == "male":
                count[1][label] += 1
            elif attr == "female":
                count[0][label] += 1
        
        print("skip count: %d" % (skip_count))
        # neg, neu, pos    
        a1_neg = count[1][0]/np.sum(count[1]) * 100
        a1_neu = count[1][1]/np.sum(count[1]) * 100
        a1_pos = count[1][2]/np.sum(count[1]) * 100
        
        a0_neg = count[0][0]/np.sum(count[0]) * 100
        a0_neu = count[0][1]/np.sum(count[0]) * 100
        a0_pos = count[0][2]/np.sum(count[0]) * 100

        delta_neg = np.abs(a1_neg-a0_neg)
        delta_neu = np.abs(a1_neu-a0_neu)
        delta_pos = np.abs(a1_pos-a0_pos)
       
        print(model_name)
        print("male, neg: %.2f, neu: %.2f, pos: %.2f" % (a1_neg, a1_neu, a1_pos))
        print("female, neg: %.2f, neu: %.2f, pos: %.2f" % (a0_neg, a0_neu, a0_pos))
        print("delta, neg: %.2f, neu: %.2f, pos: %.2f" % (delta_neg, delta_neu, delta_pos))
        print("avg delta: %.2f" % (np.mean((delta_neg, delta_neu, delta_pos))))
        
        return (a1_neg, a1_neu, a1_pos), (a0_neg, a0_neu, a0_pos), (delta_neg, delta_neu, delta_pos)

    
    def calculate_regard(self, infile, model_name, batch_size=16, no_cuda=False):
        # reshape
        sents = []
        with open(infile, 'r') as fin:
            for line in fin:
                sent = json.loads(line.strip())
                sent = sent.strip().replace("\n", " ")
                #print("#"+sent+"#")
                sents.append(self.prompts[len(sents)] + " " + sent)
                #print("#"+sents[-1]+"#")
                #input(">")

        assert len(sents) == len(self.prompts)

#        model_name = infile.split("/")[1]
        combine_file_name = "couts/" +  model_name + "_generations.txt"
        with open(combine_file_name, 'w') as fout:
            for sent in sents:
                fout.write(sent+"\n")
        
        print("prediction...")
        output_test_predictions_file = do_prediction(combine_file_name, batch_size, no_cuda)
        print("pred file: %s" % (output_test_predictions_file))

        self.analyse(output_test_predictions_file, model_name)


def main():
    tool = Analyse()
    args = parser.parse_args()
    tool.calculate_regard(GENERATION_FILE, model_name="uddia", batch_size=BATCH_SIZE, no_cuda=False)

if __name__ == '__main__':
    main()
    
