# Unified Detoxifying and Debiasing in Language Generation via Inference-time Adaptive Optimization

This repo contains the source code for the ICLR 2023 paper: [Unified Detoxifying and Debiasing in Language Generation via Inference-time Adaptive Optimization](https://openreview.net/forum?id=FvevdI0aA_h). The code is largely based on the DExperts [repo](https://github.com/alisawuffles/DExperts), and the requirements follow those of the DExperts repo as well.

The main algorithm of the UDDIA framework proposed in the paper can be found in the [uddia_generation.py](https://github.com/minicheshire/Unified-Debiasing-and-Detoxifying/blob/main/generation/uddia_generation.py). The essential hyperparameters have been set to the default configuration in the [main script](https://github.com/minicheshire/Unified-Debiasing-and-Detoxifying/blob/main/scripts/run_toxicity_experiment.py) of the toxicity experiments. 

Before generating, you need to prepare the prompts in the `prompts/` directory. We've also provided the 175 prompt pairs for the experiments in the Table 5 of our paper in the directory.

To generate with UDDIA, run
```bash
python -m scripts.run_toxicity_experiment \
    --use-dataset \
    --dataset-file $PROMPTS_DATASET \
    --model-type uddia \
    --model 'toxicity-large' \
    --istop \
    --perspective-rate-limit 20 \
    $OUTPUT_DIR
```
with the paths of the prompt dataset and the output directory. You can also run the [`run_uddia.sh`](https://github.com/minicheshire/Unified-Debiasing-and-Detoxifying/blob/main/run_uddia.sh) to have a try.

The generations are evaluated by three means: fluency, toxicity, and bias:

- For fluency, run the [`eval_fluency.py`](https://github.com/minicheshire/Unified-Debiasing-and-Detoxifying/blob/main/eval_fluency.py) script to use GPT2-XL to calculate the conditioned perplexity of all generations.
- For toxicity, run the [`eval_toxicity.py`](https://github.com/minicheshire/Unified-Debiasing-and-Detoxifying/blob/main/eval_toxicity.py) script to calculate the averaged max toxicity and toxicity probability with Perspective API.
- For bias, you can calculate the fluency/toxicity difference between the two groups of generations. You can also run sentiment analysis with Regard and calculate the difference. The [`evaluation_regard`](https://github.com/minicheshire/Unified-Debiasing-and-Detoxifying/tree/main/evaluation_regard) directory contains the evaluation code: download the finetuned BERT checkpoint from [here](https://drive.google.com/file/d/1K3IXhoI1M55bOXNelJDBolt72uSlysnU/view?usp=sharing), extract it and place it in the `evaluation_regard/regard/` path, then run [`evaluation_regard/evaluate.py`](https://github.com/minicheshire/Unified-Debiasing-and-Detoxifying/blob/main/evaluation_regard/evaluate.py). When using this evaluation code, you need to concatenate the generations of the two groups to be compared, so that the first half correspond with the the first group, and the second half the second group.

If you find this repository useful for your research, please cite our work:

```
@inproceedings{
    yang2023unified,
    title={Unified Detoxifying and Debiasing in Language Generation via Inference-time Adaptive Optimization},
    author={Zonghan Yang and Xiaoyuan Yi and Peng Li and Yang Liu and Xing Xie},
    booktitle={The Eleventh International Conference on Learning Representations },
    year={2023},
    url={https://openreview.net/forum?id=FvevdI0aA_h}
}
```