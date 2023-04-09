API_RATE=20
OUTPUT_DIR=generations/toxicity/uddia
PROMPTS_DATASET=prompts/nontoxic_prompts-10.jsonl

python -m scripts.run_toxicity_experiment \
    --use-dataset \
    --dataset-file $PROMPTS_DATASET \
    --model-type uddia \
    --model 'toxicity-large' \
    --istop \
    --perspective-rate-limit $API_RATE \
    $OUTPUT_DIR
