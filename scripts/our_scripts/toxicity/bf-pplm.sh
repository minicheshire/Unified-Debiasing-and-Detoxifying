API_RATE=20
OUTPUT_DIR=generations/toxicity/bf-pplm
PROMPTS_DATASET=prompts/nontoxic_prompts-1k.jsonl

python -m scripts.run_toxicity_experiment \
    --use-dataset \
    --dataset-file $PROMPTS_DATASET \
    --model-type bf-pplm \
	--model 'toxicity-large' \
    --p 0.9 \
    --batch-size 1 \
    --bf_lr 0.01 \
    --bf_iter 5 \
    --perspective-rate-limit $API_RATE \
	$OUTPUT_DIR
