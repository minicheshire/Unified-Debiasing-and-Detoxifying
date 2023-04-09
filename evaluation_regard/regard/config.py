import argparse

parser = argparse.ArgumentParser()

# Required parameters

parser.add_argument(
	"--test_file",
	default=None,
	type=str,
	required=False,
	help="Test file, if None, defaults to `test.tsv` file in data_dir."
)

parser.add_argument(
	"--data_dir",
	default="outputs",
	type=str
)

parser.add_argument(
	"--model_version",
	default=2,
	type=int,
	help="1 or 2.",
)

parser.add_argument(
	"--model_type",
	default='bert',
	type=str
)

parser.add_argument(
	"--model_name_or_path",
	default="bert_regard_v2_large/checkpoint-300",
	type=str,
)

parser.add_argument(
	"--output_dir",
	default="outputs",
	type=str
)

# Other parameters

parser.add_argument(
	"--max_seq_length",
	default=128,
	type=int,
)

parser.add_argument(
	"--do_lower_case", action="store_true"
)

parser.add_argument(
	"--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
)

parser.add_argument(
	"--per_gpu_eval_batch_size", default=8, type=int
)

parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
parser.add_argument(
	"--overwrite_output_dir", action="store_true", help="Overwrite the content of the output directory"
)

parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")

parser.add_argument(
	"--fp16",
	action="store_true",
	help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
)

parser.add_argument(
	"--fp16_opt_level",
	type=str,
	default="O1",
	help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
	"See details at https://nvidia.github.io/apex/amp.html",
)

parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
parser.add_argument("--server_ip", type=str, default="", help="For distant debugging.")
parser.add_argument("--server_port", type=str, default="", help="For distant debugging.")
parser.add_argument("--read_file", type=str, default="", help="read_file")
