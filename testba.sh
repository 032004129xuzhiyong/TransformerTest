exit 0

# 本地端口转发：本地访问远程服务
ssh -L 6006:localhost:6006 -p 22  inspur@10.193.127.200

# 实验室环境
CUDA_HOME=/public/soft/cuda/cuda-12.4  TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=4,5,6,7 accelerate launch main.py --model_name_or_path ./models--princeton-nlp--sup-simcse-roberta-base/snapshots/4bf73c6b5df517f74188c5e9ec159b2208c89c08/