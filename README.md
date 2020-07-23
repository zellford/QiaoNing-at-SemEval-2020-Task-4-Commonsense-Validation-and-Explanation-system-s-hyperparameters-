# QiaoNing-at-SemEval-2020-Task-4-Commonsense-Validation-and-Explanation-system-s-hyperparameters-

All the pretrained models' checkpoints are download from https://github.com/huggingface/transformers

Model command

Bert

python run_semeval.py \
  --model_type bert \
  --model_name_or_path bert-base-uncased \
  --task_name semeval \
  --do_train \
  --do_eval \
  --do_lower_case \
  --data_dir /root/code/data/semeval_data  \
  --max_seq_length 64 \
  --per_gpu_train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 5.0 \
  --output_dir /tmp/semeval_output/ 

Xlnet

python run_semeval.py \
  --model_type xlnet \
  --model_name_or_path xlnet-base-cased \
  --task_name semeval \
  --do_train \
  --do_eval \
  --do_lower_case \
  --data_dir /root/code/data/semeval_data  \
  --max_seq_length 128 \
  --per_gpu_train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 5.0 \
  --output_dir /tmp/outputs/semeval_xlnet_output/ 

Large-bert-uncased

python run_semeval.py \
  --model_type bert \
  --model_name_or_path /root/code/bert-large-uncased_L-24_H-1024_A-16 \
  --task_name semeval \
  --do_train \
  --do_eval \
  --do_lower_case \
  --data_dir /root/code/data/semeval_data  \
  --max_seq_length 64 \
  --per_gpu_train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 5.0 \
  --output_dir /tmp/outputs/semeval_large_bert_output/ 

large-bert-cased

python run_semeval.py \
  --model_type bert \
  --model_name_or_path /root/code/bert-large-cased_L-24_H-1024_A-16 \
  --task_name semeval \
  --do_train \
  --do_eval \
  --do_lower_case \
  --data_dir /root/code/data/semeval_data  \
  --max_seq_length 64 \
  --per_gpu_train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 5.0 \
  --output_dir /tmp/outputs/semeval_large_bert_cased_output/ 

xlm

python run_semeval.py \
  --model_type xlm \
  --model_name_or_path  xlm-mlm-en-2048  \
  --task_name semeval \
  --do_train \
  --do_eval \
  --do_lower_case \
  --data_dir /root/code/data/semeval_data  \
  --max_seq_length 64 \
  --per_gpu_train_batch_size 16 \
  --learning_rate 2e-5 \
  --num_train_epochs 5.0 \
  --output_dir /tmp/outputs/semeval_xlm_output/ 

roberta

python run_semeval.py \
  --model_type roberta \
  --model_name_or_path roberta-base \
  --task_name semeval \
  --do_train \
  --do_eval \
  --do_lower_case \
  --data_dir /root/code/data/semeval_data  \
  --max_seq_length 64 \
  --per_gpu_train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 8.0 \
  --output_dir /tmp/outputs/semeval_roberta_output/ 

roberta-large-openai-detector



python run_semeval.py \
  --model_type roberta \
  --model_name_or_path roberta-large-openai-detector \
  --task_name semeval \
  --do_train \
  --do_eval \
  --do_lower_case \
  --data_dir /root/code/data/semeval_data  \
  --max_seq_length 32 \
  --per_gpu_train_batch_size 16 \
  --learning_rate 2e-5 \
  --num_train_epochs 5.0 \
  --output_dir /tmp/outputs/semeval_roberta-large-openai-detector_output/ 

distilbert

python run_semeval.py \
  --model_type distilbert \
  --model_name_or_path distilbert-base-uncased \
  --task_name semeval \
  --do_train \
  --do_eval \
  --do_lower_case \
  --data_dir /root/code/data/semeval_data  \
  --max_seq_length 64 \
  --per_gpu_train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 5.0 \
  --output_dir /tmp/outputs/semeval_distilbert_output/ 

albert

python run_semeval.py \
  --model_type albert \
  --model_name_or_path /root/code/albert_v_xx_L12_H-4096_A-64   \
  --task_name semeval \
  --do_train \
  --do_eval \
  --do_lower_case \
  --data_dir /root/code/data/semeval_data  \
  --max_seq_length 64 \
  --per_gpu_train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 5.0 \
  --output_dir /tmp/outputs/semeval_albert_output/ 

Xlm-roberta

python run_semeval.py \
  --model_type xlmroberta \
  --model_name_or_path xlm-roberta-large \
  --task_name semeval \
  --do_train \
  --do_eval \
  --do_lower_case \
  --data_dir /root/code/data/semeval_data  \
  --max_seq_length 32 \
  --per_gpu_train_batch_size 16 \
  --learning_rate 2e-5 \
  --num_train_epochs 5.0 \
  --output_dir /tmp/outputs/semeval_xlmroberta_output/ 

flaubert

python run_semeval.py \
  --model_type flaubert \
  --model_name_or_path  flaubert-large-cased \
  --task_name semeval \
  --do_train \
  --do_eval \
  --do_lower_case \
  --data_dir /root/code/data/semeval_data  \
  --max_seq_length 32 \
  --per_gpu_train_batch_size 16 \
  --learning_rate 2e-5 \
  --num_train_epochs 5.0 \
  --output_dir /tmp/outputs/semeval_flaubert_output/ 




