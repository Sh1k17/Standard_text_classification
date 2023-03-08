train_path=../data/raw_data.txt
test_path=../data/test.txt
vocab_path=../data/vocab.pkl
checkpoint_dir=./a
do_train=True
do_eval=False
bert_config_dir=./a
bert_model_dir=./a
batch_size=32
max_seq_length=128
num_layers=2
save_path=../models/model.bin
num_train_steps=10000
epochs=10
num_warmup_steps=1000
learning_rate=1e-2
weight_decay=1e-3
n_vocab=100
n_embed=64
hidden_size=32
require_improvement=1000


#CUDA_VISIBLE_DEVICES=1 python -u run_cp.py \
python -u ../run_lstm.py \
    --train_path $train_path \
    --test_path $test_path \
    --vocab_path $vocab_path\
    --checkpoint_dir $checkpoint_dir \
    --do_train $do_train \
    --do_train $do_train \
    --do_eval $do_eval \
    --bert_config_dir $bert_config_dir\
    --bert_model_dir $bert_model_dir \
    --batch_size $batch_size \
    --max_seq_length $max_seq_length \
    --num_train_steps $num_train_steps \
    --epochs $epochs \
    --num_warmup_steps $num_warmup_steps \
    --learning_rate $learning_rate \
    --weight_decay $weight_decay \
    --n_vocab $n_vocab \
    --n_embed $n_embed \
    --hidden_size $hidden_size \
    --require_improvement $require_improvement \
    --num_layers $num_layers \
    --save_path $save_path