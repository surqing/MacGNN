#!/bin/bash

# 定义所有的超参数选项
learning_rates=(0.01)
weight_decays=(0 1e-5 1e-4)
embed_dims=(8 10 32)
epochs=(15 20 40)
sample_sizes=(1 2 4)
seeds=(0 2023 10003)
taus=(1 1.9 3.8)

# 进行超参数组合迭代
for lr in "${learning_rates[@]}"; do
  for wd in "${weight_decays[@]}"; do
    for ed in "${embed_dims[@]}"; do
      for ep in "${epochs[@]}"; do
        for ss in "${sample_sizes[@]}"; do
          for seed in "${seeds[@]}"; do
            for tau in "${taus[@]}"; do
              echo "Running with learning_rate=$lr, weight_decay=$wd, embed_dim=$ed, epoch=$ep, sample_size=$ss, seed=$seed, tau=$tau"
              python main.py --model_name macgnn --dataset_name kuairec --learning_rate $lr --weight_decay $wd --embed_dim $ed --epoch $ep --sample_size $ss --seed $seed --tau $tau --runs 5
            done
          done
        done
      done
    done
  done
done
