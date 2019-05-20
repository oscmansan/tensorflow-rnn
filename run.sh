#!/usr/bin/env bash
module load cuda/10.0 cudnn/7.4
source venv/bin/activate
srun -c 8 --mem 8GB --gres gpu:turing:1 python src/main.py