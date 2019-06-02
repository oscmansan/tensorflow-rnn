#!/usr/bin/env bash
module load cuda/10.0 cudnn/7.4
source venv/bin/activate
srun -p gpi.compute -t 1-00 -c 4 --mem 4GB --gres gpu:turing:1 python -m src.main