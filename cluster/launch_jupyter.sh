#!/bin/bash

#SBATCH -p cs -A cs -q csug
#SBATCH -c4 --mem=10g
#SBATCH --gres gpu:1

module load nvidia/cuda-11.0
module load nvidia/cudnn-v8.0.180-forcuda11.0

jupyter lab --ip='*' --port=8327 --no-browser --NotebookApp.max_buffer_size=10000000000 --NotebookApp.allow_origin='*' --NotebookApp.ip='0.0.0.0'