#!/bin/bash

#SBATCH -p cs -A cs -q csug
#SBATCH -c4 --mem=2g
#SBATCH --gres gpu:0

kaggle competitions download -c birdclef-2021
