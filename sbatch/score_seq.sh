#!/bin/bash

#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --account=jgray21_gpu
#SBATCH --time=2:00:00
#SBATCH --ntasks-per-node=6
#SBATCH --gres=gpu:1

csv_path=$1 # ex: data/tm/Hie2022_C143_Tm.csv
score_method=$2 # ex: iglm
progen_model=${3:-None} # optional args if score_method is progen
device=${4:-cpu} # cuda:0

if [ "$score_method" = "iglm" ]; then
	source /home/mchungy1/data_jgray21/mchungy1/miniconda3/bin/activate /home/mchungy1/scr16_jgray21/mchungy1/conda_envs/iglm
fi

if [ "$score_method" = "antiberty" ]; then
	source /home/mchungy1/data_jgray21/mchungy1/miniconda3/bin/activate /home/mchungy1/scr16_jgray21/mchungy1/conda_envs/antiberty
fi

if [ "$score_method" = "progen" ]; then
        source /home/mchungy1/data_jgray21/mchungy1/miniconda3/bin/activate /scratch16/jgray21/mchungy1/conda_envs/progen
fi

if [ "$progen_model" = "None" ] || [ -z "$progen_model" ]; then
    python scripts/score_seq.py --csv-path $csv_path --score-method $score_method --device $device
else
    python scripts/score_seq.py --csv-path $csv_path --score-method $score_method --model-variant $progen_model --device $device
fi
