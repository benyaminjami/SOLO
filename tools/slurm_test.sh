#!/usr/bin/env bash
#SBATCH --job-name=solo
#SBATCH --time=1:00:00
## SBATCH --qos=normal

### e.g. request 4 nodes with 1 gpu each, totally 4 gpus (WORLD_SIZE==4)
### Note: --gres=gpu:x should equal to ntasks-per-node
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=3
#SBATCH --mem=30GB
#SBATCH --gres=gpu:3
#SBATCH --partition=rtx6000
#SBATCH --cpus-per-task=8
##SBATCH --hint=nomultithread
#SBATCH --output=slurm_logs/%j.out
#SBATCH --error=slurm_logs/%j.err
#SBATCH --export=ALL

set -x
# PARTITION=t4
JOB_NAME=solo
CONFIG=configs/solo/solo_r50_fpn_1x_coco.py
CHECKPOINT=/checkpoint/benjami/10856402/epoch_2.pth


GPUS=${GPUS:-3}
GPUS_PER_NODE=${GPUS_PER_NODE:-3}
CPUS_PER_TASK=${CPUS_PER_TASK:-8}
SRUN_ARGS=${SRUN_ARGS:-""}


PYTHONPATH=/h/benjami/.conda/envs/openmmlab/bin/python \
srun --job-name=${JOB_NAME} \
    --gres=gpu:${GPUS_PER_NODE} \
    --ntasks=${GPUS} \
    --ntasks-per-node=${GPUS_PER_NODE} \
    --cpus-per-task=${CPUS_PER_TASK} \
    --kill-on-bad-exit=1 \
    ${SRUN_ARGS} \
    python -u tools/test.py ${CONFIG} ${CHECKPOINT} --launcher="slurm" &
wait
