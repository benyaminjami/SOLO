#!/usr/bin/env bash
#SBATCH --job-name=solo
#SBATCH --time=8:00:00
## SBATCH --qos=normal

### e.g. request 4 nodes with 1 gpu each, totally 4 gpus (WORLD_SIZE==4)
### Note: --gres=gpu:x should equal to ntasks-per-node
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --mem=60GB
#SBATCH --gres=gpu:4
#SBATCH --partition=a40
#SBATCH --cpus-per-task=8
##SBATCH --hint=nomultithread
#SBATCH --output=slurm_logs/%j.out
#SBATCH --error=slurm_logs/%j.err
#SBATCH --export=ALL

set -x
# PARTITION=t4
JOB_NAME=solo
CONFIG=configs/solo/solo_r50_fpn_1x_coco.py


# Check if first_job_id.txt exists and use its value as WORK_DIR
if [ -f first_job_id.txt ]; then
    FIRST_JOB_ID=$(cat first_job_id.txt)
    WORK_DIR=/checkpoint/${USER}/${FIRST_JOB_ID}
else
    # This is the first submission, so use the current SLURM_JOB_ID and save it to first_job_id.txt
    echo ${SLURM_JOB_ID} > first_job_id.txt
    WORK_DIR=/checkpoint/${USER}/${SLURM_JOB_ID}
fi

GPUS=${GPUS:-4}
GPUS_PER_NODE=${GPUS_PER_NODE:-4}
CPUS_PER_TASK=${CPUS_PER_TASK:-8}
SRUN_ARGS=${SRUN_ARGS:-""}

# trap handler - resubmit ourselves
handler()
{
echo "function handler called at $(date)"
# do whatever cleanup you want here;
# checkpoint, sync, etc
sbatch ${BASH_SOURCE[0]}
}
# register signal handler
trap handler SIGUSR1
#

PYTHONPATH=/h/benjami/.conda/envs/openmmlab/bin/python \
srun --job-name=${JOB_NAME} \
    --gres=gpu:${GPUS_PER_NODE} \
    --ntasks=${GPUS} \
    --ntasks-per-node=${GPUS_PER_NODE} \
    --cpus-per-task=${CPUS_PER_TASK} \
    --kill-on-bad-exit=1 \
    ${SRUN_ARGS} \
    python -u tools/train.py ${CONFIG} --work-dir=${WORK_DIR} --launcher="slurm" --auto-scale-lr &
wait
