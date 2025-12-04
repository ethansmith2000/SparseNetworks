#!/bin/bash

set -euo pipefail

SCRIPT_DIR="/home/ethan/SparseNetworks"
TRAIN_SCRIPT="${SCRIPT_DIR}/train_gpt.py"
LOG_DIR="${SCRIPT_DIR}/slurm_logs"
CONFIG_DIR="${SCRIPT_DIR}/sweep_configs"

mkdir -p "${LOG_DIR}"
mkdir -p "${CONFIG_DIR}"

# Edit these lists to define your sweep
lora_ranks=(64 128)
sparse_inits=("per_block_xavier" "global_xavier")
permute_inits=("classic" "variance_matched")
down_permute_in_modes=("none" "lora")  # "none" maps to Python None in the script

for lora_rank in "${lora_ranks[@]}"; do
  for sparse_init in "${sparse_inits[@]}"; do
    for permute_init in "${permute_inits[@]}"; do
      for down_mode in "${down_permute_in_modes[@]}"; do

        # SLURM job name (you can keep this short if you like)
        job_name="ethan-cifar"

        # Per-job JSON override file must be unique per combo, otherwise
        # later iterations overwrite earlier configs and all jobs look identical.
        cfg_file="${CONFIG_DIR}/${job_name}-lr${lora_rank}-sinit${sparse_init}-pinit${permute_init}-dpm${down_mode}.json"
        if [[ "${down_mode}" == "none" ]]; then
          down_json_val="null"
        else
          down_json_val="\"${down_mode}\""
        fi
        cat > "${cfg_file}" <<JSON
{
  "lora_rank": ${lora_rank},
  "sparse_weight_init_mode": "${sparse_init}",
  "permute_init_mode": "${permute_init}",
  "down_permute_in_mode": ${down_json_val}
}
JSON

        sbatch <<EOF
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=20
#SBATCH --job-name=${job_name}
#SBATCH --partition=queue1gpu
#SBATCH --time=6-09:59:59
#SBATCH --output=${LOG_DIR}/${job_name}-%j.out

cd ${SCRIPT_DIR}

# TODO: activate your environment / modules here if needed, e.g.:
# source ~/.bashrc
# conda activate your_env

source /home/ethan/leo-train-template/.venv/bin/activate

python ${TRAIN_SCRIPT} --override_json "${cfg_file}"
EOF

      done
    done
  done
done


