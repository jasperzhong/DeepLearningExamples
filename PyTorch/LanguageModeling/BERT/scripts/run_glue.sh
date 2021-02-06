#!/bin/bash

# Copyright (c) 2019 NVIDIA CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -e

# echo "Container nvidia build = " $NVIDIA_BUILD_ID
WORKSPACE=$HOME/repos/DeepLearningExamples/PyTorch/LanguageModeling/BERT
CODEDIR=${23:-$WORKSPACE}
RESULTS_DIR=$CODEDIR/results
init_checkpoint=${1:-$RESULTS_DIR/checkpoints-lans-2k/ckpt_125000.pt}
BERT_PREP_WORKING_DIR=$HOME/datasets
data_dir=${2:-"$BERT_PREP_WORKING_DIR/glue_data/MRPC/"}
vocab_file=${9:-$WORKSPACE/vocab/vocab}
config_file=${12:-$CODEDIR/bert_base_config.json}
out_dir=${10:-$WORKSPACE/results/MRPC}
task_name=${6:-"mrpc"}
num_gpu=${7:-"8"}
batch_size=${8:-"16"}
gradient_accumulation_steps=${9:-"1"}
learning_rate=${10:-"2.4e-5"}
warmup_proportion=${11:-"0.1"}
epochs=${12:-"3"}
max_steps=${13:-"-1.0"}
precision=${14:-"fp16"}
seed=${15:-"2"}
mode=${16:-"train eval"}

mkdir -p $out_dir

if [ "$mode" = "eval" ] ; then
  num_gpu=1
fi

use_fp16=""
if [ "$precision" = "fp16" ] ; then
  echo "fp16 activated!"
  use_fp16="--fp16"
fi

if [ "$num_gpu" = "1" ] ; then
  export CUDA_VISIBLE_DEVICES=0
  mpi_command=""
else
  unset CUDA_VISIBLE_DEVICES
  mpi_command=" -m torch.distributed.launch --nproc_per_node=$num_gpu"
fi

CMD="python3 $mpi_command run_glue.py "
CMD+="--task_name ${task_name} "
if [[ $mode == *"train"* ]] ; then
  CMD+="--do_train "
  CMD+="--train_batch_size=$batch_size "
fi
if [[ $mode == *"eval"* ]] || [[ $mode == *"prediction"* ]]; then
  if [[ $mode == *"eval"* ]] ; then
    CMD+="--do_eval "
  fi
  if [[ $mode == *"prediction"* ]] ; then
    CMD+="--do_predict "
  fi
  CMD+="--eval_batch_size=$batch_size "
fi

CMD+="--gradient_accumulation_steps=$gradient_accumulation_steps "
CMD+="--do_lower_case "
CMD+="--data_dir $data_dir "
CMD+="--bert_model bert-large-uncased "
CMD+="--seed $seed "
CMD+="--init_checkpoint $init_checkpoint "
CMD+="--warmup_proportion $warmup_proportion "
CMD+="--max_seq_length 128 "
CMD+="--learning_rate $learning_rate "
CMD+="--num_train_epochs $epochs "
CMD+="--max_steps $max_steps "
CMD+="--vocab_file=$vocab_file "
CMD+="--config_file=$config_file "
CMD+="--output_dir $out_dir "
CMD+="$use_fp16"

LOGFILE=$out_dir/logfile

$CMD |& tee $LOGFILE
