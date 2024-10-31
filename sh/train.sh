#!/bin/bash

eval "$(conda shell.bash hook)"
conda activate deep-diff

echo $CONDA_PREFIX

# set the python path
PROJECT_PATH=$(pwd)/..
export PYTHONPATH="$PROJECT_PATH:$PYTHONPATH"
 
# set default values for non-required arguments

# https://stackoverflow.com/questions/402377/using-getopts-to-process-long-and-short-command-line-options

die() { echo "$*" >&2; exit 2; }  # complain to STDERR and exit with error
needs_arg() { if [ -z "$OPTARG" ]; then die "No arg for --$OPT option"; fi; }

while getopts c:i:d:e:u:-: OPT; do
  # support long options: https://stackoverflow.com/a/28466267/519360
  if [ "$OPT" = "-" ]; then   # long option: reformulate OPT and OPTARG
    OPT="${OPTARG%%=*}"       # extract long option name
    OPTARG="${OPTARG#$OPT}"   # extract long option argument (may be empty)
    OPTARG="${OPTARG#=}"      # if long option argument, remove assigning `=`
  fi
  case "$OPT" in
    c | config-file )     needs_arg ; CONFIG_FILE="${OPTARG}" ;;
    i | run-id )          needs_arg ; RUN_ID="$OPTARG" ;;
    d | device )          needs_arg ; DEVICE="$OPTARG" ;;
    e | checkpoint-path ) CHECKPOINT_PATH="$OPTARG" ;;
    u | use-checkpoint )  needs_arg ; USE_CHECKPOINT="$OPTARG" ;;
    ??* )                 die "Illegal option --$OPT" ;;  # bad long option
    ? )                   exit 2 ;;  # bad short option (error reported via getopts)
  esac
done
shift $((OPTIND-1)) # remove parsed options and args from $@ list

mkdir -p ../logs/${RUN_ID}

if [ -z "$CHECKPOINT_PATH" ]; then
  python ../scripts/train_warp.py --config-file ${CONFIG_FILE} \
                                  --run-id ${RUN_ID} \
                                  --device ${DEVICE} \
                                  --use-checkpoint ${USE_CHECKPOINT} \
                                  2>&1 | tee ../logs/${RUN_ID}/${RUN_ID}.log
else
  python ../scripts/train_warp.py --config-file ${CONFIG_FILE} \
                                  --run-id ${RUN_ID} \
                                  --device ${DEVICE} \
                                  --use-checkpoint ${USE_CHECKPOINT} \
                                  --checkpoint ${CHECKPOINT_PATH} \
                                  2>&1 | tee ../logs/${RUN_ID}/${RUN_ID}.log

fi

conda deactivate
