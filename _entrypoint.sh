#!/bin/bash
set -e

CUR_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$CUR_DIR"

function train_multi () {
    accelerate launch --config_file accelerate_multi.yaml \
    src/main.py -t train -c res/config.train.yml
}

function train_single () {
    accelerate launch --config_file accelerate_single.yaml \
    src/main.py -t train -c res/config.train.yml
}

if [[ "$1" = "train" ]]; then
    shift 1
    train_single
else
    eval "$@"
fi

# prevent docker exit
tail -f /dev/null
