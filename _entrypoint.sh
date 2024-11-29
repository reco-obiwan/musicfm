#!/bin/bash
set -e

CUR_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$CUR_DIR"

function train_prod () {
    accelerate launch --config_file accelerate_prod.yaml \
    src/main.py -t train -c res/config.train.yml
}


if [[ "$1" = "train" ]]; then
    shift 1
    # train_dev
    train_prod
else
    eval "$@"
fi

# prevent docker exit
tail -f /dev/null
