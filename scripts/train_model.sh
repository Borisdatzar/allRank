#!/usr/bin/env bash

DIR=$(dirname $0)
PROJECT_DIR="$(cd $DIR/..; pwd)"

ARCH_VERSION=${1:cpu}

docker build --build-arg arch_version=${ARCH_VERSION} --progress=plain -t allrank:latest $PROJECT_DIR
docker run -e PYTHONPATH=/allrank -v $PROJECT_DIR:/allrank allrank:latest /bin/sh -c 'python allrank/main.py --config-file-name /allrank/scripts/deep_playback_config.json --run-id deep_model_train_with_hci --job-dir /allrank/neuralNDCG'
