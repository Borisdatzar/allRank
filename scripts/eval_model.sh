#!/usr/bin/env bash

DIR=$(dirname $0)
PROJECT_DIR="$(cd $DIR/..; pwd)"

ARCH_VERSION=${1:cpu}

docker build --build-arg arch_version=${ARCH_VERSION} --progress=plain -t allrank:latest $PROJECT_DIR
docker run -e PYTHONPATH=/allrank -v $PROJECT_DIR:/allrank allrank:latest /bin/sh -c 'python allrank/eval_on_test.py --config-file-name /allrank/scripts/playback_config.json --run-id model_train --job-dir /allrank/neuralNDCG'
