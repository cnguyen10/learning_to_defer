#!/bin/sh
DEVICE_ID=1
export CUDA_VISIBLE_DEVICES=${DEVICE_ID}

BASEDIR=$(cd $(dirname $0) && pwd)

# Learning to defer
python3 -W ignore "learn2defer.py" --experiment-name "Learning to defer" --run-description "" --train-files "/sda2/datasets/cifar10/train_random_noise_0.3_44.json,/sda2/datasets/cifar10/train_random_noise_0.3_61.json" --test-file "/sda2/datasets/cifar10/test_random_noise_0.3_44.json,/sda2/datasets/cifar10/test_random_noise_0.3_61.json" --train-groundtruth-file "/sda2/datasets/cifar10/train.json" --test-groundtruth-file "/sda2/datasets/cifar10/test.json" --num-classes "10" --num-clusters "2" --lr "0.01" --batch-size "128" --num-epochs "200" --jax-platform "cuda" --device-id "${DEVICE_ID}" --mem-frac "0.5" --tqdm --train --tracking-uri "http://127.0.0.1:8080" --artifact-path "${BASEDIR}/logdir"