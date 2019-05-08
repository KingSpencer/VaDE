#!/bin/bash
for i in {1..5}
do
    python VaDE_dp.py \
        -bnpyPath "/home/zifeng/Research/bnpy" \
        -outputPath "/home/zifeng/Research/DPVAE/mnist_results_new8" \
        -rootPath "/home/zifeng/Research/DPVAE" \
        -epoch 10 \
        -Kmax 50 \
        -dataset mnist \
        -scale 0.005 \
        -batch_iter 3 \
        -logFile \
        -useLocal \
        -rep $i \
        -sf 0.1 \
        -learningRate 2e-4
done


