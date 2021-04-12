#!/bin/bash

python doDecode.py \
    --model_class_name probabilisticModels.InverseGaussian \
    --fig_filename_pattern ../../figures/decoding_invGaussian_randomized_ISIs{:d}.{:s}
