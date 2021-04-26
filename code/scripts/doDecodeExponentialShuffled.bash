#!/bin/bash

python doDecode.py \
    --randomize_ISIs \
    --model_class_name probabilisticModels.Exponential \
    --fig_filename_pattern ../../figures/decoding_exponential_randomized_ISIs{:d}.{:s}
