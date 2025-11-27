#!/usr/bin/env bash

export PYTHONIOENCODING=utf-8

function generate_data() {
    lang=$1

    echo "$lang"

    output_file_suffix="_slice"

    # for RG-1
    uv run augment_with_slice.py \
        --language $lang \
        --query_type last_n_lines \
        --skip_if_no_cfc False \
        --output_file_suffix "rg1${output_file_suffix}"

    # for oracle experiment
    uv run augment_with_slice.py \
        --language $lang \
        --query_type groundtruth \
        --skip_if_no_cfc False \
        --output_file_suffix "oracle${output_file_suffix}"
}

generate_data python
