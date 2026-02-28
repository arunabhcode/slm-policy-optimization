#!/bin/bash

# Set CUDA_HOME if not already set
if [ -z "$CUDA_HOME" ]; then
    # Try common locations
    for cuda_path in /usr/local/cuda /usr/cuda /opt/cuda; do
        if [ -d "$cuda_path" ]; then
            export CUDA_HOME="$cuda_path"
            break
        fi
    done
fi

# Disable DeepSpeed CUDA op compilation if CUDA_HOME still not found
if [ -z "$CUDA_HOME" ]; then
    echo "Warning: CUDA_HOME not found, disabling DeepSpeed ops"
    export DS_BUILD_OPS=0
fi

echo "CUDA_HOME=$CUDA_HOME"

python src/open_r1/main.py "$@"
