#!/bin/sh
# Local model roots
GRPO_ROOT="/home/tiny/slm-policy-optimization/data/OpenRS-GRPO"
GSPO_ROOT="/home/tiny/slm-policy-optimization/data/OpenRS-GSPO"
BASE_MODEL_ARGS="dtype=bfloat16,max_model_length=32768,gpu_memory_utilization=0.8,generation_parameters={max_new_tokens:32768,temperature:0.6,top_p:0.95}"

# Define evaluation tasks
# TASKS="aime24 math_500 amc23 minerva olympiadbench"
TASKS="aime24 math_500 amc23"

# Function to get local checkpoint path for a given experiment and step
get_model_path() {
    exp=$1
    step=$2

    # Experiment 1 checkpoints
    if [ "$exp" = "1" ]; then
        case $step in
            50) echo "$GRPO_ROOT/checkpoint-50" ;;
            100) echo "$GRPO_ROOT/checkpoint-100" ;;
            *) echo "unknown" ;;
        esac
    # Experiment 2 checkpoints
    elif [ "$exp" = "2" ]; then
        case $step in
            50) echo "$GSPO_ROOT/checkpoint-50" ;;
            100) echo "$GSPO_ROOT/checkpoint-100" ;;
            150) echo "$GSPO_ROOT/checkpoint-150" ;;
            200) echo "$GSPO_ROOT/checkpoint-200" ;;
            *) echo "unknown" ;;
        esac
    else
        echo "unknown"
    fi
}

# Function to get steps for a given experiment
get_steps() {
    exp=$1

    case $exp in
        1) echo "50 100" ;;
        2) echo "50 100 150 200" ;;
        *) echo "" ;;
    esac
}

# Function to run evaluations for a given step and local checkpoint
run_evaluation() {
    experiment=$1
    step=$2
    model_path=$(get_model_path "$experiment" "$step")
    output_dir="logs/evals/Exp${experiment}_${step}"

    # Check if local checkpoint is valid
    if [ "$model_path" = "unknown" ]; then
        echo "Error: Unknown local checkpoint for experiment $experiment, step $step"
        return 1
    fi

    # Set model args with the local checkpoint path
    model_args="pretrained=$model_path,$BASE_MODEL_ARGS"

    echo "----------------------------------------"
    echo "Running evaluations for experiment $experiment, step $step"
    echo "Model path: $model_path"
    echo "Output directory: $output_dir"

    # Create output directory if it doesn't exist
    mkdir -p "$output_dir"

    # Run evaluations for each task
    for task in $TASKS; do
        echo "Evaluating task: $task"
        lighteval vllm "$model_args" "custom|$task|0|0" \
            --custom-tasks src/open_r1/evaluate.py \
            --use-chat-template \
            --output-dir "$output_dir"
    done
    echo "----------------------------------------"
}

# Function to run an experiment
run_experiment() {
    exp_num=$1
    steps=$(get_steps "$exp_num")
    
    # Check if experiment exists
    if [ -z "$steps" ]; then
        echo "Error: Experiment $exp_num not defined"
        return 1
    fi
    
    echo "========================================"
    echo "Running Experiment $exp_num"
    echo "Steps: $steps"
    echo "========================================"
    
    # Run evaluation for each step in the experiment
    for step in $steps; do
        run_evaluation "$exp_num" "$step"
    done
}

# Function to list all available experiments and local checkpoints
list_configurations() {
    echo "Available Experiments:"

    for exp_num in 1 2; do
        steps=$(get_steps "$exp_num")
        echo "  Experiment $exp_num: Steps = $steps"

        # List local checkpoints for this experiment
        echo "  Checkpoints:"
        for step in $steps; do
            model_path=$(get_model_path "$exp_num" "$step")
            echo "    Step $step: $model_path"
        done
        echo ""
    done
}

# Main function to run experiments
main() {
    if [ $# -eq 0 ] || [ "$1" = "--help" ] || [ "$1" = "-h" ]; then
        echo "Usage: $0 [options] <experiment_number> [experiment_number2 ...]"
        echo "Options:"
        echo "  --list, -l    List all available experiments and checkpoints"
        echo "  --help, -h    Show this help message"
        echo ""
        list_configurations
        exit 0
    fi

    if [ "$1" = "--list" ] || [ "$1" = "-l" ]; then
        list_configurations
        exit 0
    fi

    for exp_num in "$@"; do
        if [ "$exp_num" = "1" ] || [ "$exp_num" = "2" ]; then
            run_experiment "$exp_num"
        else
            echo "Error: Experiment $exp_num not defined"
        fi
    done
}

# Execute main function with command line arguments
main "$@"
