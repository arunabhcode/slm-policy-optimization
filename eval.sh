#!/bin/sh
# Base model configuration
MODEL="knoveleng/OpenRS-GRPO"
BASE_MODEL_ARGS="pretrained=$MODEL,dtype=bfloat16,max_model_length=32768,gpu_memory_utilization=0.8,generation_parameters={max_new_tokens:32768,temperature:0.6,top_p:0.95}"

# Define evaluation tasks
# TASKS="aime24 math_500 amc23 minerva olympiadbench"
TASKS="aime24 math_500 amc23"

# Function to get revision for a given experiment and step
get_revision() {
    exp=$1
    step=$2
    
    # Experiment 1 revisions
    if [ "$exp" = "1" ]; then
        case $step in
            100) echo "0439c48c3686728b5cc5a20820d601d2908c6ee1" ;;
            200) echo "26f1b83b6c41b2c1b6407648b0de08c09f92687c" ;;
            300) echo "52bf539b00ab80ff4f9b09092b242d56560e7259" ;;
            400) echo "998f5ed67e0457c504c49580d85d82de1bcb6f35" ;;
            *) echo "unknown" ;;
        esac
    # Experiment 2 revisions
    elif [ "$exp" = "2" ]; then
        case $step in
            50)  echo "21e477217641fec6cd81f9caa220f5cccfb6a874" ;;
            100) echo "f653a8b948a6047011ee97b181ae0ad1d78d6e6a" ;;
            150) echo "e25175ada88bb8ef4187480b35631bf397196538" ;;
            200) echo "2cb5c97863c5936d88b571b4b7eaf7cbbaf5af2f" ;;
            250) echo "9d6ebc12f8d98824b3a072a930260a62f9189285" ;;
            300) echo "5e24d1cabac521742d1e6ebd244af9fc2fa91a89" ;;
            350) echo "195beba1db4d7a3539c0dc0223df9ce2486744d8" ;;
            400) echo "f3f07cd9f9b2abd8f8760dc4bef5b7165a940ffe" ;;
            *) echo "unknown" ;;
        esac
    # Experiment 3 revisions
    elif [ "$exp" = "3" ]; then
        case $step in
            50)  echo "1793695d9e827f1024422af3698f9a070b935698" ;;
            100) echo "b224d0ff3f66ced97e621b2ad1eeafd53235240c" ;;
            150) echo "8c574c77d411bd3d39107a03d4f79db3838c5577" ;;
            200) echo "d91ed4d06af444d7213f8fc225a01c9d43e1ce18" ;;
            250) echo "7f3c01cb25ca20730f85a079787d859fd5c2be8f" ;;
            300) echo "6f19980625be4c3316c6919ef12bd168a97c7cf4" ;;
            350) echo "86ba8acb918d462e6a26f6866f573ded4afdde4a" ;;
            400) echo "e88c97c2c4035abf0e00091ce214d055aad2c270" ;;
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
        1) echo "100 200 300 400" ;;
        2) echo "50 100 150 200 250 300 350 400" ;;
        3) echo "50 100 150 200 250 300 350 400" ;;
        *) echo "" ;;
    esac
}

# Function to run evaluations for a given step and revision
run_evaluation() {
    experiment=$1
    step=$2
    revision=$(get_revision "$experiment" "$step")
    output_dir="logs/evals/Exp${experiment}_${step}"
    
    # Check if revision is valid
    if [ "$revision" = "unknown" ]; then
        echo "Error: Unknown revision for experiment $experiment, step $step"
        return 1
    fi
    
    # Set model args with the specific revision
    model_args="$BASE_MODEL_ARGS,revision=$revision"
    
    echo "----------------------------------------"
    echo "Running evaluations for experiment $experiment, step $step"
    echo "Revision: rev${experiment}_${step} = $revision"
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

# Function to list all available experiments and revisions
list_configurations() {
    echo "Available Experiments:"
    
    for exp_num in 1 2 3; do
        steps=$(get_steps "$exp_num")
        echo "  Experiment $exp_num: Steps = $steps"
        
        # List revisions for this experiment
        echo "  Revisions:"
        for step in $steps; do
            revision=$(get_revision "$exp_num" "$step")
            echo "    Step $step: rev${exp_num}_${step} = $revision"
        done
        echo ""
    done
}

# Main function to run experiments
main() {
    if [ $# -eq 0 ] || [ "$1" = "--help" ] || [ "$1" = "-h" ]; then
        echo "Usage: $0 [options] <experiment_number> [experiment_number2 ...]"
        echo "Options:"
        echo "  --list, -l    List all available experiments and revisions"
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
        if [ "$exp_num" = "1" ] || [ "$exp_num" = "2" ] || [ "$exp_num" = "3" ]; then
            run_experiment "$exp_num"
        else
            echo "Error: Experiment $exp_num not defined"
        fi
    done
}

# Execute main function with command line arguments
main "$@"
