#!/bin/bash

# Function to start a model with retries and logging
start_model() {
    local model_name="$1"
    local command="$2"
    local log_file="$3"
    local attempt_counter_var_name="$4" # Name of the counter variable
    local max_attempts=2
    local timeout=3600 # 1 hour

    # Use eval for reading and incrementing the counter via its name
    while [ "$(eval echo "\$$attempt_counter_var_name")" -lt "$max_attempts" ]; do
        eval "$attempt_counter_var_name=\$(( $(eval echo "\$$attempt_counter_var_name") + 1 ))"
        echo "Starting $model_name (Attempt $(eval echo "\$$attempt_counter_var_name") of $max_attempts)"

        log_dir="$(dirname "$log_file")"
        mkdir -p "$log_dir"
        # Ensure log file exists and is empty for the new attempt
        >"$log_file"

        # Start the command in the background
        nohup bash -c "$command" > "$log_file" 2>&1 &
        local pid=$!

        local start_time=$(date +%s)
        while true; do
            # Check if log file exists and contains the success message
            if [ -f "$log_file" ] && grep -q "INFO:     Application startup complete." "$log_file"; then
                echo "$model_name started successfully with PID $pid"
                return 0 # Success
            fi

            # Check if process still exists
            if ! kill -0 "$pid" 2>/dev/null; then
                echo "$model_name process (PID $pid) exited unexpectedly. Review logs at $log_file"
                # Optional: tail the log file for context
                tail -n 20 "$log_file"
                break # Exit inner loop as process is gone
            fi

            local current_time=$(date +%s)
            local elapsed_time=$((current_time - start_time))

            if [ "$elapsed_time" -ge "$timeout" ]; then
                echo "Timeout reached for $model_name (PID $pid). Killing process."
                kill -9 "$pid" 2>/dev/null || true # Force kill
                break # Exit inner loop due to timeout
            fi

            sleep 5 # Check every 5 seconds
        done # End of inner monitoring loop

        echo "Failed to start/confirm $model_name on attempt $(eval echo "\$$attempt_counter_var_name"). Retrying if possible..." | tee -a error_log.txt
    done # End of outer retry loop

    echo "Failed to start $model_name after $max_attempts attempts. Check logs at $log_file and error_log.txt." | tee -a error_log.txt
    return 1 # Indicate failure
}

# --- Script Execution Starts Here ---

# Activate your vLLM Python environment here if not already activated
# Example: 
# if [ -z "$VIRTUAL_ENV" ]; then
#    echo "Activating vLLM environment..."
#    source /path/to/your/vllm-env/bin/activate
# fi

# Environment variables for vLLM (optional, adjust as needed)
# export VLLM_CPU_OMP_THREADS_BIND="0-8" # Example: Bind OpenMP threads for CPU execution
# export VLLM_CPU_KVCACHE_SPACE=8        # Example: Set KV cache space for CPU

# --- Configuration for the model to serve ---
# Replace with your desired model and port
MODEL_TO_SERVE="${1:-facebook/opt-125m}" # Use first argument as model, or default
PORT_NUMBER="${2:-8001}"                  # Use second argument as port, or default
MAX_MODEL_LEN="${3:-2048}"               # Use third argument as max length, or default
HOST_IP="0.0.0.0"

# Construct the command to run the vLLM server
# Using python -m vllm.entrypoints.openai.api_server for an OpenAI-compatible API.
VLLM_COMMAND="vllm serve \"${MODEL_TO_SERVE}\" --max-model-len \"${MAX_MODEL_LEN}\" --port \"${PORT_NUMBER}\" --host \"${HOST_IP}\""

# Define log file path (ensure the directory exists or is created by start_model)
# Creates a logs subdirectory in the same directory as the script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
LOG_DIR="$SCRIPT_DIR/../logs" # Place logs in project root/logs
LOG_FILE_PATH="$LOG_DIR/vllm_server_${MODEL_TO_SERVE//\//-}_port${PORT_NUMBER}_$(hostname).log"

# Initialize retry counter for this model
# Ensure this variable name is unique if you plan to run multiple models with this script structure.
model_1_retry_counter=0

# --- Main loop to start the model ---
echo "Attempting to start vLLM server for model: $MODEL_TO_SERVE on port $PORT_NUMBER"
echo "Log file will be at: $LOG_FILE_PATH"

# Pass the *name* of the counter variable as a string to start_model
if ! start_model "$MODEL_TO_SERVE" "$VLLM_COMMAND" "$LOG_FILE_PATH" "model_1_retry_counter"; then
    echo "Main script: Failed to start $MODEL_TO_SERVE. Exiting."
    exit 1
fi

echo "Main script: Model $MODEL_TO_SERVE should be running. Monitor logs at $LOG_FILE_PATH."
# The script will exit here, but the nohup process continues in the background.
# You can add 'wait' here if you want the script to wait for the background process,
# but typically for servers, you let them run and the script exits. 