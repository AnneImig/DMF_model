#!/bin/bash

# Absolute paths to executables
glmexec="pestpp-glm"
psoexec="pestpp-pso"

# Number of agents (workers)
nagents=3
hostname=$(hostname)
port=4003

queen_dir="./queen"

# ----------------------
# Start the PSO Queen
# ----------------------
(
    cd "$queen_dir" || exit
    echo "Starting PSO Queen in $queen_dir"
    $psoexec ../control.pst /H :$port
) &
queen_pid=$!
echo "Queen PID: $queen_pid"

# Wait for Queen to start
echo "Waiting for Queen to listen on port $port..."
while ! nc -z localhost $port; do
    sleep 1
done
echo "Queen is now listening on port $port."

# ----------------------
# Start Worker Agents
# ----------------------
for ((i=1; i<=nagents; i++)); do
    hive_dir="$queen_dir/hive${i}"
    if [ -d "$hive_dir" ]; then
        (
            cd "$hive_dir" || exit
            echo "Starting worker $i in $hive_dir"
            $glmexec ../../control.pst /H $hostname:$port
        ) &
    else
        echo "Directory $hive_dir does not exist!"
    fi
done

# ----------------------
# Wait for Queen to finish
# ----------------------
wait $queen_pid
echo "PSO run complete!"
