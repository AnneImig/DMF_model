#!/bin/bash

# Absolute paths to executables
glmexec="pestpp-glm"
psoexec="pestpp-pso"
# Number of agents
nagents=3
hostname=$(hostname)
port=4003

queen_dir="/Users/anneimig/DMF_model/PEST++PSO/queen"

# Start the queen first
if [ -d "$queen_dir" ]; then
    (
        cd "$queen_dir" || exit
        echo "Starting PSO queen in $queen_dir"
        $psoexec control.pst :$port
    ) &
    queen_pid=$!
    echo "Queen PID: $queen_pid"
else
    echo "Directory $queen_dir does not exist!"
    exit 1
fi

# Wait for the queen to open the port
echo "Waiting for queen to start on port $port..."
while ! nc -z localhost $port; do
    sleep 1
done
echo "Queen is now listening on port $port."

# Start agents
for ((i=1; i<=nagents; i++)); do
    hive_dir="./queen/hive${i}"
    if [ -d "$hive_dir" ]; then
        (
            cd "$hive_dir" || exit
            echo "Starting worker $i in $hive_dir"
            $glmexec control.pst /H $hostname:$port
        ) &
    else
        echo "Directory $hive_dir does not exist!"
    fi
done

# Wait for queen to finish
wait $queen_pid
