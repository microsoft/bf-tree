#!/bin/bash


# Define a function to clean up and exit
cleanup_and_exit() {
    echo "Interrupted. Cleaning up..."

    # Add any cleanup commands here
    docker rm -f "$container_name"
    rm -rf "$temp_dir"

    echo "Cleanup done. Exiting."
    exit 1
}

cleanup() {
    echo "Cleaning up..."
    docker rm -f "$container_name" &> /dev/null
    rm -rf "$temp_dir"
}

# Trap SIGINT (Ctrl+C) and call the cleanup_and_exit function
trap cleanup_and_exit SIGINT

cargo build --release

docker build -t bftree-benchmark-img .

container_name="bf-tree-bench"

memory_size=("400")
# sut=("Sled" "BfTree" "RocksDB")
sut=("Sled")

for s in "${sut[@]}"; do
	for config in "${memory_size[@]}"; do
		container_name="bf-tree-bench-$config-$s"

		# Run docker with the current configuration
		docker run -it -m ${config}m -e SHUMAI_FILTER="basic.*-$s" -e LD_LIBRARY_PATH=/usr/local/lib --name "$container_name" bftree-benchmark-img

		exit_status=$?

		 # Check if Docker exited because of an OOM error or other issues
        if [ $exit_status -ne 0 ]; then
            echo "Error: Docker process exited with status $exit_status. Possible OOM issue."
            cleanup
            exit $exit_status
        fi

		# Copy the benchmark directory from the container
		temp_dir=$(mktemp -d)
		docker cp "$container_name:target/benchmark" "$temp_dir"

		rsync -av --ignore-existing "$temp_dir/benchmark/" target/benchmark/

		# Clean up
		docker rm "$container_name"
		rm -rf "$temp_dir"
	done
done
