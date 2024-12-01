import subprocess
import numpy as np

def get_free_gpu():
    try:
        # Run nvidia-smi command and get the output
        gpu_stats = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,nounits,noheader"])
        gpu_stats = gpu_stats.decode("utf-8").strip().split("\n")
        
        # Parse the output
        gpu_memory_info = []
        for gpu in gpu_stats:
            memory_used, memory_total = map(int, gpu.split(","))
            memory_free = memory_total - memory_used
            gpu_memory_info.append(memory_free)
            
        # If no GPUs are found, return None
        if not gpu_memory_info:
            return None
            
        # Find the GPU with maximum free memory
        free_memory_gpu_index = int(np.argmax(gpu_memory_info))
        
        return free_memory_gpu_index
        
    except (subprocess.CalledProcessError, FileNotFoundError):
        # Return None if nvidia-smi is not available or fails
        return None

if __name__ == "__main__":
    free_gpu = get_free_gpu()
    if free_gpu is not None:
        print(f"GPU {free_gpu} has the most free memory")
    else:
        print("No GPU available")
