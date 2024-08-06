

import psutil
import gc
import os

def check_memory_usage():
    # Get total and available memory
    memory_info = psutil.virtual_memory()
    total_memory = memory_info.total
    available_memory = memory_info.available
    used_memory = total_memory - available_memory
    used_memory_percent = (used_memory / total_memory) * 100

    print(f"Total Memory: {total_memory / (1024 ** 3):.2f} GB")
    print(f"Available Memory: {available_memory / (1024 ** 3):.2f} GB")
    print(f"Used Memory: {used_memory / (1024 ** 3):.2f} GB ({used_memory_percent:.2f}%)")

    # Check if used memory exceeds 90%
    if used_memory_percent > 90:
        print("Memory usage exceeds 90%. Freeing up memory...")
        gc.collect()  # Manually invoke garbage collector
        print("Memory freed up.")

# Example usage
check_memory_usage()


