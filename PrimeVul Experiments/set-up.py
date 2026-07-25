import torch
from datasets import load_dataset

def verify_environment():
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        gpu_name = torch.xpu.get_device_name(0)
        print(f"PyTorch is natively using the Intel GPU: {gpu_name}")
    else:
        print("PyTorch cannot find the Intel XPU. It will use the CPU.")
        return False
    return True

def download_and_inspect_dataset():
    dataset_name = "Leopo1d/OpenVul_Ground_Truth_Vulnerability_Information"
    local_cache_path = "./data/raw"
    print(f"\n--- Downloading Dataset to {local_cache_path} ---")
    
    dataset = load_dataset(
        dataset_name, 
        split="ground_truth_info", 
        cache_dir=local_cache_path
    )
    
    print(f"Successfully loaded {len(dataset)} rows.")
    print("\n--- Column Names ---")
    print(dataset.column_names)
    
    row_zero = dataset[0]
    
    for key, value in row_zero.items():
        str_val = str(value)
        if len(str_val) > 100:
            str_val = str_val[:100] + "... [TRUNCATED]"
        print(f"{key}: {str_val}")

if __name__ == "__main__":
    if verify_environment():
        download_and_inspect_dataset()