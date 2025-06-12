import torch

# Load model checkpoint
checkpoint = torch.load('best_model.pt', map_location='cpu')

# Check job_branch weight shape
if 'job_branch.0.weight' in checkpoint:
    job_weight_shape = checkpoint['job_branch.0.weight'].shape
    print(f"Number of jobs in trained model: {job_weight_shape[1]}")
else:
    print("job_branch.0.weight not found in checkpoint")

# List all keys to understand structure
print("\nAll model keys:")
for key in sorted(checkpoint.keys()):
    if 'job' in key:
        print(f"{key}: {checkpoint[key].shape}")