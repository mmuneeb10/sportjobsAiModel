import torch
import numpy as np

# Load model and check predictions
checkpoint = torch.load('best_model.pt', map_location='cpu')

print("Model keys sample:")
for i, key in enumerate(list(checkpoint.keys())[:5]):
    print(f"  {key}: {checkpoint[key].shape}")

# Check if classifier weights are biased
classifier_weight = checkpoint['classifier.3.weight']
classifier_bias = checkpoint['classifier.3.bias']

print("\nClassifier output layer:")
print(f"Weight shape: {classifier_weight.shape}")
print(f"Bias values: {classifier_bias}")

# Check which class has highest bias
print(f"\nClass biases:")
print(f"  ACCEPT (0): {classifier_bias[0]:.4f}")
print(f"  INTERVIEW (1): {classifier_bias[1]:.4f}")
print(f"  SHORTLIST (2): {classifier_bias[2]:.4f}")
print(f"  REJECT (3): {classifier_bias[3]:.4f}")

# Check weight norms per class
weight_norms = torch.norm(classifier_weight, dim=1)
print(f"\nWeight norms per class:")
for i, norm in enumerate(weight_norms):
    print(f"  Class {i}: {norm:.4f}")

# Simple prediction test
print("\n" + "="*50)
print("Testing with random input:")
random_input = torch.randn(1, 256)  # Classifier expects 256 features
logits = torch.matmul(random_input, classifier_weight.T) + classifier_bias
probs = torch.softmax(logits, dim=1)

print(f"Random logits: {logits[0]}")
print(f"Probabilities:")
for i, prob in enumerate(probs[0]):
    print(f"  Class {i}: {prob:.2%}")
    
predicted = torch.argmax(probs, dim=1).item()
print(f"\nPredicted class: {predicted}")

# Check semantic branch weights
if 'semantic_branch.0.weight' in checkpoint:
    semantic_weight = checkpoint['semantic_branch.0.weight']
    print(f"\nSemantic branch input: {semantic_weight.shape}")