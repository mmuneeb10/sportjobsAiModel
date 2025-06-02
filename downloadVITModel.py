import os
import torch
from transformers import ViTForImageClassification, ViTImageProcessor

def download_and_save_vit_model(model_name="google/vit-base-patch16-224", output_dir="./local_vit_model"):
    """
    Downloads a pre-trained Vision Transformer (ViT) model and processor and saves them locally.
    
    Args:
        model_name (str): The name of the ViT model to download.
                          Options include: google/vit-base-patch16-224, google/vit-large-patch16-224,
                          facebook/deit-base-patch16-224, etc.
        output_dir (str): Directory where the model and processor will be saved.
    """
    print(f"Downloading model: {model_name}")
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")
    
    # Download and save processor
    print("Downloading image processor...")
    processor = ViTImageProcessor.from_pretrained(model_name)
    processor.save_pretrained(output_dir)
    print(f"Image processor saved to {output_dir}")
    
    # Download and save model
    print("Downloading model (this may take a while)...")
    model = ViTForImageClassification.from_pretrained(model_name)
    model.save_pretrained(output_dir)
    print(f"Model saved to {output_dir}")
    
    print(f"Model and processor files successfully saved to {output_dir}")
    return model, processor

if __name__ == "__main__":
    # You can change the model here if needed
    # Some options include:
    # - "google/vit-base-patch16-224": Base ViT model (85M parameters)
    # - "google/vit-large-patch16-224": Large ViT model (307M parameters)
    # - "google/vit-huge-patch14-224": Huge ViT model (632M parameters)
    # - "facebook/deit-base-patch16-224": Data-efficient image transformer
    # - "facebook/deit-small-patch16-224": Smaller Data-efficient image transformer
    
    model_name = "google/vit-base-patch16-224"  # Default model
    output_directory = "./local_vit_model"
    
    print(f"Starting download of {model_name}...")
    model, processor = download_and_save_vit_model(model_name, output_directory)
    
    # Quick verification that the model works
    print("\nVerifying model works correctly...")
    from PIL import Image
    import requests
    from io import BytesIO
    
    # Download a sample image
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(BytesIO(requests.get(url).content))
    
    # Process image and run inference
    inputs = processor(images=image, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    print("Model verification successful!")
    print(f"\nSummary:")
    print(f"- Model: {model_name}")
    print(f"- Saved to: {os.path.abspath(output_directory)}")
    print(f"- Model size: {sum(p.numel() for p in model.parameters()):,} parameters")
    print("\nYou can now use the local model by loading from this directory:")
    print(f"    processor = ViTImageProcessor.from_pretrained('{os.path.abspath(output_directory)}')")
    print(f"    model = ViTForImageClassification.from_pretrained('{os.path.abspath(output_directory)}')")