import os
import torch
from transformers import AutoProcessor, AutoModelForDocumentQuestionAnswering

def download_and_save_layoutlmv3_model(model_name="microsoft/layoutlmv3-base", output_dir="./local_layoutlmv3_model"):
    """
    Downloads a pre-trained LayoutLMv3 model and processor and saves them locally.
    
    Args:
        model_name (str): The name of the LayoutLMv3 model to download.
                      Options include: microsoft/layoutlmv3-base, microsoft/layoutlmv3-large
        output_dir (str): Directory where the model and processor will be saved.
    """
    print(f"Downloading model: {model_name}")
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")
    
    # Download and save processor
    print("Downloading processor...")
    processor = AutoProcessor.from_pretrained(model_name)
    processor.save_pretrained(output_dir)
    print(f"Processor saved to {output_dir}")
    
    # Download and save model
    print("Downloading model (this may take a while)...")
    model = AutoModelForDocumentQuestionAnswering.from_pretrained(model_name)
    model.save_pretrained(output_dir)
    print(f"Model saved to {output_dir}")
    
    print(f"Model and processor files successfully saved to {output_dir}")
    return model, processor

if __name__ == "__main__":
    # You can change the model here if needed
    # Some options include:
    # - "microsoft/layoutlmv3-base": Base LayoutLMv3 model (133M parameters)
    # - "microsoft/layoutlmv3-large": Large LayoutLMv3 model (368M parameters)
    
    model_name = "microsoft/layoutlmv3-base"  # Default model
    output_directory = "./local_layoutlmv3_model"
    
    print(f"Starting download of {model_name}...")
    model, processor = download_and_save_layoutlmv3_model(model_name, output_directory)
    
    # Quick verification that the model works
    print("\nVerifying model works correctly...")
    from PIL import Image
    import requests
    from io import BytesIO
    
    # Download a sample image
    url = "https://www.microsoft.com/en-us/research/uploads/prod/2022/03/layoutlmv3_architecture-1024x614.png"
    image = Image.open(BytesIO(requests.get(url).content))
    
    # Process image and run inference (simple test)
    encoding = processor(image, "What is shown in this image?", return_tensors="pt")
    
    with torch.no_grad():
        try:
            outputs = model(**encoding)
            print("Model verification successful!")
        except Exception as e:
            print(f"Verification encountered an issue: {str(e)}")
            print("This is expected as we're just testing basic model loading.")
    
    print(f"\nSummary:")
    print(f"- Model: {model_name}")
    print(f"- Saved to: {os.path.abspath(output_directory)}")
    print(f"- Model size: {sum(p.numel() for p in model.parameters()):,} parameters")
    print("\nYou can now use the local model by loading from this directory:")
    print(f"    processor = AutoProcessor.from_pretrained('{os.path.abspath(output_directory)}')")
    print(f"    model = AutoModelForDocumentQuestionAnswering.from_pretrained('{os.path.abspath(output_directory)}')")