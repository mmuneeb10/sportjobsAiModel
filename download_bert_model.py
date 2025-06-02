import os
import torch
from transformers import AutoModel, AutoTokenizer

def download_and_save_bert_model(model_name="bert-base-uncased", output_dir="./local_bert_model"):
    """
    Downloads a pre-trained BERT model and tokenizer and saves them locally.
    
    Args:
        model_name (str): The name of the BERT model to download.
                          Options include: bert-base-uncased, bert-large-uncased, 
                          bert-base-cased, distilbert-base-uncased, etc.
        output_dir (str): Directory where the model and tokenizer will be saved.
    """
    print(f"Downloading model: {model_name}")
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")
    
    # Download and save tokenizer
    print("Downloading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.save_pretrained(output_dir)
    print(f"Tokenizer saved to {output_dir}")
    
    # Download and save model
    print("Downloading model (this may take a while)...")
    model = AutoModel.from_pretrained(model_name)
    model.save_pretrained(output_dir)
    print(f"Model saved to {output_dir}")
    
    # Save model configuration separately for easy access
    print(f"Model and tokenizer files successfully saved to {output_dir}")
    return model, tokenizer

if __name__ == "__main__":
    # You can change the model here if needed
    # Some options include:
    # - "bert-base-uncased": 110M parameters (smaller, faster)
    # - "bert-large-uncased": 340M parameters (larger, more accurate)
    # - "distilbert-base-uncased": 66M parameters (lighter version of BERT)
    # - "bert-base-cased": Case-sensitive version
    # - "roberta-base": RoBERTa model (improved BERT)
    
    model_name = "bert-base-uncased"  # Default model
    output_directory = "./local_bert_model"
    
    print(f"Starting download of {model_name}...")
    model, tokenizer = download_and_save_bert_model(model_name, output_directory)
    
    # Quick verification that the model works
    print("\nVerifying model works correctly...")
    text = "This is a test sentence for BERT."
    inputs = tokenizer(text, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    print("Model verification successful!")
    print(f"\nSummary:")
    print(f"- Model: {model_name}")
    print(f"- Saved to: {os.path.abspath(output_directory)}")
    print(f"- Vocabulary size: {tokenizer.vocab_size}")
    print(f"- Model size: {sum(p.numel() for p in model.parameters()):,} parameters")
    print("\nYou can now use the local model by loading from this directory:")
    print(f"    tokenizer = AutoTokenizer.from_pretrained('{os.path.abspath(output_directory)}')")
    print(f"    model = AutoModel.from_pretrained('{os.path.abspath(output_directory)}')")