import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import sys
import os
import re

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Model paths
BASE_MODEL_NAME = "microsoft/phi-2"
BASE_MODEL_CACHE = "./phi2_model"
LORA_MODEL_PATH = "./cv_jd_finetuned"


def read_file(file_path):
    """Read content from various file formats"""
    try:
        ext = os.path.splitext(file_path)[1].lower()
        
        if ext == '.txt':
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                return f.read()
                
        elif ext in ['.doc', '.docx']:
            try:
                import docx
                doc = docx.Document(file_path)
                return "\n".join([para.text for para in doc.paragraphs])
            except:
                return f"Error reading Word document: {file_path}"
                
        elif ext == '.pdf':
            try:
                import fitz  # PyMuPDF
                doc = fitz.open(file_path)
                text = ""
                for page in doc:
                    text += page.get_text()
                doc.close()
                return text
            except:
                return f"Error reading PDF: {file_path}"
                
        else:
            # Try reading as text file
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                return f.read()
                
    except Exception as e:
        return f"Error reading file: {str(e)}"


def load_model():
    """Load Phi-2 model with LoRA adapters"""
    print("Loading Phi-2 base model...")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL_NAME, 
        cache_dir=BASE_MODEL_CACHE,
        local_files_only=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        cache_dir=BASE_MODEL_CACHE,
        device_map={"": device},  # Simple device mapping
        torch_dtype=torch.float32,  # Full precision for Mac
        local_files_only=True,
        low_cpu_mem_usage=True
    )
    
    # Load LoRA adapters if they exist
    if os.path.exists(LORA_MODEL_PATH):
        print("Loading LoRA adapters...")
        try:
            # Try loading with different approaches
            from peft import PeftModel, PeftConfig
            
            # First check if adapter_config.json exists
            config_path = os.path.join(LORA_MODEL_PATH, "adapter_config.json")
            if os.path.exists(config_path):
                # Load config first
                peft_config = PeftConfig.from_pretrained(LORA_MODEL_PATH)
                # Then load model with config
                model = PeftModel.from_pretrained(model, LORA_MODEL_PATH, config=peft_config)
                print("LoRA adapters loaded successfully!")
            else:
                print("Warning: adapter_config.json not found, trying direct load...")
                model = PeftModel.from_pretrained(model, LORA_MODEL_PATH)
        except Exception as e:
            print(f"Warning: Could not load LoRA adapters: {str(e)}")
            print("Using base model only...")
    else:
        print(f"Warning: LoRA model not found at {LORA_MODEL_PATH}")
        print("Using base model only...")
    
    model.eval()
    return model, tokenizer


def predict_cv_status(model, tokenizer, jd_text, cv_text):
    """Predict CV status using the fine-tuned model"""
    
    # Create prompt in the same format as training
    prompt = f"""### Instruction:
Analyze the following CV and Job Description to determine if the candidate should be accepted, interviewed, shortlisted, or rejected.

### Job Description:
{jd_text[:1000]}  # Limit to prevent token overflow

### CV:
{cv_text[:1000]}  # Limit to prevent token overflow

### Response:
Based on the analysis, the candidate should be:"""

    # Tokenize
    inputs = tokenizer(
        prompt, 
        return_tensors="pt",
        truncation=True,
        max_length=1024
    ).to(device)
    
    # Generate
    print("\nGenerating prediction...")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            temperature=0.7,
            do_sample=True,
            top_p=0.95,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode response
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract the response part
    response_part = full_response.split("### Response:")[-1].strip()
    response_text = response_part.split("Based on the analysis, the candidate should be:")[-1].strip()
    
    # Extract status from response
    status = "UNKNOWN"
    confidence = 0.0
    
    # Check for keywords
    response_lower = response_text.lower()
    if "accept" in response_lower:
        status = "ACCEPT"
        confidence = 0.9
    elif "interview" in response_lower:
        status = "INTERVIEW"
        confidence = 0.85
    elif "shortlist" in response_lower:
        status = "SHORTLIST"
        confidence = 0.8
    elif "reject" in response_lower:
        status = "REJECT"
        confidence = 0.9
    else:
        # Try to find any status word
        for s in ["ACCEPT", "INTERVIEW", "SHORTLIST", "REJECT"]:
            if s in response_text.upper():
                status = s
                confidence = 0.75
                break
    
    return status, confidence, response_text


def main():
    if len(sys.argv) != 3:
        print("Usage: python detect_phi2.py <job_description_file> <cv_file>")
        print("Example: python detect_phi2.py job.txt resume.pdf")
        sys.exit(1)
    
    jd_path = sys.argv[1]
    cv_path = sys.argv[2]
    
    # Check files exist
    if not os.path.exists(jd_path):
        print(f"Error: Job description file not found: {jd_path}")
        sys.exit(1)
        
    if not os.path.exists(cv_path):
        print(f"Error: CV file not found: {cv_path}")
        sys.exit(1)
    
    # Load model
    model, tokenizer = load_model()
    
    # Read files
    print(f"\nReading job description: {jd_path}")
    jd_text = read_file(jd_path)
    
    print(f"Reading CV: {cv_path}")
    cv_text = read_file(cv_path)
    
    # Show previews
    print(f"\nJD Preview: {jd_text[:200]}...")
    print(f"CV Preview: {cv_text[:200]}...")
    
    # Predict
    status, confidence, response = predict_cv_status(model, tokenizer, jd_text, cv_text)
    
    # Display results
    print("\n" + "="*60)
    print("PHI-2 CV EVALUATION RESULTS")
    print("="*60)
    print(f"Decision: {status}")
    print(f"Confidence: {confidence:.2%}")
    print(f"\nModel Response:")
    print(response)
    print("="*60)
    
    # Interpretation
    if status == "ACCEPT":
        print("\n✓ Candidate is suitable for the position")
    elif status == "INTERVIEW":
        print("\n~ Candidate should be interviewed")
    elif status == "SHORTLIST":
        print("\n~ Candidate is shortlisted for further review")
    elif status == "REJECT":
        print("\n✗ Candidate does not match requirements")
    else:
        print("\n? Unable to determine clear decision")


if __name__ == "__main__":
    main()