import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from qlora_dataloader import get_qlora_dataloader
import os

# Offline mode set karo
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# Paths
MODEL_NAME = "microsoft/phi-2"
MODEL_PATH = "./phi2_model"
OUTPUT_DIR = "./cv_jd_finetuned"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=MODEL_PATH, local_files_only=True)
tokenizer.pad_token = tokenizer.eos_token

print("Loading model...")
# Mac pe bitsandbytes nahi hai toh normal load karo
try:
    from transformers import BitsAndBytesConfig
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        cache_dir=MODEL_PATH,
        quantization_config=bnb_config,
        device_map="auto",
        local_files_only=True
    )
except:
    print("Loading without quantization (for Mac/CPU)...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        cache_dir=MODEL_PATH,
        device_map="auto",
        torch_dtype=torch.float16,
        local_files_only=True
    )

# LoRA setup
model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=4,  # Bohot kam rakha hai for faster training
    lora_alpha=8,
    target_modules=["Wqkv", "fc1", "fc2"],  # Phi-2 specific modules
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)

print("Adding LoRA...")
model = get_peft_model(model, lora_config)

# Data loading
print("\nLoading your CV-JD data...")
train_loader, val_loader, dataset = get_qlora_dataloader(
    base_cv_dir="./CVS",
    tokenizer=tokenizer,
    batch_size=1,
    max_length=256  # Bohot kam for testing
)

# Training
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=1,  # Sirf 1 epoch for testing
    per_device_train_batch_size=1,
    logging_steps=5,
    save_steps=50,
    eval_strategy="no",  # Correct parameter name
    fp16=False,  # Mac pe CPU mode ke liye
    optim="adamw_torch",
    learning_rate=3e-4,
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_loader.dataset,
    eval_dataset=None,  # Skip validation for speed
    tokenizer=tokenizer,
)

print(f"\nStarting training with {len(train_loader.dataset)} samples...")
trainer.train()

print("\nSaving model...")
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"\nDone! Model saved at: {OUTPUT_DIR}")

# Quick test
print("\n--- Quick Test ---")
test_prompt = """### Instruction:
Analyze the following CV and Job Description to determine if the candidate should be accepted, interviewed, shortlisted, or rejected.

### Job Description:
Software Engineer required with Python experience.

### CV:
5 years Python developer with Django expertise.

### Response:
Based on the analysis, the candidate should be:"""

inputs = tokenizer(test_prompt, return_tensors="pt")
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=10)
    
print("Model output:", tokenizer.decode(outputs[0], skip_special_tokens=True))