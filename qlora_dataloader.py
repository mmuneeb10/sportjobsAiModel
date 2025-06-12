import os
import torch
from torch.utils.data import Dataset, DataLoader
from PyPDF2 import PdfReader
import docx
import random

class CVJDDataset(Dataset):     
    def __init__(self, base_cv_dir, tokenizer, max_length=2048):
        self.base_cv_dir = base_cv_dir
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Status mapping same as train.py
        self.status_map = {
            "ACCEPT": 0,
            "INTERVIEW": 1, 
            "SHORTLIST": 2,
            "REJECT": 3,
        }
        
        # Load all CV-JD pairs with their labels
        self.data_pairs = self._load_data_from_folders()
        
    def _read_file(self, filepath):
        """Extract text from PDF, DOCX, or TXT files"""
        try:
            ext = os.path.splitext(filepath)[1].lower()
            
            if ext == '.pdf':
                reader = PdfReader(filepath)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() + "\n"
                return text.strip()
                
            elif ext in ['.docx', '.doc']:
                doc = docx.Document(filepath)
                return "\n".join([para.text for para in doc.paragraphs if para.text.strip()])
                
            elif ext == '.txt':
                with open(filepath, 'r', encoding='utf-8') as f:
                    return f.read()
                    
            else:
                return ""
                
        except Exception as e:
            print(f"Error reading {filepath}: {e}")
            return ""
    
    def _load_data_from_folders(self):
        """Load CV-JD pairs from the folder structure"""
        data_pairs = []
        
        # Iterate through job folders (e.g., JOB1, JOB2, etc.)
        for job_folder in os.listdir(self.base_cv_dir):
            job_path = os.path.join(self.base_cv_dir, job_folder)
            
            if not os.path.isdir(job_path) or job_folder.startswith('.'):
                continue
                
            # Find job description file
            jd_file = None
            for file in os.listdir(job_path):
                if 'job_description' in file.lower() or 'jobdescription' in file.lower():
                    jd_file = os.path.join(job_path, file)
                    break
                    
            if not jd_file:
                # Try to find any .docx or .txt file as JD
                for file in os.listdir(job_path):
                    if file.endswith(('.docx', '.txt')) and not os.path.isdir(os.path.join(job_path, file)):
                        jd_file = os.path.join(job_path, file)
                        break
                        
            if not jd_file:
                print(f"No job description found in {job_folder}")
                continue
                
            # Read job description once
            jd_text = self._read_file(jd_file)
            
            # Process each status folder
            for status_folder in ['ACCEPT', 'INTERVIEW', 'SHORTLIST', 'REJECT']:
                status_path = os.path.join(job_path, status_folder)
                
                if not os.path.exists(status_path) or not os.path.isdir(status_path):
                    continue
                    
                # Get label for this status
                label = self.status_map[status_folder]
                
                # Process each CV in the status folder
                for cv_file in os.listdir(status_path):
                    cv_path = os.path.join(status_path, cv_file)
                    
                    if os.path.isdir(cv_path) or cv_file.startswith('.'):
                        continue
                        
                    # Check if it's a supported file type
                    if not cv_file.endswith(('.pdf', '.docx', '.doc', '.txt')):
                        continue
                        
                    # Read CV text
                    cv_text = self._read_file(cv_path)
                    
                    if cv_text and jd_text:
                        data_pairs.append({
                            'cv_text': cv_text,
                            'jd_text': jd_text,
                            'label': label,
                            'status': status_folder,
                            'job_name': job_folder,
                            'cv_path': cv_path
                        })
                        
        print(f"Loaded {len(data_pairs)} CV-JD pairs")
        
        # Print distribution
        status_counts = {}
        for pair in data_pairs:
            status = pair['status']
            status_counts[status] = status_counts.get(status, 0) + 1
            
        print("\nData distribution:")
        for status, count in status_counts.items():
            print(f"  {status}: {count}")
            
        return data_pairs
    
    def __len__(self):
        return len(self.data_pairs)
    
    def __getitem__(self, idx):
        pair = self.data_pairs[idx]
        
        # Create prompt for fine-tuning
        prompt = f"""### Instruction:
Analyze the following CV and Job Description to determine if the candidate should be accepted, interviewed, shortlisted, or rejected.

### Job Description:
{pair['jd_text']}

### CV:
{pair['cv_text']}

### Response:
Based on the analysis, the candidate should be: {pair['status']}"""
        
        # Tokenize
        encoding = self.tokenizer(
            prompt,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # For QLoRA training, we need input_ids and labels
        # Labels are same as input_ids but with -100 for non-response tokens
        labels = encoding['input_ids'].clone()
        
        # Find where response starts and mask everything before it
        response_start_text = "### Response:"
        tokenized_response_start = self.tokenizer.encode(response_start_text, add_special_tokens=False)
        
        # Simple approach: mask first 80% of tokens (instruction part)
        # In production, you'd want to find exact response position
        mask_until = int(len(labels[0]) * 0.8)
        labels[0, :mask_until] = -100
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': labels.squeeze()
        }

def get_qlora_dataloader(base_cv_dir, tokenizer, batch_size=4, max_length=2048):
    """Create data loaders for QLoRA training"""
    
    # Create dataset
    dataset = CVJDDataset(base_cv_dir, tokenizer, max_length)
    
    # Split into train and validation (80-20)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False
    )
    
    return train_loader, val_loader, dataset