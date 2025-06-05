import torch
from torch import nn
import torch.nn.functional as F
from transformers import ViTImageProcessor, ViTModel, BertModel, BertTokenizer
import numpy as np
import os
import fitz  # PyMuPDF for PDF processing
from PIL import Image
import io
import re
import docx  # For reading Word documents
import warnings
import logging
from datetime import datetime
from collections import Counter

# Suppress the overflowing tokens warning
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)

# Force offline mode to use local models
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# Paths to local models
VIT_MODEL_PATH = "./local_vit_model"
BERT_MODEL_PATH = "./local_bert_model"
MODEL_WEIGHTS_PATH = "./best_model.pt"  # Using best_model.pt for highest accuracy
DETECT_DIR = "./detect"

# Define status/label mapping 
def map_label_to_status(label):
    """Maps numeric class label to status string"""
    status_map = {
        0: "ACCEPT",
        1: "INTERVIEW",
        2: "SHORTLIST",
        3: "REJECT",
    }
    return status_map.get(label, "Unknown")

def create_page_image(file_path, page_num=0, scale=1.5):
    """
    Extracts a specific page from a document and converts it to a PIL Image
    """
    if file_path.lower().endswith('.pdf'):
        doc = None
        try:
            if not os.path.exists(file_path):
                print(f"PDF file does not exist: {file_path}")
                return Image.new('RGB', (224, 224), color='white')
            
            if not os.access(file_path, os.R_OK):
                print(f"PDF file is not readable: {file_path}")
                return Image.new('RGB', (224, 224), color='white')
            
            try:
                doc = fitz.open(file_path)
            except fitz.FileDataError as fde:
                print(f"PDF file is corrupted or invalid: {file_path} - {str(fde)}")
                return Image.new('RGB', (224, 224), color='white')
            except Exception as open_error:
                print(f"Cannot open PDF {file_path}: {str(open_error)}")
                return Image.new('RGB', (224, 224), color='white')
            
            if len(doc) == 0:
                print(f"PDF has no pages: {file_path}")
                doc.close()
                return Image.new('RGB', (224, 224), color='white')
            
            if page_num >= len(doc):
                page_num = 0
                
            page = doc[page_num]
            
            try:
                pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
            except Exception as render_error:
                print(f"Error rendering PDF page {page_num} from {file_path}: {str(render_error)}")
                doc.close()
                return Image.new('RGB', (224, 224), color='white')
            
            try:
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            except Exception as img_error:
                print(f"Error converting PDF page to image from {file_path}: {str(img_error)}")
                doc.close()
                return Image.new('RGB', (224, 224), color='white')
            
            try:
                img = img.resize((224, 224), Image.Resampling.LANCZOS)
            except AttributeError:
                img = img.resize((224, 224), Image.LANCZOS)
            
            doc.close()
            return img
        
        except Exception as e:
            print(f"Unexpected error creating image from PDF {file_path}: {str(e)}")
            if doc is not None:
                try:
                    doc.close()
                except:
                    pass
            return Image.new('RGB', (224, 224), color='white')
    else:
        return Image.new('RGB', (224, 224), color='white')

def extract_text_from_file(file_path):
    """Extract text content from various file types"""
    file_ext = os.path.splitext(file_path)[1].lower()
    
    try:
        if not os.path.exists(file_path):
            print(f"File does not exist: {file_path}")
            return f"File not found: {os.path.basename(file_path)}"
        
        if not os.access(file_path, os.R_OK):
            print(f"File is not readable: {file_path}")
            return f"File not readable: {os.path.basename(file_path)}"
        
        # PDF files
        if file_ext == '.pdf':
            doc = None
            try:
                doc = fitz.open(file_path)
                text = ""
                for page_num in range(len(doc)):
                    try:
                        page = doc[page_num]
                        page_text = page.get_text()
                        text += page_text + "\n"
                    except Exception as page_error:
                        print(f"Error reading page {page_num} from {file_path}: {str(page_error)}")
                        continue
                
                if not text.strip():
                    for page_num in range(len(doc)):
                        try:
                            page = doc[page_num]
                            blocks = page.get_text("blocks")
                            for block in blocks:
                                if block[6] == 0:
                                    text += block[4] + "\n"
                        except:
                            pass
                
                doc.close()
                
                if not text.strip():
                    return f"PDF content could not be extracted from {os.path.basename(file_path)}"
                
                return text
            except Exception as pdf_error:
                print(f"Error opening PDF {file_path}: {str(pdf_error)}")
                return f"Failed to read PDF: {os.path.basename(file_path)}"
            finally:
                if doc is not None:
                    try:
                        doc.close()
                    except:
                        pass
            
        # Text files
        elif file_ext == '.txt':
            try:
                encodings = ['utf-8', 'latin-1', 'windows-1252', 'ascii']
                for encoding in encodings:
                    try:
                        with open(file_path, 'r', encoding=encoding) as f:
                            return f.read()
                    except UnicodeDecodeError:
                        continue
                with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                    return f.read()
            except Exception as txt_error:
                print(f"Error reading text file {file_path}: {str(txt_error)}")
                return f"Failed to read text file: {os.path.basename(file_path)}"
                
        # Word documents
        elif file_ext in ['.docx', '.doc']:
            try:
                doc = docx.Document(file_path)
                return "\n".join([para.text for para in doc.paragraphs])
            except Exception as doc_error:
                print(f"Error reading Word document {file_path}: {str(doc_error)}")
                return f"Failed to read Word document: {os.path.basename(file_path)}"
        
        else:
            print(f"Unsupported file type for text extraction: {file_ext}")
            return f"Unsupported file type {file_ext}: {os.path.basename(file_path)}"
            
    except Exception as e:
        print(f"Error extracting text from {file_path}: {str(e)}")
        return f"Error reading file: {os.path.basename(file_path)}"

def extract_cv_features(file_path):
    """Extract formatting features from a CV"""
    if file_path.lower().endswith('.pdf'):
        doc = None
        try:
            if not os.path.exists(file_path) or not os.access(file_path, os.R_OK):
                return {
                    "font_count": 1,
                    "text_density": 1,
                    "bullet_points": 0,
                    "pages": 1
                }
            
            try:
                doc = fitz.open(file_path)
            except Exception as open_error:
                return {
                    "font_count": 1,
                    "text_density": 1,
                    "bullet_points": 0,
                    "pages": 1
                }
            
            if len(doc) == 0:
                doc.close()
                return {
                    "font_count": 1,
                    "text_density": 1,
                    "bullet_points": 0,
                    "pages": 1
                }
            
            font_counter = {}
            text_blocks = 0
            bullet_points = 0
            page_count = len(doc)
            
            for page_num in range(len(doc)):
                try:
                    page = doc[page_num]
                    
                    try:
                        blocks = page.get_text("dict")["blocks"]
                        text_blocks += len([b for b in blocks if b["type"] == 0])
                        
                        for block in blocks:
                            if block["type"] == 0:
                                for line in block.get("lines", []):
                                    for span in line.get("spans", []):
                                        font = span.get("font", "unknown")
                                        if font not in font_counter:
                                            font_counter[font] = 0
                                        font_counter[font] += 1
                    except:
                        pass
                    
                    try:
                        text = page.get_text()
                        bullet_points += text.count("•") + text.count("-") + text.count("*")
                    except:
                        pass
                        
                except:
                    continue
                
            font_count = len(font_counter) if font_counter else 1
            text_density = text_blocks / max(1, page_count)
            
            doc.close()
            
            return {
                "font_count": max(1, font_count),
                "text_density": max(0.1, text_density),
                "bullet_points": max(0, bullet_points),
                "pages": page_count
            }
            
        except Exception as e:
            if doc is not None:
                try:
                    doc.close()
                except:
                    pass
            return {
                "font_count": 1,
                "text_density": 1,
                "bullet_points": 0,
                "pages": 1
            }
    else:
        return {
            "font_count": 1,
            "text_density": 1,
            "bullet_points": 0,
            "pages": 1
        }

class MultimodalCVClassifier(nn.Module):
    """Multimodal CV classifier with separate branches and attention-based fusion"""
    def __init__(self, vit_model_name, bert_model_name, num_classes, num_jobs, dropout_rate=0.3):
        super(MultimodalCVClassifier, self).__init__()
        
        self.vit = ViTModel.from_pretrained(vit_model_name)
        self.bert = BertModel.from_pretrained(bert_model_name)
        
        self.vit_dim = self.vit.config.hidden_size
        self.bert_dim = self.bert.config.hidden_size
        self.format_dim = 4
        self.job_dim = num_jobs
        
        self.visual_branch = nn.Sequential(
            nn.Linear(self.vit_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )
        
        self.visual_score = nn.Linear(256, 1)
        
        self.semantic_branch = nn.Sequential(
            nn.Linear(self.bert_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )
        
        self.semantic_score = nn.Linear(256, 1)
        
        self.format_branch = nn.Sequential(
            nn.Linear(self.format_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )
        
        self.job_branch = nn.Sequential(
            nn.Linear(self.job_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )
        
        self.combined_dim = 256 + 256 + 64 + 64
        
        self.attention = nn.Sequential(
            nn.Linear(self.combined_dim, 4),
            nn.Softmax(dim=1)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(self.combined_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, pixel_values, input_ids, attention_mask, token_type_ids, 
                format_features=None, job_one_hot=None):
        if format_features is None:
            format_features = torch.zeros(pixel_values.size(0), self.format_dim, 
                                         device=pixel_values.device)
        
        if job_one_hot is None:
            job_one_hot = torch.zeros(pixel_values.size(0), self.job_dim,
                                     device=pixel_values.device)
        
        vit_outputs = self.vit(pixel_values=pixel_values)
        vit_features = vit_outputs.pooler_output
        visual_features = self.visual_branch(vit_features)
        visual_score_val = self.visual_score(visual_features)
        
        bert_outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        bert_features = bert_outputs.pooler_output
        semantic_features = self.semantic_branch(bert_features)
        semantic_score_val = self.semantic_score(semantic_features)
        
        format_features = self.format_branch(format_features)
        job_features = self.job_branch(job_one_hot)
        
        combined_features = torch.cat(
            (visual_features, semantic_features, format_features, job_features), dim=1
        )
        
        attention_weights = self.attention(combined_features)
        
        visual_weight = attention_weights[:, 0].unsqueeze(1).expand_as(visual_features)
        semantic_weight = attention_weights[:, 1].unsqueeze(1).expand_as(semantic_features)
        format_weight = attention_weights[:, 2].unsqueeze(1).expand_as(format_features)
        job_weight = attention_weights[:, 3].unsqueeze(1).expand_as(job_features)
        
        weighted_visual = visual_features * visual_weight
        weighted_semantic = semantic_features * semantic_weight
        weighted_format = format_features * format_weight
        weighted_job = job_features * job_weight
        
        weighted_features = torch.cat(
            (weighted_visual, weighted_semantic, weighted_format, weighted_job), dim=1
        )
        
        logits = self.classifier(weighted_features)
        
        return {
            'logits': logits,
            'visual_score': visual_score_val,
            'semantic_score': semantic_score_val,
            'attention_weights': attention_weights
        }

def sigmoid(x):
    """Sigmoid function to normalize scores to [0,1] range"""
    return 1 / (1 + np.exp(-x))


def process_text_in_chunks(bert_model, bert_tokenizer, job_description, cv_text, device, chunk_size=400):
    """Process CV and JD text in chunks to capture all content"""
    print("\nProcessing text in chunks for comprehensive semantic analysis...")
    
    # Split texts into chunks
    def split_into_chunks(text, chunk_size):
        words = text.split()
        chunks = []
        for i in range(0, len(words), chunk_size):
            chunk = ' '.join(words[i:i + chunk_size])
            chunks.append(chunk)
        return chunks
    
    # Create JD summary (first 200 words) to pair with each CV chunk
    jd_words = job_description.split()
    jd_summary = ' '.join(jd_words[:200])
    
    # Split CV into chunks
    cv_chunks = split_into_chunks(cv_text, chunk_size)
    print(f"  Processing {len(cv_chunks)} CV chunks...")
    
    semantic_scores = []
    
    # Process each CV chunk with JD summary
    for i, cv_chunk in enumerate(cv_chunks):
        encoding = bert_tokenizer(
            jd_summary,
            cv_chunk,
            add_special_tokens=True,
            max_length=512,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            return_overflowing_tokens=False,
        )
        
        input_ids = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)
        token_type_ids = encoding["token_type_ids"].to(device)
        
        with torch.no_grad():
            bert_outputs = bert_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )
            
            # Get pooled output for this chunk
            chunk_features = bert_outputs.pooler_output
            semantic_scores.append(chunk_features)
    
    # Average all chunk features
    if semantic_scores:
        avg_semantic_features = torch.stack(semantic_scores).mean(dim=0, keepdim=True)
        return avg_semantic_features
    else:
        # Fallback to empty tensor if no chunks
        return torch.zeros(1, bert_model.config.hidden_size).to(device)

def find_files_in_detect_folder():
    """Find job description and CV files in the detect folder"""
    if not os.path.exists(DETECT_DIR):
        raise ValueError(f"Detect directory '{DETECT_DIR}' does not exist!")
    
    files = os.listdir(DETECT_DIR)
    
    # Filter out hidden files
    files = [f for f in files if not f.startswith('.')]
    
    # Find job description file
    jd_file = None
    cv_file = None
    
    for file in files:
        file_lower = file.lower()
        file_path = os.path.join(DETECT_DIR, file)
        
        if not os.path.isfile(file_path):
            continue
            
        # Check if it's a job description file
        if 'jd' in file_lower or 'job' in file_lower or 'description' in file_lower:
            jd_file = file_path
        # Otherwise, assume it's a CV file
        else:
            ext = os.path.splitext(file)[1].lower()
            if ext in ['.pdf', '.txt', '.docx', '.doc']:
                cv_file = file_path
    
    if not jd_file:
        raise ValueError("No job description file found in detect folder! Please include a file with 'jd', 'job', or 'description' in the name.")
    
    if not cv_file:
        raise ValueError("No CV file found in detect folder! Please include a PDF, TXT, or DOCX file.")
    
    return jd_file, cv_file

def print_banner(text):
    """Print text with a prominent banner"""
    width = max(len(text) + 4, 80)
    print("\n" + "="*width)
    print(f"  {text}")
    print("="*width + "\n")

def log_result(cv_name, status, confidence, visual_score, semantic_score, attention_weights):
    """Log the detection result prominently"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Create prominent result display
    print("\n")
    print("*" * 100)
    print("*" * 100)
    print("*" + " " * 98 + "*")
    print("*" + f"  DETECTION RESULT - {timestamp}".center(96) + "*")
    print("*" + " " * 98 + "*")
    print("*" + f"  CV FILE: {cv_name}".center(96) + "*")
    print("*" + " " * 98 + "*")
    print("*" + " " * 98 + "*")
    print("*" + f"  PREDICTED STATUS: {status}".center(96) + "*")
    print("*" + f"  CONFIDENCE: {confidence:.2%}".center(96) + "*")
    print("*" + " " * 98 + "*")
    print("*" + "-" * 98 + "*")
    print("*" + " " * 98 + "*")
    print("*" + "  SCORING BREAKDOWN:".center(96) + "*")
    print("*" + " " * 98 + "*")
    print("*" + f"  📊 VISUAL SCORE: {visual_score:.2%}".center(96) + "*")
    print("*" + f"     (How professional/well-formatted the CV looks)".center(96) + "*")
    print("*" + " " * 98 + "*")
    print("*" + f"  🎯 SEMANTIC MATCHING SCORE: {semantic_score:.2%}".center(96) + "*")
    print("*" + f"     (How well the CV content matches the job description)".center(96) + "*")
    print("*" + " " * 98 + "*")
    print("*" + "-" * 98 + "*")
    print("*" + " " * 98 + "*")
    print("*" + f"  Model Attention Focus:".center(96) + "*")
    print("*" + f"  Visual: {attention_weights['visual']:.2f} | Semantic: {attention_weights['semantic']:.2f} | Format: {attention_weights['format']:.2f} | Job: {attention_weights['job']:.2f}".center(96) + "*")
    print("*" + " " * 98 + "*")
    print("*" * 100)
    print("*" * 100)
    print("\n")
    
    # Also save to a log file
    log_filename = "cv_detection_log.txt"
    with open(log_filename, 'a') as f:
        f.write(f"\n{'='*60}\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"CV File: {cv_name}\n")
        f.write(f"STATUS: {status}\n")
        f.write(f"Confidence: {confidence:.2%}\n")
        f.write(f"Visual Score: {visual_score:.2%}\n")
        f.write(f"Semantic Score: {semantic_score:.2%}\n")
        f.write(f"Attention Weights: Visual={attention_weights['visual']:.2f}, Semantic={attention_weights['semantic']:.2f}, Format={attention_weights['format']:.2f}, Job={attention_weights['job']:.2f}\n")
        f.write(f"{'='*60}\n")
    
    print(f"Result also logged to: {log_filename}")

def main():
    """Main function to detect CV status"""
    print_banner("CV STATUS DETECTION SYSTEM")
    print(f"Using model: {MODEL_WEIGHTS_PATH}")
    print(f"Looking for files in: {DETECT_DIR}")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Find files in detect folder
    try:
        jd_file, cv_file = find_files_in_detect_folder()
        print(f"\nFound Job Description: {os.path.basename(jd_file)}")
        print(f"Found CV: {os.path.basename(cv_file)}")
    except ValueError as e:
        print(f"\nERROR: {e}")
        return
    
    # Load models and processors
    print("\nLoading models and processors...")
    bert_tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_PATH)
    vit_processor = ViTImageProcessor.from_pretrained(VIT_MODEL_PATH)
    
    # Initialize model (assuming 10 jobs to match trained model)
    model = MultimodalCVClassifier(VIT_MODEL_PATH, BERT_MODEL_PATH, num_classes=4, num_jobs=10)
    
    # Load trained weights
    if not os.path.exists(MODEL_WEIGHTS_PATH):
        print(f"\nERROR: Model weights not found at {MODEL_WEIGHTS_PATH}")
        print("Please ensure you have trained the model first using train.py")
        return
    
    print(f"Loading model weights from {MODEL_WEIGHTS_PATH}...")
    model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location=device))
    model.to(device)
    model.eval()
    
    # Extract texts
    print("\nExtracting text from files...")
    job_description = extract_text_from_file(jd_file)
    cv_text = extract_text_from_file(cv_file)
    
    # Log text lengths for debugging
    print(f"\nText lengths:")
    print(f"  Job Description: {len(job_description)} characters")
    print(f"  CV Text: {len(cv_text)} characters")
    print(f"  Total combined: {len(job_description) + len(cv_text)} characters")
    
    # Create image from CV
    print("Processing CV image...")
    cv_image = create_page_image(cv_file)
    
    # Extract format features
    print("Extracting CV format features...")
    format_features_dict = extract_cv_features(cv_file)
    format_features = torch.tensor([
        min(format_features_dict["font_count"], 10) / 10.0,
        min(format_features_dict["text_density"], 20) / 20.0,
        min(format_features_dict["bullet_points"], 30) / 30.0,
        min(format_features_dict["pages"], 5) / 5.0
    ], dtype=torch.float32).unsqueeze(0).to(device)
    
    # Process text with BERT tokenizer - now using chunk-based approach
    print("Processing text with BERT...")
    
    # Get comprehensive semantic features using chunk processing
    chunk_semantic_features = process_text_in_chunks(
        model.bert, 
        bert_tokenizer, 
        job_description, 
        cv_text, 
        device
    )
    
    # Also create a standard encoding for the model's expected input format
    # This uses truncated versions but the semantic score will come from chunk processing
    jd_truncated = job_description[:500] if len(job_description) > 500 else job_description
    cv_truncated = cv_text[:1500] if len(cv_text) > 1500 else cv_text
    
    encoding = bert_tokenizer(
        jd_truncated,
        cv_truncated,
        add_special_tokens=True,
        max_length=512,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
        return_overflowing_tokens=False,
    )
    
    # Process image for ViT
    print("Processing image with ViT...")
    visual_inputs = vit_processor(images=cv_image, return_tensors="pt")
    
    # Move inputs to device
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)
    token_type_ids = encoding["token_type_ids"].to(device)
    pixel_values = visual_inputs.pixel_values.to(device)
    
    # Create generic job encoding (all zeros except first position)
    job_one_hot = torch.zeros(1, 10, dtype=torch.float32).to(device)
    job_one_hot[0, 0] = 1.0
    
    # Make prediction with modified semantic processing
    print("\nMaking prediction...")
    with torch.no_grad():
        # First get the visual features
        vit_outputs = model.vit(pixel_values=pixel_values)
        vit_features = vit_outputs.pooler_output
        visual_features = model.visual_branch(vit_features)
        visual_score_val = model.visual_score(visual_features)
        
        # Use our chunk-based semantic features
        semantic_features = model.semantic_branch(chunk_semantic_features)
        semantic_score_val = model.semantic_score(semantic_features)
        
        # Ensure semantic_features has the right shape [batch_size, feature_dim]
        if semantic_features.dim() == 3:
            semantic_features = semantic_features.squeeze(1)
        
        # Get format and job features
        format_features_processed = model.format_branch(format_features)
        job_features = model.job_branch(job_one_hot)
        
        # Combine all features
        combined_features = torch.cat(
            (visual_features, semantic_features, format_features_processed, job_features), dim=1
        )
        
        # Get attention weights
        attention_weights = model.attention(combined_features)
        
        # Apply attention weights
        visual_weight = attention_weights[:, 0].unsqueeze(1).expand_as(visual_features)
        semantic_weight = attention_weights[:, 1].unsqueeze(1).expand_as(semantic_features)
        format_weight = attention_weights[:, 2].unsqueeze(1).expand_as(format_features_processed)
        job_weight = attention_weights[:, 3].unsqueeze(1).expand_as(job_features)
        
        weighted_visual = visual_features * visual_weight
        weighted_semantic = semantic_features * semantic_weight
        weighted_format = format_features_processed * format_weight
        weighted_job = job_features * job_weight
        
        weighted_features = torch.cat(
            (weighted_visual, weighted_semantic, weighted_format, weighted_job), dim=1
        )
        
        # Get final prediction
        logits = model.classifier(weighted_features)
        
        visual_score = visual_score_val.item()
        semantic_score = semantic_score_val.item()
        attention_weights_np = attention_weights[0].cpu().numpy()
        
        # Get probabilities and predicted class
        probabilities = F.softmax(logits, dim=1)[0]
        pred_class = torch.argmax(probabilities).item()
        confidence = probabilities[pred_class].item()
        
        # Map to status
        status = map_label_to_status(pred_class)
        
    # Prepare attention weights dict
    attention_dict = {
        "visual": attention_weights_np[0],
        "semantic": attention_weights_np[1],
        "format": attention_weights_np[2],
        "job": attention_weights_np[3]
    }
    
    # Log the result prominently
    log_result(
        cv_name=os.path.basename(cv_file),
        status=status,
        confidence=confidence,
        visual_score=sigmoid(visual_score),
        semantic_score=sigmoid(semantic_score),
        attention_weights=attention_dict
    )
    
    # Show all probabilities
    print("\nDetailed Probabilities:")
    for i, prob in enumerate(probabilities):
        status_name = map_label_to_status(i)
        print(f"  {status_name}: {prob.item():.2%}")
    
    # Provide interpretation of scores
    print("\n" + "="*60)
    print("SCORE INTERPRETATION:")
    print("="*60)
    
    visual_score_norm = sigmoid(visual_score)
    semantic_score_norm = sigmoid(semantic_score)
    
    # Visual score interpretation
    if visual_score_norm >= 0.8:
        visual_interpretation = "Excellent - Very professional and well-formatted CV"
    elif visual_score_norm >= 0.6:
        visual_interpretation = "Good - Professional appearance with minor improvements needed"
    elif visual_score_norm >= 0.4:
        visual_interpretation = "Average - Acceptable format but could be improved"
    else:
        visual_interpretation = "Poor - Needs significant formatting improvements"
    
    # Semantic score interpretation
    if semantic_score_norm >= 0.8:
        semantic_interpretation = "Excellent Match - CV content strongly aligns with job requirements"
    elif semantic_score_norm >= 0.6:
        semantic_interpretation = "Good Match - CV shows relevant skills and experience"
    elif semantic_score_norm >= 0.4:
        semantic_interpretation = "Partial Match - Some relevant experience but gaps exist"
    else:
        semantic_interpretation = "Poor Match - CV content doesn't align well with job requirements"
    
    print(f"\nVisual Score ({visual_score_norm:.2%}): {visual_interpretation}")
    print(f"Semantic Score ({semantic_score_norm:.2%}): {semantic_interpretation}")
    print("="*60)
    
    print("\nDetection complete!")

if __name__ == "__main__":
    main()