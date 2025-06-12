import torch
from torch import nn
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer
import os
import sys
import argparse
from pathlib import Path

# Force offline mode to use local BERT model
os.environ["TRANSFORMERS_OFFLINE"] = "1"


# Enhanced model with cross-attention (similar to train.py)
class EnhancedBertForCVJDMatching(nn.Module):
    def __init__(self, bert_model_name, num_classes, dropout_rate=0.2):
        super(EnhancedBertForCVJDMatching, self).__init__()

        # Load pre-trained BERT model
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.bert_dim = self.bert.config.hidden_size

        # Cross-attention between CV and JD
        self.cv_jd_attention = nn.MultiheadAttention(
            embed_dim=self.bert_dim,
            num_heads=8,
            dropout=dropout_rate
        )

        # Semantic branch (similar to train.py)
        self.semantic_branch = nn.Sequential(
            nn.Linear(self.bert_dim * 2, 768),  # CV + JD embeddings concatenated
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes),
        )

        # Semantic score prediction
        self.semantic_score = nn.Linear(256, 1)

    def forward(self, input_ids, attention_mask, token_type_ids):
        # Get BERT outputs
        bert_outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        # Get all hidden states for cross-attention
        all_hidden_states = bert_outputs.last_hidden_state

        # Separate CV and JD representations using token_type_ids
        # token_type_ids = 0 for JD, 1 for CV
        jd_mask = (token_type_ids == 0).float()
        cv_mask = (token_type_ids == 1).float()

        # Get average embeddings for JD and CV
        jd_embeddings = (all_hidden_states * jd_mask.unsqueeze(-1)).sum(dim=1) / jd_mask.sum(dim=1, keepdim=True).clamp(min=1)
        cv_embeddings = (all_hidden_states * cv_mask.unsqueeze(-1)).sum(dim=1) / cv_mask.sum(dim=1, keepdim=True).clamp(min=1)

        # Apply cross-attention between CV and JD
        cv_attended, _ = self.cv_jd_attention(
            cv_embeddings.unsqueeze(0),  # Query
            jd_embeddings.unsqueeze(0),  # Key
            jd_embeddings.unsqueeze(0),  # Value
        )
        cv_attended = cv_attended.squeeze(0)

        # Concatenate original and attended features
        combined_semantic = torch.cat([cv_embeddings, cv_attended], dim=1)

        # Process through semantic branch
        semantic_features = self.semantic_branch(combined_semantic)

        # Get semantic score
        semantic_score_val = self.semantic_score(semantic_features)

        # Classification
        logits = self.classifier(semantic_features)

        return logits, semantic_score_val


def read_file(file_path):
    """Read content from various file formats"""
    try:
        path = Path(file_path)
        ext = path.suffix.lower()

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
            return f"Unsupported file type: {ext}"
    except Exception as e:
        return f"Error reading file: {str(e)}"


def evaluate_cv(jd_text, cv_text, model, tokenizer, device, max_length=768):
    """Evaluate CV against job description with enhanced semantic processing"""
    
    # Status mapping (matching train.py)
    status_labels = {
        0: "ACCEPT",
        1: "INTERVIEW",
        2: "SHORTLIST",
        3: "REJECT",
    }

    # Tokenize the JD and CV pair
    encoding = tokenizer(
        jd_text,
        cv_text,
        add_special_tokens=True,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    # Move tensors to device
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)
    token_type_ids = encoding["token_type_ids"].to(device)

    # Set model to evaluation mode
    model.eval()

    # Get predictions
    with torch.no_grad():
        logits, semantic_score = model(input_ids, attention_mask, token_type_ids)

        # Get probabilities
        probabilities = F.softmax(logits, dim=1)

        # Get predicted class
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class].item()

        # Normalize semantic score
        semantic_match = torch.sigmoid(semantic_score).item()

    # Prepare detailed results
    result = {
        "status": status_labels[predicted_class],
        "confidence": confidence,
        "semantic_match": semantic_match,
        "probabilities": {
            status_labels[i]: prob.item() 
            for i, prob in enumerate(probabilities[0])
        }
    }

    return result


def load_model(model_path="enhanced_cv_model.pt"):
    """Load the enhanced model"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load BERT tokenizer
    bert_model_name = "./local_bert_model"
    tokenizer = BertTokenizer.from_pretrained(bert_model_name)

    # Initialize model
    num_classes = 4  # ACCEPT, INTERVIEW, SHORTLIST, REJECT
    model = EnhancedBertForCVJDMatching(bert_model_name, num_classes)

    # Load saved weights
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Model loaded from {model_path}")
    else:
        print(f"Warning: Model file {model_path} not found. Using untrained model.")

    model.to(device)
    return model, tokenizer, device


def main():
    parser = argparse.ArgumentParser(description='Evaluate CV against Job Description')
    parser.add_argument('--jd', type=str, required=True, help='Path to job description file')
    parser.add_argument('--cv', type=str, required=True, help='Path to CV file')
    parser.add_argument('--model', type=str, default='enhanced_cv_model.pt', help='Path to model file')
    
    args = parser.parse_args()

    # Load model
    model, tokenizer, device = load_model(args.model)

    # Read files
    print("\nReading job description...")
    jd_text = read_file(args.jd)
    if jd_text.startswith("Error"):
        print(f"Error: {jd_text}")
        return

    print("Reading CV...")
    cv_text = read_file(args.cv)
    if cv_text.startswith("Error"):
        print(f"Error: {cv_text}")
        return

    # Display file info
    print(f"\nJob Description: {args.jd}")
    print(f"CV: {args.cv}")
    print(f"JD Preview: {jd_text[:200]}...")
    print(f"CV Preview: {cv_text[:200]}...")

    # Evaluate
    print("\nEvaluating CV against Job Description...")
    result = evaluate_cv(jd_text, cv_text, model, tokenizer, device)

    # Display results
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"Decision: {result['status']}")
    print(f"Confidence: {result['confidence']:.2%}")
    print(f"CV-JD Match Score: {result['semantic_match']:.2%}")
    print("\nDetailed Probabilities:")
    for status, prob in result['probabilities'].items():
        print(f"  {status}: {prob:.2%}")
    print("="*60)

    # Interpretation
    if result['semantic_match'] > 0.7:
        print("\n✓ Strong match between CV and Job Description")
    elif result['semantic_match'] > 0.5:
        print("\n~ Moderate match between CV and Job Description")
    else:
        print("\n✗ Weak match between CV and Job Description")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("Usage: python detect.py --jd <job_description_file> --cv <cv_file>")
        print("Example: python detect.py --jd job.txt --cv resume.pdf")
        sys.exit(1)
    
    main()