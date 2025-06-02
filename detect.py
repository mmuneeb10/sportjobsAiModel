import torch
from torch import nn
from transformers import BertModel, BertTokenizer
import os

# Force offline mode to use local BERT model
os.environ["TRANSFORMERS_OFFLINE"] = "1"


# Define the model class (same as in server.py)
class BertForTextPairClassification(nn.Module):
    def __init__(self, bert_model_name, num_classes, dropout_rate=0.1):
        super(BertForTextPairClassification, self).__init__()

        # Load pre-trained BERT model
        self.bert = BertModel.from_pretrained(bert_model_name)

        # Feed-forward classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes),
        )

    def forward(self, input_ids, attention_mask, token_type_ids):
        # Get BERT outputs
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        # Use the [CLS] token representation
        pooled_output = outputs.pooler_output

        # Pass through classifier
        logits = self.classifier(pooled_output)

        return logits


# Function to evaluate a CV against a job description
def evaluate_cv(jd, cv, model, tokenizer, device):
    # Map numeric predictions back to status labels
    status_labels = {
        0: "Accepted",
        1: "Interview Scheduled",
        2: "Shortlisted",
        3: "On Hold",
        4: "Rejected",
    }

    # Tokenize the JD and CV pair
    encoding = tokenizer(
        jd,
        cv,
        add_special_tokens=True,
        max_length=256,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    # Move tensors to the appropriate device
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)
    token_type_ids = encoding["token_type_ids"].to(device)

    # Set model to evaluation mode
    model.eval()

    # Get predictions
    with torch.no_grad():
        outputs = model(input_ids, attention_mask, token_type_ids)

        # Get probabilities using softmax
        probabilities = torch.nn.functional.softmax(outputs, dim=1)

        # Get the most likely class
        _, predicted_class = torch.max(outputs, dim=1)
        prediction = predicted_class.item()

        # Get the confidence score (probability of the predicted class)
        confidence = probabilities[0][prediction].item()

    # Return prediction and confidence
    return status_labels[prediction], confidence


def load_model(model_path="final_model.pt"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load the BERT tokenizer from local directory
    bert_model_name = "./local_bert_model"
    tokenizer = BertTokenizer.from_pretrained(bert_model_name)

    # Initialize the model
    num_classes = 5  # Same number of classes as in training
    model = BertForTextPairClassification(bert_model_name, num_classes)

    # Load the saved model state
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)

    return model, tokenizer, device


def main():
    # Load the model, tokenizer, and determine device
    model, tokenizer, device = load_model()

    # Hardcoded job description (can be replaced as needed)
    job_description = """Head Coach for professional basketball team. Will oversee all aspects of team performance including training sessions, game strategy, player development, and scouting. Minimum 5 years coaching experience at collegiate or professional level required. Strong leadership skills and proven track record of success essential."""

    # Example CV for testing
    # test_cv = """Marketing professional with 10 years of experience at Nike and Adidas.
    # Led digital campaigns increasing engagement by 45%. Managed relationships with 20+
    # professional athletes across football and basketball leagues. MBA from Stanford University
    # with focus on Sports Business."""

    test_cv = """Enthusiastic and results-driven Physiotherapist with a Master’s degree in Physiotherapy and a focus on sports rehabilitation. Passionate about working with elite athletes, especially in tennis, to enhance performance and recovery. Seeking to apply expertise in injury management and rehabilitation at a leading tennis academy.

Completed a Master's in Physiotherapy with a sports specialization from the University of California, Los Angeles in 2021. Gained practical experience working as a Sports Physiotherapist Intern at XYZ Tennis Academy in Los Angeles from June 2020 to August 2021. During the internship, assisted in treating injuries and improving performance for junior and professional tennis players. Designed and implemented rehabilitation programs to address tennis-related injuries and conducted injury assessments to identify concerns, crafting personalized treatment plans.

Certified in Myofascial Release Techniques, CPR and First Aid, and Instrument-assisted Soft Tissue Mobilization (IASTM).

Skilled in injury assessment and rehabilitation, sports injury prevention, hands-on treatment techniques, and possess strong communication and interpersonal skills.

An avid tennis player with a deep passion for racquet sports, and regularly attend conferences and workshops on sports physiotherapy."""

    print("Job Description:")
    print(job_description)
    print("\nCV:")
    print(test_cv)

    # Evaluate the CV against the job description
    status, confidence = evaluate_cv(job_description, test_cv, model, tokenizer, device)

    # Display results
    print("\nEvaluation Results:")
    print(f"Status: {status}")
    print(f"Confidence: {confidence:.2%}")

    # Interactive mode - allow user to enter custom CVs
    while True:
        print("\nEnter a CV to evaluate (or 'quit' to exit):")
        user_cv = input()

        if user_cv.lower() == "quit":
            break

        status, confidence = evaluate_cv(
            job_description, user_cv, model, tokenizer, device
        )

        print("\nEvaluation Results:")
        print(f"Status: {status}")
        print(f"Confidence: {confidence:.2%}")


if __name__ == "__main__":
    main()
