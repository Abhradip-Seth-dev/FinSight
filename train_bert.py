"""BERT model training and inference services for classifying financial transactions.
"""

import json
import logging
import os
import pickle
import re
import warnings
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset
from transformers import (
    BertForSequenceClassification,
    BertTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

warnings.filterwarnings("ignore")

# Configure logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

# ──────────────────────────────────────────────
# CONFIGURATION CONSTANTS
# ──────────────────────────────────────────────
PROJECT_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRANSACTION_CSV_PATH = os.path.join(PROJECT_BASE_DIR, "upi_transactions_2024.csv")
BERT_MODEL_DIR = os.path.join(PROJECT_BASE_DIR, "bert_model")
LABEL_ENCODER_PATH = os.path.join(BERT_MODEL_DIR, "label_encoder.pkl")

MAX_SEQUENCE_LENGTH = 64
TRAINING_BATCH_SIZE = 32
TRAINING_EPOCHS = 3
INITIAL_LEARNING_RATE = 2e-5
VALIDATION_SPLIT_SIZE = 0.15
RANDOM_SEED = 42

COMPUTATION_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device: {COMPUTATION_DEVICE}")


# ──────────────────────────────────────────────
# DATASET HELPER
# ──────────────────────────────────────────────
class TransactionDataset(Dataset):
    """Custom PyTorch Dataset for tokenizing and loading transaction texts and labels."""

    def __init__(
        self,
        transaction_texts: List[str],
        transaction_labels: List[int],
        bert_tokenizer: BertTokenizer,
        max_seq_len: int = MAX_SEQUENCE_LENGTH,
    ) -> None:
        """Initialize the dataset and tokenize inputs.

        Args:
            transaction_texts: List of transaction strings.
            transaction_labels: Encoded integer labels.
            bert_tokenizer: Pretrained BERT tokenizer instance.
            max_seq_len: Maximum token sequence length.
        """
        self.encodings = bert_tokenizer(
            transaction_texts,
            truncation=True,
            padding="max_length",
            max_length=max_seq_len,
        )
        self.labels = transaction_labels

    def __len__(self) -> int:
        """Return the total number of transactions."""
        return len(self.labels)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        """Fetch tokenized tensors for a single sample.

        Args:
            index: The index of the item.

        Returns:
            A dictionary containing model input tensors (input_ids, attention_mask)
            and label tensor.
        """
        encoded_item = {key: torch.tensor(val[index]) for key, val in self.encodings.items()}
        encoded_item["labels"] = torch.tensor(self.labels[index], dtype=torch.long)
        return encoded_item


# ──────────────────────────────────────────────
# DATA CONVERTER FROM ROW
# ──────────────────────────────────────────────
def convert_row_to_bert_input(transaction_row: pd.Series) -> str:
    """Convert a transaction dataframe row into a natural-language description.

    Args:
        transaction_row: A pandas Series containing transaction features.

    Returns:
        A text representation of the transaction features suitable for BERT.
    """
    text_segments = []

    # Process transaction type
    if "transaction type" in transaction_row.index:
        text_segments.append(f"type {str(transaction_row['transaction type']).strip()}")

    # Process amount
    if "amount (INR)" in transaction_row.index:
        text_segments.append(f"amount {transaction_row['amount (INR)']} INR")

    # Process bank fields
    for column_name in ("sender_bank", "receiver_bank"):
        if column_name in transaction_row.index:
            text_segments.append(f"{column_name.replace('_', ' ')} {str(transaction_row[column_name]).strip()}")

    # Process device type
    if "device_type" in transaction_row.index:
        text_segments.append(f"device {str(transaction_row['device_type']).strip()}")

    # Process temporal features
    if "hour_of_day" in transaction_row.index:
        text_segments.append(f"hour {transaction_row['hour_of_day']}")
    if "day_of_week" in transaction_row.index:
        text_segments.append(f"day {str(transaction_row['day_of_week']).strip()}")
    if "is_weekend" in transaction_row.index:
        text_segments.append("weekend" if transaction_row["is_weekend"] == 1 else "weekday")

    # Process fraud tag
    if "fraud_flag" in transaction_row.index:
        text_segments.append("fraud" if transaction_row["fraud_flag"] == 1 else "legit")

    # Process transaction status
    if "transaction_status" in transaction_row.index:
        text_segments.append(f"status {str(transaction_row['transaction_status']).strip()}")

    return " ".join(text_segments)


# ──────────────────────────────────────────────
# METRICS COMPUTATION
# ──────────────────────────────────────────────
def evaluate_model_metrics(evaluation_predictions) -> Dict[str, float]:
    """Compute the accuracy score of the BERT classifier predictions.

    Args:
        evaluation_predictions: Tuple of predictions and labels.

    Returns:
        A dictionary containing the calculated accuracy metric.
    """
    prediction_logits, true_labels = evaluation_predictions
    predicted_classes = np.argmax(prediction_logits, axis=-1)
    accuracy_score = (predicted_classes == true_labels).mean()
    return {"accuracy": float(accuracy_score)}


# ──────────────────────────────────────────────
# BERT TRAINING ENGINE
# ──────────────────────────────────────────────
def train_bert_classifier() -> None:
    """Train the BERT Sequence Classification model using the transaction CSV dataset."""
    logger.info("=" * 60)
    logger.info("  BERT Transaction Classifier – Training Initiated")
    logger.info("=" * 60)

    # 1. Load data
    logger.info(f"[1/6] Loading transaction dataset from:\n      {TRANSACTION_CSV_PATH}")
    if not os.path.exists(TRANSACTION_CSV_PATH):
        raise FileNotFoundError(f"Missing training data at: {TRANSACTION_CSV_PATH}")

    transactions_df = pd.read_csv(TRANSACTION_CSV_PATH)
    logger.info(f"      Loaded rows: {len(transactions_df):,} | Columns: {list(transactions_df.columns)}")

    # 2. Check for category label column
    required_columns = {"merchant_category"}
    missing_columns = required_columns - set(transactions_df.columns)
    if missing_columns:
        logger.error(f"Missing mandatory column: {missing_columns}")
        raise ValueError(f"CSV is missing columns: {missing_columns}")

    # 3. Clean labels and prepare textual input structures
    logger.info("[2/6] Structuring rows into training text documents...")
    transactions_df = transactions_df.dropna(subset=["merchant_category"]).reset_index(drop=True)
    processed_texts = transactions_df.apply(convert_row_to_bert_input, axis=1).tolist()

    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(transactions_df["merchant_category"]).tolist()
    number_of_classes = len(label_encoder.classes_)
    logger.info(f"      Total unique categories ({number_of_classes}): {list(label_encoder.classes_)}")

    # 4. Train / Val split
    logger.info("[3/6] Splitting dataset into train and validation sets...")
    train_texts, validation_texts, train_labels, validation_labels = train_test_split(
        processed_texts,
        encoded_labels,
        test_size=VALIDATION_SPLIT_SIZE,
        random_state=RANDOM_SEED,
        stratify=encoded_labels,
    )
    logger.info(f"      Train set size: {len(train_texts):,} | Val set size: {len(validation_texts):,}")

    # 5. Tokenize
    logger.info("[4/6] Loading pre-trained Tokenizer (bert-base-uncased)...")
    bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    train_dataset = TransactionDataset(train_texts, train_labels, bert_tokenizer)
    validation_dataset = TransactionDataset(validation_texts, validation_labels, bert_tokenizer)

    # 6. Initialize Model
    logger.info("[5/6] Initializing Sequence Classification Model...")
    sequence_classification_model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased", num_labels=number_of_classes
    )
    sequence_classification_model.to(COMPUTATION_DEVICE)

    # 7. Configure trainer
    logger.info("[6/6] Fine-tuning the classifier...")
    os.makedirs(BERT_MODEL_DIR, exist_ok=True)

    training_arguments = TrainingArguments(
        output_dir=BERT_MODEL_DIR,
        num_train_epochs=TRAINING_EPOCHS,
        per_device_train_batch_size=TRAINING_BATCH_SIZE,
        per_device_eval_batch_size=TRAINING_BATCH_SIZE * 2,
        learning_rate=INITIAL_LEARNING_RATE,
        weight_decay=0.01,
        warmup_ratio=0.1,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        logging_steps=100,
        fp16=(COMPUTATION_DEVICE == "cuda"),
        report_to="none",  # Deactivate weights & biases and tensorboard reporting
        seed=RANDOM_SEED,
    )

    huggingface_trainer = Trainer(
        model=sequence_classification_model,
        args=training_arguments,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        compute_metrics=evaluate_model_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )
    
    huggingface_trainer.train()

    # 8. Save artifacts
    huggingface_trainer.save_model(BERT_MODEL_DIR)
    bert_tokenizer.save_pretrained(BERT_MODEL_DIR)
    with open(LABEL_ENCODER_PATH, "wb") as file_handle:
        pickle.dump(label_encoder, file_handle)

    evaluation_metrics = huggingface_trainer.evaluate()
    logger.info(f"\n✅ Training complete! Validation Accuracy: {evaluation_metrics.get('eval_accuracy', 0):.4f}")
    logger.info(f"   Model artifacts saved to: {BERT_MODEL_DIR}")


# ──────────────────────────────────────────────
# MODEL INFERENCE SERVICES (Called by web app)
# ──────────────────────────────────────────────

# Lazy-loaded singletons for memory efficiency
_global_bert_model: Optional[BertForSequenceClassification] = None
_global_bert_tokenizer: Optional[BertTokenizer] = None
_global_label_encoder: Optional[LabelEncoder] = None


def _initialize_inference_assets() -> None:
    """Load model, tokenizer, and label encoder dynamically on the first prediction request."""
    global _global_bert_model, _global_bert_tokenizer, _global_label_encoder
    if _global_bert_model is not None:
        return

    if not os.path.isdir(BERT_MODEL_DIR):
        logger.error(f"Inference directory {BERT_MODEL_DIR} not found. Model must be trained first.")
        raise RuntimeError("No trained model found. Please run: python train_bert.py")

    logger.info("[BERT] Loading model artifacts into memory...")
    _global_bert_tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_DIR)
    _global_bert_model = BertForSequenceClassification.from_pretrained(BERT_MODEL_DIR)
    _global_bert_model.eval()
    _global_bert_model.to(COMPUTATION_DEVICE)

    with open(LABEL_ENCODER_PATH, "rb") as file_handle:
        _global_label_encoder = pickle.load(file_handle)
    logger.info("[BERT] Inference model setup completed.")


def predict_transaction_categories(transaction_texts: List[str]) -> List[str]:
    """Run model inference to assign category labels to transaction descriptions.

    Args:
        transaction_texts: List of natural language transaction text strings.

    Returns:
        List of decoded category names predicted for each text snippet.
    """
    _initialize_inference_assets()
    tokenized_encodings = _global_bert_tokenizer(
        transaction_texts,
        truncation=True,
        padding="max_length",
        max_length=MAX_SEQUENCE_LENGTH,
        return_tensors="pt",
    ).to(COMPUTATION_DEVICE)

    with torch.no_grad():
        prediction_logits = _global_bert_model(**tokenized_encodings).logits
    
    predicted_class_indices = torch.argmax(prediction_logits, dim=-1).cpu().numpy()
    return _global_label_encoder.inverse_transform(predicted_class_indices).tolist()


def _parse_uploaded_bank_statement(file_path: str) -> List[dict]:
    """Parse raw transaction entries from a bank statement file.

    Accepts CSV, XLSX, XLS, PDF, or Plain Text.

    Args:
        file_path: Path to the bank statement file.

    Returns:
        A list of transaction record dictionaries.
    """
    file_extension = os.path.splitext(file_path)[-1].lower()

    if file_extension == ".csv":
        logger.info(f"Parsing CSV bank statement: {file_path}")
        statement_df = pd.read_csv(file_path)
    elif file_extension in (".xlsx", ".xls"):
        logger.info(f"Parsing Excel bank statement: {file_path}")
        statement_df = pd.read_excel(file_path)
    elif file_extension == ".pdf":
        logger.info(f"Parsing PDF bank statement (Extracting Text): {file_path}")
        extracted_text = _extract_text_from_pdf(file_path)
        return _parse_raw_text_to_transactions(extracted_text)
    else:
        logger.info(f"Treating file as plain text statement: {file_path}")
        with open(file_path, "r", errors="ignore") as file_handle:
            extracted_text = file_handle.read()
        return _parse_raw_text_to_transactions(extracted_text)

    return statement_df.to_dict(orient="records")


def _extract_text_from_pdf(file_path: str) -> str:
    """Extract plain text from PDF using pdftotext tool, falling back to OCR if needed.

    Args:
        file_path: Path to the PDF file.

    Returns:
        Extracted raw text content of the PDF.
    """
    import shutil
    import subprocess

    # Strategy 1: Attempt using pdftotext utility (from Poppler)
    pdftotext_executable = shutil.which("pdftotext")
    if pdftotext_executable is None:
        for candidate_path in ("/opt/homebrew/bin/pdftotext", "/usr/local/bin/pdftotext"):
            if os.path.isfile(candidate_path):
                pdftotext_executable = candidate_path
                break

    if pdftotext_executable:
        try:
            logger.info("Attempting extraction using pdftotext command-line tool...")
            subprocess_result = subprocess.run(
                [pdftotext_executable, "-layout", file_path, "-"],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if subprocess_result.returncode == 0 and subprocess_result.stdout.strip():
                logger.info(f"[PDF] Extracted {len(subprocess_result.stdout)} characters using pdftotext")
                return subprocess_result.stdout
        except Exception as exc:
            logger.warning(f"[PDF] pdftotext tool execution failed: {exc}. Falling back to OCR.")

    # Strategy 2: Fallback to pdf2image + PyTesseract OCR
    logger.info("Falling back to PDF-to-image conversion and OCR scanning...")
    try:
        import cv2
        from pdf2image import convert_from_path
        import pytesseract

        pdf_pages = convert_from_path(file_path, dpi=200)
        extracted_page_texts = []
        # Limit to the first 5 pages to avoid potential performance issues/timeouts
        for index, pdf_page in enumerate(pdf_pages[:5]):
            page_image = np.array(pdf_page)
            bgr_image = cv2.cvtColor(page_image, cv2.COLOR_RGB2BGR)
            grayscale_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
            extracted_text = pytesseract.image_to_string(grayscale_image, config="--psm 6")
            extracted_page_texts.append(extracted_text)
            logger.info(f"[PDF OCR] Processed page {index + 1}: {len(extracted_text)} characters parsed.")
        return "\n".join(extracted_page_texts)
    except Exception as exc:
        logger.error(f"[PDF OCR] Scan failed entirely: {exc}", exc_info=True)
        return ""


def _parse_raw_text_to_transactions(raw_text: str) -> List[dict]:
    """Parse potential transaction amounts out of raw unstructured text blocks.

    Args:
        raw_text: Parsed string from PDF or OCR output.

    Returns:
        List of transactions containing amount dictionaries.
    """
    if not raw_text.strip():
        return []

    # Pattern 1: Match currency symbols prefixed amounts (e.g. INR 1,500.00, Rs 500, ₹2500)
    detected_amounts = re.findall(r"(?:INR|Rs\.?|₹)\s*([\d,]+(?:\.\d{1,2})?)", raw_text, re.IGNORECASE)

    # Pattern 2: Match debit/credit column prefixes (e.g. Dr 125.00, Cr. 2,000)
    if not detected_amounts:
        detected_amounts = re.findall(
            r"(?:Dr\.?|Cr\.?|debit|credit)\s+([\d,]+(?:\.\d{1,2})?)",
            raw_text,
            re.IGNORECASE,
        )

    # Pattern 3: Standalone numbers that match monetary values
    if not detected_amounts:
        potential_amount_candidates = re.findall(r"\b(\d{1,8}(?:,\d{3})*\.\d{2})\b", raw_text)
        detected_amounts = [amt for amt in potential_amount_candidates if float(amt.replace(",", "")) >= 1.0]

    logger.info(f"[PARSE] Detected {len(detected_amounts)} transactions from raw text ({len(raw_text)} chars)")

    transaction_records = []
    for amount_string in detected_amounts:
        try:
            cleaned_amount = float(amount_string.replace(",", ""))
            transaction_records.append({"amount (INR)": cleaned_amount})
        except ValueError:
            pass
    return transaction_records


def analyze_statement_transactions(file_path: str) -> Dict[str, Union[int, dict, str]]:
    """Analyze a statement file to classify transactions and aggregate category metrics.

    Args:
        file_path: Path to the bank statement file.

    Returns:
        Summary dictionary containing transaction counts and category totals.
    """
    try:
        transaction_records = _parse_uploaded_bank_statement(file_path)
        if not transaction_records:
            return {"error": "No transactions found in file."}

        # Build training format representation strings for inference
        processed_texts = []
        for record in transaction_records:
            record_series = pd.Series(record)
            processed_texts.append(convert_row_to_bert_input(record_series))

        # Predict in mini-batches to prevent GPU OOM scenarios
        inference_batch_size = 64
        all_predicted_categories = []
        for index in range(0, len(processed_texts), inference_batch_size):
            all_predicted_categories.extend(
                predict_transaction_categories(processed_texts[index : index + inference_batch_size])
            )

        # Aggregate transactions into a dictionary
        transaction_amounts = [
            float(rec.get("amount (INR)", rec.get("amount", 0))) for rec in transaction_records
        ]
        category_summary = {}
        for predicted_category, transaction_amount in zip(all_predicted_categories, transaction_amounts):
            if predicted_category not in category_summary:
                category_summary[predicted_category] = {"count": 0, "total_amount": 0.0}
            category_summary[predicted_category]["count"] += 1
            category_summary[predicted_category]["total_amount"] = round(
                category_summary[predicted_category]["total_amount"] + transaction_amount, 2
            )

        return {
            "total_transactions": len(transaction_records),
            "categories": category_summary,
        }

    except RuntimeError as exc:
        # Gracefully handle situations where BERT is not yet fine-tuned
        logger.error(f"Inference run failed: {exc}")
        return {"error": str(exc), "categories": {}}
    except Exception as exc:
        logger.error(f"Execution failed: {exc}", exc_info=True)
        return {"error": f"Analysis failed: {exc}", "categories": {}}


# ──────────────────────────────────────────────
# COMMAND LINE RUNNER
# ──────────────────────────────────────────────
if __name__ == "__main__":
    train_bert_classifier()