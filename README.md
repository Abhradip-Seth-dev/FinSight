# FinSight — Transaction Monitoring & Document Analysis Platform

FinSight is a modern, production-ready web application designed for financial document processing, salary eligibility evaluation, and transactional bank statement classification. It combines OCR document scanning with a fine-tuned BERT transformer model to provide intelligent insights through a sleek, glassmorphic interactive dashboard.

---

## 🌟 Key Features

* **Salary Eligibility Checker (OCR)**
  * Uploads payslips in image formats (PNG, JPG) or PDFs.
  * Preprocesses documents using OpenCV (grayscale conversion, cubic scaling, Otsu's thresholding) to maximize OCR character recognition.
  * Extracts salary components (Basic Pay, HRA, Taxes/TDS) and checks loan eligibility based on basic salary thresholds.

* **Bank Statement Category Classifier (NLP)**
  * Uploads bank statements (CSV, Excel, PDF, or plain text).
  * Automatically extracts transaction amounts using customized regular expression matchers.
  * Runs inference on transaction contexts using a fine-tuned **BERT Sequence Classification model** (`bert-base-uncased`) to categorize expenses (e.g., dining, utilities, investment, etc.).
  * Generates custom Canvas-based interactive donut and bar charts for immediate visual breakdown.

* **Secure Authentication & Management**
  * Built-in registration, credentials verification, and session state tracking using Flask.
  * Persists user records securely in a MySQL database.

---

## 🛠️ Technology Stack

* **Frontend**: HTML5, Vanilla JavaScript, CSS3 (Modern Glassmorphism aesthetics, responsive layouts).
* **Backend**: Flask, MySQL Connector.
* **Computer Vision & OCR**: OpenCV (`cv2`), PyTesseract, `pdf2image` (Poppler).
* **Deep Learning & Data Science**: PyTorch, Hugging Face Transformers (BERT), Pandas, NumPy, Scikit-learn.

---

## 📁 Repository Structure

```text
├── app.py                     # Flask web server (routes, auth, uploads handling)
├── ocr_utils.py               # Document pre-processing & PyTesseract OCR extraction
├── train_bert.py              # BERT training pipelines, dataset utilities, and inference services
├── templates/
│   ├── login.html             # Login and Registration portal interface
│   └── dashboard.html         # Interactive user dashboard (Charts, Drag & Drop uploads)
├── uploads/                   # Folder holding uploaded documents temporarily
├── .env                       # Environment variables config
└── .gitignore                 # Target build exclusions
```

---

## 🚀 Getting Started

### Prerequisites

1. **Python 3.8+**
2. **MySQL Server**
3. **Tesseract OCR** (For text extraction)
   * macOS: `brew install tesseract`
   * Windows: Install executable from [UB Mannheim](https://github.com/UB-Mannheim/tesseract/wiki).
4. **Poppler** (For PDF conversion)
   * macOS: `brew install poppler`

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Abhradip-Seth-dev/FinSight.git
   cd FinSight
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   *(Note: Make sure to install `torch`, `transformers`, `opencv-python`, `pytesseract`, `pdf2image`, `mysql-connector-python`, `pandas`, `openpyxl`, `scikit-learn`)*

3. Configure your MySQL Database:
   * Create a database named `transaction_db`.
   * Create a `users` table:
     ```sql
     CREATE TABLE users (
         uid VARCHAR(50) PRIMARY KEY,
         username VARCHAR(50) UNIQUE NOT NULL,
         password VARCHAR(100) NOT NULL,
         created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
     );
     ```
   * Update the credentials in `get_database_connection()` inside [app.py](app.py).

4. Train the BERT model (if utilizing custom categorization):
   * Ensure `upi_transactions_2024.csv` is present in the base directory.
   * Run the training script:
     ```bash
     python3 train_bert.py
     ```

5. Launch the Web Server:
   ```bash
   python3 app.py
   ```
   Open your browser and navigate to `http://127.0.0.1:5000`.
