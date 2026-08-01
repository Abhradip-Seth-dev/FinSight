"""Flask Web Server for financial document analysis."""

import logging
import os

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request, Response

from ocr_utils import assess_salary_eligibility
from train_bert import analyze_statement_transactions

# Load environment variables from .env file if present
load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

flask_app = Flask(__name__)
flask_app.secret_key = os.getenv("SECRET_KEY", "finsight-secret-key")

# Ensure the uploads directory exists
UPLOAD_DIRECTORY = "uploads"
os.makedirs(UPLOAD_DIRECTORY, exist_ok=True)


@flask_app.route("/")
def render_dashboard_page() -> str:
    """Render the main FinSight dashboard.

    Returns:
        The rendered dashboard template.
    """
    return render_template("dashboard.html")


@flask_app.route("/upload", methods=["POST"])
def handle_file_upload() -> Response:
    """Process and analyze uploaded PDF or image files based on document types.

    Supports 'Passslip' for salary checks and 'Bank Statement' for transaction classification.

    Returns:
        A JSON response containing the respective analysis results or execution status.
    """
    uploaded_file = request.files.get("file")
    document_type = request.form.get("doc_type")

    if not uploaded_file:
        logger.warning("Upload request received with empty file attachment.")
        return jsonify({"status": "error", "msg": "No file received."})

    # Save incoming files securely into UPLOAD_DIRECTORY
    destination_file_path = os.path.join(UPLOAD_DIRECTORY, f"{document_type}_{uploaded_file.filename}")
    uploaded_file.save(destination_file_path)
    logger.info(f"File uploaded successfully to: {destination_file_path}")

    # Process based on selected document options, always clean up after
    try:
        if document_type == "Passslip":
            try:
                payslip_analysis_result = assess_salary_eligibility(destination_file_path)
                return jsonify({"status": "success", "ocr": payslip_analysis_result})
            except Exception as exc:
                logger.error(f"Payslip processing failed for {destination_file_path}: {exc}", exc_info=True)
                error_message = str(exc)
                if "poppler" in error_message.lower() or "pdfinfo" in error_message.lower():
                    error_message = (
                        "Poppler is not installed. Please install poppler using: "
                        "brew install poppler  and restart the web server."
                    )
                return jsonify({"status": "error", "msg": error_message})

        elif document_type == "Bank Statement":
            try:
                statement_analysis_result = analyze_statement_transactions(destination_file_path)
                return jsonify({"status": "success", "categories": statement_analysis_result})
            except Exception as exc:
                logger.error(f"Bank statement analysis failed for {destination_file_path}: {exc}", exc_info=True)
                return jsonify({"status": "error", "msg": str(exc)})

        return jsonify({"status": "success"})
    finally:
        # Clean up the uploaded file to preserve disk space
        if os.path.exists(destination_file_path):
            os.remove(destination_file_path)
            logger.info(f"Cleaned up uploaded file: {destination_file_path}")


if __name__ == "__main__":
    flask_app.run(debug=True)