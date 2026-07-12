"""Flask Web Server for user authentication and document analysis tasks.
"""

import logging
import os
import uuid
from typing import Dict, Union

import mysql.connector
from flask import Flask, jsonify, redirect, render_template, request, session, Response
from ocr_utils import assess_salary_eligibility
from train_bert import analyze_statement_transactions

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

flask_app = Flask(__name__)
flask_app.secret_key = "secret123"

# Ensure the uploads directory exists
UPLOAD_DIRECTORY = "uploads"
os.makedirs(UPLOAD_DIRECTORY, exist_ok=True)


def get_database_connection() -> mysql.connector.MySQLConnection:
    """Establish and return a connection to the MySQL database.

    Returns:
        A connection object for MySQL database.
    """
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="somu2006",
        database="transaction_db",
    )


@flask_app.route("/")
def render_login_page() -> str:
    """Render and return the portal login interface page.

    Returns:
        The rendered template of the login HTML page.
    """
    return render_template("login.html")


@flask_app.route("/login", methods=["POST"])
def handle_login() -> Response:
    """Process user login requests by verifying credentials against the database.

    Returns:
        A JSON response signaling authentication success or failure.
    """
    request_payload = request.get_json()
    logger.info(f"Login attempt initiated for username: {request_payload.get('username')}")

    try:
        db_connection = get_database_connection()
        db_cursor = db_connection.cursor(dictionary=True)
        db_cursor.execute(
            "SELECT * FROM users WHERE username=%s AND password=%s",
            (request_payload["username"], request_payload["password"]),
        )
        user_record = db_cursor.fetchone()
        db_connection.close()

        if user_record:
            session["user"] = user_record
            logger.info(f"Successful authentication for user: {user_record['username']}")
            return jsonify(
                {
                    "status": "success",
                    "username": user_record["username"],
                    "date": str(user_record["created_date"]),
                }
            )

        logger.warning(f"Failed authentication attempt for username: {request_payload.get('username')}")
        return jsonify({"status": "error"})

    except Exception as exc:
        logger.error(f"Error during login authentication execution: {exc}", exc_info=True)
        return jsonify({"status": "error", "msg": str(exc)})


@flask_app.route("/register", methods=["POST"])
def handle_registration() -> Response:
    """Process user registration requests by verifying user uniqueness.

    Returns:
        A JSON response signaling account creation success, name conflict, or failure.
    """
    request_payload = request.get_json()
    logger.info(f"Registration request received for username: {request_payload.get('username')}")

    try:
        db_connection = get_database_connection()
        db_cursor = db_connection.cursor()
        
        # Verify if username already exists in database
        db_cursor.execute("SELECT * FROM users WHERE username=%s", (request_payload["username"],))
        if db_cursor.fetchone():
            db_connection.close()
            logger.warning(f"Registration aborted. Username already taken: {request_payload.get('username')}")
            return jsonify({"status": "exists"})

        # Generate a unique User ID identifier
        user_id = "U" + str(uuid.uuid4())[:8].upper()  # Generates e.g. U3F2A1B9
        db_cursor.execute(
            "INSERT INTO users (uid, username, password) VALUES (%s, %s, %s)",
            (user_id, request_payload["username"], request_payload["password"]),
        )
        db_connection.commit()
        db_connection.close()

        logger.info(f"Account registered successfully. User ID: {user_id}")
        return jsonify({"status": "success"})

    except Exception as exc:
        logger.error(f"Error during registration execution: {exc}", exc_info=True)
        return jsonify({"status": "error", "msg": str(exc)})


@flask_app.route("/dashboard")
def render_dashboard_page() -> Union[Response, str]:
    """Render the dashboard interface for authenticated sessions.

    Returns:
        The rendered dashboard template or a redirect to the login index.
    """
    if "user" not in session:
        logger.info("Unauthorized access attempt. Redirecting to login.")
        return redirect("/")
    return render_template("dashboard.html", user=session["user"])


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

    # Process based on selected document options
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


@flask_app.route("/logout")
def handle_logout() -> Response:
    """Clear session data and redirect user back to the login interface page.

    Returns:
        A redirection response to the index root route.
    """
    logger.info(f"Session terminated for user: {session.get('user', {}).get('username')}")
    session.clear()
    return redirect("/")


if __name__ == "__main__":
    flask_app.run(debug=True)