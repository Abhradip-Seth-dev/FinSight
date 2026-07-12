"""OCR utility module for extracting text from document images/PDFs and parsing salary details.
"""

import logging
import os
import re
import shutil
from typing import Dict, List, Optional, Union

import cv2
import numpy as np
import pytesseract
from pdf2image import convert_from_path

# Configure logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

# Setup environment paths for external dependencies (Poppler and Tesseract)
POPPLER_SEARCH_PATHS = ("/opt/homebrew/bin", "/usr/local/bin")
for search_path in POPPLER_SEARCH_PATHS:
    if search_path not in os.environ.get("PATH", "") and os.path.isfile(
        os.path.join(search_path, "pdfinfo")
    ):
        os.environ["PATH"] = f"{search_path}:{os.environ.get('PATH', '')}"
        logger.info(f"Poppler path configured: {search_path}")
        break

tesseract_executable_path = shutil.which("tesseract")
if tesseract_executable_path is None:
    # Check common installations paths for Tesseract
    CANDIDATE_TESSERACT_PATHS = (
        "/opt/homebrew/bin/tesseract",
        "/usr/local/bin/tesseract",
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    )
    for candidate_path in CANDIDATE_TESSERACT_PATHS:
        if os.path.isfile(candidate_path):
            tesseract_executable_path = candidate_path
            break

if tesseract_executable_path:
    pytesseract.pytesseract.tesseract_cmd = tesseract_executable_path
    logger.info(f"Tesseract executable path set to: {tesseract_executable_path}")
else:
    logger.warning(
        "Tesseract executable not found. OCR processing will fail. "
        "Please install Tesseract and make sure it is in your system path."
    )


def extract_text_from_file(file_path: str) -> str:
    """Extract text from an image or a PDF page using PyTesseract OCR.

    Args:
        file_path: The absolute or relative path to the input image or PDF file.

    Returns:
        The extracted raw text string from the file.

    Raises:
        FileNotFoundError: If the file does not exist at file_path.
        Exception: If OCR or image parsing fails.
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Target file not found at: {file_path}")

    logger.info(f"Extracting text from document: {file_path}")
    try:
        file_extension = file_path.lower().split(".")[-1]
        if file_extension == "pdf":
            # Convert the first page of the PDF to an image for processing
            pdf_pages = convert_from_path(file_path, dpi=300)
            if not pdf_pages:
                logger.error(f"Failed to read any pages from PDF: {file_path}")
                return ""
            document_image = np.array(pdf_pages[0])
            document_image = cv2.cvtColor(document_image, cv2.COLOR_RGB2BGR)
        else:
            document_image = cv2.imread(file_path)

        if document_image is None:
            logger.error(f"Failed to load image from path: {file_path}")
            return ""

        # Preprocess the image to enhance OCR accuracy
        grayscale_image = cv2.cvtColor(document_image, cv2.COLOR_BGR2GRAY)
        # Upscale the image for better character recognition
        scaled_image = cv2.resize(
            grayscale_image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC
        )
        # Apply Otsu's thresholding to get clean binarized image
        _, thresholded_image = cv2.threshold(
            scaled_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        # Run Tesseract OCR on the binarized image with single uniform block of text layout (PSM 6)
        extracted_text = pytesseract.image_to_string(thresholded_image, config="--psm 6")

        logger.debug("--- OCR EXTRACTED TEXT ---")
        logger.debug(extracted_text)
        logger.debug("--------------------------")

        return extracted_text

    except Exception as exc:
        logger.error(f"Error occurred during OCR extraction on {file_path}: {exc}", exc_info=True)
        raise


def parse_basic_salary(extracted_text: str) -> Optional[int]:
    """Parse the basic salary figure from the OCR extracted text.

    Args:
        extracted_text: Raw text string retrieved from the document.

    Returns:
        The detected basic salary as an integer, or None if not found.
    """
    text_lines = extracted_text.split("\n")
    for line in text_lines:
        # Search for lines containing variation of the term 'basic'
        if re.search(r"bas", line, re.IGNORECASE):
            # Extract all numeric strings from the matching line
            numeric_matches = re.findall(r"\d[\d,]*", line)
            if numeric_matches:
                # Remove commas and convert matching numbers to integers
                parsed_numbers = [int(num.replace(",", "")) for num in numeric_matches]
                max_salary_candidate = max(parsed_numbers)
                # Ignore numbers too small to be a basic salary
                if max_salary_candidate > 100:
                    logger.info(f"Basic salary parsed from line '{line.strip()}': {max_salary_candidate}")
                    return max_salary_candidate
    logger.warning("Basic salary keyword or value was not detected in the extracted text.")
    return None


def parse_salary_components(extracted_text: str) -> Dict[str, int]:
    """Extract individual salary components (Basic, HRA, Tax) from the text.

    Args:
        extracted_text: Raw text string retrieved from the document.

    Returns:
        A dictionary mapping component names to their parsed integer values.
    """
    parsed_components = {}
    component_regex_patterns = {
        "Basic": [r"basic\s*(?:salary|pay|sal)?", r"base\s*(?:salary|pay)?"],
        "HRA": [r"h\.?r\.?a", r"house\s*rent\s*allowance"],
        "Tax": [r"tax(?:es)?", r"income\s*tax", r"tds"],
    }

    text_lines = extracted_text.split("\n")
    for line in text_lines:
        for component_label, regex_patterns in component_regex_patterns.items():
            for pattern in regex_patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    numeric_matches = re.findall(r"\d[\d,]*", line)
                    if numeric_matches:
                        # Clean and filter out numbers smaller than 100
                        filtered_large_numbers = [
                            int(num.replace(",", ""))
                            for num in numeric_matches
                            if int(num.replace(",", "")) > 100
                        ]
                        if filtered_large_numbers:
                            max_component_value = max(filtered_large_numbers)
                            if component_label not in parsed_components:
                                parsed_components[component_label] = max_component_value
                                logger.info(
                                    f"Parsed salary component '{component_label}': {max_component_value} "
                                    f"from line: '{line.strip()}'"
                                )
                                break  # Move to next line once a label matches to prevent duplicate parsing
    return parsed_components


def assess_salary_eligibility(file_path: str) -> Dict[str, Union[bool, int, str, Dict[str, int]]]:
    """Analyze a salary slip document to determine loan eligibility.

    Assumes the user is eligible if the basic salary is greater than or equal to Rs. 20,000.

    Args:
        file_path: Path to the salary slip file.

    Returns:
        A dictionary containing the parsed metrics and eligibility status.
    """
    extracted_text = extract_text_from_file(file_path)
    basic_salary_amount = parse_basic_salary(extracted_text)
    salary_components = parse_salary_components(extracted_text)

    if basic_salary_amount is None:
        logger.warning(f"Eligibility check incomplete: Basic salary not found in {file_path}")
        return {"found": False, "message": "Basic salary not found in document."}

    is_eligible = basic_salary_amount >= 20000
    status_symbol = "Eligible ✓" if is_eligible else "Not Eligible ✗"
    message_text = f"Basic Salary: Rs.{basic_salary_amount} — {status_symbol}"

    logger.info(f"Eligibility check complete. Eligible: {is_eligible}, Basic: {basic_salary_amount}")
    return {
        "found": True,
        "basic": basic_salary_amount,
        "eligible": is_eligible,
        "message": message_text,
        "components": salary_components,
    }