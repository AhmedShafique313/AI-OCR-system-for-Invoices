import os
import cv2
import pytesseract
import pandas as pd
import re
from collections import defaultdict

# Configure Tesseract path if needed
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Function to extract text from an image
def extract_text_from_image(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    text = pytesseract.image_to_string(gray)
    return text

# Function to dynamically extract fields from text
def extract_invoice_details(text):
    data = defaultdict(str)  # Store extracted fields dynamically

    patterns = {
        "Date": r"(?:Date|Invoice Date|Dated)[:\s]*([\d{2}/\d{2}/\d{4}]|\d{4}-\d{2}-\d{2})",
        "Invoice Number": r"(?:Invoice No|Invoice Number|Inv#)[:\s]*([\w-]+)",
        "Vendor": r"(?:Vendor|Supplier|Company)[:\s]*(.+)",
        "Type": r"(?:Type)[:\s]*(.+)",
        "Part Number": r"(?:Part Number|Item No|SKU)[:\s]*([\w-]+)",
        "Quantity": r"(?:Qty|Quantity)[:\s]*(\d+)",
        "Description": r"(?:Description|Item Details)[:\s]*(.+)",
        "Price": r"(?:Unit Price|Price)[:\s]*([\d.,]+)",
        "Total Price": r"(?:Total Amount|Total Price|Grand Total)[:\s]*([\d.,]+)"
    }

    for field, pattern in patterns.items():
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            data[field] = match.group(1).strip()

    return data

# Process multiple invoices
def process_invoices(invoice_images):
    extracted_data = []

    for img_path in invoice_images:
        text = extract_text_from_image(img_path)
        details = extract_invoice_details(text)
        extracted_data.append(details)

    return extracted_data

# List of invoice image files
invoice_files = [
    "/content/invoice_1.jpg",
    "/content/Invoices (1)_page-0006.jpg",
    "/content/Invoices (1)_page-0004.jpg",
    "/content/Invoices (1)_page-0003.jpg",
    "/content/Invoices (1)_page-0002.jpg",
    "/content/Invoices (1)_page-0001.jpg"
]

# Extract invoice details dynamically
invoice_data = process_invoices(invoice_files)

# Convert data to CSV
df = pd.DataFrame(invoice_data)
df.to_csv("extracted_invoices.csv", index=False)

print("Invoice data extracted and saved to CSV!")
