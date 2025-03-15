import cv2
import pytesseract
import google.generativeai as genai
from dotenv import load_dotenv

image_path = r"C:\Users\Infinity AI Systems\Documents\Projects\AI OCR System for Invoices\PNG Images\invoice_1_crop.png"
image = cv2.imread(image_path)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

text = pytesseract.image_to_string(gray)
print("Extracted Text:\n", text)


genai.configure(api_key=)

def extract_invoice_details(text):
    prompt = """
    Extract the following details from the given invoice text:
    - Date
    - Invoice Number
    - Vendor
    - Type
    - Bill to
    - QTY SHP
    - Item
    - Part Number
    - Quantity
    - Description
    - Price
    - Total Price

    Here is the invoice text:
    """ + text + """

    Return the extracted details in JSON format.
    """

    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content(prompt)
    return response.text

# Extract and display details
details = extract_invoice_details(invoice_text)
print(details)
