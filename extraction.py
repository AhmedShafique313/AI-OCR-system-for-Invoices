import os
import cv2
import pytesseract
import google.generativeai as genai
from dotenv import load_dotenv
load_dotenv()

gemini_api_key = os.environ.get('GEMINI_API_KEY')

image_path = r"C:\Users\Infinity AI Systems\Documents\Projects\AI OCR System for Invoices\PNG Images\invoice_1_crop.png"
image = cv2.imread(image_path)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

text = pytesseract.image_to_string(gray)
print("Extracted Text:\n", text)


genai.configure(api_key=gemini_api_key)

def extract_invoice_details(text):
    prompt = """
    Extract the following details from the given invoice text:
    - Date: It should be the invoice date
    - Invoice Number: It maybe written as INVOICE or Invoice Number
    - Vendor
    - Type: It will find from the item description whether it should be the oil or gas also, it will find from the buyer or bill to
    - Bill to
    - QTY SHP
    - Item
    - Part Number
    - Quantity: The quantity you can find easily by taking comparision of the unit price of the item to the extended price of that item
    - Description
    - Price
    - Total Price

    Here is the invoice text:
    """ + text + """

    Return the required details in JSON format as provided above.
    """

    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content(prompt)
    return response.text

# Extract and display details
details = extract_invoice_details(text)
print(details)
