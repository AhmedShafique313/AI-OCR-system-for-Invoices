import json, os, re
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
load_dotenv()

gemini_api_key = os.environ.get('GEMINI_API_KEY')

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=gemini_api_key
)

def process_with_gemini(extracted_text):
    prompt = f""" THE EXTRACTED TEXT IS THE TEXT OF AN INVOCE
    Extract the following structured data from the OCR text,The Terms can be capital or not you have to figure it out yourself:
    - INVOICE NUMBER (Its shows invoice number)
    - DATE CREATED (On which the invoice created)
    - VENDOR (It should be complete company name)
    - SALE TYPE (It should be the part number only)
    - DESCRIPTION (as a list, should be small and concise)
    - Quantity (It is the quantity of the product do not confuse it with any price value, it can  start fron QTY or Qty and may or may not have SHP that is quantity of the product its an integer)
    - UNIT PRICE (It is the price of one unit, It can be replaced by its synonym and its not case sensitive)
    - EXTD PRICE or Total (It can be replaced by its synonym and its not case sensitive)
    Return the data in such format that it can be store in the csv file.
    OCR Text:
    {extracted_text}
    """

    response = llm.invoke(prompt)
    return extract_json(response)

def extract_json(response):
    """Extracts JSON from Gemini API response."""
    response_text = response.content if hasattr(response, "content") else str(response)
    json_match = re.search(r"```json\n(.*?)\n```", response_text, re.DOTALL)

    if json_match:
        json_string = json_match.group(1)
    else:
        json_string = response_text

    try:
        return json.loads(json_string)
    except json.JSONDecodeError:
        return None
