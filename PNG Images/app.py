from fastapi import FastAPI, UploadFile, File, HTTPException
import pandas as pd
import json
import re
import easyocr
from PIL import Image
import cv2
import os
import numpy as np
from dotenv import load_dotenv
from io import BytesIO
from langchain_google_genai import ChatGoogleGenerativeAI


# Initialize FastAPI app
app = FastAPI()

# Initialize OCR reader
reader = easyocr.Reader(["en"])

def OCR(image: bytes):
    """Extracts text from the uploaded image using EasyOCR."""
    image = Image.open(BytesIO(image)).convert("RGB")  # Ensure RGB format
    image = np.array(image)  # Convert PIL image to NumPy array

    if len(image.shape) == 3 and image.shape[2] == 3:  
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)  # Convert only if RGB
    else:
        gray = image  # If already grayscale, use as is

    results = reader.readtext(gray)
    extracted_text = [text[1] for text in results]  # Extract only text part
    return extracted_text

def llm_response(extracted_text: str):
    """Sends extracted text to Gemini API for structured data extraction."""
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key='AIzaSyCcGqDgr0okwX3zivih9YRXciiCEixQx1c')
    
    prompt = f"""
    THE EXTRACTED TEXT IS FROM AN INVOICE.
    Extract the following structured data:
    - INVOICE (Invoice number)
    - DATE CREATED
    - VENDOR (Company name)
    - SALE TYPE
    - DELIVER TO
    - DESCRIPTION (as a list)
    - QUANTITY (Extract as an integer)
    - UNIT PRICE (Single item price)
    - EXTD PRICE or TOTAL (Total price)

    Return the data in **valid JSON format**.

    OCR Text:
    {extracted_text}
    """
    response = llm.invoke(prompt)
    return response.content  # Extracts actual text

def into_df(response: str):
    """Extracts JSON from response and converts it into a DataFrame."""
    json_match = re.search(r"```json\n(.*?)\n```", response, re.DOTALL)
    json_string = json_match.group(1) if json_match else response

    try:
        extracted_json = json.loads(json_string)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON format from LLM response")

    rows = []
    max_length = max([len(v) if isinstance(v, list) else 1 for v in extracted_json.values()])  

    for i in range(max_length):
        row = {key: (value[i] if isinstance(value, list) and i < len(value) else value)
               for key, value in extracted_json.items()}
        rows.append(row)

    new_df = pd.DataFrame(rows)
    new_df = new_df[new_df["UNIT PRICE"].notna() & (new_df["UNIT PRICE"] != "")]

    if new_df.empty:
        raise HTTPException(status_code=400, detail="No valid data to process.")

    if "Quantity" in new_df.columns:
        new_df["Quantity"] = pd.to_numeric(new_df["Quantity"], errors="coerce").fillna(0).astype(int)

    return new_df

@app.post("/process-invoice/")
async def process_invoice(file: UploadFile = File(...)):
    """Processes an uploaded invoice image, extracts text, and returns structured data."""
    image_bytes = await file.read()
    extracted_text = OCR(image_bytes)
    
    if not extracted_text:
        raise HTTPException(status_code=400, detail="No text extracted from image.")
    
    llm_result = llm_response("\n".join(extracted_text))
    df = into_df(llm_result)
    
    return {
        "message": "Invoice processed successfully",
        "data": df.to_dict(orient="records")
    }

@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {"status": "API is running successfully"}
