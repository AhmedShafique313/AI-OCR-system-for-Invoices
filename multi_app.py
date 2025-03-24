import streamlit as st  
import pandas as pd
import json
import re
import easyocr
from PIL import Image
import cv2
import numpy as np
from langchain_google_genai import ChatGoogleGenerativeAI

# Initialize OCR reader
reader = easyocr.Reader(["en"])

# Session state initialization
if "processed" not in st.session_state:
    st.session_state.processed = False
if "df_store" not in st.session_state or st.session_state.df_store is None:
    st.session_state.df_store = pd.DataFrame()


def OCR(image):
    """Extracts text from the uploaded image using EasyOCR."""
    image = Image.open(image).convert("RGB")
    image = np.array(image)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
    
    results = reader.readtext(gray)
    extracted_text = [text[1] for text in results]
    return extracted_text


def llm_response(extracted_text):
    """Sends extracted text to Gemini API for structured data extraction."""
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key='YOUR_GOOGLE_API_KEY')
    
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
    return response.content 


def into_df(response):
    """Extracts JSON from response and appends it to a global DataFrame."""
    json_match = re.search(r"```json\n(.*?)\n```", response, re.DOTALL)
    json_string = json_match.group(1) if json_match else response

    try:
        extracted_json = json.loads(json_string)
    except json.JSONDecodeError:
        st.error("❌ Error: Invalid JSON format")
        return

    rows = []
    max_length = max([len(v) if isinstance(v, list) else 1 for v in extracted_json.values()])

    for i in range(max_length):
        row = {key: (value[i] if isinstance(value, list) and i < len(value) else value)
               for key, value in extracted_json.items()}
        rows.append(row)

    new_df = pd.DataFrame(rows)
    new_df = new_df[new_df["UNIT PRICE"].notna() & (new_df["UNIT PRICE"] != "")]

    if new_df.empty:
        st.error("❌ No valid data to append.")
        return

    if "Quantity" in new_df.columns:
        new_df["Quantity"] = pd.to_numeric(new_df["Quantity"], errors="coerce").fillna(0).astype(int)

    st.session_state.df_store = pd.concat([st.session_state.df_store, new_df], ignore_index=True)


st.title("📄 Multi-Invoice Data Extraction")

uploaded_files = st.file_uploader("Upload multiple invoice images", type=["png", "jpg", "jpeg"], accept_multiple_files=True)

if uploaded_files:
    for uploaded_file in uploaded_files:
        image = Image.open(uploaded_file)
        st.image(image, caption=f"Uploaded Invoice: {uploaded_file.name}", use_column_width=True)
        
        with st.spinner(f"Processing {uploaded_file.name}..."):
            extracted_text = OCR(uploaded_file)
        
        if extracted_text:
            st.success(f"✅ Text extracted from {uploaded_file.name}!")
            with st.spinner("Processing structured data..."):
                response = llm_response(extracted_text)
                into_df(response)
            st.success(f"✅ Data from {uploaded_file.name} added to table!")
    
    if not st.session_state.df_store.empty:
        st.dataframe(st.session_state.df_store)
        csv = st.session_state.df_store.to_csv(index=False).encode("utf-8")
        st.download_button(label="📥 Download CSV", data=csv, file_name="extracted_data.csv", mime="text/csv")
