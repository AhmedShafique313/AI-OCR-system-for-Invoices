import streamlit as st  
import pandas as pd
import json
import re
import os
import numpy as np
from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai
from PIL import Image

# Configure Google Gemini API

genai.configure(api_key="AIzaSyDX8E1vneEcUU2CvdFf5Ltg-7TjunK3zcU")  # API key for google.generativeai

def upload_to_gemini(path, mime_type=None):
    """Uploads the given file to Gemini."""
    file = genai.upload_file(path, mime_type=mime_type)
    return file

# Create the model
model = genai.GenerativeModel(
    model_name="gemini-2.0-flash",
    generation_config={
        "temperature": 1,
        "top_p": 0.95,
        "top_k": 40,
        "max_output_tokens": 8192,
        "response_mime_type": "text/plain",
    },
)

def extract_text_from_image(image_path):
    file = upload_to_gemini(image_path, mime_type="image/png")
    chat_session = model.start_chat(
        history=[
            {"role": "user", "parts": [file, "Extract structured invoice data including Invoice Number, Date, Vendor, sale type, Item, Quantity, Price, and Total. Format the response as CSV-compatible values."]},
        ]
    )
    response = chat_session.send_message("Extract and format the invoice details clearly and give a plain text.")
    extracted_text = response.text
    return extracted_text

def llm_response(extracted_text):
    """Sends extracted text to Gemini API for structured data extraction."""
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key='AIzaSyCcGqDgr0okwX3zivih9YRXciiCEixQx1c') # API key for langchain_google_genai
    
    prompt = f"""
    Extract the following structured data from the OCR text,The Terms can be capital or not you have to figure it out yourself:
    - INVOICE (Its shows invoice number it is also not case sensitive)
    - DATE CREATED
    - VENDOR (company name)
    - SALE TYPE
    - DELIVER TO
    - DESCRIPTION (as a list, should be small and concise)
    - Quantity (It is the quantity of the product do not confuse it with any price value, it can  start fron QTY or Qty and may or may not have SHP that is quantity of the product its an integer)
    - UNIT PRICE (It is the price of one unit, It can be replaced by its synonym and its not case sensitive)
    - EXTD PRICE or Total (It can be replaced by its synonym and its not case sensitive)
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

st.title("📄 Invoice Data Extraction")

uploaded_file = st.file_uploader("Upload an invoice image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.session_state.uploaded_image = uploaded_file
    st.image(image, caption="Uploaded Invoice", use_column_width=True)

    action = st.radio("Do you want to process this image?", 
                      ["Yes, Process", "No, Upload Another"], 
                      index=None) 
    
    if "processed" not in st.session_state:
        st.session_state.processed = False

    if "df_store" not in st.session_state:
        st.session_state.df_store = pd.DataFrame()
    
    if action == "Yes, Process" and not st.session_state.processed:
        with st.spinner("Extracting text..."):
            extracted_text = extract_text_from_image(uploaded_file)
        
        if extracted_text:
            st.success("✅ Text extracted successfully!")

            with st.spinner("Processing structured data..."):
                response = llm_response(extracted_text)
                into_df(response)

            st.success("✅ Data processed and added to table!")
            st.dataframe(st.session_state.df_store)
            st.session_state.processed = True
            
            if not st.session_state.df_store.empty:
                csv = st.session_state.df_store.to_csv(index=False).encode("utf-8")
                st.download_button(label="📥 Download CSV", data=csv, file_name="extracted_data.csv", mime="text/csv")

    elif action == "No, Upload Another":
        st.session_state.uploaded_image = None
        st.session_state.processed = False
        st.session_state.df_store = pd.DataFrame()
        st.rerun()

    else:
        st.warning("🔄 Please Select an Option")