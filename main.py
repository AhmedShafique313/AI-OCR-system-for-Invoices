# import main as st
# import pandas as pd
# import json
# import re
# import easyocr
# from PIL import Image
# from io import BytesIO
# from langchain_google_genai import ChatGoogleGenerativeAI
# import cv2, os
# import numpy as np
# from dotenv import load_dotenv
# load_dotenv()

# gemini_api_key = os.environ.get('GEMINI_API_KEY')

# reader = easyocr.Reader(['en'])

# # Initialize session state for df_store if not already set
# if "processed" not in st.session_state:
#     st.session_state.processed = False
# if "df_store" not in st.session_state:
#     st.session_state.df_store = None
# if "uploaded_image" not in st.session_state:
#     st.session_state.uploaded_image = None

# def OCR(image):
#     """Extracts text from the uploaded image using EasyOCR."""
#     # Convert BytesIO to OpenCV format
#     image = Image.open(image)
#     image = np.array(image)  # Convert to NumPy array (RGB)
    
#     # Convert RGB to grayscale for better OCR performance
#     gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

#     results = reader.readtext(gray)  # Run OCR on grayscale
#     results = reader.readtext(image)
#     extracted_text = [text[1] for text in results]  # Extract only the text part
#     return extracted_text


# def llm_response(extracted_text):
#     """Sends extracted text to Gemini API for structured data extraction."""
#     api = "AIzaSyAvPY9KmVoZMmTHEe0CpBnnXhy-IBXI7js"
#     llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=gemini_api_key)
    
#     prompt = f"""
#     THE EXTRACTED TEXT IS FROM AN INVOICE.
#     Extract the following structured data:
#     - INVOICE (Invoice number)
#     - DATE CREATED
#     - VENDOR (Company name)
#     - SALE TYPE
#     - DELIVER TO
#     - DESCRIPTION (as a list)
#     - QUANTITY (Extract as an integer)
#     - UNIT PRICE (Single item price)
#     - EXTD PRICE or TOTAL (Total price)

#     Return the data in **valid JSON format**.

#     OCR Text:
#     {extracted_text}
#     """

#     response = llm.invoke(prompt)
#     return response.content  # Extracts actual text


# def into_df(response):
#     """Extracts JSON from response and appends it to a global DataFrame."""
#     global df_store

#     # 🔹 Extract JSON from Response
#     json_match = re.search(r"```json\n(.*?)\n```", response, re.DOTALL)
#     json_string = json_match.group(1) if json_match else response

#     # 🔹 Convert JSON to DataFrame
#     try:
#         extracted_json = json.loads(json_string)
#     except json.JSONDecodeError:
#         st.error("❌ Error: Invalid JSON format")
#         return

#     rows = []
#     max_length = max([len(v) if isinstance(v, list) else 1 for v in extracted_json.values()])  

#     for i in range(max_length):
#         row = {key: (value[i] if isinstance(value, list) and i < len(value) else value)
#                for key, value in extracted_json.items()}
#         rows.append(row)

#     new_df = pd.DataFrame(rows)

#     # 🔹 Filter out rows where "UNIT PRICE" is NaN or empty
#     new_df = new_df[new_df["UNIT PRICE"].notna() & (new_df["UNIT PRICE"] != "")]

#     if new_df.empty:
#         st.error("❌ No valid data to append.")
#         return

#     # 🔹 Convert "QUANTITY" column to integer
#     if "Quantity" in new_df.columns:
#         new_df["Quantity"] = pd.to_numeric(new_df["Quantity"], errors="coerce").fillna(0).astype(int)

#     # 🔹 Append new data to session state DataFrame
#     st.session_state.df_store = pd.concat([st.session_state.df_store, new_df], ignore_index=True)


# # Streamlit UI
# st.title("📄 Invoice Data Extraction")

# # File uploader
# uploaded_file = st.file_uploader("Upload an invoice image", type=["png", "jpg", "jpeg"])

# if uploaded_file:
#     # Display uploaded image
#     image = Image.open(uploaded_file)
#     st.session_state.uploaded_image = uploaded_file
#     st.image(image, caption="Uploaded Invoice", use_column_width=True)

#     # Ask if user wants to process or upload another image
#     action = st.radio("Do you want to process this image?", 
#                   [ "Yes, Process", "No, Upload Another"], 
#                   index=None) 
#     if not st.session_state.processed:
#         if action == "Yes, Process":
#             if not st.session_state.processed:
#                 with st.spinner("Extracting text..."):
#                     extracted_text = OCR(uploaded_file)
            
#             if extracted_text:
#                 st.success("✅ Text extracted successfully!")

#                 with st.spinner("Processing structured data..."):
#                     response = llm_response(extracted_text)
#                     into_df(response)

#                 st.success("✅ Data processed and added to table!")
#                 st.dataframe(st.session_state.df_store)
#                 st.session_state.processed = True
#                 if not st.session_state.df_store.empty:
#                     csv = st.session_state.df_store.to_csv(index=False).encode("utf-8")
#                     st.download_button(label="📥 Download CSV", data=csv, file_name="extracted_data.csv", mime="text/csv")

#     elif action == "No, Upload Another":
        
#         st.session_state.uploaded_image = None
#         st.session_state.processed = False
#         st.session_state.df_store = None
#         st.experimental_rerun()  # Clear the state and ask for new upload

#     else:
#         st.warning("🔄 Please Select an Option ")

# # Download CSV button


import streamlit as st  
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

# Load environment variables
load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")

# Initialize OCR reader
reader = easyocr.Reader(["en"])

# ✅ Initialize session state correctly
if "processed" not in st.session_state:
    st.session_state.processed = False
if "df_store" not in st.session_state or st.session_state.df_store is None:
    st.session_state.df_store = pd.DataFrame()
if "uploaded_image" not in st.session_state:
    st.session_state.uploaded_image = None

# 🟢 OCR Function
# 🟢 OCR Function (Fixed)
def OCR(image):
    """Extracts text from the uploaded image using EasyOCR."""
    image = Image.open(image).convert("RGB")  # Ensure RGB format
    image = np.array(image)  # Convert PIL image to NumPy array

    # ✅ Check if the image has multiple channels before converting to grayscale
    if len(image.shape) == 3 and image.shape[2] == 3:  
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)  # Convert only if RGB
    else:
        gray = image  # If already grayscale, use as is

    # ✅ Run OCR on the processed image
    results = reader.readtext(gray)
    extracted_text = [text[1] for text in results]  # Extract only text part
    return extracted_text


# 🟢 Gemini LLM Response Function
def llm_response(extracted_text):
    """Sends extracted text to Gemini API for structured data extraction."""
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=gemini_api_key)
    
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

# 🟢 Convert LLM Response to DataFrame
def into_df(response):
    """Extracts JSON from response and appends it to a global DataFrame."""
    # 🔹 Extract JSON from Response
    json_match = re.search(r"```json\n(.*?)\n```", response, re.DOTALL)
    json_string = json_match.group(1) if json_match else response

    # 🔹 Convert JSON to DataFrame
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

    # 🔹 Filter out rows where "UNIT PRICE" is NaN or empty
    new_df = new_df[new_df["UNIT PRICE"].notna() & (new_df["UNIT PRICE"] != "")]

    if new_df.empty:
        st.error("❌ No valid data to append.")
        return

    # 🔹 Convert "QUANTITY" column to integer
    if "Quantity" in new_df.columns:
        new_df["Quantity"] = pd.to_numeric(new_df["Quantity"], errors="coerce").fillna(0).astype(int)

    # 🔹 Append new data to session state DataFrame
    st.session_state.df_store = pd.concat([st.session_state.df_store, new_df], ignore_index=True)

# 🟢 Streamlit UI
st.title("📄 Invoice Data Extraction")

# File uploader
uploaded_file = st.file_uploader("Upload an invoice image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.session_state.uploaded_image = uploaded_file
    st.image(image, caption="Uploaded Invoice", use_column_width=True)

    # Ask if user wants to process or upload another image
    action = st.radio("Do you want to process this image?", 
                  ["Yes, Process", "No, Upload Another"], 
                  index=None) 
    
    if not st.session_state.processed:
        if action == "Yes, Process":
            if not st.session_state.processed:
                with st.spinner("Extracting text..."):
                    extracted_text = OCR(uploaded_file)
            
            if extracted_text:
                st.success("✅ Text extracted successfully!")

                with st.spinner("Processing structured data..."):
                    response = llm_response(extracted_text)
                    into_df(response)

                st.success("✅ Data processed and added to table!")
                st.dataframe(st.session_state.df_store)
                st.session_state.processed = True
                
                # ✅ Enable CSV download if data exists
                if not st.session_state.df_store.empty:
                    csv = st.session_state.df_store.to_csv(index=False).encode("utf-8")
                    st.download_button(label="📥 Download CSV", data=csv, file_name="extracted_data.csv", mime="text/csv")

    elif action == "No, Upload Another":
        # ✅ Properly Reset States
        st.session_state.uploaded_image = None
        st.session_state.processed = False
        st.session_state.df_store = pd.DataFrame()  # Reset DataFrame
        st.rerun()  # ✅ Use `st.rerun()` instead of `st.experimental_rerun()`

    else:
        st.warning("🔄 Please Select an Option")

# ✅ End of the script
