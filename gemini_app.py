import os
import csv
import streamlit as st
import google.generativeai as genai

# Set your API key
genai.configure(api_key="AIzaSyDX8E1vneEcUU2CvdFf5Ltg-7TjunK3zcU")  # Replace with your actual API key

def upload_to_gemini(path, mime_type=None):
    """Uploads the given file to Gemini."""
    file = genai.upload_file(path, mime_type=mime_type)
    return file

# Create the model
model = genai.GenerativeModel(
    model_name="gemini-1.5-pro",
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
            {"role": "user", "parts": [file, "Extract structured invoice data including Invoice Number, Date, Item, Quantity, Price, and Total. Format the response as CSV-compatible values."]},
        ]
    )
    response = chat_session.send_message("Extract and format the invoice details clearly.")
    return response.text

def save_to_csv(extracted_text, output_csv):
    """Saves the extracted text to a CSV file."""
    lines = extracted_text.strip().split("\n")
    
    with open(output_csv, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["Invoice Number", "Date", "Item", "Quantity", "Price", "Total"])
        
        for line in lines:
            columns = line.split(",")  # Assuming extracted text is comma-separated
            if len(columns) == 6:
                writer.writerow(columns)
    
    return output_csv

# Streamlit UI
st.title("Invoice OCR Extractor")

uploaded_file = st.file_uploader("Upload an Invoice Image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    with open("temp_image.png", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    if st.button("Process Image"):
        extracted_text = extract_text_from_image("temp_image.png")
        csv_file = save_to_csv(extracted_text, "extracted_invoice_data.csv")
        st.success("Extraction completed and saved to CSV.")
        
        with open(csv_file, "rb") as f:
            st.download_button(
                label="Download CSV",
                data=f,
                file_name="extracted_invoice_data.csv",
                mime="text/csv"
            )
