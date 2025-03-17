import easyocr

reader = easyocr.Reader(['en'])

def extract_text(file_path):
    results = reader.readtext(file_path)
    return [text[1] for text in results]

# extract_text(file_path= r'C:\Users\Infinity AI Systems\Documents\Projects\AI OCR System for Invoices\uploads\invoice_2.png')