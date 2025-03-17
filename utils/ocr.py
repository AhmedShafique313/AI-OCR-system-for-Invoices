import easyocr

reader = easyocr.Reader(['en'])

def extract_text(file_path):
    """Extracts text from an image using EasyOCR."""
    results = reader.readtext(file_path)
    return [text[1] for text in results]
