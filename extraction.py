import cv2
import pytesseract

image_path = r"C:\Users\Infinity AI Systems\Documents\Projects\AI OCR System for Invoices\image\invoice_1.jpg"  # Change this to your uploaded image path
image = cv2.imread(image_path)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

text = pytesseract.image_to_string(gray)
print("Extracted Text:\n", text)