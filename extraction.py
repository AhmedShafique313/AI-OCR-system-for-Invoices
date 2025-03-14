import cv2
import pytesseract
from google.colab.patches import cv2_imshow

# Install dependencies (Run these only once)
!apt-get install -y tesseract-ocr
!pip install pytesseract

# Load the image
image_path = "/content/Invoices (1)_page-0005.jpg"  # Change this to your uploaded image path
image = cv2.imread(image_path)

# Convert to grayscale (optional but improves OCR accuracy)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Display the image
cv2_imshow(gray)

# Perform OCR
text = pytesseract.image_to_string(gray)
print("Extracted Text:\n", text)
