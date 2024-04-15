import streamlit as st
import cv2
from PIL import Image
import numpy as np

def load_image(image_file):
    img = Image.open(image_file)
    return img

def process_image(img):
    # Convert the image to an array
    img_array = np.array(img)

    # Convert the image to grayscale
    gray_image = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    
    # Convert back to PIL image
    return Image.fromarray(gray_image)

def main():
    st.title("Image Processing App")

    # File uploader allows user to add their own image
    uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        image = load_image(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Process the image
        processed_image = process_image(image)
        
        # Display the processed image
        st.image(processed_image, caption='Processed Image in Grayscale', use_column_width=True)

if __name__ == "__main__":
    main()
