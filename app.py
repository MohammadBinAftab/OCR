import streamlit as st
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch
from PIL import Image
import io

# Streamlit app title
st.title("OCR Model Deployment")

# File uploader in Streamlit
uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Read the uploaded image file directly
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Convert the image to bytes
    image_bytes = io.BytesIO()
    image.save(image_bytes, format=image.format)
    image_file = image_bytes.getvalue()

    # Load the model and tokenizer
    with st.spinner('Loading model...'):
        tokenizer = AutoTokenizer.from_pretrained('ucaslcl/GOT-OCR2_0', trust_remote_code=True)
        model = AutoModelForSeq2SeqLM.from_pretrained(
            'ucaslcl/GOT-OCR2_0', 
            trust_remote_code=True, 
            low_cpu_mem_usage=True, 
            device_map='auto', 
            use_safetensors=True
        )
        model = model.eval()

    # Ensure model and operations are forced to the CPU
    device = torch.device('cpu')
    model = model.to(device)
    st.warning("Running on CPU.")

    # Perform OCR
    with st.spinner('Running OCR...'):
        try:
            # Assuming the model expects tokens and the image is processed accordingly
            inputs = tokenizer(image_file, return_tensors="pt").to(device)
            outputs = model.generate(**inputs)
            result = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            st.success("OCR Result:")
            st.write(result)

            # Option to download the result as a text file
            st.download_button('Download Result', result, file_name="ocr_result.txt", mime="text/plain")

        except Exception as e:
            st.error(f"Error: {e}")
