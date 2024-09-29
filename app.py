import streamlit as st
from transformers import AutoModel, AutoTokenizer
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
        model = AutoModel.from_pretrained(
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

    # Perform OCR using the model's custom method
    with st.spinner('Running OCR...'):
        try:
            # Assuming the model's `.chat()` method processes the image bytes and performs OCR
            result = model.chat(tokenizer, image_file, ocr_type='format', render=True, save_render_file='./demo.html')
            
            st.success("OCR Result:")
            st.write(result)

            # Option to download the result as an HTML file
            with open('./demo.html', 'rb') as f:
                st.download_button('Download Result', f, file_name="ocr_result.html", mime="text/html")

        except Exception as e:
            st.error(f"Error: {e}")
