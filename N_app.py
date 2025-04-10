import streamlit as st
import pdfplumber
from transformers import pipeline
import json
import numpy as np

# Print numpy version for debugging
print("Numpy version:", np.__version__)

# Function to extract text from PDF using pdfplumber
def extract_text_from_pdf(file):
    """Extract text from the uploaded PDF using pdfplumber."""
    try:
        with pdfplumber.open(file) as pdf:
            text = ""
            for page in pdf.pages:
                text += page.extract_text()
        return text
    except Exception as e:
        st.error(f"Error extracting text from PDF: {e}")
        return None

# Load the Named Entity Recognition (NER) model using Hugging Face pipeline
def load_ner_model():
    """Load the pre-trained NER model using Hugging Face transformers."""
    try:
        ner_pipeline = pipeline("ner", model="bert-base-cased")
        st.success("NER model loaded successfully!")
        return ner_pipeline
    except Exception as e:
        st.error(f"Error loading NER model: {e}")
        return None

# Function to perform Named Entity Recognition (NER) and return JSON format
def perform_ner(text, ner_pipeline):
    """Run NER on the provided text and return the recognized entities in JSON format."""
    try:
        entities = ner_pipeline(text)
        return json.dumps(entities, indent=4)
    except Exception as e:
        st.error(f"Error performing NER: {e}")
        return None

# Streamlit UI
def main():
    st.title("Named Entity Recognition (NER) Extraction")

    # Sidebar for file upload
    st.sidebar.header("Upload a PDF Document")
    uploaded_file = st.sidebar.file_uploader("Drag and drop a file here", type="pdf")

    # If a file is uploaded
    if uploaded_file:
        st.write("Processing file:", uploaded_file.name)

        # Extract text from the uploaded PDF
        text = extract_text_from_pdf(uploaded_file)
        if text:
            # Display the first 1000 characters of the extracted text
            st.subheader("Extracted Text (first 1000 characters):")
            st.write(text[:1000])

            # Load NER model
            ner_pipeline = load_ner_model()

            if ner_pipeline:
                # Perform NER on the extracted text
                st.subheader("Extracted Entities (in JSON format):")
                result = perform_ner(text, ner_pipeline)

                if result:
                    st.json(result)  # Display the NER results in JSON format
                else:
                    st.error("Failed to extract entities.")
        else:
            st.error("Failed to extract text from the PDF.")

if __name__ == "__main__":
    main()
