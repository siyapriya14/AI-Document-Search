import streamlit as st
import os

st.title("AI Document Search Project")
st.write("This is my Infosys Springboard Individual Project")

# Upload file
uploaded_file = st.file_uploader("Upload a PDF or TXT file", type=["txt"])

if uploaded_file:
    os.makedirs("data", exist_ok=True)
    file_path = os.path.join("data", uploaded_file.name)

    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success(f"File saved successfully: {uploaded_file.name}")

    # Read text
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    st.subheader("Extracted Text (Preview)")
    st.text(text[:1000])   # first 1000 characters

    # Question input
    question = st.text_input("Ask a question from the document")

    if st.button("Submit"):
        if question:
            if question.lower() in text.lower():
                st.success("Answer found in document ✅")
                st.write("Relevant information exists in the document.")
            else:
                st.warning("Answer not found ❌")
        else:
            st.warning("Please enter a question")