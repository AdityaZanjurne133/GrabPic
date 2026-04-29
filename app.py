from main import setup_deck, query_image

import streamlit as st
import os
import sys
import yaml
import base64
from PIL import Image, ImageOps
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.getenv("BASE_DIR")
sys.path.append(BASE_DIR)

with open("paths_config.yaml", "r") as f:
    paths_config = yaml.safe_load(f)

st.set_page_config(page_title="GrabPic", layout="wide")

st.title("📸 GrabPic")
st.markdown("Find all your photos effortlessly.")

def load_image(path, size=(300, 300)):
    img = Image.open(path)
    img = ImageOps.exif_transpose(img)
    img.thumbnail(size)
    return img

def get_image_bytes(path):
    with open(path, "rb") as f:
        return f.read()

def display_retrieved_images(results):
    st.success(f"Found {len(results)} images!")

    # -------- Gallery UI --------
    st.markdown("### 🖼️ Your Photos")

    cols = st.columns(4)
    for idx, img_path in enumerate(results):
        img = load_image(img_path)
        with cols[idx % 4]:
            st.image(img, use_container_width=True)
            with open(img_path, "rb") as f:
                st.download_button(
                    label="Download",
                    data=f,
                    file_name=os.path.basename(img_path),
                    mime="image/jpeg"
                )

# Session state
if os.path.exists(os.path.join(BASE_DIR, paths_config["vector_store_path"])) \
    and len(os.listdir(os.path.join(BASE_DIR, paths_config["vector_store_path"]))) > 0:
    st.session_state.processed = True
else:
    st.session_state.processed = False

# Sidebar Navigation
page = st.sidebar.radio("Navigation", ["1. Setup", "2. Find My Photos"])

# -------------------------
# PAGE 1: SETUP
# -------------------------
if page == "1. Setup":
    st.header("Step 1: Process Your Photo Collection")
    st.markdown("Provide a folder path containing your images. This is a one-time process (Skip if already done).")

    folder_path = st.text_input("Enter folder path:")

    if st.button("Process Images"):
        if not folder_path or not os.path.exists(folder_path):
            st.error("Invalid folder path. Please try again.")
        else:
            with st.spinner("Processing images... This may take a while ⏳"):
                setup_deck(folder_path)
                # pass
            st.session_state.processed = True
            st.success("✅ Images processed successfully!")

# -------------------------
# PAGE 2: FIND IMAGES
# -------------------------
elif page == "2. Find My Photos":
    st.header("Step 2: Upload Your Photo")

    if not st.session_state.processed:
        st.warning("⚠️ Please complete Step 1 before searching.")
    else:
        uploaded_image = st.file_uploader("Upload your image", type=["jpg", "jpeg", "png"])
        
        st.session_state.uploaded_image = uploaded_image

        if uploaded_image is not None:
            # use_container_width=True => width='stretch' ; use_container_width=False => width='content'
            st.image(uploaded_image, caption="Uploaded Image", width='stretch')
            results = None

            if st.button("Find My Photos"):
                with st.spinner("Searching for your photos... 🔍"):
                    results = query_image(uploaded_image)
                    st.session_state.results = results

                if not st.session_state.results:
                    st.info("No matching images found.")
                else:
                    results = st.session_state.results
                    display_retrieved_images(results=results)
            
            if not results and 'results' in st.session_state.keys() and st.session_state.results:
                results = st.session_state.results
                display_retrieved_images(results=results)
        else:
            st.session_state.results = None
                

# -------------------------
# FOOTER
# -------------------------
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit")
