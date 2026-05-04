from main import setup_deck, query_image

import streamlit as st
import os
import sys
import yaml
from PIL import Image, ImageOps
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

def get_available_decks():
    base_path = os.path.join(BASE_DIR, paths_config["user_photo_decks_root_path"])
    
    if not os.path.exists(base_path):
        return []
    
    return [
        d for d in os.listdir(base_path)
        if os.path.isdir(os.path.join(base_path, d))
    ]

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
# if os.path.exists(os.path.join(BASE_DIR, paths_config["vector_store_path"])) \
#     and len(os.listdir(os.path.join(BASE_DIR, paths_config["vector_store_path"]))) > 0:
#     st.session_state.processed = True
# else:
#     st.session_state.processed = False

if "processed" not in st.session_state:
    photo_decks = get_available_decks()
    st.session_state.processed = len(photo_decks) > 0

if "selected_deck" not in st.session_state:
    st.session_state.selected_deck = None

# Sidebar Navigation
page = st.sidebar.radio("Navigation", ["Setup", "Find My Photos"])

# -------------------------
# PAGE 1: SETUP
# -------------------------
if page == "Setup":
    st.header("Process Your Photo Collection")
    st.markdown("Provide a folder path containing your images. This is a one-time process (Skip if already done).")

    folder_path = st.text_input("Enter folder path:")
    photo_deck_name = st.text_input("Enter a name for the collection:")

    if st.button("Process Images"):
        photo_decks = get_available_decks()
        if not folder_path or not os.path.exists(folder_path):
            st.error("Invalid folder path. Please try again.")
        elif not photo_deck_name:
            st.error("Please provide a deck name.")
        elif photo_deck_name in photo_decks:
            st.error("A deck with that name already exists. Try a different name.")
        else:
            with st.spinner("Processing images... This may take a while ⏳"):
                setup_deck(folder_path, photo_deck_name)
                # pass
            st.session_state.processed = True
            st.success("✅ Images processed successfully!")

# -------------------------
# PAGE 2: FIND IMAGES
# -------------------------
elif page == "Find My Photos":
    st.header("Upload Your Photo")
    photo_decks = get_available_decks()

    if "processed" not in st.session_state or not st.session_state.processed:
        st.warning("⚠️ No photo collections found. Please create one first.")
    else:
        # Deck selection
        selected_deck = st.selectbox("Select a Photo Collection", photo_decks)
        st.session_state.selected_deck = selected_deck

        uploaded_image = st.file_uploader("Upload your image", type=["jpg", "jpeg", "png"])
        
        st.session_state.uploaded_image = uploaded_image

        if uploaded_image is not None:
            # use_container_width=True => width='stretch' ; use_container_width=False => width='content'
            st.image(uploaded_image, caption="Uploaded Image", width='stretch')
            results = None

            if st.button("Find My Photos"):
                with st.spinner("Searching for your photos... 🔍"):
                    results = query_image(uploaded_image, st.session_state.selected_deck)
                    st.session_state.results = results

                if not st.session_state.results:
                    st.info("No matching images found.")
                else:
                    results = st.session_state.results
                    display_retrieved_images(results=results)
            
            if not results and 'results' in st.session_state and st.session_state.results:
                results = st.session_state.results
                display_retrieved_images(results=results)
        else:
            st.session_state.results = None
                

# -------------------------
# FOOTER
# -------------------------
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit")
