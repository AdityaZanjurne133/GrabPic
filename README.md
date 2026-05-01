# 📸 GrabPic

GrabPic is a simple and efficient photo retrieval application that helps you find all images containing a specific person from a large collection of photos.

Upload a picture of yourself, and GrabPic will return all matching images from your dataset using a preprocessed vector store.

## 🚀 Features

* 🔍 **Face-based image retrieval**
* 📂 **One-time dataset processing**
* ⚡ **Fast similarity search using embeddings**
* 🖼️ **Clean Streamlit UI with gallery view**
* ⬇️ **Download matched images**

## 🏗️ Project Structure

```
GrabPic/
│
├── app.py                  # Streamlit UI
├── main.py                 # Backend logic (query + processing)
├── src/                    # Scripts aiding in processing and query
├── paths_config.yaml       # Paths configuration
├── .env                    # Environment variables (user-defined)
├── pyproject.toml          # Project dependencies
├── models/                 # Models being used in the app
└── data/                   # Stores all the images and embeddings
```


## ⚙️ Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/AdityaZanjurne133/GrabPic.git
cd GrabPic
```

### 2. Install `uv` (if not already installed)

```bash
pip install uv
```

Or using curl (recommended if using Linux-based OS):

```bash
curl -Ls https://astral.sh/uv/install.sh | sh
```

### 3. Create environment & install dependencies

```bash
uv init
uv venv
uv sync
```

### 4. Activate virtual environment

```bash
source .venv/bin/activate   # Linux / Mac
# OR
.venv\\Scripts\\activate    # Windows
```

### 5. Configure environment variables

Create a `.env` file in the root directory:

```
BASE_DIR=/absolute/path/to/your/project/root
```

> ⚠️ `BASE_DIR` must point to the root of this project.

### 6. Download model file

Download the required model file from:

```
wget -O models/yunet.onnx https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx
```

Place it inside:

```
$BASE_DIR/models/
```

<!-- Update `paths_config.yaml` if needed. -->

### 7. Run the application

```bash
streamlit run app.py
```

## 🧠 How It Works

### Step 1: Process Image Dataset

* Provide a folder path containing images
* Faces are detected and embeddings are generated
* Data is stored in a vector database

### Step 2: Query

* Upload a selfie
* The app extracts the face embedding
* Performs similarity search against stored embeddings
* Returns matching images

<!-- ## 📌 Notes

* Dataset processing is a **one-time operation**, you can query any number of times once it is done.
* Ensure good quality images for better accuracy
* Large datasets may take time to process -->

## 🛠️ Tech Stack

* Python
* Streamlit
* PIL and OpenCV (Image Processing)
* PyTorch (Embeddings)
* FAISS (Vector store)
* Scikit-learn (Clustering using DBSCAN)


## 🧩 Future Improvements
* Multiple decks
* Authentication
* Cloud deployment

---

<!-- ## 📜 License

This project is for educational and research purposes. -->
