import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from uuid import uuid4
from facenet_pytorch.models.inception_resnet_v1 import InceptionResnetV1
import torch
import cv2
import os
import sys
from dotenv import load_dotenv
import pickle
from tqdm import tqdm

from utils import get_face_tensor, image_clahe
from create_face_embeddings import create_face_embedding as embedding_model

load_dotenv()

BASE_DIR = os.getenv("BASE_DIR")
sys.path.append(BASE_DIR)

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# embedding_model = InceptionResnetV1(pretrained='vggface2').eval().to(device)

sample_image_path = os.path.join(BASE_DIR, "data/extracted_faces/20260124_140806/face_1932_1072_282_304.jpg")
# sample_image = cv2.imread(sample_image_path)

index = faiss.IndexFlatL2(len(embedding_model(sample_image_path)))

vector_store = FAISS(
    embedding_function=embedding_model,
    index=index,
    docstore=InMemoryDocstore(),
    index_to_docstore_id={},
)

def create_vector_store():
    pass

if __name__ == "__main__":
    # with open("data/face_embeddings_eps_0.7_min_samples_5/extracted_emb_data.pkl", "rb") as f:
    with open("data/face_embeddings_eps_0.7_min_samples_5/face_embedding_with_label.pkl", "rb") as f:
        face_label_data = pickle.load(f)
        f.close()
    # print(face_label_data[0])

    documents = []
    for d in face_label_data:
        doc = Document(page_content = os.path.join(BASE_DIR, "data/extracted_faces", d["image"], d["face"]), metadata={"label": d["label"]})
        documents.append(doc)
    
    uuids = [str(uuid4()) for _ in range(len(documents))]

    vector_store.add_documents(documents=documents, ids=uuids)
    # vector_store.delete(ids=[uuids[-1]])

    vector_store.save_local(os.path.join(BASE_DIR, "data/face_embeddings_eps_0.7_min_samples_5/faiss_index"))
