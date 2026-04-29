import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from uuid import uuid4
import os
import sys
from dotenv import load_dotenv
from tqdm import tqdm

from src.utils import read_pickle_file
from src.create_face_embeddings import create_face_embedding as embedding_model

load_dotenv()

BASE_DIR = os.getenv("BASE_DIR")
sys.path.append(BASE_DIR)

sample_image_path = os.path.join(BASE_DIR, "sample.jpeg")

index = faiss.IndexFlatL2(len(embedding_model(sample_image_path)))

vector_store = FAISS(
    embedding_function=embedding_model,
    index=index,
    docstore=InMemoryDocstore(),
    index_to_docstore_id={},
)

def create_vector_store(face_embeddings_data_with_labels_output_file_path, faces_save_path, vector_store_path):
    face_label_data = read_pickle_file(face_embeddings_data_with_labels_output_file_path)

    documents = []
    for d in tqdm(face_label_data, desc="Creating documents:"):
        doc = Document(page_content = os.path.join(faces_save_path, d["image"], d["face"]), metadata={"label": d["label"]})
        documents.append(doc)
    
    uuids = [str(uuid4()) for _ in range(len(documents))]

    print("Adding documents to vector store...")
    vector_store.add_documents(documents=documents, ids=uuids)
    # vector_store.delete(ids=[uuids[-1]])

    vector_store.save_local(vector_store_path)
    print(f"\nSuccessfully saved vector store in {vector_store_path}!")

if __name__ == "__main__":
    face_embeddings_data_with_labels_output_file_path = os.path.join(BASE_DIR, "data/face_embeddings/face_embeddings_data_with_labels.pkl")
    faces_save_path = os.path.join(BASE_DIR, "data/extracted_faces")
    vector_store_path = os.path.join(BASE_DIR, "data/face_embeddings_vector_store")
    create_vector_store(face_embeddings_data_with_labels_output_file_path, faces_save_path, vector_store_path)
