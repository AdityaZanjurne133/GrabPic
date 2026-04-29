import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from src.utils import read_pickle_file, write_pickle_file
from src.create_face_embeddings import create_face_embedding as embedding_model

import os
from dotenv import load_dotenv
import sys

load_dotenv()

BASE_DIR = os.getenv("BASE_DIR")
sys.path.append(BASE_DIR)

def get_query_label(face_path, vector_store_path):
    vector_store = FAISS.load_local(
        vector_store_path, embedding_model, allow_dangerous_deserialization=True
    )
    labels = []

    '''Query the vector store'''
    results = vector_store.similarity_search_with_score(
        os.path.join(BASE_DIR, face_path),
        k=5,
        # filter={"source": "tweet"},
    )
    for res, score in results:
        # print(f"* [SIM={score:3f}] {res.page_content} [{res.metadata}]")
        labels.append([score, res.metadata["label"]])
    
    # Get the label with highest confidence
    labels = sorted(labels, reverse=True)

    # print(labels)

    return labels[0][1]

def get_query_images(query_face_path, face_embeddings_data_with_labels_output_file_path, all_images_path, vector_store_path, output_path):
    print("Fetching the label for query image...")
    face_label = get_query_label(query_face_path, vector_store_path)
    print(f"Label: {face_label}")

    output_path = os.path.join(output_path, f"{face_label}.pkl")
    if os.path.exists(output_path):
        print("Image paths were already found in some previous trial.")
        final_face_image_paths = read_pickle_file(output_path)
        return final_face_image_paths
    
    face_emb_data = read_pickle_file(face_embeddings_data_with_labels_output_file_path)
    
    face_in_images = [d["image"] for d in face_emb_data if d["label"] == face_label]

    images_list = os.listdir(all_images_path)

    final_face_image_paths = [os.path.join(all_images_path, f) for f in images_list if f.split(".")[0] in face_in_images]

    write_pickle_file(data=final_face_image_paths, output_path=output_path)
    print(f"Image paths found and written to {output_path}")
    
    return final_face_image_paths


if __name__ == "__main__":
    '''Convert test image to .jpg'''
    # query_images_path = os.path.join(BASE_DIR, "data/query_images/uploaded")
    # jpg_images_path = os.path.join(BASE_DIR, "data/query_images/jpg_format")

    # convert_to_jpg(source_dir=query_images_path, output_dir=jpg_images_path)
    
    face_path = "data/query_faces/20260417_214104/face_275_1012_432_535.jpg"

    face_label = get_query_label(face_path)
    print("Face Label:", face_label)

    face_embeddings_data_with_labels_output_file_path = os.path.join(BASE_DIR, "data/face_embeddings/face_embeddings_data_with_labels.pkl")
    all_images_path = os.path.join(BASE_DIR, "data/all_photos")
    query_image_faces_output_folder = os.path.join(BASE_DIR, "data/query_faces_in_images")    # <Face label>.pkl will be appended later
    vector_store_path = os.path.join(BASE_DIR, "data/face_embeddings_vector_store")

    image_files = get_query_images(face_path, face_embeddings_data_with_labels_output_file_path, all_images_path, vector_store_path, query_image_faces_output_folder)
    print(image_files)
