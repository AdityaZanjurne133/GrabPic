import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from utils import read_pickle_file, write_pickle_file
from create_face_embeddings import create_face_embedding as embedding_model

import os
from dotenv import load_dotenv
import sys

load_dotenv()

BASE_DIR = os.getenv("BASE_DIR")
sys.path.append(BASE_DIR)

vector_store = FAISS.load_local(
    os.path.join(BASE_DIR, "data/face_embeddings_eps_0.7_min_samples_5/faiss_index"), embedding_model, allow_dangerous_deserialization=True
)

def get_query_label(face_path):
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

def get_query_images(query_face_path, embedding_and_labels_data_file_path, all_images_path, output_path):
    face_label = get_query_label(query_face_path)

    output_path = os.path.join(output_path, f"{face_label}.pkl")
    if os.path.exists(output_path):
        print("Image paths were already found in some previous trial.")
        final_face_image_paths = read_pickle_file(output_path)
        return final_face_image_paths
    
    face_emb_data = read_pickle_file(embedding_and_labels_data_file_path)
    
    face_in_images = [d["image"] for d in face_emb_data if d["label"] == face_label]

    images_list = os.listdir(all_images_path)

    final_face_image_paths = [os.path.join(all_images_path, f) for f in images_list if f.split(".")[0] in face_in_images]

    write_pickle_file(data=final_face_image_paths, output_path=output_path)
    print(f"Image paths found and written to {output_path}")
    
    return final_face_image_paths


if __name__ == "__main__":
    '''Convert test image to .jpg'''
    # query_images_path = os.path.join(BASE_DIR, "data/test/query_images")
    # jpg_images_path = os.path.join(BASE_DIR, "data/test/input_images")

    # convert_to_jpg(source_dir=query_images_path, output_dir=jpg_images_path)
    
    # face_path = "data/test/extracted_faces/20260417_214104/face_275_1013_432_535.jpg"
    # face_path = "data/test/extracted_faces/20260417_080614/face_190_1008_532_579.jpg"
    # face_path = "data/test/extracted_faces/20251016_161106_309_1x-1/face_209_351_80_96.jpg"
    # face_path = "data/test/extracted_faces/20251016_161106_309_1x-1/face_418_367_81_101.jpg"
    face_path = "data/test/extracted_faces/20251016_161106_309_1x-1/face_573_444_81_101.jpg"
    # face_path = "data/test/extracted_faces/20251016_161106_309_1x-1/face_787_1875_73_99.jpg"
    # face_path = "data/test/extracted_faces/20251016_161106_309_1x-1/face_990_1838_87_110.jpg"
    # face_path = "data/test/extracted_faces/1749983182144/face_154_31_33_39.jpg"

    face_label = get_query_label(face_path)
    print("Face Label:", face_label)

    embedding_and_labels_data_file_path = os.path.join(BASE_DIR, "data/face_embeddings_eps_0.7_min_samples_5/face_embedding_with_label.pkl")
    all_images_path = os.path.join(BASE_DIR, "data/all_photos")
    final_image_paths_output_file = os.path.join(BASE_DIR, "data/face_embeddings_eps_0.7_min_samples_5")    # <Face label>.pkl will be appended later

    image_files = get_query_images(face_path, embedding_and_labels_data_file_path, all_images_path, final_image_paths_output_file)
    print(image_files)
