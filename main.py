from src.utils import convert_to_jpg
from src.extract_faces import extract_faces
from src.create_face_embeddings import extract_embeddings
from src.cluster_embeddings import cluster_embeddings
from src.vector_store import create_vector_store
from src.query import get_query_images

import yaml
import os
from dotenv import load_dotenv
import sys

load_dotenv()

BASE_DIR = os.getenv("BASE_DIR")
sys.path.append(BASE_DIR)

with open("paths_config.yaml", "r") as f:
    paths_config = yaml.safe_load(f)

def setup_deck():
    try:
        # print("\n1. Convert all images to JPEG format")
        # uploaded_images_path = os.path.join(BASE_DIR, paths_config["uploaded_images_path"])
        all_images_path = os.path.join(BASE_DIR, paths_config["all_images_path"])
        # os.makedirs(all_images_path, exist_ok=True)

        # convert_to_jpg(uploaded_images_path, all_images_path)

        print("\n2. Extract faces from all the photos")
        faces_save_path = os.path.join(BASE_DIR, paths_config["faces_save_path"])
        os.makedirs(faces_save_path, exist_ok=True)

        extract_faces(all_images_path, faces_save_path)

        print("\n3. Create embeddings from faces")
        face_embeddings_data_output_folder = os.path.join(BASE_DIR, paths_config["face_embeddings_data_output_folder"])
        os.makedirs(face_embeddings_data_output_folder, exist_ok=True)

        face_embeddings_data_output_file_path = os.path.join(face_embeddings_data_output_folder, paths_config["face_embeddings_data_output_file_path"])
        extract_embeddings(faces_save_path, face_embeddings_data_output_file_path)

        print("\n4. Cluster embeddings")
        face_embeddings_data_with_labels_output_file_path = os.path.join(face_embeddings_data_output_folder, paths_config["face_embeddings_data_with_labels_output_file_path"])
        cluster_embeddings(face_embeddings_data_output_file_path, face_embeddings_data_with_labels_output_file_path)

        print("\n5. Create vector store")
        vector_store_path = os.path.join(BASE_DIR, paths_config["vector_store_path"])
        create_vector_store(face_embeddings_data_with_labels_output_file_path, faces_save_path, vector_store_path)

    except Exception as e:
        print(f"[ERROR]: {e}")

def query_image():
    
    face_embeddings_data_output_folder = os.path.join(BASE_DIR, paths_config["face_embeddings_data_output_folder"])
    face_embeddings_data_with_labels_output_file_path = os.path.join(face_embeddings_data_output_folder, paths_config["face_embeddings_data_with_labels_output_file_path"])
    all_images_path = os.path.join(BASE_DIR, paths_config["all_images_path"])

    query_image_folder = os.path.join(BASE_DIR, paths_config["query_image_folder"])
    converted_query_image_folder = os.path.join(BASE_DIR, paths_config["converted_query_image_folder"])
    convert_to_jpg(query_image_folder, converted_query_image_folder)

    query_image_faces_output_folder = os.path.join(BASE_DIR, paths_config["query_image_faces_output_folder"])    # <Face label>.pkl will be appended later
    os.makedirs(query_image_faces_output_folder, exist_ok=True)

    extract_faces(converted_query_image_folder, query_image_faces_output_folder)

    query_in_image_paths_output_folder = os.path.join(BASE_DIR, paths_config["query_in_image_paths_output_folder"])
    os.makedirs(query_in_image_paths_output_folder, exist_ok=True)

    query_image_face_folder_name = os.listdir(query_image_faces_output_folder)[0]
    query_image_face_file_name = os.listdir(os.path.join(query_image_faces_output_folder, query_image_face_folder_name))[0]
    query_face_path = os.path.join(query_image_faces_output_folder, query_image_face_folder_name, query_image_face_file_name)

    vector_store_path = os.path.join(BASE_DIR, paths_config["vector_store_path"])

    image_files = get_query_images(query_face_path, face_embeddings_data_with_labels_output_file_path, all_images_path, vector_store_path, query_in_image_paths_output_folder)
    print(image_files)

if __name__ == "__main__":
    # setup_deck()
    query_image()
    