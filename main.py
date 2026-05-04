from src.utils import copy_all_images, convert_to_jpg, empty_dir
from src.extract_faces import extract_faces
from src.create_face_embeddings import extract_embeddings
from src.cluster_embeddings import cluster_embeddings
from src.vector_store import create_vector_store
from src.query import get_query_images

import yaml
import os
import traceback
from dotenv import load_dotenv
import sys

load_dotenv()

BASE_DIR = os.getenv("BASE_DIR")
sys.path.append(BASE_DIR)

with open("paths_config.yaml", "r") as f:
    paths_config = yaml.safe_load(f)

def setup_deck(source_dir, photo_deck_name):
    try:
        photo_deck_root_path = os.path.join(BASE_DIR, paths_config["user_photo_decks_root_path"], photo_deck_name)

        print("\n1. Copying all images to data folder...")
        uploaded_images_path = os.path.join(photo_deck_root_path, paths_config["uploaded_images_path"])
        os.makedirs(uploaded_images_path, exist_ok=True)

        copy_all_images(source_dir, uploaded_images_path)

        print("\n2. Convert all images to JPEG format")
        all_images_path = os.path.join(photo_deck_root_path, paths_config["all_images_path"])
        os.makedirs(all_images_path, exist_ok=True)

        convert_to_jpg(uploaded_images_path, all_images_path)

        print("\n3. Extract faces from all the photos")
        faces_save_path = os.path.join(photo_deck_root_path, paths_config["faces_save_path"])
        os.makedirs(faces_save_path, exist_ok=True)

        extract_faces(all_images_path, faces_save_path)

        print("\n4. Create embeddings from faces")
        face_embeddings_data_output_folder = os.path.join(photo_deck_root_path, paths_config["face_embeddings_data_output_folder"])
        os.makedirs(face_embeddings_data_output_folder, exist_ok=True)

        face_embeddings_data_output_file_path = os.path.join(face_embeddings_data_output_folder, paths_config["face_embeddings_data_output_file_path"])
        extract_embeddings(faces_save_path, face_embeddings_data_output_file_path)

        print("\n5. Cluster embeddings")
        face_embeddings_data_with_labels_output_file_path = os.path.join(face_embeddings_data_output_folder, paths_config["face_embeddings_data_with_labels_output_file_path"])
        cluster_embeddings(face_embeddings_data_output_file_path, face_embeddings_data_with_labels_output_file_path)

        print("\n6. Create vector store")
        vector_store_path = os.path.join(photo_deck_root_path, paths_config["vector_store_path"])
        create_vector_store(face_embeddings_data_with_labels_output_file_path, faces_save_path, vector_store_path)

    except Exception as e:
        error_str = traceback.format_exc()
        print(error_str)

def query_image(uploaded_image, photo_deck_name):

    photo_deck_root_path = os.path.join(BASE_DIR, paths_config["user_photo_decks_root_path"], photo_deck_name)
    
    face_embeddings_data_output_folder = os.path.join(photo_deck_root_path, paths_config["face_embeddings_data_output_folder"])
    face_embeddings_data_with_labels_output_file_path = os.path.join(face_embeddings_data_output_folder, paths_config["face_embeddings_data_with_labels_output_file_path"])
    all_images_path = os.path.join(photo_deck_root_path, paths_config["all_images_path"])

    query_image_folder = os.path.join(photo_deck_root_path, paths_config["query_image_folder"])
    converted_query_image_folder = os.path.join(photo_deck_root_path, paths_config["converted_query_image_folder"])

    os.makedirs(query_image_folder, exist_ok=True)
    os.makedirs(converted_query_image_folder, exist_ok=True)
    
    # Empty the query image folders to avoid processing wrong images
    empty_dir(query_image_folder)
    empty_dir(converted_query_image_folder)

    # Save the uploaded image
    save_path = os.path.join(photo_deck_root_path, paths_config["query_image_folder"], uploaded_image.name)
    with open(save_path, "wb") as f:
        f.write(uploaded_image.getbuffer())
    
    print("\n1. Convert query image to JPEG format")
    convert_to_jpg(query_image_folder, converted_query_image_folder)

    query_image_faces_output_folder = os.path.join(photo_deck_root_path, paths_config["query_image_faces_output_folder"])    # <Face label>.pkl will be appended later
    os.makedirs(query_image_faces_output_folder, exist_ok=True)

    print("\n2. Extract faces from the query image")
    extract_faces(converted_query_image_folder, query_image_faces_output_folder)

    query_in_image_paths_output_folder = os.path.join(photo_deck_root_path, paths_config["query_in_image_paths_output_folder"])
    os.makedirs(query_in_image_paths_output_folder, exist_ok=True)

    query_image_face_folder_name = os.listdir(query_image_faces_output_folder)[0]
    query_image_face_file_name = os.listdir(os.path.join(query_image_faces_output_folder, query_image_face_folder_name))[0]
    query_face_path = os.path.join(query_image_faces_output_folder, query_image_face_folder_name, query_image_face_file_name)

    vector_store_path = os.path.join(photo_deck_root_path, paths_config["vector_store_path"])

    image_files = get_query_images(query_face_path, face_embeddings_data_with_labels_output_file_path, all_images_path, vector_store_path, query_in_image_paths_output_folder)
    # print(image_files)

    return image_files

if __name__ == "__main__":
    # setup_deck()
    # query_image()
    pass
    