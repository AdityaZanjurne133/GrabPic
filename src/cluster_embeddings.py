from sklearn.cluster import DBSCAN
import numpy as np
import cv2
import os
from dotenv import load_dotenv
import sys

load_dotenv()

BASE_DIR = os.getenv("BASE_DIR")
sys.path.append(BASE_DIR)

from src.utils import plot_faces, read_pickle_file, write_pickle_file

def cluster_embeddings(face_embeddings_data_output_file_path, face_embeddings_data_with_labels_output_file_path):
    face_emb_data = read_pickle_file(face_embeddings_data_output_file_path)

    # Get all the face embeddings as a list
    face_embeddings = []

    face_embeddings = np.array(
        [np.array(d["embedding"], dtype=np.float64) for d in face_emb_data],
        dtype=np.float64
    )
    
    clustering = DBSCAN(eps=0.7, min_samples=5).fit(face_embeddings)

    labels = clustering.labels_

    for i in range(len(labels)):
        face_emb_data[i]["label"] = int(labels[i])

    write_pickle_file(data=face_emb_data, output_path=face_embeddings_data_with_labels_output_file_path)
    print(f"\nSuccessfully saved labeled embeddings data in {face_embeddings_data_with_labels_output_file_path}!")

if __name__ == "__main__":
    face_embeddings_data_output_file_path = os.path.join(BASE_DIR, "data/face_embeddings_eps_0.7_min_samples_5/extracted_emb_data.pkl")
    face_embeddings_data_with_labels_output_file_path = os.path.join(BASE_DIR, "data/face_embeddings_eps_0.7_min_samples_5/face_embedding_with_label.pkl")
    extracted_faces_path = os.path.join(BASE_DIR, "data/extracted_faces")
    
    cluster_embeddings(face_embeddings_data_output_file_path, face_embeddings_data_with_labels_output_file_path)
    
    # faces = []

    # face_emb_data = read_pickle_file(face_embeddings_data_output_file_path)
    # labels = cluster_embeddings(face_embeddings_data_output_file_path, face_embeddings_data_with_labels_output_file_path)

    # for i in range(len(labels)):
    #     if labels[i] == 3:
    #         face_path = os.path.join(extracted_faces_path, face_emb_data[i]["image"], face_emb_data[i]["face"])
    #         faces.append(cv2.imread(face_path))

    # plot_faces(faces)