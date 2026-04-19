from sklearn.cluster import DBSCAN
import torch
import pickle
import numpy as np
import cv2
import os
from dotenv import load_dotenv
import sys

load_dotenv()

BASE_DIR = os.getenv("BASE_DIR")
sys.path.append(BASE_DIR)

from utils import plot_faces

def get_face_embedding_and_metadata(embedding_data_file_path):
    # Read face embeddings and related metadata from the pickle file
    face_emb_data = []

    with open(embedding_data_file_path, "rb") as f:
        face_emb_data = pickle.load(f)

    return face_emb_data

def cluster_embeddings(embedding_data_file_path, embedding_and_labels_data_file_path):
    face_emb_data = get_face_embedding_and_metadata(embedding_data_file_path)

    # Get all the face embeddings as a list
    face_embeddings = []
    # for data in face_emb_data:
    #     face_embeddings.append(np.array(data["embedding"]))
    # face_embeddings = np.array(face_embeddings)

    face_embeddings = np.array(
        [np.array(d["embedding"], dtype=np.float64) for d in face_emb_data],
        dtype=np.float64
    )

    # print(face_embeddings.shape)
    
    clustering = DBSCAN(eps=0.7, min_samples=5).fit(face_embeddings)

    labels = clustering.labels_

    with open(embedding_and_labels_data_file_path, "wb") as f:
        pickle.dump(face_emb_data, f)

    return labels

if __name__ == "__main__":
    embedding_data_file_path = os.path.join(BASE_DIR, "data/face_embeddings_eps_0.7_min_samples_5/extracted_emb_data.pkl")
    embedding_and_labels_data_file_path = os.path.join(BASE_DIR, "data/face_embeddings_eps_0.7_min_samples_5/face_embedding_with_label.pkl")
    extracted_faces_path = os.path.join(BASE_DIR, "data/extracted_faces")
    faces = []

    face_emb_data = get_face_embedding_and_metadata(embedding_data_file_path)
    labels = cluster_embeddings(embedding_data_file_path, embedding_and_labels_data_file_path)

    for i in range(len(labels)):
        if labels[i] == 3:
            face_path = os.path.join(extracted_faces_path, face_emb_data[i]["image"], face_emb_data[i]["face"])
            faces.append(cv2.imread(face_path))

    plot_faces(faces)