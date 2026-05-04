from facenet_pytorch.models.inception_resnet_v1 import InceptionResnetV1
import torch
import cv2
import os
import sys
from dotenv import load_dotenv
from tqdm import tqdm

from src.utils import get_face_tensor, image_clahe, write_pickle_file

load_dotenv()

BASE_DIR = os.getenv("BASE_DIR")
sys.path.append(BASE_DIR)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

embedding_model = InceptionResnetV1(pretrained='vggface2').eval().to(device)

def create_face_embedding(face_path):

    face = cv2.imread(face_path)
    # face = image_clahe(face)
    face_tensor = get_face_tensor(face)
    face_tensor = face_tensor.unsqueeze(0)
    face_tensor = face_tensor.to(device)

    with torch.no_grad():
        embedding = embedding_model(face_tensor)
    
    return embedding.detach().cpu().numpy().flatten()

def extract_embeddings(faces_save_path, face_embeddings_data_output_file_path):
    face_emb_data = []

    for img_folder in os.listdir(faces_save_path):
        for face in tqdm(os.listdir(os.path.join(faces_save_path, img_folder)), desc=f"Creating embedding of faces in {img_folder}"):
            face_emb = create_face_embedding(os.path.join(faces_save_path, img_folder, face))
            face_emb_data.append({
                "embedding": face_emb.astype("float64").tolist(),
                "image": img_folder,
                "face": face
            })
    
    print(f"All embeddings created. Saving to {face_embeddings_data_output_file_path} ...")

    write_pickle_file(data=face_emb_data, output_path=face_embeddings_data_output_file_path)
    
    print(f"Face embedding data written to {face_embeddings_data_output_file_path}")

if __name__ == "__main__":
    '''Testing create_face_embedding()'''
    # face_path = os.path.join(BASE_DIR, "data/extracted_faces/20260124_140806/face_1932_1072_282_304.jpg")
    # emb = create_face_embedding(face_path)
    # print(emb)

    '''Testing extract_embeddings()'''
    faces_save_path = os.path.join(BASE_DIR, "data/extracted_faces")
    face_embeddings_data_output_file_path = os.path.join(BASE_DIR, "data/face_embeddings/face_embeddings_data.pkl")
    extract_embeddings(faces_save_path, face_embeddings_data_output_file_path)