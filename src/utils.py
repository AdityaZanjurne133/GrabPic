import cv2
import shutil
import os
import torch
import numpy as np

def get_face_tensor(face):
    # Resize to model input size
    face = cv2.resize(face, (160, 160))

    # Convert BGR → RGB
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

    # # Convert to tensor
    # transform = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize(
    #         mean=[0.5, 0.5, 0.5],
    #         std=[0.5, 0.5, 0.5]
    #     )
    # ])

    # face_tensor = transform(face)

    # Convert to float and normalize [0,1]
    face = face.astype(np.float32) / 255.0

    # Normalize to [-1, 1]
    face = (face - 0.5) / 0.5

    # HWC → CHW
    face = np.transpose(face, (2, 0, 1))

    # To tensor
    face_tensor = torch.tensor(face, dtype=torch.float32)

    return face_tensor

def empty_dir(path):
    for item in os.listdir(path):
        item_path = os.path.join(path, item)

        if os.path.isfile(item_path) or os.path.islink(item_path):
            os.remove(item_path)      # delete file
        elif "virtual" not in item_path and os.path.isdir(item_path):
            shutil.rmtree(item_path) # delete folder recursively


def image_clahe(image):
    # CLAHE for contrast
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l = clahe.apply(l)

    lab = cv2.merge((l, a, b))
    image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    return image