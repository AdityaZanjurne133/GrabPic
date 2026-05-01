import cv2
import shutil
import os
import torch
import numpy as np
from PIL import Image, ImageOps
import pillow_heif
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle

# For .heic files
# Register HEIF opener with Pillow
pillow_heif.register_heif_opener()

def convert_to_jpg(source_dir, output_dir):

    files = os.listdir(source_dir)

    for file in tqdm(files, desc="converting files to .jpg:"):
        input_file = os.path.join(source_dir, file)
        output_file = os.path.join(output_dir, file.split(".")[0] + ".jpg")
        try:
            img = Image.open(input_file)

            # To avoid rotation
            img = ImageOps.exif_transpose(img)

            # Show or process
            # img.show()

            # Convert to RGB (important for some ops)
            img = img.convert("RGB")

            # Save as JPEG/PNG if needed
            img.save(output_file, format="JPEG", quality=95)

        except Exception as e:
            print(f"\nError occured while converting {file} to .jpg format:\n{e}")

    print("\nSuccessfully converted all files to .jpg!")


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

def plot_faces(faces):
    if faces is not None:
        # empty_dir("/kaggle/working/")
    
        cols = 5
        n_images = len(faces)
        rows = (n_images + cols - 1) // cols
        plt.figure(figsize=(15, 3 * rows))
        
        for i, face in enumerate(faces):
            # x, y, w, h = face[:4].astype(int)
    
            # # Clip to image boundaries
            # x = max(0, x)
            # y = max(0, y)
            # x2 = min(w_img, x + w)
            # y2 = min(h_img, y + h)
    
            # cropped = image[y:y+h, x:x+w]
    
            # cv2.imwrite(f"face_{i}.jpg", cropped)
            plt.subplot(rows, cols, i + 1)
            plt.imshow(face)
            # plt.title(f"face_{x}_{y}_{w}_{h}.jpg", fontsize=8)
            plt.axis("off")
    
        # for i, img in enumerate(images):
        #     plt.subplot(rows, cols, i + 1)
        #     plt.imshow(img)
        #     plt.title(f"face_{x}_{y}_{w}_{h}.jpg", fontsize=8)
        #     plt.axis("off")
    
        plt.tight_layout()
        plt.show()

def write_pickle_file(data, output_path):
    with open(output_path, "wb") as f:
        pickle.dump(data, f)
        f.close()

def read_pickle_file(file_path):
    with open(file_path, "rb") as f:
        data = pickle.load(f)
        f.close()
    
    return data
