import cv2
import sys
import os
from dotenv import load_dotenv
from tqdm import tqdm
import numpy as np

from src.utils import empty_dir, image_clahe

load_dotenv()

BASE_DIR = os.getenv("BASE_DIR")
sys.path.append(BASE_DIR)

# Load model
model_path = os.path.join(BASE_DIR, "models/yunet.onnx")

detector = cv2.FaceDetectorYN.create(
    model=model_path,
    config="",
    input_size=(320, 320),  # will update later
    score_threshold=0.6,
    nms_threshold=0.3,
    top_k=5000
)

def resize_image(img, max_dim=1024):
    h, w = img.shape[:2]
    scale = max_dim / max(h, w)
    if scale < 1:
        img = cv2.resize(img, (int(w*scale), int(h*scale)))
    return img

def align_face(image, landmarks):
    src = np.array(landmarks, dtype=np.float32)

    dst = np.array([
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041]
    ], dtype=np.float32)

    M, _ = cv2.estimateAffinePartial2D(src, dst)
    aligned = cv2.warpAffine(image, M, (160, 160))
    
    return aligned

def extract_faces(all_images_path, face_save_path):

    images = os.listdir(all_images_path)
    skipped = []
    errors = []

    if os.path.exists(face_save_path):
        empty_dir(face_save_path)

    for img in images:
        try:
            image_path = os.path.join(all_images_path, img)
            img_faces_folder = img.split(".")[0]
            
            image = cv2.imread(image_path)
            
            # Resize image before detection to avoid out-of-memory issue
            image = resize_image(img=image, max_dim=2048)

            h_img, w_img = image.shape[:2]
            
            # IMPORTANT: set input size dynamically
            detector.setInputSize((w_img, h_img))
            
            # Detect
            _, faces = detector.detect(image)

            if faces is None or len(faces) == 0:
                print(f"No faces found in {img}. Applying CLAHE...")
                clahe_image = image_clahe(image)
                
                # Detect
                _, faces = detector.detect(clahe_image)
            
            if faces is None or len(faces) == 0:
                print(f"No faces found in {img}. Skipping...")
                skipped.append(img)
                continue

            os.makedirs(os.path.join(face_save_path, img_faces_folder), exist_ok=True)

            for face in tqdm(faces, desc=f"Detecting faces in: {img}"):
                x, y, w, h, le_x, le_y, re_x, re_y, n_x, n_y, lm_x, lm_y, rm_x, rm_y = face[:-1].astype(int)

                landmarks = np.array([
                    [le_x,le_y],
                    [re_x,re_y],
                    [n_x,n_y],
                    [lm_x,lm_y],
                    [rm_x,rm_y],
                ])

                margin = 0.25  # try 0.2–0.3

                x1 = max(0, int(x - margin * w))
                y1 = max(0, int(y - margin * h))
                x2 = min(w_img, int(x + w + margin * w))
                y2 = min(h_img, int(y + h + margin * h))

                # # Clip to image boundaries
                # x1 = max(0, x)
                # y1 = max(0, y)
                # x2 = min(w_img, x + w)
                # y2 = min(h_img, y + h)

                face = image[y1:y2, x1:x2]

                # landmarks_crop = landmarks - np.array([x1, y1])

                # aligned_face = align_face(face, landmarks_crop)

                # cv2.imwrite(os.path.join(face_save_path, img_faces_folder, f"face_{x}_{y}_{w}_{h}.jpg"), aligned_face)
                cv2.imwrite(os.path.join(face_save_path, img_faces_folder, f"face_{x}_{y}_{w}_{h}.jpg"), face)
        except Exception as e:
            print(f"Error occured while detecting faces in {img}: {e}")
            if os.path.exists(os.path.join(face_save_path, img_faces_folder)):
                empty_dir(os.path.join(face_save_path, img_faces_folder))
            errors.append(img)
    
    print(f"\nSuccessfully extracted faces in {len(os.listdir(face_save_path))} images!")

    if len(skipped) > 0:
        print(f"\nCouldn't find faces in {len(skipped)} images:")
        for file in skipped:
            print(file)
    
    if len(errors) > 0:
        
        for file in errors:
            print(file)

if __name__ == "__main__":
    '''For the main files extraction'''
    # all_images_path = os.path.join(BASE_DIR, "data/all_photos")
    # face_save_path = os.path.join(BASE_DIR, "data/extracted_faces")

    '''For testing files'''
    all_images_path = os.path.join(BASE_DIR, "data/query_images/jpg_format")
    face_save_path = os.path.join(BASE_DIR, "data/query_faces")

    extract_faces(all_images_path, face_save_path)
