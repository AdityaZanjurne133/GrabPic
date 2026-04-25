import cv2
import sys
import os
from dotenv import load_dotenv
from tqdm import tqdm
from utils import empty_dir, image_clahe

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

def extract_faces(images_path, face_save_path):

    images = os.listdir(images_path)
    skipped = []
    errors = []

    if os.path.exists(face_save_path):
        empty_dir(face_save_path)

    for img in images:
        try:
            image_path = os.path.join(images_path, img)
            img_faces_folder = img.split(".")[0]
            
            image = cv2.imread(image_path)
            h_img, w_img = image.shape[:2]
            
            # IMPORTANT: set input size dynamically
            detector.setInputSize((w_img, h_img))
            
            # Detect
            _, faces = detector.detect(image)

            if faces is None or len(faces) == 0:
                print(f"No faces found in {img}. Applying CLAHE...")
                image = image_clahe(image)
                
                # Detect
                _, faces = detector.detect(image)
            
            if faces is None or len(faces) == 0:
                print(f"No faces found in {img}. Skipping...")
                skipped.append(img)
                continue

            os.makedirs(os.path.join(face_save_path, img_faces_folder), exist_ok=True)

            for face in tqdm(faces, desc=f"Detecting faces in: {img}"):
                x, y, w, h = face[:4].astype(int)

                # Clip to image boundaries
                x = max(0, x)
                y = max(0, y)
                x2 = min(w_img, x + w)
                y2 = min(h_img, y + h)

                face = image[y:y+h, x:x+w]

                cv2.imwrite(os.path.join(face_save_path, img_faces_folder, f"face_{x}_{y}_{w}_{h}.jpg"), face)
        except Exception as e:
            print(f"Error occured while detecting faces in {img}: {e}")
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
    # images_path = os.path.join(BASE_DIR, "data/all_photos")
    # face_save_path = os.path.join(BASE_DIR, "data/extracted_faces")

    '''For testing files'''
    images_path = os.path.join(BASE_DIR, "data/test/input_images")
    face_save_path = os.path.join(BASE_DIR, "data/test/extracted_faces")

    extract_faces(images_path, face_save_path)
