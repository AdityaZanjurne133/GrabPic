import cv2
import shutil
import os

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