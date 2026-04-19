from utils import convert_to_jpg

import os
from dotenv import load_dotenv
import sys

load_dotenv()

BASE_DIR = os.getenv("BASE_DIR")
sys.path.append(BASE_DIR)

if __name__ == "__main__":
    query_images_path = os.path.join(BASE_DIR, "data/test/query_images")
    jpg_images_path = os.path.join(BASE_DIR, "data/test/input_images")

    convert_to_jpg(source_dir=query_images_path, output_dir=jpg_images_path)
