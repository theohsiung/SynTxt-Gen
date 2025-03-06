import os
from tqdm import tqdm
import cv2
from skimage import io

path = "./datasets/bg_data/bg_img"
fileList = os.listdir(path)

for i in tqdm(fileList):
    try:
        img = io.imread(os.path.join(path, i))
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGRA)
        cv2.imencode('.png', img)[1].tofile(os.path.join(path, i))
    except Exception as e:
        print(f"Error processing delete {i}")
        os.remove(os.path.join(path, i))
        continue
    # print(img.shape)
