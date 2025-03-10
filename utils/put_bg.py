import os
import cv2
import numpy as np
from typing import Tuple
import time
from .pick_color import pick_random_color_not_in_bg
"""
> Usage:
    from utils.put_bg import put_bg
    bg, textimg_with_bg1, textimg_with_bg2 = put_bg(image1 = image1, image2 = image2)

> Inputs:
    - image1: cv2.imread    mask_s
    - image2: cv2.imread    mask_t
> Returns:
    - bg: cv2.imread        t_b
    - result1: cv2.imread    i_s
    - result2: cv2.imread    t_f
"""

def pick_bg(bg_dir: str = "./datasets/bg_data/bg_img") -> np.ndarray:
    """
    隨機挑選指定資料夾內的背景圖片
    """
    bg_name = np.random.choice(os.listdir(bg_dir))
    bg_path = os.path.join(bg_dir, bg_name)
    if not os.path.exists(bg_path):
        raise FileNotFoundError(f"背景圖片 {bg_name} 不存在於 {bg_dir} 中")
    bg = cv2.imread(bg_path)
    time.sleep(0.01)
    while bg is None:
        bg_name = np.random.choice(os.listdir(bg_dir))
        bg_path = os.path.join(bg_dir, bg_name)
        bg = cv2.imread(bg_path)
    return bg


def put_bg(image1: np.ndarray, image2: np.ndarray, bg_dir: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    將圖片中的黑色區域（接近 (0, 0, 0)）換成背景圖片，
    而保留白色區域（接近 (255, 255, 255)）。

    **Parameters**
    - image: np.ndarray     原始圖片(mask_s 或 mask_t)
    - bg_dir: str           背景圖片資料夾路徑

    **Returns**
    - bg: np.ndarray        背景圖片(t_b)
    - result: np.ndarray    替換後的圖片(i_s 或 t_f)
    """
    # 複製原始圖片
    img1 = image1.copy()
    img2 = image2.copy()
    # 定義黑色區域閾值 (可以根據實際情況調整)
    lower_black = np.array([0, 0, 0], dtype=np.uint8)
    upper_black = np.array([200, 200, 200], dtype=np.uint8)
    lower_white = np.array([0, 0, 200], dtype=np.uint8)
    upper_white = np.array([180, 30, 255], dtype=np.uint8)
    # 產生遮罩：黑色區域會被標記為 255
    mask1 = cv2.inRange(img1, lower_black, upper_black)
    mask2 = cv2.inRange(img2, lower_black, upper_black)
    hsv1 = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
    hsv2 = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)
    text_mask1 = cv2.inRange(hsv1, lower_white, upper_white)
    text_mask2 = cv2.inRange(hsv2, lower_white, upper_white)

    # 選取一張背景圖片並調整尺寸與原圖一致
    bg = pick_bg(bg_dir=bg_dir)
    bg = cv2.resize(bg, (img1.shape[1], img1.shape[0]))
    # 生成隨機顏色 (R, G, B)
    random_color = pick_random_color_not_in_bg(bg, distance_threshold=20)

    # 建立指定大小的圖片，並填充隨機顏色
    rand_color_txt = np.full((img1.shape[0], img1.shape[1], 3), random_color, dtype=np.uint8)

    # 將原圖中黑色區域替換為背景圖片中的對應像素
    # 注意：白色區域保留原本內容
    text_w_bg1 = img1.copy()
    text_w_bg1[mask1 > 0] = bg[mask1 > 0]
    text_w_bg1 = text_w_bg1.copy()
    text_w_bg1[text_mask1 > 0] = rand_color_txt[text_mask1 > 0]
    text_w_bg2 = img2.copy()
    text_w_bg2[mask2 > 0] = bg[mask2 > 0]
    text_w_bg2 = text_w_bg2.copy()
    text_w_bg2[text_mask2 > 0] = rand_color_txt[text_mask2 > 0]
    return bg, text_w_bg1, text_w_bg2

if __name__ == "__main__":
    # ---------------------------------
    # simple usage for function testing
    # ---------------------------------

    # 請自行替換成你的圖片路徑
    input_image1_path = "test1.png"
    image1 = cv2.imread(input_image1_path)
    if image1 is None:
        raise FileNotFoundError(f"找不到圖片: {input_image1_path}")
    input_image2_path = "test2.png"
    image2 = cv2.imread(input_image2_path)
    if image2 is None:
        raise FileNotFoundError(f"找不到圖片: {input_image2_path}")

    bg, result_img1, result_img2 = put_bg(image1 = image1, image2 = image2, bg_dir = "./datasets/bg_data/bg_img")
    cv2.imwrite("./bg.jpg", bg)
    cv2.imwrite("./result1.jpg", result_img1)
    cv2.imwrite("./result2.jpg", result_img2)
    print("已將白色區域替換成背景，圖片儲存為 result.jpg")
