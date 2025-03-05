import os
import cv2
import numpy as np
from typing import Tuple

"""
> Usage:
    from utils.put_bg import put_bg
    bg, textimg_with_bg = put_bg(image = image)

> Inputs:
    - image: cv2.imread    mask_s / mask_t
> Returns:
    - bg: cv2.imread        t_b
    - result: cv2.imread    i_s / t_f
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
    if bg is None:
        raise ValueError(f"無法讀取背景圖片：{bg_path}")
    return bg


def put_bg(image: np.ndarray, bg_dir: str) -> Tuple[np.ndarray, np.ndarray]:
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
    img = image.copy()

    # 定義黑色區域閾值 (可以根據實際情況調整)
    lower_black = np.array([0, 0, 0], dtype=np.uint8)
    upper_black = np.array([200, 200, 200], dtype=np.uint8)
    # 產生遮罩：黑色區域會被標記為 255
    mask = cv2.inRange(img, lower_black, upper_black)

    # 選取一張背景圖片並調整尺寸與原圖一致
    bg = pick_bg(bg_dir=bg_dir)
    bg = cv2.resize(bg, (img.shape[1], img.shape[0]))

    # 將原圖中黑色區域替換為背景圖片中的對應像素
    # 注意：白色區域保留原本內容
    text_w_bg = img.copy()
    text_w_bg[mask > 0] = bg[mask > 0]

    return bg, text_w_bg

if __name__ == "__main__":
    # ---------------------------------
    # simple usage for function testing
    # ---------------------------------

    # 請自行替換成你的圖片路徑
    input_image_path = "test.png"
    image = cv2.imread(input_image_path)
    if image is None:
        raise FileNotFoundError(f"找不到圖片: {input_image_path}")

    bg, result_img = put_bg(image = image, bg_dir = "../datasets/bg_data/bg_img")
    cv2.imwrite("./tmp/bg.jpg", bg)
    cv2.imwrite("./tmp/result.jpg", result_img)
    print("已將白色區域替換成背景，圖片儲存為 result.jpg")
