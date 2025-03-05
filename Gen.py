import os
import json
import cv2
import random
import time
import numpy as np
from collections import defaultdict, Counter
from wand.image import Image
from wand.font import Font
from wand.drawing import Drawing
from wand.color import Color
from utils.put_bg import put_bg
from tqdm import tqdm
from utils.Image_transformer import ImageTransformer
from utils.PYR_2_RGB import PYR_2_RGB
import argparse

def get_font_paths(directory):
    """获取所有字体文件的路径"""
    return [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(".ttf")]

def chunk_list(lst, chunk_size):
    """将列表分割成指定大小的块"""
    for i in range(0, len(lst), chunk_size):
        yield lst[i:i + chunk_size]

def create_text_images(texts, font_path, max_width=800, fixed_height=150):
    """创建相同尺寸的单行文字图像，确保两张图片独立，且大小一致"""
    font_size = 80  # 初始字體大小
    time.sleep(0.05)
    # 計算最大文字寬度，確保兩張圖片統一
    max_text_width = 0
    for text in texts:
        with Drawing() as draw, Image(width=max_width, height=fixed_height) as img:
            draw.font = font_path
            draw.font_size = font_size
            metrics = draw.get_font_metrics(img, text)
            max_text_width = max(max_text_width, int(metrics.text_width))

    # 確保文字不會超出最大寬度
    if max_text_width > max_width:
        scaling_factor = max_width / max_text_width
        font_size = int(font_size * scaling_factor * 0.8)

    # 獨立建立 mask_s 和 mask_t
    text_images = []
    for text in texts:
        with Image(width=max_text_width, height=fixed_height, background=Color('black')) as img:
            with Drawing() as draw:
                draw.clear()
                time.sleep(0.05)
                draw.font = font_path
                draw.font_size = font_size
                draw.text_alignment = 'center'
                draw.fill_color = Color('white')  # 設定文字顏色
                draw.text(int(max_text_width / 2), int(fixed_height / 2), text)  # 文字居中
                draw(img)  # 繪製文字

            # **確保圖片獨立，避免重疊**
            uniform_img = img.clone()

            # **立即清除原始 img，防止影像疊加**
            img.destroy()

            text_images.append(uniform_img)

    return text_images  # 返回兩张独立的图片

        
def apply_arc_distortion(image, arc_angle):
    """对图像应用弧形变形"""
    if arc_angle != 0:
        image.virtual_pixel = 'black'
        image.distort('arc', [arc_angle])
    return image

def replace_white_with_color(image, target_rgb=(0, 255, 0)):
    """
    將圖片中的白色區域替換為指定的 RGB 顏色。
    :param image_path: 圖片路徑
    :param target_rgb: 目標 RGB 顏色 (tuple)
    :return: 修改後的圖片
    """
    # 讀取圖片
    img = image

    # 定義白色區域閾值（白色範圍: 近似255, 255, 255）
    lower_white = np.array([200, 200, 200], dtype=np.uint8)  # 白色範圍下限
    upper_white = np.array([255, 255, 255], dtype=np.uint8)  # 白色範圍上限

    # 創建遮罩，找到白色區域
    mask = cv2.inRange(img, lower_white, upper_white)

    # 將白色區域替換為指定顏色
    img[mask == 255] = target_rgb

    return img

def process_text_images(TEXT_DIR="txt_text", DATA_DIR="test_img",FONT_DIR="./fonts/english_ttf", arc_angle=[0, 60, 120]):
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(os.path.join(DATA_DIR, "i_s"), exist_ok=True)
    os.makedirs(os.path.join(DATA_DIR, "mask_s"), exist_ok=True)
    os.makedirs(os.path.join(DATA_DIR, "mask_t"), exist_ok=True)
    os.makedirs(os.path.join(DATA_DIR, "mask_3d_s"), exist_ok=True)
    os.makedirs(os.path.join(DATA_DIR, "mask_3d_t"), exist_ok=True)
    os.makedirs(os.path.join(DATA_DIR, "t_b"), exist_ok=True)
    os.makedirs(os.path.join(DATA_DIR, "t_f"), exist_ok=True)
    TEMP_DIR = "temp_images"
    os.makedirs(TEMP_DIR, exist_ok=True)

    # 旋轉模式及其確切數量
    rotation_distribution = {
        (1, ("phi")): 4,
        (1, ("theta")): 4,
        (1, ("gamma")): 4,
        (2, ("phi", "theta")): 4,
        (2, ("phi", "gamma")): 1,
        (2, ("gamma", "theta")): 1,
        (3, ("phi", "theta", "gamma")): 2
    }

    texts = os.listdir(TEXT_DIR)
    # total_images = len(texts) 
    fonts = get_font_paths(FONT_DIR)
    font_groups = list(chunk_list(fonts, 2))

    for idx, text in tqdm(enumerate(texts)):
        font_group = font_groups[idx % len(font_groups)]
        file_name = text.split('.')[0] #00000
        with open(os.path.join(TEXT_DIR, text), "r", encoding="utf-8") as f:
            text = f.read()
        text = text.split(' ') # ['src str', 'tgt str']
        
        # 定義旋轉角度範圍並確保均衡分佈
        angle_options = [30, random.randint(45, 60), random.randint(65, 75), random.randint(285, 295), random.randint(300, 315), 330]
        arc_options = random.sample(arc_angle, 2)
        index = 0
        count = 0
        for font in font_group:
        # for font in fonts:
            # print(font)
            mask_s, mask_t = create_text_images([text[0], text[1]], font)
            for arc in arc_options:
                # print(arc)
                mask_s = apply_arc_distortion(mask_s, arc)
                mask_t = apply_arc_distortion(mask_t, arc)
                w = mask_s.width
                h = mask_s.height
                # Output Checked ✅
                temp_s_path = os.path.join(TEMP_DIR, f"temp_s_{idx}.jpg")
                temp_t_path = os.path.join(TEMP_DIR, f"temp_t_{idx}.jpg")
                mask_s.save(filename=temp_s_path)
                mask_t.save(filename=temp_t_path)
        # ----------------------- check font -----------------------
                s = cv2.imread(temp_s_path)
                if np.all(s == 0):
                    print(f"⚠️ 字體 {font} 產生的 mask_s 是全黑的")
        #             count += 1
        #         t = cv2.imread(temp_t_path)
        #         if np.all(t == 0):
        #             print(f"⚠️ 字體 {font} 產生的 mask_t 是全黑的")
        # print(count)
        # ----------------------- check font -----------------------
                for key,value in rotation_distribution.items():
                    # print(key)
                    num_axes = key[0]
                    axes = key[1]
                    for i in range(value):
                        index += 1
                        if num_axes == 1:
                            angles = {axes: random.choice(angle_options)}
                        # ----------------------- check normal vector -----------------------
                        #     angles = {'gamma':0, 'phi':0, 'theta':0}
                        # else:
                        #     continue
                        # ----------------------- check normal vector -----------------------
                        elif num_axes == 2:
                            angles = {axes[0]: random.choice(angle_options), axes[1]: random.choice(angle_options)}
                        elif num_axes == 3:
                            angles = {axes[0]: random.choice(angle_options), axes[1]: random.choice(angle_options), axes[2]: random.choice(angle_options)}  
                        # print(angles)
                        src_mask = ImageTransformer(temp_s_path, (w, h))
                        tgt_mask = ImageTransformer(temp_t_path, (w, h))
                        r_src_mask = src_mask.rotate_along_axis(phi=angles.get("phi", 0), theta=angles.get("theta", 0), gamma=angles.get("gamma", 0), dx=5)
                        r_tgt_mask = tgt_mask.rotate_along_axis(phi=angles.get("phi", 0), theta=angles.get("theta", 0), gamma=angles.get("gamma", 0), dx=5) 
                        rgb = PYR_2_RGB(angles.get("theta", 0), angles.get("phi", 0), angles.get("gamma", 0))
                        r_src_3d_mask = r_src_mask
                        r_tgt_3d_mask = r_tgt_mask
                        # background & text color rendering
                        bg, i_s, t_f = put_bg(image1 = r_src_mask, image2 = r_tgt_mask, bg_dir = "./datasets/bg_data/bg_img")
                        cv2.imwrite(f'./{DATA_DIR}/i_s/{file_name}_{index}.png', i_s)
                        cv2.imwrite(f'./{DATA_DIR}/t_f/{file_name}_{index}.png', t_f)
                        cv2.imwrite(f'./{DATA_DIR}/t_b/{file_name}_{index}.png', bg)
                        cv2.imwrite(f'./{DATA_DIR}/mask_s/{file_name}_{index}.png', r_src_mask)
                        cv2.imwrite(f'./{DATA_DIR}/mask_t/{file_name}_{index}.png', r_tgt_mask)
                        # rgb = rgb.astype(np.int16) # for phi, theta, gamma alignment
                        # rgb[1] = rgb[1] * (-1)
                        r_src_3d_mask = replace_white_with_color(r_src_3d_mask, rgb)
                        r_tgt_3d_mask = replace_white_with_color(r_tgt_3d_mask, rgb)
                        cv2.imwrite(f'./{DATA_DIR}/mask_3d_s/{file_name}_{index}.png', r_src_3d_mask)
                        cv2.imwrite(f'./{DATA_DIR}/mask_3d_t/{file_name}_{index}.png', r_tgt_3d_mask)
                os.remove(temp_s_path)
                os.remove(temp_t_path)
        os.rmdir(TEMP_DIR)
    print(f"finish generating")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="生成 3D 旋轉文字圖像")
    parser.add_argument("--text_dir", type=str, default="txt_test")
    parser.add_argument("--data_dir", type=str, default="SynTxt3D_50k_1")
    args = parser.parse_args()
    
    process_text_images(TEXT_DIR=args.text_dir, DATA_DIR=args.data_dir)
