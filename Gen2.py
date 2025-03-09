import os
import json
import cv2
import random
import glob
import time
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from collections import defaultdict, Counter
from wand.image import Image
from wand.font import Font
from wand.drawing import Drawing
from wand.color import Color
from utils.put_bg import put_bg
from tqdm import tqdm
from utils.Image_transformer import ImageTransformer
from utils.spherical_2_rgb import spherical2RGB
import argparse

def get_font_paths(directory):
    """获取所有字体文件的路径"""
    return [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(".ttf")]

def chunk_list(lst, chunk_size):
    """将列表分割成指定大小的块"""
    for i in range(0, len(lst), chunk_size):
        yield lst[i:i + chunk_size]

def create_text_images(texts, font_path, max_width=800, fixed_height=150, trim_fuzz=40):
    """创建单行文字图像，裁切掉多余背景（利用 fuzz 參數讓邊緣更乾淨），但所有输出影像尺寸保持一致"""
    font_size = 80  # 初始字體大小

    #----------------------------------------------
    try:
        with Drawing() as draw, Image(width=max_width, height=fixed_height) as img:
            draw.font = font_path
            draw.font_size = font_size
            metrics = draw.get_font_metrics(img, "TEST_STRING")
            if metrics.text_width == 0:
                raise ValueError(f"⚠️ {font_path}，請檢查字體是否損壞")
    except Exception as e:
        print(f"❌ 字體載入失敗: {font_path}, 錯誤: {e}")
        return None
    #----------------------------------------------
    # 計算所有文字中的最大寬度
    max_text_width = 0
    for text in texts:
        with Drawing() as draw, Image(width=max_width, height=fixed_height) as img:
            draw.font = font_path
            draw.font_size = font_size
            metrics = draw.get_font_metrics(img, text)
            max_text_width = max(max_text_width, int(metrics.text_width))
    
    # 如有必要，縮放字體大小，避免文字超出指定最大寬度
    if max_text_width > max_width:
        scaling_factor = max_width / max_text_width
        font_size = int(font_size * scaling_factor * 0.8)

    text_images = []
    for text in texts:
        # 在指定尺寸的黑色背景上繪製文字
        with Image(width=max_text_width, height=fixed_height, background=Color('black')) as img:
            with Drawing() as draw:
                draw.clear()
                time.sleep(0.01)
                draw.font = font_path
                draw.font_size = font_size
                draw.text_alignment = 'center'
                draw.fill_color = Color('white')
                draw.text(int(max_text_width / 2), int(fixed_height / 2), text)
                draw(img)
            # 確保圖片獨立
            uniform_img = img.clone()
            
            # 裁切掉多餘的黑色背景，並用 fuzz 參數讓邊緣更乾淨
            trimmed = uniform_img.clone()
            trimmed.trim(fuzz=trim_fuzz)
            # trimmed.repage()  # 重置影像頁面資訊，避免殘留 offset
            
            # 將裁切後的圖像置中合成到固定尺寸的黑色背景上
            with Image(width=max_text_width, height=fixed_height, background=Color('black')) as canvas:
                offset_x = (max_text_width - trimmed.width) // 2
                offset_y = (fixed_height - trimmed.height) // 2
                canvas.composite(trimmed, left=offset_x, top=offset_y)
                final_img = canvas.clone()
            text_images.append(final_img)
    return text_images


def apply_arc_distortion(image1, image2, arc_angle):
    """
    對兩個圖像同時應用弧形變形，確保兩者彎曲的角度保持一致。
    若變成全黑則減小 arc_angle 直到兩者皆不全黑；若 arc_angle 為 0，則返回原圖。

    參數:
      image1, image2: 要處理的兩個影像 (Wand Image 對象)
      arc_angle: 初始弧形角度

    回傳:
      (distorted_image1, distorted_image2)
    """
    if arc_angle == 0:
        return image1, image2

    # 設定虛擬像素模式避免填充色干擾
    image1.virtual_pixel = 'black'
    image2.virtual_pixel = 'black'
    current_arc = arc_angle

    while current_arc > 0:
        temp1 = image1.clone()
        temp2 = image2.clone()
        temp1.distort('arc', [current_arc])
        temp2.distort('arc', [current_arc])

        # 轉換為 OpenCV 格式檢查是否全黑
        img_bytes1 = temp1.make_blob(format='PNG')
        nparr1 = np.frombuffer(img_bytes1, np.uint8)
        img_cv1 = cv2.imdecode(nparr1, cv2.IMREAD_GRAYSCALE)

        img_bytes2 = temp2.make_blob(format='PNG')
        nparr2 = np.frombuffer(img_bytes2, np.uint8)
        img_cv2 = cv2.imdecode(nparr2, cv2.IMREAD_GRAYSCALE)

        # 檢查兩張影像都不是全黑
        if not np.all(img_cv1 == 0) and not np.all(img_cv2 == 0):
            return temp1, temp2

        # 若任一影像全黑，將 arc_angle 減半再重試
        current_arc //= 2

    # 若調整後仍無法獲得有效影像，則返回原始影像
    return image1, image2


def replace_white_with_color(image, target_rgb=(0, 255, 0)):
    """
    將圖片中的白色區域替換為指定的 RGB 顏色。
    """
    img = image
    lower_white = np.array([200, 200, 200], dtype=np.uint8)
    upper_white = np.array([255, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(img, lower_white, upper_white)
    img[mask == 255] = target_rgb
    return img

def get_angle(axis, angle_options):
    if axis == "gamma":
        # 對 gamma 軸隨機選取 330~360 或 0~30
        return random.choice([random.randint(330, 360), random.randint(0, 30)])
    else:
        return random.choice(angle_options)

def compute_common_bbox(img1, img2, threshold=200):
    """
    計算兩張圖片中白色文字的共同邊界 (top, bottom, left, right)
    參數:
      img1, img2: 輸入影像，必須為灰階或單通道影像
      threshold: 判定白色的閾值 (預設200，可依需要調整)
    回傳:
      (top, bottom, left, right) 四個邊界座標
    """
    # 若影像為彩色，先轉換為灰階
    if len(img1.shape) > 2:
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    if len(img2.shape) > 2:
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # 找出兩張影像中像素值大於 threshold 的座標
    # np.where 回傳 (rows, cols)
    coords1 = np.where(img1 >= threshold)
    coords2 = np.where(img2 >= threshold)

    # 檢查是否有找到任何白色像素
    if coords1[0].size == 0 or coords2[0].size == 0:
        raise ValueError("至少一張影像中找不到足夠的白色文字區域")

    # 分別計算每張影像的邊界 (注意：第一個維度為 row, 即 y；第二個為 col, 即 x)
    top1, bottom1 = np.min(coords1[0]), np.max(coords1[0])
    left1, right1 = np.min(coords1[1]), np.max(coords1[1])
    
    top2, bottom2 = np.min(coords2[0]), np.max(coords2[0])
    left2, right2 = np.min(coords2[1]), np.max(coords2[1])
    
    # 取聯集：以兩張影像中最極端的邊界為共同裁切區域
    top   = min(top1, top2)
    bottom = max(bottom1, bottom2)
    left  = min(left1, left2)
    right = max(right1, right2)

    return top, bottom, left, right

# 範例：使用計算出的邊界來裁切兩張圖片
def crop_images_to_common_bbox(img1, img2, img3, img4, img5, img6, img7, top, bottom, left, right):
    cropped1 = img1[top:bottom+1, left:right+1]
    cropped2 = img2[top:bottom+1, left:right+1]
    cropped3 = img3[top:bottom+1, left:right+1] 
    cropped4 = img4[top:bottom+1, left:right+1]
    cropped5 = img5[top:bottom+1, left:right+1]
    cropped6 = img6[top:bottom+1, left:right+1]
    cropped7 = img7[top:bottom+1, left:right+1]
    return cropped1, cropped2, cropped3, cropped4, cropped5, cropped6, cropped7

def process_text_images(TEXT_DIR="txt_text", DATA_DIR="test_img", FONT_DIR="./fonts/english_ttf", arc_angle=[0, 60, 120], file_range=[25000, 50000]):
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(os.path.join(DATA_DIR, "i_s"), exist_ok=True)
    os.makedirs(os.path.join(DATA_DIR, "mask_s"), exist_ok=True)
    os.makedirs(os.path.join(DATA_DIR, "mask_t"), exist_ok=True)
    os.makedirs(os.path.join(DATA_DIR, "mask_3d_s"), exist_ok=True)
    os.makedirs(os.path.join(DATA_DIR, "mask_3d_t"), exist_ok=True)
    os.makedirs(os.path.join(DATA_DIR, "t_b"), exist_ok=True)
    os.makedirs(os.path.join(DATA_DIR, "t_f"), exist_ok=True)
    os.makedirs(os.path.join(DATA_DIR, "txt"), exist_ok=True)

    # 原本的 3D 旋轉角度分佈配置，總共 20 個選項
    rotation_distribution = {
        (1, ("phi",)): 4,
        (1, ("theta",)): 4,
        (1, ("gamma",)): 4,
        (2, ("phi", "theta")): 4,
        (2, ("phi", "gamma")): 1,
        (2, ("gamma", "theta")): 1,
        (3, ("phi", "theta", "gamma")): 2
    }
    # 將配置展平成列表（長度 20），後續依據 text index 輪替使用
    rotation_configs = []
    for key, count in rotation_distribution.items():
        for i in range(count):
            rotation_configs.append(key)

    # angle_options 為可選的旋轉角度
    angle_options = [30, random.randint(45, 60), random.randint(65, 70), random.randint(290, 295), random.randint(300, 315), 330]

    texts = glob.glob(os.path.join(TEXT_DIR, "*.txt"))
    filtered_files = sorted([
        f for f in texts if file_range[0] <= int(os.path.basename(f)[:-4]) <= file_range[1]
    ])
    
    fonts = get_font_paths(FONT_DIR)
    font_groups = list(chunk_list(fonts, 69))

    for idx, text_file in tqdm(enumerate(filtered_files), total=len(filtered_files), desc="Generating data", leave=True):
        # 從 font_group 中選取一個字體（例如：第一個）
        font_group = font_groups[idx % len(font_groups)]
        font = font_group[idx % len(font_group)]
        file_name = os.path.basename(text_file)[:-4]  # 去除 .txt

        with open(text_file, "r", encoding="utf-8") as f:
            content = f.read()
        parts = content.split(' ')  # 預期格式： [src_text, tgt_text]
        if len(parts) < 2:
            print(f"⚠️ 文件格式錯誤：{text_file}")
            continue

        # 選擇一個 arc 值
        arc = random.choice(arc_angle)

        # 產生文字圖片（僅產生一組 pair 的 mask_s 與 mask_t）
        mask_s, mask_t = create_text_images([parts[0], parts[1]], font)
        if mask_s is None or mask_t is None:
            continue

        # 應用弧形變形
        mask_s, mask_t = apply_arc_distortion(mask_s, mask_t, arc)

        # mask_s.save(filename = f'./temp/{file_name}_mask_s.png')
        # mask_t.save(filename = f'./temp/{file_name}_mask_t.png')
        w = mask_s.width
        h = mask_s.height

        # 轉換為 OpenCV 圖片
        mask_s_bytes = mask_s.make_blob(format='PNG')
        mask_t_bytes = mask_t.make_blob(format='PNG')
        nparr_s = np.frombuffer(mask_s_bytes, np.uint8)
        nparr_t = np.frombuffer(mask_t_bytes, np.uint8)
        mask_s_cv = cv2.imdecode(nparr_s, cv2.IMREAD_COLOR)
        mask_t_cv = cv2.imdecode(nparr_t, cv2.IMREAD_COLOR)

        # 根據當前 text 的索引，從 rotation_configs 中選取旋轉配置
        config = rotation_configs[idx % len(rotation_configs)]
        # 根據配置決定旋轉角度組合
        if config[0] == 1:
            angles = {config[1][0]: get_angle(config[1][0], angle_options)}
        elif config[0] == 2:
            angles = {config[1][0]: get_angle(config[1][0], angle_options), config[1][1]: get_angle(config[1][1], angle_options)}
        elif config[0] == 3:
            angles = {config[1][0]: get_angle(config[1][0], angle_options), config[1][1]: get_angle(config[1][1], angle_options), config[1][2]: get_angle(config[1][2], angle_options)}

        # 3D 旋轉處理
        src_mask = ImageTransformer(mask_s_cv, (w, h))
        tgt_mask = ImageTransformer(mask_t_cv, (w, h))
        r_src_mask = src_mask.rotate_along_axis(phi=angles.get("phi", 0), 
                                                  theta=angles.get("theta", 0), 
                                                  gamma=angles.get("gamma", 0), 
                                                  dx=5)
        r_tgt_mask = tgt_mask.rotate_along_axis(phi=angles.get("phi", 0), 
                                                  theta=angles.get("theta", 0), 
                                                  gamma=angles.get("gamma", 0), 
                                                  dx=5)
        
        top, bottom, left, right = compute_common_bbox(r_src_mask, r_tgt_mask)
        # 根據旋轉角度計算背景色
        _, bgr = spherical2RGB(theta=angles.get("theta", 0), phi=angles.get("phi", 0))
        r_src_3d_mask = r_src_mask
        r_tgt_3d_mask = r_tgt_mask

        # 背景與文字渲染
        bg, i_s, t_f = put_bg(image1=r_src_mask, image2=r_tgt_mask, bg_dir="./datasets/bg_data/bg_img")
        i_s, t_f, bg, r_src_mask, r_tgt_mask, r_src_3d_mask, r_tgt_3d_mask = crop_images_to_common_bbox(i_s, t_f, bg, r_src_mask, r_tgt_mask, r_src_3d_mask, r_tgt_3d_mask, top, bottom, left, right)
        
        # 為每個 text 僅輸出一組圖片
        # output_index = idx + 1
        cv2.imwrite(f'./{DATA_DIR}/i_s/{file_name}.png', i_s)
        cv2.imwrite(f'./{DATA_DIR}/t_f/{file_name}.png', t_f)
        cv2.imwrite(f'./{DATA_DIR}/t_b/{file_name}.png', bg)
        cv2.imwrite(f'./{DATA_DIR}/mask_s/{file_name}.png', r_src_mask)
        cv2.imwrite(f'./{DATA_DIR}/mask_t/{file_name}.png', r_tgt_mask)
        r_src_3d_mask = replace_white_with_color(r_src_3d_mask, bgr)
        r_tgt_3d_mask = replace_white_with_color(r_tgt_3d_mask, bgr)
        cv2.imwrite(f'./{DATA_DIR}/mask_3d_s/{file_name}.png', r_src_3d_mask)
        cv2.imwrite(f'./{DATA_DIR}/mask_3d_t/{file_name}.png', r_tgt_3d_mask)
        with open(f"./{DATA_DIR}/txt/{file_name}.txt", "w", encoding="utf-8") as f:
            f.write(f"{parts[0]} {parts[1]}")
        with open(f"./{DATA_DIR}/i_t.txt", "a", encoding="utf-8") as f:
            f.write(f"{file_name}.png {parts[1]}\n")
    print("finish generating")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="生成 3D 旋轉文字圖像")
    parser.add_argument("--text_dir", type=str, default="train-50k-2/txt")
    parser.add_argument("--data_dir", type=str, default="SynTxt3D_50k_gen2")
    parser.add_argument("--file_range", type=list, default=[0, 50000])
    args = parser.parse_args()
    
    process_text_images(TEXT_DIR=args.text_dir, DATA_DIR=args.data_dir, file_range=args.file_range)
