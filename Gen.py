import os
import json
import cv2
import random
from collections import defaultdict
from wand.image import Image
from wand.font import Font
from wand.drawing import Drawing
from wand.color import Color
from tqdm import tqdm
from utils.Image_transformer import ImageTransformer
import argparse

def process_text_images(
    TEXT_FILE="test.txt",
    IMG_DIR="test_img",
    PARAM_DIR="test_para",
    split=False,
    scale_factors=[0.4, 0.5, 0.75, 1.0],
    arc_angle=[0, 120],
    position=["upper left", "upper middle", "upper right",
              "middle left", "middle", "middle right",
              "bottom left", "bottom middle", "bottom right"]
):
    """
    讀取指定文字檔，並使用預設的字型與參數生成圖像及對應 JSON 檔案。
    
    參數:
        TEXT_FILE: 文字檔路徑 (預設 "test.txt")
        IMG_DIR: 圖片儲存目錄 (預設 "test_img")
        PARAM_DIR: JSON 參數儲存目錄 (預設 "test_para")
        scale_factors: 縮放比例列表 (預設 [0.4, 0.5, 0.75, 1.0])
        arc_angle: 弧形變形角度列表 (預設 [0, 120])
        position: 貼圖位置列表 (預設包含九個位置)
    """
    # 固定字型目錄及臨時目錄
    FONT_DIR_LOWER = "fonts/english_ttf_l"
    FONT_DIR_UPPER = "fonts/english_ttf_c"
    TEMP_DIR = "temp_images"
    
    # 確保目錄存在
    os.makedirs(IMG_DIR, exist_ok=True)
    os.makedirs(PARAM_DIR, exist_ok=True)
    os.makedirs(TEMP_DIR, exist_ok=True)

    # 3D旋轉角度列表（選取前 3 個作為變化）
    rotate_3D_angle = [0, 30, 45, 60, 75, 285, 290, 300, 330]
    
    # 初始化計數器
    angle_counter = defaultdict(int)
    position_counter = defaultdict(int)
    
    def apply_arc_distortion(image, arc_angle_value):
        """對圖像應用弧形變形"""
        if arc_angle_value != 0:
            image.virtual_pixel = 'black'
            image.distort('arc', [arc_angle_value])
        return image

    def load_texts(file_path):
        """讀取文字檔中所有非空行"""
        with open(file_path, "r", encoding="utf-8") as f:
            return [line.strip() for line in f.readlines() if line.strip()]

    def get_font_paths(directory):
        """取得指定目錄下所有 .ttf 字型檔路徑"""
        return [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(".ttf")]

    def split_text(text):
        """
        將文字拆分為上下兩部分，
        當 split=False 時，回傳 (text, "")。
        """
        words = text.split()
        if not split:
            return text, ""
        if len(words) < 2:
            raise ValueError("Text is too short to split")
        mid_index = len(words) // 2
        top_text = " ".join(words[:mid_index])
        bottom_text = " ".join(words[mid_index:])
        return top_text, bottom_text

    def combine_images(image1, image2):
        """將兩張圖片上下合併"""
        max_width = max(image1.width, image2.width)
        total_height = image1.height + image2.height
        combined = Image(width=max_width, height=total_height, background="black")
        combined.composite(image1, left=(max_width - image1.width) // 2, top=0)
        combined.composite(image2, left=(max_width - image2.width) // 2, top=image1.height)
        return combined

    def create_text_image(text, font_path, max_width=800):
        """建立單行文字圖片"""
        font_size = 80  
        with Drawing() as draw:
            draw.font = font_path
            draw.font_size = font_size
            with Image(width=max_width, height=100) as img:
                metrics = draw.get_font_metrics(img, text)
                text_width = metrics.text_width
                if text_width > max_width:
                    scaling_factor = max_width / text_width
                    font_size = int(font_size * scaling_factor * 0.8)
                    draw.font_size = font_size
                img.background_color = Color('black')
                img.font = Font(font_path, font_size, color='white')
                img.caption(text, gravity='center')
                img.trim()
                return img.clone()

    def chunk_list(lst, chunk_size):
        """將列表切分成指定大小的塊"""
        for i in range(0, len(lst), chunk_size):
            yield lst[i:i + chunk_size]

    # 讀取文字檔與設定每個文字使用字型數量
    texts = load_texts(TEXT_FILE)
    fonts_per_text = 9  # 每個文字使用 9 種字型

    total_texts = len(texts)
    total_images = total_texts * fonts_per_text * 3  # 每個 text-font 組合產生 3 張圖片
    target_count_per_angle = (total_images) // len(rotate_3D_angle)
    target_count_per_position = total_images // len(position)

    for idx, text in enumerate(tqdm(texts, desc="Processing texts")):
        # 根據索引決定大小寫
        if idx % 2 == 0:
            text_processed = text.lower()
            case = "lower"
        else:
            text_processed = text.upper()
            case = "upper"
        
        # 根據文字大小寫選擇對應字型目錄
        current_font_dir = FONT_DIR_UPPER if text_processed.isupper() else FONT_DIR_LOWER
        fonts = get_font_paths(current_font_dir)
        # 為每個文字選擇一個縮放比例（確保整體一致）
        scale_factor = scale_factors[idx % len(scale_factors)]
        
        # 將字型列表分組，每組包含 fonts_per_text 個字型
        font_groups = list(chunk_list(fonts, fonts_per_text))
        font_group = font_groups[idx % len(font_groups)]
        
        for font_idx, font_path in enumerate(font_group):
            font_name = os.path.basename(font_path).replace(".ttf", "")
            # 拆分文字（若 split=False，bottom_text 為空）
            top_text, bottom_text = split_text(text_processed)
            top_img = create_text_image(top_text, font_path)
            bottom_img = create_text_image(bottom_text, font_path)
            combined_img = combine_images(top_img, bottom_img)
            
            # 選擇弧形變形角度，這裡使用函數參數 arc_angle
            current_arc_angle = arc_angle[font_idx % len(arc_angle)]
            final_img = apply_arc_distortion(combined_img, current_arc_angle)
            
            # 調整圖片寬度為 512（若不等，按比例調整高度）
            if final_img.width != 512:
                new_height = int(final_img.height * (512 / final_img.width))
                final_img.resize(width=512, height=new_height)
            else:
                new_height = final_img.height
            
            # 儲存臨時圖片
            temp_img_path = os.path.join(TEMP_DIR, f"temp_{idx}_{font_idx}.jpg")
            final_img.save(filename=temp_img_path)
            
            # 根據 position 計算目前可用的貼圖位置
            available_positions = []
            for pos in position:
                if position_counter[pos] < target_count_per_position:
                    available_positions.append(pos)
            if not available_positions:
                available_positions = position
            chosen_position = random.choice(available_positions)
            position_counter[chosen_position] += 1

            # 根據使用次數選取 3 個 3D 旋轉角度
            available_angles = []
            for angle in rotate_3D_angle:
                if angle_counter[angle] < target_count_per_angle:
                    available_angles.append(angle)
            if len(available_angles) < 3:
                available_angles = rotate_3D_angle
            selected_rotate_angles = random.sample(available_angles, 3)
            
            # 針對該字型圖片分別應用 3 種旋轉角度
            for angle in selected_rotate_angles:
                angle_counter[angle] += 1
                
                # 進行 3D 旋轉
                it = ImageTransformer(temp_img_path, (512, new_height))
                rotated_img_path = os.path.join(TEMP_DIR, f"rotated_{idx}_{font_idx}_{angle}.jpg")
                rotated_img = it.rotate_along_axis(phi=angle, dx=5)
                cv2.imwrite(rotated_img_path, rotated_img)
                with Image(filename=rotated_img_path) as rotated_wand_img:
                    rotated_img_wand = rotated_wand_img.clone()
                
                # 根據縮放比例調整旋轉後圖片尺寸
                new_width_scaled = int(rotated_img_wand.width * scale_factor)
                new_height_scaled = int(rotated_img_wand.height * scale_factor)
                rotated_img_wand.resize(new_width_scaled, new_height_scaled)
                
                # 根據貼圖位置計算左上角座標（背景尺寸固定為 512x512）
                bg_width, bg_height = 512, 512
                if chosen_position == "upper left":
                    paste_left, paste_top = 0, 0
                elif chosen_position == "upper middle":
                    paste_left, paste_top = (bg_width - new_width_scaled) // 2, 0
                elif chosen_position == "upper right":
                    paste_left, paste_top = bg_width - new_width_scaled, 0
                elif chosen_position == "middle left":
                    paste_left, paste_top = 0, (bg_height - new_height_scaled) // 2
                elif chosen_position == "middle":
                    paste_left, paste_top = (bg_width - new_width_scaled) // 2, (bg_height - new_height_scaled) // 2
                elif chosen_position == "middle right":
                    paste_left, paste_top = bg_width - new_width_scaled, (bg_height - new_height_scaled) // 2
                elif chosen_position == "bottom left":
                    paste_left, paste_top = 0, bg_height - new_height_scaled
                elif chosen_position == "bottom middle":
                    paste_left, paste_top = (bg_width - new_width_scaled) // 2, bg_height - new_height_scaled
                elif chosen_position == "bottom right":
                    paste_left, paste_top = bg_width - new_width_scaled, bg_height - new_height_scaled
                
                # 將旋轉後的圖片貼到背景上
                background = Image(width=512, height=512, background=Color('black'))
                background.composite(rotated_img_wand, left=paste_left, top=paste_top)
                
                # 計算 bounding box
                bbox = {
                    "top_left": [paste_left, paste_top],
                    "bottom_right": [paste_left + new_width_scaled, paste_top + new_height_scaled]
                }
                
                # 生成輸出檔名（此處檔名中不包含大小寫資訊）
                img_filename = f"{text_processed}_{angle}_{current_arc_angle}_{scale_factor}_{chosen_position}_{font_name}.png".replace(" ", "_")
                img_path = os.path.join(IMG_DIR, img_filename)
                json_path = os.path.join(PARAM_DIR, img_filename.replace(".png", ".json"))
                
                # 儲存圖片及對應 JSON 參數檔案
                background.save(filename=img_path)
                params = {
                    "text": text_processed,
                    "rotate_angle": angle,
                    "arc_angle": current_arc_angle,
                    "scale_factor": scale_factor,
                    "position": chosen_position,
                    "bounding_box": bbox,
                    "font": font_name, 
                    "image": img_filename,
                    "split": split,
                    "case": case
                }
                with open(json_path, "w", encoding="utf-8") as json_file:
                    json.dump(params, json_file, ensure_ascii=False, indent=4)
                
                os.remove(rotated_img_path)
            os.remove(temp_img_path)

if __name__ == "__main__":
    '''
         z
        /
       /
      /
     /
    ---------------> x
    |
    |
    |
    |
    |
    v
    y

    x axis rotate: theta
    y axis rotate: phi
    z axis rotate: gamma
    '''
    parser = argparse.ArgumentParser(description="生成帶有 3D 旋轉與弧形變形的文字圖像")
    parser.add_argument("--text_file", type=str, default="test.txt", help="path to text file for rendering")
    parser.add_argument("--img_dir", type=str, default="test_img", help="save directory for images")
    parser.add_argument("--param_dir", type=str, default="test_para", help="save directory for parameters JSON file")
    parser.add_argument("--split", type=bool, default=False, help="split text into two lines")
    parser.add_argument("--scale_factors", type=str, default="0.4,0.5,0.75,1.0", 
                        help="縮放比例，使用逗號分隔 (例如: 0.4,0.5,0.75,1.0)")
    parser.add_argument("--arc_angle", type=str, default="0,120", 
                        help="弧形變形角度，使用逗號分隔 (例如: 0,120)")
    parser.add_argument("--position", type=str, 
                        default="upper left,upper middle,upper right,middle left,middle,middle right,bottom left,bottom middle,bottom right", 
                        help="貼圖位置，使用逗號分隔 (例如: upper left,upper middle,upper right,...)")
    
    args = parser.parse_args()
    
    # 轉換參數字串成對應的 list 格式
    scale_factors = [float(s) for s in args.scale_factors.split(",")]
    arc_angles = [int(s) for s in args.arc_angle.split(",")]
    positions = [p.strip() for p in args.position.split(",")]
    
    process_text_images(
        TEXT_FILE=args.text_file,
        IMG_DIR=args.img_dir,
        PARAM_DIR=args.param_dir,
        split=args.split,
        scale_factors=scale_factors,
        arc_angle=arc_angles,
        position=positions
    )
