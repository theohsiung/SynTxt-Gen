import os
import json
import cv2
from wand.image import Image
from wand.font import Font
from wand.drawing import Drawing
from wand.color import Color
from tqdm import tqdm
from utils.Image_transformer import ImageTransformer
import random
from collections import defaultdict

# 设置文件和目录
TEXT_FILE = "test.txt"
FONT_DIR_LOWER = "fonts/english_ttf_l"
FONT_DIR_UPPER = "fonts/english_ttf_c"
IMG_DIR = "test_img"
PARAM_DIR = "test_para"
TEMP_DIR = "temp_images"

# 确保目录存在
os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs(PARAM_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

# 是否拆分文字
split = False  

# 配置参数：3D旋转角度（選取前3個作為變化）、
# 縮放比例（保持不變）、
# 弧形變形角度（3個）與貼圖位置（9個）
rotate_3D_angle = [0, 30, 45, 60, 75, 285, 290, 300, 330]
scale_factors = [0.4, 0.5, 0.75, 1.0]
arc_angles = [0, 120]
positions = [
    "upper left", "upper middle", "upper right",
    "middle left", "middle", "middle right",
    "bottom left", "bottom middle", "bottom right"
]

# 添加全局计数器
angle_counter = defaultdict(int)

# 在文件开头添加position计数器
position_counter = defaultdict(int)

# 计算每个position的目标使用次数
total_images = len(texts) * fonts_per_text * 3  # 每个text-font组合会生成3张图片
target_count_per_position = total_images // len(positions)

def apply_arc_distortion(image, arc_angle):
    """对图像应用弧形变形"""
    if arc_angle != 0:
        image.virtual_pixel = 'black'
        image.distort('arc', [arc_angle])
    return image

def load_texts(file_path):
    """读取文本文件中的所有文字"""
    with open(file_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f.readlines() if line.strip()]

def get_font_paths(directory):
    """获取所有字体文件的路径"""
    return [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(".ttf")]

def split_text(text):
    """将文字拆分为两部分，确保上下长度差异不大"""
    words = text.split()
    if not split:
        return text, ""
    if split and len(words) < 2:
        assert False, "Text is too short to split"

    mid_index = len(words) // 2
    top_text = " ".join(words[:mid_index])
    bottom_text = " ".join(words[mid_index:])
    return top_text, bottom_text

def combine_images(image1, image2):
    """将两张图片上下合并"""
    max_width = max(image1.width, image2.width)
    total_height = image1.height + image2.height

    combined = Image(width=max_width, height=total_height, background="black")
    combined.composite(image1, left=(max_width - image1.width) // 2, top=0)
    combined.composite(image2, left=(max_width - image2.width) // 2, top=image1.height)
    return combined

def create_text_image(text, font_path, max_width=800):
    """创建单行文字的图像"""
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
    """将列表分割成指定大小的块"""
    for i in range(0, len(lst), chunk_size):
        yield lst[i:i + chunk_size]

if __name__ == "__main__":
    texts = load_texts(TEXT_FILE)
    fonts_per_text = 9  # 每個 text 使用 9 種字型
    
    # 计算每个角度应该出现的目标次数
    total_texts = len(texts)
    total_fonts_per_text = fonts_per_text
    target_count_per_angle = (total_texts * total_fonts_per_text * 3) // len(rotate_3D_angle)

    for idx, text in enumerate(tqdm(texts, desc="Processing texts")):
        # 根据索引决定大小写
        if idx % 2 == 0:  # 偶数索引使用小写
            text = text.lower()
            case = "lower"
        else:  # 奇数索引使用大写
            text = text.upper()
            case = "upper"
            
        current_font_dir = FONT_DIR_UPPER if text.isupper() else FONT_DIR_LOWER
        fonts = get_font_paths(current_font_dir)
        # 為每個 text 選擇一個縮放比例（保持每個 text 整體一致）
        scale_factor = scale_factors[idx % len(scale_factors)]
        
        # 遍历前 9 個字型
        fonts = get_font_paths(current_font_dir)
        font_groups = list(chunk_list(fonts, fonts_per_text))
        font_group = font_groups[idx % len(font_groups)]  # 选择对应的字体组

        for font_idx, font_path in enumerate(font_group):
        # for font_idx, font_path in enumerate(fonts[:fonts_per_text]):
            font_name = os.path.basename(font_path).replace(".ttf", "")
            # 拆分文字（若 split=False，则 bottom_text 為空）
            top_text, bottom_text = split_text(text)
            top_img = create_text_image(top_text, font_path)
            bottom_img = create_text_image(bottom_text, font_path)
            combined_img = combine_images(top_img, bottom_img)
            
            # 根据当前字型索引选取弧形变形角度（3 个角度循环）
            arc_angle = arc_angles[font_idx % len(arc_angles)]
            final_img = apply_arc_distortion(combined_img, arc_angle)
            
            # 确保宽度为 512（若不等，则按比例调整高度）
            if final_img.width != 512:
                new_height = int(final_img.height * (512 / final_img.width))
                final_img.resize(width=512, height=new_height)
            else:
                new_height = final_img.height
            
            # 保存临时图片（包含 text 索引与字型索引，以免冲突）
            temp_img_path = os.path.join(TEMP_DIR, f"temp_{idx}_{font_idx}.jpg")
            final_img.save(filename=temp_img_path)
            
            # 新的position选择逻辑
            available_positions = []
            for pos in positions:
                if position_counter[pos] < target_count_per_position:
                    available_positions.append(pos)
            
            # 如果所有position都达到目标次数，则使用所有position
            if not available_positions:
                available_positions = positions
            
            # 随机选择position
            position = random.choice(available_positions)
            
            # 根据使用次数选择3个角度
            available_angles = []
            for angle in rotate_3D_angle:
                if angle_counter[angle] < target_count_per_angle:
                    available_angles.append(angle)
            
            # 如果可用角度不足3个，则使用所有角度
            if len(available_angles) < 3:
                available_angles = rotate_3D_angle
            
            # 随机选择3个角度
            selected_rotate_angles = random.sample(available_angles, 3)
            
            # 对该字型的图像，分别应用3种3D旋转角度
            for angle in selected_rotate_angles:
                # 更新计数器
                angle_counter[angle] += 1
                
                # 进行3D旋转
                it = ImageTransformer(temp_img_path, (512, new_height))
                rotated_img_path = os.path.join(TEMP_DIR, f"rotated_{idx}_{font_idx}_{angle}.jpg")
                rotated_img = it.rotate_along_axis(phi=angle, dx=5)
                cv2.imwrite(rotated_img_path, rotated_img)
                # 重新加载旋转后的图像
                with Image(filename=rotated_img_path) as rotated_wand_img:
                    rotated_img = rotated_wand_img.clone()
                
                # 根据缩放比例调整旋转后图像的尺寸
                new_width_scaled = int(rotated_img.width * scale_factor)
                new_height_scaled = int(rotated_img.height * scale_factor)
                rotated_img.resize(new_width_scaled, new_height_scaled)
                
                # 根据选择的位置计算貼圖左上角坐标
                bg_width, bg_height = 512, 512
                if position == "upper left":
                    paste_left, paste_top = 0, 0
                elif position == "upper middle":
                    paste_left, paste_top = (bg_width - new_width_scaled) // 2, 0
                elif position == "upper right":
                    paste_left, paste_top = bg_width - new_width_scaled, 0
                elif position == "middle left":
                    paste_left, paste_top = 0, (bg_height - new_height_scaled) // 2
                elif position == "middle":
                    paste_left, paste_top = (bg_width - new_width_scaled) // 2, (bg_height - new_height_scaled) // 2
                elif position == "middle right":
                    paste_left, paste_top = bg_width - new_width_scaled, (bg_height - new_height_scaled) // 2
                elif position == "bottom left":
                    paste_left, paste_top = 0, bg_height - new_height_scaled
                elif position == "bottom middle":
                    paste_left, paste_top = (bg_width - new_width_scaled) // 2, bg_height - new_height_scaled
                elif position == "bottom right":
                    paste_left, paste_top = bg_width - new_width_scaled, bg_height - new_height_scaled
                
                # 创建背景，并将旋转后的图像贴到背景上，依据贴圖位置对齐
                background = Image(width=512, height=512, background=Color('black'))
                background.composite(rotated_img, left=paste_left, top=paste_top)
                
                # 计算 bounding box
                bbox = {
                    "top_left": [paste_left, paste_top],
                    "bottom_right": [paste_left + new_width_scaled, paste_top + new_height_scaled]
                }
                
                # 生成文件名（注意：此处文件名中不再包含大小寫資訊）
                img_filename = f"{text}_{angle}_{arc_angle}_{scale_factor}_{position}_{font_name}.png".replace(" ", "_")
                img_path = os.path.join(IMG_DIR, img_filename)
                json_path = os.path.join(PARAM_DIR, img_filename.replace(".png", ".json"))
                
                # 保存图像
                background.save(filename=img_path)
                
                # 记录 JSON 参数
                params = {
                    "text": text,
                    "rotate_angle": angle,
                    "arc_angle": arc_angle,
                    "scale_factor": scale_factor,
                    "position": position,
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

    # 可选：在程序结束时打印每个position的使用次数
    # print("\nPosition usage statistics:")
    # for pos in positions:
    #     print(f"{pos}: {position_counter[pos]}")
