import numpy as np
import cv2
import math
"""
Usage:

    from utils.PYR_2_RGB import PYR_2_RGB
    rgb = PYR_2_RGB(theta, phi, gamma)

        # theta, phi, gamma 為 pitch, yaw, roll 角度
        # rgb 為一個 numpy array，形狀為 (3,)，數值為 [0, 255] 的整數

        note:
        # rotation 可以不用理會，測試用的。
"""
def normal_vector(theta: float, phi: float, gamma: float) -> np.array:
    """
    Convert pitch-yaw-roll angles to normal vector.
    此處僅使用 pitch 與 yaw（roll 不影響表面法線）。
    輸入角度單位為度。
    """
    theta = np.deg2rad(theta)
    phi = np.deg2rad(phi-70)
    gamma = np.deg2rad(gamma)
    x = np.cos(theta) * np.cos(phi)
    y = np.sin(theta)
    z = np.cos(theta) * np.sin(phi)
    return np.array([x, y, z])

def PYR_2_RGB(theta: float, phi: float, gamma: float) -> np.array:
    """
    Convert pitch-yaw-roll angles to RGB color based on the normal vector.
    將法向量的每個分量從 [-1, 1] 線性映射到 [0, 255]。
    """
    normal = normal_vector(theta, phi, gamma)
    rgb = ((normal + 1) / 2) * 255
    return rgb.astype(np.uint8)

def rotation_matrix(axis: np.array, angle_deg: float) -> np.array:
    """
    生成一個繞指定軸 (必須為單位向量) 旋轉指定角度（度）的旋轉矩陣。
    """
    angle = np.deg2rad(angle_deg)
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    u = axis.reshape(3, 1)
    I = np.eye(3)
    R = cos_a * I + (1 - cos_a) * (u @ u.T) + sin_a * np.array([[ 0, -u[2,0], u[1,0]],
                                                                   [ u[2,0], 0, -u[0,0]],
                                                                   [-u[1,0], u[0,0], 0]])
    return R

def rotate_rgb(rgb: np.array, angle_deg: float = 30) -> np.array:
    """
    將 RGB 顏色（以 [0, 255] 表示，型態 float）逆時針旋轉指定角度，
    這裡旋轉是指在 RGB 空間中以灰色軸 (1,1,1) 為旋轉軸進行旋轉。
    """
    # 先轉換到浮點數，並縮放到 [0,1]
    rgb_norm = rgb.astype(np.float32) / 255.0
    # 將 RGB 從 [0,1] 轉回到 [-1,1]（原本 mapping 用的是 (n+1)/2）
    color_vector = rgb_norm * 2 - 1

    # 定義灰色軸 (1,1,1) 並正規化
    axis = np.array([1, 1, 1], dtype=np.float32)
    axis = axis / np.linalg.norm(axis)

    # 取得旋轉矩陣
    R = rotation_matrix(axis, angle_deg)
    rotated_vector = R @ color_vector

    # 將結果從 [-1,1] 映射回 [0,1]，再乘以 255
    rotated_rgb = ((rotated_vector + 1) / 2) * 255
    # clip 到 [0,255] 並轉為 uint8
    return np.clip(rotated_rgb, 0, 255).astype(np.uint8)

if __name__ == "__main__":
    # 這裡測試使用 pitch=0, yaw=0, roll=0 得到的 RGB 顏色，
    # 然後對這個 RGB 顏色進行旋轉 60 度（逆時針），
    # 並將結果顯示在一張圖片上。
    theta = 90
    phi = 0
    gamma = 0
    rgb_color = PYR_2_RGB(theta, phi, gamma)
    # 將顏色旋轉 60 度
    rotated_color = rotate_rgb(rgb_color, angle_deg=0)

    # 產生一張 100x100 的圖片，填滿旋轉後的顏色
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[:, :] = rotated_color
    cv2.imwrite("rotated_color.png", img)
    # print("Original RGB:", rgb_color)
    print("Rotated RGB:", rotated_color)

    # 進一步畫一個球體，讓每個像素根據其法向量轉換後的顏色經過旋轉處理
    size = 200
    sphere_img = np.full((size, size, 3), 255, dtype=np.uint8)  # 背景設為白色
    center = (size - 1) / 2.0
    radius = center

    for i in range(size):
        for j in range(size):
            # 將像素映射到 [-1, 1]
            X = (j - center) / radius
            Y = -(i - center) / radius
            if X * X + Y * Y <= 1.0:
                z = math.sqrt(1.0 - X*X - Y*Y)
                # 使用原本的方法計算 pitch 與 yaw
                theta_rad = math.asin(Y)
                theta_deg = math.degrees(theta_rad)
                phi_rad = math.atan2(z, X)
                phi_deg = math.degrees(phi_rad)
                gamma = 0
                rgb_val = PYR_2_RGB(theta_deg, phi_deg, gamma)
                # 將得到的 RGB 色轉旋轉 60 度
                rotated_val = rotate_rgb(rgb_val, angle_deg=0)
                sphere_img[i, j] = rotated_val

    cv2.imwrite("sphere.png", sphere_img)
    print("已生成旋轉後球體顏色圖： sphere.png")
