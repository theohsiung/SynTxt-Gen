import numpy as np
import cv2
"""
Usage:

    from utils.spherical_2_rgb import spherical2RGB
    rgb, bgr = spherical2RGB(theta, phi)

        # rgb, bgr 為一個 numpy array，形狀為 (3,)，數值為 [0, 255] 的整數
"""
def spherical2RGB(theta: float, phi: float) -> np.array:
  theta = (theta + 180) % 360 - 90
  phi = -1 * ((phi + 180) % 360 - 180)
  theta = np.deg2rad(theta)
  phi = np.deg2rad(phi)

  #sphere coordinate
  nx = np.sin(theta) * np.cos(phi)
  ny = np.sin(theta) * np.sin(phi)
  nz = np.cos(theta)

  # Cartesian coordinate
  px = (1+ny)/2 * 255
  py = (1+nz)/2 * 255
  pz = (1+nx)/2 * 255

  rgb = np.array([px,py,pz], dtype=np.uint8)
  bgr = np.array([pz,py,px], dtype=np.uint8)
  return rgb, bgr

if __name__ == "__main__":
    # print center point
    immmg = np.zeros((100, 100, 3), dtype=np.uint8)
    _, bgr = spherical2RGB(theta=0, phi=0)
    immmg[:,:]= bgr
    cv2.imwrite("o_point.png", immmg)

    #print sphere show all point
    width, height = 100, 100
    img = np.zeros((height, width, 3), dtype=np.uint8)

    # 圓心與半徑（在圖像座標中）
    center_x, center_y = width // 2, height // 2
    radius = min(center_x, center_y)

    # 依照題目要求：θ 在垂直方向上 -90 (上邊) 到 90 (下邊)，
    # φ 在水平方向上 -90 (左邊) 到 90 (右邊)。
    # 圓心 (θ, φ) = (0, 0)。
    for y in range(height):
        for x in range(width):
            # 判斷該像素是否在圓內
            if (x - center_x) ** 2 + (y - center_y) ** 2 <= radius ** 2:
                # 將像素座標映射到球面座標中的 θ 與 φ 值
                # 當 y = center_y 時，θ = 0；y = center_y - radius 對應 θ = -90（上邊）
                # y = center_y + radius 對應 θ = 90（下邊）
                theta_val = (y - center_y) / radius * 90
                # 當 x = center_x 時，φ = 0；x = center_x - radius 對應 φ = -90（左邊）
                # x = center_x + radius 對應 φ = 90（右邊）
                phi_val = (x - center_x) / radius * 90

                # 取得該球面座標對應的顏色
                _, bgr = spherical2RGB(theta_val, phi_val)
                img[y, x] = bgr

    # 顯示圖像
    cv2.imwrite("Spherical Circle.png", img)