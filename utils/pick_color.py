import cv2
import numpy as np
import random

def get_average_color_bgr(bg):
    """
    計算背景 (BGR) 的平均顏色，回傳 (B, G, R) (float 或 np.uint8 皆可)。
    bg: shape = (H, W, 3)
    """
    # (H, W, 3) 攤平成 (H*W, 3)，對每一通道做平均
    # mean(axis=0) => 得到 [B_avg, G_avg, R_avg]
    avg_bgr = np.mean(bg.reshape(-1, 3), axis=0)
    # 為了後續計算方便，維持 float 即可；若想最後轉回整數可用 round()
    return avg_bgr  # shape=(3,)

def build_valid_color_table_single_color(
    anchor_color,
    threshold=30,
    sample_step=4,
    use_lab=False
):
    """
    建立一個以「單一錨定色」(anchor_color) 為基準、篩選出距離大於 threshold 的 BGR 顏色表。

    參數:
    - anchor_color: 形如 (B, G, R)，可為 float 或 uint8
    - threshold: 距離門檻，大於此值才視為有效顏色
    - sample_step: 降採樣步長，預設=4 => 0,4,8,...,252
    - use_lab: 若為 True，則先把 anchor_color 與候選顏色都轉成 Lab，再計算歐幾里得距離

    回傳:
    - valid_colors: shape=(N, 3) 的 np.ndarray(uint8)，表示可用之 BGR 顏色
    """

    # -- (1) 準備錨定色 (anchor_color) --
    # 若要在 Lab 空間下計算，就先把錨定色轉成 shape=(1,1,3) 後 cvtColor
    if use_lab:
        # 確保 anchor_color 是 float32
        anchor_bgr_reshaped = np.array(anchor_color, dtype=np.float32).reshape(1,1,3)
        anchor_lab = cv2.cvtColor(anchor_bgr_reshaped, cv2.COLOR_BGR2Lab).reshape(3).astype(np.float32)
    else:
        anchor_lab = np.array(anchor_color, dtype=np.float32)  # 直接用 BGR

    # -- (2) 建立候選顏色集 (降採樣) --
    color_range = np.arange(0, 256, sample_step, dtype=np.uint8)
    B, G, R = np.meshgrid(color_range, color_range, color_range, indexing='ij')
    candidate_colors = np.stack([B, G, R], axis=-1).reshape(-1, 3).astype(np.uint8)  # shape=(num_candidates,3)

    # 若要在 Lab 空間下計算，則要把候選顏色也轉成 Lab
    if use_lab:
        candidate_reshaped = candidate_colors.reshape(-1,1,3)
        candidate_lab = cv2.cvtColor(candidate_reshaped, cv2.COLOR_BGR2Lab).reshape(-1,3).astype(np.float32)
    else:
        candidate_lab = candidate_colors.astype(np.float32)

    # -- (3) 計算錨定色與所有候選的距離，篩出距離 > threshold --
    diff = candidate_lab - anchor_lab  # shape=(num_candidates, 3)
    dist_sq = np.sum(diff**2, axis=1)
    threshold_sq = threshold**2

    valid_mask = dist_sq > threshold_sq
    valid_colors = candidate_colors[valid_mask]  # shape=(K,3)
    return valid_colors

def pick_color_from_valid_table(valid_colors):
    """從 valid_colors 裡隨機挑一個 (B, G, R)。"""
    idx = random.randint(0, len(valid_colors) - 1)
    return valid_colors[idx]

def pick_random_color_not_in_bg(bg, distance_threshold=10):
    avg_color_bgr = get_average_color_bgr(bg)
    valid_colors = build_valid_color_table_single_color(
        anchor_color=avg_color_bgr,
        threshold=distance_threshold,
        sample_step=4,
        use_lab=False
    )
    random_color_bgr = pick_color_from_valid_table(valid_colors)
    return random_color_bgr


# --------------------------------------------------------------
# 以下示範整合使用
# --------------------------------------------------------------
def main():
    # 1) 讀取背景圖片 (示例)
    bg = cv2.imread("background.jpg")

    # 2) 建立可用顏色表
    #    - threshold: 與背景顏色距離 > 30 才算合格
    #    - sample_step=4: 以 4 的步長掃 RGB 空間，減少計算量
    #    - convert_to_lab=False: 直接在 BGR 空間比距離，若有需要可改 True
    valid_colors = build_valid_color_table(
        bg, 
        threshold=30,
        sample_step=4,
        convert_to_lab=False
    )

    print(f"可用顏色總數: {len(valid_colors)}")

    # 3) 假設你要產 10 萬張圖片，每張都在可用顏色表裡隨機挑一個顏色
    for i in range(10_0000):
        color_bgr = pick_color_from_valid_table(valid_colors)
        # color_bgr 形式: [B, G, R]
        # 這時就可以用這個顏色進行後續繪製、合成等操作
        # ...

    print("顏色挑選完成。")

if __name__ == "__main__":
    main()
