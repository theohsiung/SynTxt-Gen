import numpy as np
import plotly.graph_objects as go
from flask import Flask, render_template_string, request

app = Flask(__name__)

def spherical2RGB(theta: float, phi: float) -> np.array:
    """
    將輸入的 theta 與 phi（單位：度）經過轉換後，
    計算球面上單位向量，並將其對應到 [0, 255] 的 RGB 顏色。
    """
    theta = (theta + 180) % 360 - 90
    phi = -1 * ((phi + 180) % 360 - 180)
    theta = np.deg2rad(theta)
    phi = np.deg2rad(phi)
    
    # 計算球面法向量
    nx = np.sin(theta) * np.cos(phi)
    ny = np.sin(theta) * np.sin(phi)
    nz = np.cos(theta)
    
    # 將向量分量映射到 [0, 255]
    px = (1 + ny) / 2 * 255
    py = (1 + nz) / 2 * 255
    pz = (1 + nx) / 2 * 255
    
    rgb = np.array([px, py, pz], dtype=np.uint8)
    bgr = np.array([pz, py, px], dtype=np.uint8)
    return rgb, bgr

@app.route('/')
def index():
    # 若提供 total 參數，則依據 total 設定網格解析度，否則使用預設值（約 100 個點）
    total_param = request.args.get("total", None)
    if total_param is not None:
        try:
            total = int(total_param)
            n_points = max(2, int(np.sqrt(total)))  # 每邊至少 2 個點
            n_theta = n_phi = n_points
        except ValueError:
            n_theta, n_phi = 10, 10
    else:
        try:
            n_theta = int(request.args.get("n_theta", 10))
            n_phi = int(request.args.get("n_phi", 10))
        except ValueError:
            n_theta, n_phi = 10, 10

    # 建立角度範圍，theta 與 phi 均從 -90° 到 90°
    theta_vals = np.linspace(-90, 90, n_theta)  # 垂直方向角
    phi_vals = np.linspace(-90, 90, n_phi)        # 水平方向角
    theta_grid, phi_grid = np.meshgrid(theta_vals, phi_vals)
    
    # --- 計算 3D 座標 ---
    # 為保持與 spherical2RGB() 一致，先依照相同邏輯調整角度
    theta_adj = (theta_grid + 180) % 360 - 90
    phi_adj = -1 * ((phi_grid + 180) % 360 - 180)
    theta_rad = np.deg2rad(theta_adj)
    phi_rad = np.deg2rad(phi_adj)
    
    # 球面座標 (半徑 = 1)
    x = np.sin(theta_rad) * np.cos(phi_rad)
    y = np.sin(theta_rad) * np.sin(phi_rad)
    z = np.cos(theta_rad)
    
    # 攤平成 1D 陣列供 Plotly 使用
    x_flat = x.flatten()
    y_flat = y.flatten()
    z_flat = z.flatten()
    
    # --- 計算每個頂點的顏色 ---
    vertex_colors = []
    for t, p in zip(theta_grid.flatten(), phi_grid.flatten()):
        rgb, _ = spherical2RGB(t, p)
        vertex_colors.append(f"rgb({int(rgb[0])},{int(rgb[1])},{int(rgb[2])})")
    
    # --- 使用 Scatter3d 顯示每個頂點 ---
    scatter = go.Scatter3d(
        x = x_flat,
        y = y_flat,
        z = z_flat,
        mode = 'markers',
        marker = dict(
            size = 5,      # 可調整點的大小
            color = vertex_colors,
            opacity = 1.0
        )
    )
    
    fig = go.Figure(data=[scatter])
    fig.update_layout(
        scene = dict(aspectmode='data'),
        margin = dict(l=0, r=0, b=0, t=0)
    )
    
    graph_html = fig.to_html(full_html=False)
    return render_template_string('''
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>3D 球面散點圖</title>
</head>
<body>
  <h1>3D 球面散點圖：每個頂點僅為一個點，不連成面</h1>
  <p>目前網格：n_theta={{ n_theta }}, n_phi={{ n_phi }} (共 {{ n_theta * n_phi }} 個點)</p>
  <p>如欲調整總點數，可使用 URL 參數 total，例如：?total=100</p>
  {{ graph_html|safe }}
</body>
</html>
''', graph_html=graph_html, n_theta=n_theta, n_phi=n_phi)

if __name__ == '__main__':
    app.run(debug=True)
"""usage
# on your browser

http://127.0.0.1:5000/?total=1000
    
    # total = the number of the point
    # total could be any integer

"""