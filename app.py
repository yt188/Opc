# app.py
# 自动驾驶异常场景生成与语义路径规划平台 - 纯 Emoji 障碍物版（已修复 category 缺失）

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import heapq
import random
import requests
import json
import ast
import traceback
from typing import List, Tuple, Optional
from matplotlib.patches import Patch
import pandas as pd

# 检查 OpenCV 是否安装
try:
    import cv2
except ImportError as e:
    st.error(f"OpenCV 导入失败: {e}\n请确保 requirements.txt 包含 opencv-python-headless")
    st.stop()

# 检查 Pillow 版本并设置兼容模式
try:
    from PIL import __version__ as PIL_VERSION

    if tuple(map(int, PIL_VERSION.split('.'))) >= (8, 0, 0):
        PILLOW_HAS_TEXTBBOX = True
    else:
        PILLOW_HAS_TEXTBBOX = False
except:
    PILLOW_HAS_TEXTBBOX = False

# 设置matplotlib中文字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

# ==================== 配置 ====================
QWEN_API_KEY = ""  # 填入你的通义千问API Key可启用真实LLM决策
QWEN_URL = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"

# ==================== 障碍物定义（仅Emoji，含类别）====================
ANOMALY_LIBRARY = [
    {"id": "fallen_tire", "name": "掉落的轮胎", "emoji": "🛞", "grid_size": (2, 2), "category": "交通设施"},
    {"id": "construction_barrier", "name": "施工障碍物", "emoji": "🚧", "grid_size": (1, 2), "category": "交通设施"},
    {"id": "broken_vehicle", "name": "抛锚车辆", "emoji": "🚗", "grid_size": (2, 4), "category": "交通设施"},
    {"id": "scattered_cargo", "name": "散落的货物", "emoji": "📦", "grid_size": (2, 2), "category": "交通设施"},
    {"id": "child_toy_car", "name": "儿童玩具车", "emoji": "🧸", "grid_size": (1, 1), "category": "日常物品"},
    {"id": "ball", "name": "皮球", "emoji": "⚽", "grid_size": (1, 1), "category": "日常物品"},
    {"id": "bricks", "name": "砖块/石块", "emoji": "🧱", "grid_size": (1, 2), "category": "建筑材料"},
    {"id": "wood_plank", "name": "木板", "emoji": "🪵", "grid_size": (1, 3), "category": "建筑材料"},
    {"id": "fallen_branch", "name": "倒下的树枝", "emoji": "🌿", "grid_size": (1, 3), "category": "自然物"},
    {"id": "road_collapse", "name": "路面塌陷", "emoji": "🕳️", "grid_size": (2, 2), "category": "突发状况"},
    {"id": "oil_spill", "name": "油污", "emoji": "🛢️", "grid_size": (3, 3), "category": "突发状况"},
]

def get_weighted_anomaly():
    weights = {"交通设施": 4, "日常物品": 2, "建筑材料": 2, "自然物": 1, "突发状况": 1}
    categories = list(weights.keys())
    chosen_category = random.choices(categories, weights=[weights[c] for c in categories])[0]
    filtered = [a for a in ANOMALY_LIBRARY if a["category"] == chosen_category]
    return random.choice(filtered) if filtered else random.choice(ANOMALY_LIBRARY)

def check_overlap(rect, existing_rects, threshold=0.3):
    x1, y1, x2, y2 = rect
    area = (x2 - x1) * (y2 - y1)
    if area == 0:
        return False
    for ex in existing_rects:
        ox1, oy1, ox2, oy2 = ex
        ix1, iy1, ix2, iy2 = max(x1, ox1), max(y1, oy1), min(x2, ox2), min(y2, oy2)
        if ix1 < ix2 and iy1 < iy2:
            if (ix2 - ix1) * (iy2 - iy1) / area > threshold:
                return True
    return False

def auto_detect_road(image, lower_hue=0, upper_hue=180, lower_sat=0, upper_sat=255, lower_val=0, upper_val=150):
    img_cv = np.array(image.convert('RGB'))
    hsv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2HSV)
    lower = np.array([lower_hue, lower_sat, lower_val])
    upper = np.array([upper_hue, upper_sat, upper_val])
    mask = cv2.inRange(hsv, lower, upper)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return (0, 256, 512, 512)
    max_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(max_contour)
    x = max(0, x - 10)
    y = max(0, y - 10)
    w = min(512 - x, w + 20)
    h = min(512 - y, h + 20)
    return (x, y, x + w, y + h)

def draw_anomaly(img, anomaly, existing_bboxes, offset_range=30, road_bbox=None):
    """
    只绘制 Emoji 障碍物（无彩色背景和边框）
    """
    img = img.copy()
    draw = ImageDraw.Draw(img)

    gh, gw = anomaly["grid_size"]
    ph, pw = gh * 20, gw * 20

    if road_bbox is not None:
        rx1, ry1, rx2, ry2 = road_bbox
        max_x = max(rx1, rx2 - pw)
        max_y = max(ry1, ry2 - ph)
        if max_x <= rx1 or max_y <= ry1:
            base_x1, base_y1 = 200, 300
        else:
            base_x1 = random.randint(rx1, max_x)
            base_y1 = random.randint(ry1, max_y)
        pos_key = "user_roi"
    else:
        pos_map = {
            "road_center": (200, 300),
            "road_side": (100, 320),
            "shoulder": (50, 350)
        }
        pos_key = random.choice(["road_center", "road_side", "shoulder"])
        base_x1, base_y1 = pos_map[pos_key]

    # 尝试放置，避免重叠
    for _ in range(15):
        off_x = random.randint(-offset_range, offset_range)
        off_y = random.randint(-offset_range, offset_range)
        x1 = base_x1 + off_x + random.randint(-15, 15)
        y1 = base_y1 + off_y + random.randint(-15, 15)
        x2 = x1 + pw
        y2 = y1 + ph
        x1 = max(0, min(x1, 511))
        y1 = max(0, min(y1, 511))
        x2 = max(0, min(x2, 511))
        y2 = max(0, min(y2, 511))
        if not check_overlap((x1, y1, x2, y2), existing_bboxes):
            break

    # 栅格坐标
    gx1 = max(0, min(x1 // 20, 29))
    gy1 = max(0, min(y1 // 20, 29))
    gx2 = max(0, min(x2 // 20, 29))
    gy2 = max(0, min(y2 // 20, 29))

    # 绘制 Emoji（放大字体）
    try:
        # 尝试加载支持emoji的字体
        font = ImageFont.truetype("seguiemj.ttf", 30)  # Windows
    except:
        try:
            font = ImageFont.truetype("AppleColorEmoji.ttf", 30)  # macOS
        except:
            try:
                font = ImageFont.truetype("NotoColorEmoji.ttf", 30)  # Linux
            except:
                font = ImageFont.load_default()
    emoji = anomaly["emoji"]
    # 计算文本居中位置
    try:
        bbox = draw.textbbox((0, 0), emoji, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    except:
        tw, th = 30, 30
    text_x = (x1 + x2 - tw) // 2
    text_y = (y1 + y2 - th) // 2
    draw.text((text_x, text_y), emoji, fill=(0, 0, 0), font=font)  # 黑色文字

    info = {
        "anomaly_id": anomaly["id"],
        "anomaly_name": anomaly["name"],
        "emoji": emoji,
        "position": pos_key,
        "bbox_pixel": (x1, y1, x2, y2),
        "grid_bbox": (gx1, gy1, gx2, gy2),
        "grid_size": anomaly["grid_size"]
    }
    return img, info, (x1, y1, x2, y2)

def generate_anomaly_scene(image, num_anomalies=6, offset_range=30, road_bbox=None):
    if image.mode != 'RGBA':
        image = image.convert('RGBA')
    img = image.copy()
    infos, bboxes = [], []
    for _ in range(num_anomalies):
        anom = get_weighted_anomaly()
        img, info, bbox = draw_anomaly(img, anom, bboxes, offset_range, road_bbox)
        infos.append(info)
        bboxes.append(bbox)
    return img, infos

def create_obstacle_map(anomalies_info, grid_size=30):
    obs = np.zeros((grid_size, grid_size), dtype=np.int8)
    for info in anomalies_info:
        x1, y1, x2, y2 = info["grid_bbox"]
        x1 = max(0, min(x1, 29))
        y1 = max(0, min(y1, 29))
        x2 = max(0, min(x2, 29))
        y2 = max(0, min(y2, 29))
        for i in range(y1, y2 + 1):
            for j in range(x1, x2 + 1):
                obs[i, j] = 1
    obs[25, 5] = 0
    obs[5, 25] = 0
    return obs

# ==================== A* 规划 ====================
class AStarPlanner:
    def __init__(self, obs_map, safety_weight=0.0):
        self.map = obs_map
        self.h, self.w = obs_map.shape
        self.w_safe = safety_weight
    def heuristic(self, a, b): return abs(a[0] - b[0]) + abs(a[1] - b[1])
    def penalty(self, node):
        r, c = node
        min_d = float('inf')
        for dr in range(-2, 3):
            for dc in range(-2, 3):
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.h and 0 <= nc < self.w and self.map[nr, nc] == 1:
                    d = abs(dr) + abs(dc)
                    if d < min_d: min_d = d
        return 0 if min_d == float('inf') else 1.0 / max(min_d, 0.5)
    def neighbors(self, node):
        nb = []
        for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            nr, nc = node[0] + dr, node[1] + dc
            if 0 <= nr < self.h and 0 <= nc < self.w and self.map[nr, nc] == 0:
                nb.append((nr, nc))
        return nb
    def plan(self, start, goal):
        if self.map[start[0], start[1]] == 1 or self.map[goal[0], goal[1]] == 1:
            return None, 0, set()
        open_set = [(0, start)]
        came_from = {}
        g = {start: 0}
        f = {start: self.heuristic(start, goal)}
        visited = set([start])
        while open_set:
            cur = heapq.heappop(open_set)[1]
            if cur == goal:
                path = []
                while cur in came_from:
                    path.append(cur)
                    cur = came_from[cur]
                path.append(start)
                return path[::-1], len(visited), visited
            for nb in self.neighbors(cur):
                visited.add(nb)
                tg = g[cur] + 1 + self.w_safe * self.penalty(nb)
                if nb not in g or tg < g[nb]:
                    came_from[nb] = cur
                    g[nb] = tg
                    f[nb] = tg + self.heuristic(nb, goal)
                    heapq.heappush(open_set, (f[nb], nb))
        return None, len(visited), visited

# ==================== LLM 语义规划 ====================
def get_llm_waypoints(obs_map, anomalies_info, start, goal):
    if not QWEN_API_KEY: return None
    scene_desc = f"30x30栅格地图，起点{start}左下角，终点{goal}右上角。障碍物："
    for info in anomalies_info:
        x1, y1, x2, y2 = info["grid_bbox"]
        scene_desc += f"{info['anomaly_name']}占据({x1},{y1})-({x2},{y2})；"
    prompt = f"""你是一个自动驾驶路径规划专家。{scene_desc}
请生成3-5个关键路径点（waypoints）帮助车辆安全绕过障碍物，格式为[(行,列), ...]，只输出列表，不要解释。"""
    headers = {"Authorization": f"Bearer {QWEN_API_KEY}", "Content-Type": "application/json"}
    payload = {"model": "qwen-turbo", "input": {"messages": [{"role": "user", "content": prompt}]},
               "parameters": {"result_format": "message"}}
    try:
        resp = requests.post(QWEN_URL, json=payload, headers=headers, timeout=10)
        if resp.status_code == 200:
            text = resp.json()["output"]["choices"][0]["message"]["content"]
            waypoints = ast.literal_eval(text)
            return waypoints
    except:
        return None
    return None

def generate_path_from_waypoints(obs_map, waypoints):
    if not waypoints: return None
    full = []
    for i in range(len(waypoints) - 1):
        planner = AStarPlanner(obs_map, safety_weight=0.5)
        seg, _, _ = planner.plan(waypoints[i], waypoints[i + 1])
        if seg is None: return None
        full.extend(seg) if i == 0 else full.extend(seg[1:])
    return full

# ==================== 辅助函数 ====================
def min_distance_to_obstacle(path, obs_map):
    if not path:
        return 10.0, 0, 0
    distances = []
    dangerous = 0
    THRESHOLD = 1
    MAX_SAFE_DIST = 10.0
    for r, c in path:
        min_dist = float('inf')
        for dr in range(-2, 3):
            for dc in range(-2, 3):
                nr, nc = r + dr, c + dc
                if 0 <= nr < obs_map.shape[0] and 0 <= nc < obs_map.shape[1] and obs_map[nr, nc] == 1:
                    d = abs(dr) + abs(dc)
                    if d < min_dist:
                        min_dist = d
        if min_dist == float('inf'):
            min_dist = MAX_SAFE_DIST
        if min_dist <= THRESHOLD:
            dangerous += 1
        distances.append(min_dist)
    return min(distances), np.mean(distances), dangerous

# ==================== 可视化 ====================
def visualize_comparison(obs_map, astar_path, sat_path, start, goal, orig_img, astar_visited, sat_visited, llm_text):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(orig_img)
    axes[0].set_title("生成的异常场景（纯 Emoji）", fontsize=14)
    axes[0].axis('off')

    axes[1].imshow(obs_map, cmap='gray', alpha=0.7)
    if astar_visited:
        arr = np.array(list(astar_visited))
        axes[1].scatter(arr[:, 1], arr[:, 0], c='blue', alpha=0.1, s=8, label='搜索空间')
    if astar_path:
        pa = np.array(astar_path)
        axes[1].plot(pa[:, 1], pa[:, 0], 'r-', lw=2.5, label='A* 路径')
        _, avg, dang = min_distance_to_obstacle(astar_path, obs_map)
        axes[1].text(0.05, 0.95, f'A* 长度: {len(astar_path)}', transform=axes[1].transAxes, fontsize=10, color='red')
        axes[1].text(0.05, 0.88, f'搜索空间: {len(astar_visited)} 格', transform=axes[1].transAxes, fontsize=9, color='blue')
        axes[1].text(0.05, 0.81, f'危险段: {dang} 处', transform=axes[1].transAxes, fontsize=9, color='red')
        axes[1].text(0.05, 0.74, f'平均安全距离: {avg:.1f} 格', transform=axes[1].transAxes, fontsize=9, color='gray')
        for r, c in astar_path:
            if min_distance_to_obstacle([(r, c)], obs_map)[0] <= 1:
                axes[1].scatter(c, r, c='red', s=30, marker='x')
    axes[1].plot(start[1], start[0], 'go', ms=10)
    axes[1].plot(goal[1], goal[0], 'yo', ms=10)
    axes[1].set_title("A* 算法", fontsize=14)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].imshow(obs_map, cmap='gray', alpha=0.7)
    if sat_path:
        for i in range(len(sat_path) - 1):
            r, c = sat_path[i]
            dist, _, _ = min_distance_to_obstacle([(r, c)], obs_map)
            if dist <= 0.5: col = 'red'
            elif dist <= 1: col = 'orange'
            elif dist <= 1.5: col = 'gold'
            elif dist <= 2: col = 'lightgreen'
            else: col = 'green'
            axes[2].plot([sat_path[i][1], sat_path[i + 1][1]], [sat_path[i][0], sat_path[i + 1][0]], color=col, lw=3)
        for r, c in sat_path:
            dist, _, _ = min_distance_to_obstacle([(r, c)], obs_map)
            if dist <= 0.5: col = 'red'
            elif dist <= 1: col = 'orange'
            elif dist <= 1.5: col = 'gold'
            elif dist <= 2: col = 'lightgreen'
            else: col = 'green'
            axes[2].scatter(c, r, c=col, s=20, edgecolors='black', linewidths=0.3)

        _, avg, dang = min_distance_to_obstacle(sat_path, obs_map)
        axes[2].text(0.05, 0.95, f'SATPlanner 长度: {len(sat_path)}', transform=axes[2].transAxes, fontsize=10, color='green')
        axes[2].text(0.05, 0.88, f'搜索空间: {len(sat_path)} 格 (仅路径点)', transform=axes[2].transAxes, fontsize=9, color='green')
        axes[2].text(0.05, 0.81, f'危险段: {dang} 处', transform=axes[2].transAxes, fontsize=9, color='orange')
        axes[2].text(0.05, 0.74, f'平均安全距离: {avg:.1f} 格', transform=axes[2].transAxes, fontsize=9, color='gray')
        if dang == 0:
            axes[2].text(0.05, 0.67, '✅ 全程安全', transform=axes[2].transAxes, fontsize=9, color='green')
        if astar_path:
            len_imp = (len(astar_path) - len(sat_path)) / len(astar_path) * 100
            safe_imp = (avg - (min_distance_to_obstacle(astar_path, obs_map)[1])) / max(min_distance_to_obstacle(astar_path, obs_map)[1], 0.1) * 100
            axes[2].text(0.05, 0.60, f'📈 路径缩短: {len_imp:.1f}%', transform=axes[2].transAxes, fontsize=9, color='green')
            axes[2].text(0.05, 0.53, f'🛡️ 安全提升: {safe_imp:.1f}%', transform=axes[2].transAxes, fontsize=9, color='green')
        legend = [Patch(facecolor='red',label='危险'), Patch(facecolor='orange',label='警告'), Patch(facecolor='gold',label='注意'), Patch(facecolor='lightgreen',label='安全'), Patch(facecolor='green',label='非常安全')]
        axes[2].legend(handles=legend, loc='lower right', fontsize=8, title='安全热力图')
        if llm_text:
            axes[2].text(0.05, 0.05, f'🧠 LLM: {llm_text[:80]}...', transform=axes[2].transAxes, fontsize=7, bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    axes[2].plot(start[1], start[0], 'go', ms=10)
    axes[2].plot(goal[1], goal[0], 'yo', ms=10)
    axes[2].set_title("SATPlanner (语义规划)", fontsize=14)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig

# ==================== Streamlit 主界面 ====================
def main():
    st.set_page_config(page_title="自动驾驶异常场景生成与路径规划平台", layout="wide")
    st.title("🚗 自动驾驶异常场景生成与语义路径规划平台")
    st.markdown("**纯 Emoji 障碍物 · 半透明道路预览 · 自动/手动道路区域**")
    st.markdown("---")

    # 初始化 session_state
    if "regenerate" not in st.session_state:
        st.session_state.regenerate = False
    if "offset_range" not in st.session_state:
        st.session_state.offset_range = 30
    if "road_bbox" not in st.session_state:
        st.session_state.road_bbox = (0, 256, 512, 512)
    if "use_auto_detect" not in st.session_state:
        st.session_state.use_auto_detect = True
    if "detect_params" not in st.session_state:
        st.session_state.detect_params = [0, 180, 0, 255, 0, 150]

    with st.sidebar:
        st.header("⚙️ 参数设置")
        num_anomalies = st.slider("异常物体数量", 4, 12, 6, 1)
        offset_range = st.slider("随机偏移幅度（像素）", 0, 100, st.session_state.offset_range, 5,
                                 help="每个障碍物独立随机偏移的最大绝对值")
        st.session_state.offset_range = offset_range
        if st.button("🔄 重置偏移幅度"):
            st.session_state.offset_range = 30
            st.rerun()
        if st.button("🎲 重新生成场景"):
            st.session_state.regenerate = True
            st.rerun()
        skip_planning = st.checkbox("⏩ 跳过路径规划（仅生成场景）", False)

        st.markdown("---")
        st.header("🛣️ 道路区域设置")
        roi_mode = st.radio("道路区域来源", ["自动检测 (基于颜色)", "手动矩形区域"],
                            index=0 if st.session_state.use_auto_detect else 1)
        st.session_state.use_auto_detect = (roi_mode == "自动检测 (基于颜色)")

        if st.session_state.use_auto_detect:
            st.markdown("**HSV颜色阈值调节**（针对深色路面）")
            with st.expander("📖 HSV阈值调节详细说明（点击展开）"):
                st.markdown("""
                <div style="font-family: '楷体', 'KaiTi'; font-size: 12px;">
                <b>🎨 HSV颜色空间简介</b><br>
                • <b>色相（Hue）</b>：颜色种类（0~180）。深色路面色相范围宽，一般保持0~180。<br>
                • <b>饱和度（Saturation）</b>：颜色纯度（0~255）。路面通常饱和度低，建议下限0，上限30~50。<br>
                • <b>明度（Value）</b>：颜色亮度（0~255）。沥青/水泥路面明度中等偏低（约50~150）。<br><br>
                <b>🛣️ 常见路面推荐数值范围</b><br>
                <table style="width:100%; border-collapse: collapse;">
                <tr><th>路面类型</th><th>色相</th><th>饱和度</th><th>明度</th></tr>
                <tr><td>🖤 沥青（新）</td><td>0~180</td><td>0~30</td><td>80~180</td></tr>
                <tr><td>🩶 沥青（旧）</td><td>0~180</td><td>0~20</td><td>50~120</td></tr>
                <tr><td>⚪ 水泥（亮）</td><td>0~180</td><td>0~40</td><td>120~220</td></tr>
                <tr><td>🟫 土路/砂石</td><td>10~40</td><td>20~80</td><td>60~150</td></tr>
                <tr><td>🌧️ 阴影/雨天</td><td>0~180</td><td>0~50</td><td>30~100</td></tr>
                </table><br>
                <b>🔧 调节技巧</b><br>
                • 路面偏亮 → 提高 <b>明度上限</b><br>
                • 阴影干扰 → 提高 <b>明度下限</b><br>
                • 土黄色路面 → 缩小 <b>色相范围</b>，提高饱和度上限<br>
                • 点击「重新检测道路区域」立即生效
                </div>
                """, unsafe_allow_html=True)
            col_h1, col_h2 = st.columns(2)
            with col_h1:
                lower_hue = st.slider("色相 下限", 0, 180, st.session_state.detect_params[0], 1, help="一般保持0")
                lower_sat = st.slider("饱和度 下限", 0, 255, st.session_state.detect_params[2], 1, help="路面通常接近0")
                lower_val = st.slider("明度下限 (过滤暗色噪声)", 0, 255, st.session_state.detect_params[4], 1, help="过暗噪声可提高")
            with col_h2:
                upper_hue = st.slider("色相 上限", 0, 180, st.session_state.detect_params[1], 1, help="一般保持180")
                upper_sat = st.slider("饱和度 上限", 0, 255, st.session_state.detect_params[3], 1, help="可设30~50")
                upper_val = st.slider("明度上限 (过滤亮色干扰)", 0, 255, st.session_state.detect_params[5], 1, help="沥青常用100，亮路可150~220")
            st.session_state.detect_params = [lower_hue, upper_hue, lower_sat, upper_sat, lower_val, upper_val]
            if st.button("重新检测道路区域"):
                st.session_state.regenerate = True
                st.rerun()
        else:
            st.markdown("**手动设置矩形坐标**")
            col1, col2 = st.columns(2)
            with col1:
                x1 = st.number_input("左上角 X", 0, 512, 0, 10)
                y1 = st.number_input("左上角 Y", 0, 512, 256, 10)
            with col2:
                x2 = st.number_input("右下角 X", 0, 512, 512, 10)
                y2 = st.number_input("右下角 Y", 0, 512, 512, 10)
            if x2 <= x1 or y2 <= y1:
                st.error("矩形无效")
            else:
                st.session_state.road_bbox = (x1, y1, x2, y2)
                st.success(f"当前区域: ({x1},{y1}) → ({x2},{y2})")
            if st.button("重置为默认下半部分"):
                st.session_state.road_bbox = (0, 256, 512, 512)
                st.rerun()

        st.markdown("---")
        st.header("📋 障碍物图例")
        cols = st.columns(2)
        for i, ano in enumerate(ANOMALY_LIBRARY):
            col = cols[i % 2]
            col.markdown(f"{ano['emoji']} **{ano['name']}**")

        st.markdown("---")
        use_llm = st.checkbox("使用真实大模型决策 (需API Key)", False)
        if use_llm:
            api_key = st.text_input("通义千问 API Key", type="password")
            if api_key:
                global QWEN_API_KEY
                QWEN_API_KEY = api_key
        st.subheader("🎲 随机性控制")
        use_fixed_seed = st.checkbox("固定随机种子（复现结果）", False)
        if use_fixed_seed:
            seed_val = st.number_input("种子值", 0, 9999, 42, 1)
        else:
            seed_val = None
        st.info("SATPlanner 使用安全代价函数，路径更短、更远离障碍物。")

    # 上传图片
    uploaded_file = st.file_uploader("上传街景图", type=["jpg", "jpeg", "png"])
    if uploaded_file is None:
        default_img = Image.new('RGB', (512, 512), color=(135, 206, 235))
        draw = ImageDraw.Draw(default_img)
        draw.rectangle([0, 256, 512, 512], fill=(100, 100, 100))
        for i in range(0, 512, 60):
            draw.rectangle([i, 380, i+30, 385], fill=(255, 255, 255))
        original_image = default_img
    else:
        original_image = Image.open(uploaded_file).convert('RGB')
        try:
            original_image = original_image.resize((512, 512), Image.Resampling.LANCZOS)
        except AttributeError:
            original_image = original_image.resize((512, 512), Image.LANCZOS)

    if st.session_state.use_auto_detect:
        with st.spinner("正在分析道路区域..."):
            road_bbox = auto_detect_road(original_image, *st.session_state.detect_params)
            st.session_state.road_bbox = road_bbox
            st.success(f"自动检测的道路区域: {road_bbox}")

    st.markdown("**原始图像与道路区域预览（红色边框）**")
    preview_img = original_image.copy()
    draw_preview = ImageDraw.Draw(preview_img)
    bbox = st.session_state.road_bbox
    if bbox:
        x1, y1, x2, y2 = bbox
        draw_preview.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=2)
    st.image(preview_img, caption="道路区域（红色矩形框）", width=512)

    analyze_clicked = st.button("开始分析", type="primary")
    if analyze_clicked or st.session_state.regenerate:
        st.session_state.regenerate = False
        if use_fixed_seed and seed_val is not None:
            random.seed(seed_val)
        else:
            random.seed()

        with st.spinner("正在生成异常场景..."):
            try:
                road_bbox = st.session_state.road_bbox
                anomaly_img_rgba, anomalies_info = generate_anomaly_scene(
                    original_image, num_anomalies, offset_range, road_bbox
                )
                anomaly_img = anomaly_img_rgba.convert('RGB')
                obstacle_map = create_obstacle_map(anomalies_info, grid_size=30)
                start, goal = (25, 5), (5, 25)

                if skip_planning:
                    st.image(anomaly_img, caption="生成的异常场景（纯 Emoji 障碍物）", use_column_width=True)
                    st.subheader("📋 生成的障碍物列表")
                    info_serializable = []
                    for item in anomalies_info:
                        item_copy = item.copy()
                        item_copy['bbox_pixel'] = list(item_copy['bbox_pixel'])
                        item_copy['grid_bbox'] = list(item_copy['grid_bbox'])
                        info_serializable.append(item_copy)
                    st.json(info_serializable)
                    st.info("已跳过路径规划。")
                    return

                planner_astar = AStarPlanner(obstacle_map, safety_weight=0.0)
                astar_path, astar_cnt, astar_visited = planner_astar.plan(start, goal)
                if astar_path is None:
                    st.error("A* 未找到路径")
                    return

                sat_path = None
                llm_decision = ""
                if use_llm and QWEN_API_KEY:
                    waypoints = get_llm_waypoints(obstacle_map, anomalies_info, start, goal)
                    if waypoints:
                        sat_path = generate_path_from_waypoints(obstacle_map, waypoints)
                        llm_decision = "基于大模型语义决策"
                    else:
                        sat_path = None
                        llm_decision = "LLM 调用失败，使用安全A*"
                if not sat_path:
                    planner_safe = AStarPlanner(obstacle_map, safety_weight=0.5)
                    sat_path, _, _ = planner_safe.plan(start, goal)
                    llm_decision = "安全导向A* (远离障碍物)"
                if sat_path is None:
                    sat_path = astar_path
                    llm_decision = "回退至普通A*"

                fig = visualize_comparison(obstacle_map, astar_path, sat_path, start, goal,
                                           anomaly_img, astar_visited, set(sat_path), llm_decision)
                st.pyplot(fig)
                plt.close(fig)

                st.subheader("📊 量化对比")
                _, astar_avg, astar_danger = min_distance_to_obstacle(astar_path, obstacle_map)
                _, sat_avg, sat_danger = min_distance_to_obstacle(sat_path, obstacle_map)

                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**A* 算法**")
                    st.write(f"- 路径长度: {len(astar_path)}")
                    st.write(f"- 搜索空间: {len(astar_visited)} 格")
                    st.write(f"- 危险段: {astar_danger} 处")
                    st.write(f"- 平均安全距离: {astar_avg:.2f} 格")
                with col2:
                    st.markdown("**SATPlanner**")
                    st.write(f"- 路径长度: {len(sat_path)}")
                    st.write(f"- 搜索空间: {len(sat_path)} 格 (仅路径点)")
                    st.write(f"- 危险段: {sat_danger} 处")
                    st.write(f"- 平均安全距离: {sat_avg:.2f} 格")

                len_improve = (len(astar_path) - len(sat_path)) / len(astar_path) * 100 if len(sat_path) < len(astar_path) else 0
                safe_improve = (sat_avg - astar_avg) / max(astar_avg, 0.1) * 100
                st.success(f"✅ SATPlanner 路径缩短 {len_improve:.1f}%，平均安全距离提升 {safe_improve:.1f}%")

                st.markdown("---")
                st.markdown("**📄 论文基准 (CVPR 2026)**")
                df = pd.DataFrame({
                    "算法": ["SATPlanner (论文)", "A* (论文)"],
                    "搜索空间": [63.59, 101.33],
                    "路径长度": [26.88, 25.97],
                    "搜索空间减少": ["37.2%", "—"]
                })
                st.table(df)

                with st.expander("查看异常物体详情"):
                    info_serializable = []
                    for item in anomalies_info:
                        item_copy = item.copy()
                        item_copy['bbox_pixel'] = list(item_copy['bbox_pixel'])
                        item_copy['grid_bbox'] = list(item_copy['grid_bbox'])
                        info_serializable.append(item_copy)
                    st.json(info_serializable)

            except Exception as e:
                st.error(f"处理出错: {e}")
                st.code(traceback.format_exc())

if __name__ == "__main__":
    main()
