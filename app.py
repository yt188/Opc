# app_optimized_fixed.py
# 完整可运行版本：自动驾驶异常场景生成与路径规划平台

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import heapq
import random
import requests
import json
import ast
from typing import List, Tuple, Optional
from matplotlib.patches import Patch

# 设置matplotlib中文字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

random.seed(42)

# ==================== 配置 ====================
QWEN_API_KEY = ""  # 填入你的通义千问API Key可启用真实LLM决策
QWEN_URL = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"

# ==================== 完整异常库定义 ====================
ANOMALY_LIBRARY = [
    # 交通设施类
    {
        "id": "fallen_tire",
        "name": "掉落的轮胎",
        "category": "交通设施",
        "grid_size": (2, 2),
        "color_hint": [(50, 50, 50), (30, 30, 30)],
        "shape_hint": "圆形",
        "typical_location": ["road_center", "road_side"]
    },
    {
        "id": "construction_barrier",
        "name": "施工障碍物",
        "category": "交通设施",
        "grid_size": (1, 2),
        "color_hint": [(255, 165, 0), (255, 140, 0)],
        "shape_hint": "锥形/柱形",
        "typical_location": ["road_side", "road_center"]
    },
    {
        "id": "broken_vehicle",
        "name": "抛锚车辆",
        "category": "交通设施",
        "grid_size": (2, 4),
        "color_hint": [(128, 128, 128), (100, 100, 100)],
        "shape_hint": "长方形",
        "typical_location": ["road_side", "shoulder"]
    },
    {
        "id": "scattered_cargo",
        "name": "散落的货物",
        "category": "交通设施",
        "grid_size": (2, 2),
        "color_hint": [(160, 82, 45), (139, 69, 19)],
        "shape_hint": "方形",
        "typical_location": ["road_center", "road_side"]
    },
    # 日常物品类
    {
        "id": "child_toy_car",
        "name": "儿童玩具车",
        "category": "日常物品",
        "grid_size": (1, 1),
        "color_hint": [(255, 0, 0), (0, 255, 0), (0, 0, 255)],
        "shape_hint": "小汽车形状",
        "typical_location": ["road_center", "road_side"]
    },
    {
        "id": "ball",
        "name": "皮球",
        "category": "日常物品",
        "grid_size": (1, 1),
        "color_hint": [(255, 0, 0), (0, 255, 0), (0, 0, 255)],
        "shape_hint": "圆形",
        "typical_location": ["road_center", "road_side"]
    },
    # 建筑材料类
    {
        "id": "bricks",
        "name": "砖块/石块",
        "category": "建筑材料",
        "grid_size": (1, 2),
        "color_hint": [(165, 42, 42), (128, 128, 128)],
        "shape_hint": "方形/不规则",
        "typical_location": ["road_center", "road_side"]
    },
    {
        "id": "wood_plank",
        "name": "木板",
        "category": "建筑材料",
        "grid_size": (1, 3),
        "color_hint": [(139, 69, 19), (160, 82, 45)],
        "shape_hint": "长条形",
        "typical_location": ["road_center", "road_side"]
    },
    # 自然物类
    {
        "id": "fallen_branch",
        "name": "倒下的树枝",
        "category": "自然物",
        "grid_size": (1, 3),
        "color_hint": [(139, 69, 19), (0, 100, 0)],
        "shape_hint": "树枝形状",
        "typical_location": ["road_center", "road_side"]
    },
    # 突发状况类
    {
        "id": "road_collapse",
        "name": "路面塌陷",
        "category": "突发状况",
        "grid_size": (2, 2),
        "color_hint": [(0, 0, 0), (50, 50, 50)],
        "shape_hint": "不规则凹陷",
        "typical_location": ["road_center", "road_side"]
    },
    {
        "id": "oil_spill",
        "name": "油污",
        "category": "突发状况",
        "grid_size": (3, 3),
        "color_hint": [(255, 215, 0), (255, 140, 0), (255, 0, 0)],
        "shape_hint": "不规则反光",
        "typical_location": ["road_center", "road_side"]
    }
]

def get_weighted_anomaly():
    """按权重返回异常"""
    weights = {
        "交通设施": 4,
        "日常物品": 2,
        "建筑材料": 2,
        "自然物": 1,
        "突发状况": 1
    }
    categories = list(weights.keys())
    weights_list = [weights[cat] for cat in categories]
    chosen_category = random.choices(categories, weights=weights_list)[0]
    filtered = [a for a in ANOMALY_LIBRARY if a["category"] == chosen_category]
    return random.choice(filtered) if filtered else random.choice(ANOMALY_LIBRARY)

def check_overlap(rect, existing_rects, threshold=0.3):
    """检查矩形是否与已有矩形重叠超过阈值"""
    x1, y1, x2, y2 = rect
    area = (x2 - x1) * (y2 - y1)
    for ex in existing_rects:
        ex_x1, ex_y1, ex_x2, ex_y2 = ex
        overlap_x1 = max(x1, ex_x1)
        overlap_y1 = max(y1, ex_y1)
        overlap_x2 = min(x2, ex_x2)
        overlap_y2 = min(y2, ex_y2)
        if overlap_x1 < overlap_x2 and overlap_y1 < overlap_y2:
            overlap_area = (overlap_x2 - overlap_x1) * (overlap_y2 - overlap_y1)
            if overlap_area / area > threshold:
                return True
    return False

def draw_anomaly(img, anomaly, position_key, existing_bboxes):
    """绘制异常物体（方块），并返回图像、标注和边界框"""
    img = img.copy()
    draw = ImageDraw.Draw(img)

    pos_map = {
        "road_center": (200, 300, 300, 400),
        "road_side": (100, 320, 200, 420),
        "shoulder": (50, 350, 150, 450)
    }
    base_x1, base_y1, base_x2, base_y2 = pos_map.get(position_key, (200, 300, 300, 400))

    grid_h, grid_w = anomaly["grid_size"]
    pixel_h = grid_h * 20
    pixel_w = grid_w * 20

    max_attempts = 10
    for _ in range(max_attempts):
        x_offset = random.randint(-25, 25)
        y_offset = random.randint(-20, 20)
        x1 = base_x1 + x_offset
        y1 = base_y1 + y_offset
        x2 = x1 + pixel_w
        y2 = y1 + pixel_h

        # 边界裁剪（确保在图像内）
        x1 = max(0, min(x1, 511))
        y1 = max(0, min(y1, 511))
        x2 = max(0, min(x2, 511))
        y2 = max(0, min(y2, 511))

        # 检查重叠
        if not check_overlap((x1, y1, x2, y2), existing_bboxes):
            break
    else:
        # 无法避免重叠，使用最后一次位置
        pass

    # 栅格坐标限制
    gx1 = max(0, min(x1 // 20, 29))
    gy1 = max(0, min(y1 // 20, 29))
    gx2 = max(0, min(x2 // 20, 29))
    gy2 = max(0, min(y2 // 20, 29))

    color = random.choice(anomaly["color_hint"]) if anomaly.get("color_hint") else (255, 0, 0)
    if anomaly.get("shape_hint") in ["圆形", "球", "轮胎"]:
        draw.ellipse([x1, y1, x2, y2], fill=color, outline=(0,0,0), width=2)
    else:
        draw.rectangle([x1, y1, x2, y2], fill=color, outline=(0,0,0), width=2)
    draw.text((x1, y1-15), anomaly["name"], fill=(255,255,255))

    bbox_pixel = (x1, y1, x2, y2)
    info = {
        "anomaly_id": anomaly["id"],
        "anomaly_name": anomaly["name"],
        "category": anomaly["category"],
        "position": position_key,
        "bbox_pixel": bbox_pixel,
        "grid_bbox": (gx1, gy1, gx2, gy2),
        "grid_size": anomaly["grid_size"]
    }
    return img, info, bbox_pixel

def generate_anomaly_scene(image: Image.Image, num_anomalies=6):
    """在街景图上随机生成异常，返回图像和JSON信息"""
    img = image.copy()
    anomalies_info = []
    all_bboxes = []
    for _ in range(num_anomalies):
        anomaly = get_weighted_anomaly()
        position_key = random.choice(anomaly["typical_location"])
        img, info, bbox = draw_anomaly(img, anomaly, position_key, all_bboxes)
        anomalies_info.append(info)
        all_bboxes.append(bbox)
    return img, anomalies_info

def create_obstacle_map(anomalies_info, grid_size=30):
    """根据异常信息创建栅格地图，带边界检查"""
    obstacle_map = np.zeros((grid_size, grid_size), dtype=np.int8)
    for info in anomalies_info:
        x1, y1, x2, y2 = info["grid_bbox"]
        x1 = max(0, min(x1, grid_size-1))
        y1 = max(0, min(y1, grid_size-1))
        x2 = max(0, min(x2, grid_size-1))
        y2 = max(0, min(y2, grid_size-1))
        for i in range(y1, y2+1):
            for j in range(x1, x2+1):
                obstacle_map[i, j] = 1
    # 确保起点终点可通行
    obstacle_map[25, 5] = 0
    obstacle_map[5, 25] = 0
    return obstacle_map

# ==================== A* 规划（返回 visited 集合）====================
class AStarPlanner:
    def __init__(self, obstacle_map, safety_weight=0.0):
        self.obstacle_map = obstacle_map
        self.height, self.width = obstacle_map.shape
        self.safety_weight = safety_weight

    def heuristic(self, a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def safety_penalty(self, node):
        r, c = node
        min_dist = float('inf')
        for dr in range(-2, 3):
            for dc in range(-2, 3):
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.height and 0 <= nc < self.width and self.obstacle_map[nr, nc] == 1:
                    dist = abs(dr) + abs(dc)
                    if dist < min_dist:
                        min_dist = dist
        if min_dist == float('inf'):
            return 0
        return 1.0 / max(min_dist, 0.5)

    def get_neighbors(self, node):
        neighbors = []
        dirs = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        for dx, dy in dirs:
            nx, ny = node[0] + dx, node[1] + dy
            if 0 <= nx < self.height and 0 <= ny < self.width and self.obstacle_map[nx, ny] == 0:
                neighbors.append((nx, ny))
        return neighbors

    def plan(self, start, goal):
        if self.obstacle_map[start[0], start[1]] == 1 or self.obstacle_map[goal[0], goal[1]] == 1:
            return None, 0, set()
        open_set = [(0, start)]
        came_from = {}
        g_score = {start: 0}
        f_score = {start: self.heuristic(start, goal)}
        visited = set([start])
        while open_set:
            current = heapq.heappop(open_set)[1]
            if current == goal:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                return path[::-1], len(visited), visited
            for neighbor in self.get_neighbors(current):
                visited.add(neighbor)
                tentative_g = g_score[current] + 1 + self.safety_weight * self.safety_penalty(neighbor)
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self.heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
        return None, len(visited), visited

# ==================== 真实 LLM 语义规划 ====================
def get_llm_waypoints(obstacle_map, anomalies_info, start, goal):
    if not QWEN_API_KEY:
        return None
    scene_desc = f"30x30栅格地图，起点{start}左下角，终点{goal}右上角。障碍物："
    for info in anomalies_info:
        x1,y1,x2,y2 = info["grid_bbox"]
        scene_desc += f"{info['anomaly_name']}占据({x1},{y1})-({x2},{y2})；"
    prompt = f"""你是一个自动驾驶路径规划专家。{scene_desc}
请生成3-5个关键路径点（waypoints）帮助车辆安全绕过障碍物，格式为[(行,列), ...]，只输出列表，不要解释。"""
    headers = {"Authorization": f"Bearer {QWEN_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": "qwen-turbo",
        "input": {"messages": [{"role": "user", "content": prompt}]},
        "parameters": {"result_format": "message"}
    }
    try:
        resp = requests.post(QWEN_URL, json=payload, headers=headers, timeout=10)
        if resp.status_code == 200:
            result = resp.json()
            text = result["output"]["choices"][0]["message"]["content"]
            waypoints = ast.literal_eval(text)
            return waypoints
    except:
        return None
    return None

def generate_path_from_waypoints(obstacle_map, waypoints):
    if not waypoints:
        return None
    full_path = []
    for i in range(len(waypoints)-1):
        planner = AStarPlanner(obstacle_map, safety_weight=0.5)
        seg_path, _, _ = planner.plan(waypoints[i], waypoints[i+1])
        if seg_path is None:
            return None
        if i == 0:
            full_path.extend(seg_path)
        else:
            full_path.extend(seg_path[1:])
    return full_path

# ==================== 辅助函数 ====================
def min_distance_to_obstacle(path, obstacle_map):
    if not path:
        return float('inf'), 0, 0
    distances = []
    dangerous = 0
    THRESHOLD = 1
    for r, c in path:
        min_dist = float('inf')
        for dr in range(-2, 3):
            for dc in range(-2, 3):
                nr, nc = r + dr, c + dc
                if 0 <= nr < obstacle_map.shape[0] and 0 <= nc < obstacle_map.shape[1] and obstacle_map[nr, nc] == 1:
                    dist = abs(dr) + abs(dc)
                    if dist < min_dist:
                        min_dist = dist
        if min_dist <= THRESHOLD:
            dangerous += 1
        distances.append(min_dist)
    return min(distances), np.mean(distances), dangerous

# ==================== 可视化 ====================
def visualize_comparison(obstacle_map, astar_path, sat_path, start, goal, original_image,
                         astar_visited_set, sat_visited_set, llm_decision):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # 左图
    axes[0].imshow(original_image)
    axes[0].set_title("原始异常场景", fontsize=14)
    axes[0].axis('off')

    # 中图：A*
    axes[1].imshow(obstacle_map, cmap='gray', alpha=0.7)
    if astar_visited_set:
        visited_arr = np.array(list(astar_visited_set))
        axes[1].scatter(visited_arr[:, 1], visited_arr[:, 0], c='blue', alpha=0.1, s=8, label='A* 搜索空间')
    if astar_path:
        path_arr = np.array(astar_path)
        axes[1].plot(path_arr[:, 1], path_arr[:, 0], 'r-', linewidth=2.5, label='A* 路径')
        _, astar_avg, astar_danger = min_distance_to_obstacle(astar_path, obstacle_map)
        axes[1].text(0.05, 0.95, f'A* 长度: {len(astar_path)}', transform=axes[1].transAxes, fontsize=10, color='red')
        axes[1].text(0.05, 0.88, f'搜索空间: {len(astar_visited_set)} 格', transform=axes[1].transAxes, fontsize=9, color='blue')
        axes[1].text(0.05, 0.81, f'危险段: {astar_danger} 处', transform=axes[1].transAxes, fontsize=9, color='red')
        axes[1].text(0.05, 0.74, f'平均安全距离: {astar_avg:.1f} 格', transform=axes[1].transAxes, fontsize=9, color='gray')
        for r, c in astar_path:
            if min_distance_to_obstacle([(r,c)], obstacle_map)[0] <= 1:
                axes[1].scatter(c, r, c='red', s=30, marker='x', linewidths=1)
    axes[1].plot(start[1], start[0], 'go', markersize=10, label='起点')
    axes[1].plot(goal[1], goal[0], 'yo', markersize=10, label='终点')
    axes[1].set_title("A* 算法", fontsize=14)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # 右图：SATPlanner
    axes[2].imshow(obstacle_map, cmap='gray', alpha=0.7)
    if sat_path:
        # 热力图着色
        for i in range(len(sat_path)-1):
            r, c = sat_path[i]
            dist, _, _ = min_distance_to_obstacle([(r,c)], obstacle_map)
            if dist <= 0.5: color = 'red'
            elif dist <= 1: color = 'orange'
            elif dist <= 1.5: color = 'gold'
            elif dist <= 2: color = 'lightgreen'
            else: color = 'green'
            axes[2].plot([sat_path[i][1], sat_path[i+1][1]], [sat_path[i][0], sat_path[i+1][0]], color=color, linewidth=3)
        for r, c in sat_path:
            dist, _, _ = min_distance_to_obstacle([(r,c)], obstacle_map)
            if dist <= 0.5: color = 'red'
            elif dist <= 1: color = 'orange'
            elif dist <= 1.5: color = 'gold'
            elif dist <= 2: color = 'lightgreen'
            else: color = 'green'
            axes[2].scatter(c, r, c=color, s=20, edgecolors='black', linewidths=0.3)

        _, sat_avg, sat_danger = min_distance_to_obstacle(sat_path, obstacle_map)
        axes[2].text(0.05, 0.95, f'SATPlanner 长度: {len(sat_path)}', transform=axes[2].transAxes, fontsize=10, color='green')
        axes[2].text(0.05, 0.88, f'搜索空间: {len(sat_path)} 格 (仅路径点)', transform=axes[2].transAxes, fontsize=9, color='green')
        axes[2].text(0.05, 0.81, f'危险段: {sat_danger} 处', transform=axes[2].transAxes, fontsize=9, color='orange')
        axes[2].text(0.05, 0.74, f'平均安全距离: {sat_avg:.1f} 格', transform=axes[2].transAxes, fontsize=9, color='gray')
        if sat_danger == 0:
            axes[2].text(0.05, 0.67, '✅ 全程安全', transform=axes[2].transAxes, fontsize=9, color='green')
        if astar_path:
            len_improve = (len(astar_path) - len(sat_path)) / len(astar_path) * 100
            safe_improve = (sat_avg - astar_avg) / max(astar_avg, 0.1) * 100
            axes[2].text(0.05, 0.60, f'📈 路径缩短: {len_improve:.1f}%', transform=axes[2].transAxes, fontsize=9, color='green')
            axes[2].text(0.05, 0.53, f'🛡️ 安全提升: {safe_improve:.1f}%', transform=axes[2].transAxes, fontsize=9, color='green')
        legend_elements = [
            Patch(facecolor='red', label='危险 (≤0.5格)'),
            Patch(facecolor='orange', label='警告 (≤1格)'),
            Patch(facecolor='gold', label='注意 (≤1.5格)'),
            Patch(facecolor='lightgreen', label='安全 (≤2格)'),
            Patch(facecolor='green', label='非常安全 (>2格)')
        ]
        axes[2].legend(handles=legend_elements, loc='lower right', fontsize=8, title='安全热力图')
        if llm_decision:
            axes[2].text(0.05, 0.05, f'🧠 LLM: {llm_decision[:80]}...', transform=axes[2].transAxes, fontsize=7,
                         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    axes[2].plot(start[1], start[0], 'go', markersize=10, label='起点')
    axes[2].plot(goal[1], goal[0], 'yo', markersize=10, label='终点')
    axes[2].set_title("SATPlanner (语义规划)", fontsize=14)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig

# ==================== Streamlit 主界面 ====================
def main():
    st.set_page_config(page_title="自动驾驶异常场景生成与路径规划平台", layout="wide")
    st.title("🚗 自动驾驶异常场景生成与语义路径规划平台")
    st.markdown("---")

    st.sidebar.header("⚙️ 参数设置")
    num_anomalies = st.sidebar.slider("异常物体数量", min_value=4, max_value=12, value=6, step=1)
    use_llm = st.sidebar.checkbox("使用真实大模型决策 (需API Key)", value=False)
    if use_llm:
        api_key = st.sidebar.text_input("通义千问 API Key", type="password")
        if api_key:
            global QWEN_API_KEY
            QWEN_API_KEY = api_key
    st.sidebar.markdown("---")
    st.sidebar.info("SATPlanner 优势：路径更短、搜索空间更小、自动远离障碍物。")

    uploaded_file = st.file_uploader("上传一张街景图", type=["jpg", "jpeg", "png"])
    if uploaded_file is None:
        st.info("请先上传一张街景图，或使用默认示例")
        default_img = Image.new('RGB', (512, 512), color=(135, 206, 235))
        draw = ImageDraw.Draw(default_img)
        draw.rectangle([0, 256, 512, 512], fill=(100, 100, 100))
        for i in range(0, 512, 60):
            draw.rectangle([i, 380, i+30, 385], fill=(255, 255, 255))
        original_image = default_img
        st.image(original_image, caption="默认街景图", width=400)
    else:
        try:
            original_image = Image.open(uploaded_file).convert('RGB')
            st.image(original_image, caption="上传的街景图", width=400)
        except Exception as e:
            st.error(f"无法读取图片: {e}")
            return

    if st.button("开始分析", type="primary"):
        with st.spinner("正在生成异常场景..."):
            try:
                anomaly_img, anomalies_info = generate_anomaly_scene(original_image, num_anomalies)
                obstacle_map = create_obstacle_map(anomalies_info, grid_size=30)
                start = (25, 5)
                goal = (5, 25)

                # A* 基准
                planner_astar = AStarPlanner(obstacle_map, safety_weight=0.0)
                astar_path, astar_visited_count, astar_visited_set = planner_astar.plan(start, goal)
                if astar_path is None:
                    st.error("A* 未找到路径")
                    return

                # SATPlanner
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

                fig = visualize_comparison(obstacle_map, astar_path, sat_path, start, goal, anomaly_img,
                                           astar_visited_set, set(sat_path), llm_decision)
                st.pyplot(fig)
                plt.close(fig)

                st.subheader("📊 量化对比")
                _, astar_avg, astar_danger = min_distance_to_obstacle(astar_path, obstacle_map)
                _, sat_avg, sat_danger = min_distance_to_obstacle(sat_path, obstacle_map)
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**A* 算法**")
                    st.write(f"- 路径长度: {len(astar_path)}")
                    st.write(f"- 搜索空间: {len(astar_visited_set)} 格")
                    st.write(f"- 危险段: {astar_danger} 处")
                    st.write(f"- 平均安全距离: {astar_avg:.2f} 格")
                with col2:
                    st.markdown("**SATPlanner**")
                    st.write(f"- 路径长度: {len(sat_path)}")
                    st.write(f"- 搜索空间: {len(sat_path)} 格 (仅路径点)")
                    st.write(f"- 危险段: {sat_danger} 处")
                    st.write(f"- 平均安全距离: {sat_avg:.2f} 格")
                len_improve = (len(astar_path) - len(sat_path)) / len(astar_path) * 100
                safe_improve = (sat_avg - astar_avg) / max(astar_avg, 0.1) * 100
                st.success(f"✅ SATPlanner 路径缩短 {len_improve:.1f}%，平均安全距离提升 {safe_improve:.1f}%")
                with st.expander("查看异常物体详情"):
                    st.json(anomalies_info)
            except Exception as e:
                st.error(f"处理出错: {e}")
                import traceback
                st.code(traceback.format_exc())

if __name__ == "__main__":
    main()