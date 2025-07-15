import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import heapq

# Constants
ROWS, COLS = 10, 10
obstacles = [(3, 3), (3, 4), (4, 3), (6, 7), (6, 8)]

def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def astar(start, goal, obstacles):
    open_set = []
    heapq.heappush(open_set, (heuristic(start, goal), 0, start, [start]))
    visited = set()

    while open_set:
        _, cost, current, path = heapq.heappop(open_set)
        if current in visited:
            continue
        visited.add(current)

        if current == goal:
            return path

        for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
            neighbor = (current[0] + dx, current[1] + dy)
            if 0 <= neighbor[0] < ROWS and 0 <= neighbor[1] < COLS and neighbor not in obstacles:
                heapq.heappush(open_set, (
                    cost + 1 + heuristic(neighbor, goal),
                    cost + 1,
                    neighbor,
                    path + [neighbor]
                ))
    return None

def plot_grid(path, start, goal):
    grid = np.zeros((ROWS, COLS))
    for obs in obstacles:
        grid[obs] = -1

    fig, ax = plt.subplots()
    ax.imshow(grid, cmap="gray_r")

    for x in range(ROWS + 1):
        ax.axhline(x - 0.5, color='lightgray', linewidth=0.5)
    for y in range(COLS + 1):
        ax.axvline(y - 0.5, color='lightgray', linewidth=0.5)

    if path:
        for (x, y) in path:
            color = 'green' if (x, y) == goal else ('blue' if (x, y) == start else 'black')
            ax.text(y, x, '●', ha='center', va='center', color=color, fontsize=12)

    ax.set_xticks([])
    ax.set_yticks([])
    st.pyplot(fig)

# --------------------- STREAMLIT UI ---------------------
st.set_page_config(page_title="AGV Route Optimization", layout="centered")

st.title("🚗 AGV Route Optimization in E-Commerce Warehouse")

col1, col2 = st.columns(2)

with col1:
    start_x = st.number_input("Start X", min_value=0, max_value=ROWS - 1, value=0)
    start_y = st.number_input("Start Y", min_value=0, max_value=COLS - 1, value=0)

with col2:
    goal_x = st.number_input("Goal X", min_value=0, max_value=ROWS - 1, value=9)
    goal_y = st.number_input("Goal Y", min_value=0, max_value=COLS - 1, value=9)

start = (int(start_x), int(start_y))
goal = (int(goal_x), int(goal_y))

if st.button("Optimize AGV Path"):
    if start == goal:
        st.warning("Start and Goal cannot be the same.")
    elif start in obstacles or goal in obstacles:
        st.error("Start or Goal is inside an obstacle!")
    else:
        path = astar(start, goal, obstacles)
        if path:
            st.success(f"Path found! Total steps: {len(path)-1}")
            plot_grid(path, start, goal)
        else:
            st.error("No path found.")
