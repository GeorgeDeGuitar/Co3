import pynvml
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time

# 初始化 pynvml
pynvml.nvmlInit()

# 获取 GPU 设备数量
device_count = pynvml.nvmlDeviceGetCount()
if device_count == 0:
    print("No NVIDIA GPU found. Exiting...")
    exit()

# 数据存储：每个 GPU 一个列表
timestamps = []
utilization_percents = [[] for _ in range(device_count)]  # 每个 GPU 的利用率列表
devices = []
device_names = []

# 获取所有 GPU 设备
for i in range(device_count):
    device = pynvml.nvmlDeviceGetHandleByIndex(i)
    devices.append(device)
    device_name = pynvml.nvmlDeviceGetName(device)  # 已是字符串，无需解码
    device_names.append(f"GPU {i}: {device_name}")
    print(f"Monitoring {device_names[i]}")

# 设置动态图表
fig, ax = plt.subplots(figsize=(10, 6))
ax.set_title("GPU Utilization (%)")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Utilization (%)")
ax.set_ylim(0, 100)  # 利用率范围 0-100%
ax.grid(True)

# 为每个 GPU 创建折线
lines = []
colors = ['blue', 'orange', 'green', 'red', 'purple']  # 为不同 GPU 设置颜色
for i in range(device_count):
    line, = ax.plot([], [], label=device_names[i], color=colors[i % len(colors)])
    lines.append(line)
ax.legend()

# 初始化时间
start_time = time.time()

def update(frame):
    # 获取当前时间
    current_time = time.time() - start_time
    timestamps.append(current_time)

    # 获取每个 GPU 的利用率
    for i in range(device_count):
        util = pynvml.nvmlDeviceGetUtilizationRates(devices[i]).gpu
        utilization_percents[i].append(util)

    # 限制数据长度（例如只显示最近 60 秒）
    max_points = 60
    if len(timestamps) > max_points:
        timestamps.pop(0)
        for i in range(device_count):
            utilization_percents[i].pop(0)

    # 更新每条折线
    for i in range(device_count):
        lines[i].set_data(timestamps, utilization_percents[i])
    ax.set_xlim(max(0, timestamps[-1] - max_points), timestamps[-1] + 1)

    return lines

# 动画更新（每 1000ms 更新一次）
ani = animation.FuncAnimation(fig, update, interval=1000, blit=True)

# 显示图表
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

# 清理
pynvml.nvmlShutdown()