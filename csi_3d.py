import numpy as np
import plotly.graph_objects as go

# 1. 模拟或读取你的数据
f = np.fromfile(open("data", "rb"), dtype=np.complex64).reshape(-1, 52)

# 2. 计算幅度
amplitude = np.abs(f)

# 注意：如果你的包数量（行数）非常大（比如超过几万行），
# 3D 渲染可能会卡顿，建议截取前 1000 个包来作图，例如：
# amplitude = amplitude[:1000, :]

# 3. 创建 3D 表面图
fig = go.Figure(data=[go.Surface(z=amplitude, colorscale='Viridis')])

# 4. 设置坐标轴标签和标题
fig.update_layout(
    title='交互式 3D CSI 幅度图',
    scene=dict(
        xaxis_title='子载波 (Subcarrier Index)',
        yaxis_title='时间/数据包 (Packet Index)',
        zaxis_title='幅度 (Amplitude)'
    ),
    autosize=False,
    width=900, height=800,
    margin=dict(l=65, r=50, b=65, t=90)
)

# 5. 显示图像（会自动在浏览器中打开）
fig.show()