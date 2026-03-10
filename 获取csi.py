import numpy as np
import matplotlib.pyplot as plt

csi = np.fromfile(open("data"), dtype=np.complex64)
csi = csi.reshape(-1, 52)
amp_csi = np.abs(csi)
phase = np.angle(csi)
print(amp_csi.shape)

plt.figure(figsize=(14, 10))

# --- 图 1：CSI 幅度热力图 (Waterfall) ---
plt.subplot(2, 2, 1)
# aspect='auto' 确保长宽比自动适应，cmap='viridis' 是常用好看的配色
plt.imshow(amp_csi, aspect='auto', cmap='viridis', origin='lower')
plt.colorbar(label='Amplitude')
plt.title('CSI Amplitude Heatmap')
plt.xlabel('Subcarrier Index (0-51)')
plt.ylabel('Packet / Time Index')

# --- 图 2：CSI 相位热力图 ---
plt.subplot(2, 2, 2)
# 相位通常是循环的，所以使用 'hsv' 或 'twilight' 配色比较合适
plt.imshow(phase, aspect='auto', cmap='twilight', origin='lower')
plt.colorbar(label='Phase (Radians)')
plt.title('CSI Phase Heatmap')
plt.xlabel('Subcarrier Index (0-51)')
plt.ylabel('Packet / Time Index')

# --- 图 3：单帧/多帧 幅度折线图 ---
plt.subplot(2, 2, 3)
plt.plot(amp_csi[0, :], label='Packet 0 (First)', marker='.')
if amp_csi.shape[0] > 100:
    plt.plot(amp_csi[100, :], label='Packet 100', marker='.')
plt.title('CSI Amplitude over Subcarriers')
plt.xlabel('Subcarrier Index (0-51)')
plt.ylabel('Amplitude')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()

# --- 图 4：单帧/多帧 相位折线图 ---
plt.subplot(2, 2, 4)
plt.plot(phase[0, :], label='Packet 0 (First)', marker='.')
if phase.shape[0] > 100:
    plt.plot(phase[100, :], label='Packet 100', marker='.')
plt.title('CSI Phase over Subcarriers')
plt.xlabel('Subcarrier Index (0-51)')
plt.ylabel('Phase (Radians)')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()

# 自动调整子图间距并显示
plt.tight_layout()
plt.show()
