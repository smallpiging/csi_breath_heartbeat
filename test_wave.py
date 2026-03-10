# import numpy as np
# import pywt
# import matplotlib.pyplot as plt
#
# # 生成一个测试信号：正弦波 + 高斯噪声
# t = np.linspace(0, 1, 1024)
# signal = np.sin(50 * np.pi * t) + 0.5 * np.sin(80 * np.pi * t) + 0.2 * np.random.randn(len(t))
#
# # 选择小波基（Daubechies 4）
# wavelet = 'db4'
#
# # 进行多层离散小波分解
# max_level = pywt.dwt_max_level(len(signal), pywt.Wavelet(wavelet).dec_len)
# coeffs = pywt.wavedec(signal, wavelet, level=max_level)
#
# # coeffs[0] 是近似系数，coeffs[1:] 是细节系数
# print(f"分解层数: {max_level}")
# for i, c in enumerate(coeffs):
#     print(f"Level {i} 系数长度: {len(c)}")
#
# # 信号重构（验证变换的可逆性）
# reconstructed_signal = pywt.waverec(coeffs, wavelet)
#
# # 绘图
# plt.figure(figsize=(10, 8))
#
# plt.subplot(max_level + 2, 1, 1)
# plt.plot(t, signal)
# plt.title("原始信号")
#
# for i, c in enumerate(coeffs):
#     plt.subplot(max_level + 2, 1, i + 2)
#     plt.plot(c)
#     if i == 0:
#         plt.title(f"近似系数 Level {max_level}")
#     else:
#         plt.title(f"细节系数 Level {max_level - i + 1}")
#
# plt.tight_layout()
# plt.show()
#
# # 验证重构误差
# error = np.linalg.norm(signal - reconstructed_signal[:len(signal)])
# print(f"重构误差: {error:.6f}")

import numpy as np
import matplotlib.pyplot as plt
import pywt

# 生成测试信号
t = np.linspace(0, 1, 400)
signal = np.cos(2 * np.pi * 7 * t) + np.sin(2 * np.pi * 13 * t)

# 选择小波基和尺度范围
wavelet = 'cmor'  # 复Morlet小波
scales = np.arange(1, 128)

# 进行连续小波变换
coefficients, frequencies = pywt.cwt(signal, scales, wavelet, sampling_period=1/400)

# 绘制时频图
plt.figure(figsize=(10, 6))
plt.imshow(np.abs(coefficients), extent=[0, 1, 1, 128], cmap='jet', aspect='auto',
           vmax=abs(coefficients).max(), vmin=0)
plt.gca().invert_yaxis()
plt.title("连续小波变换 (CWT) 时频图")
plt.xlabel("时间 (s)")
plt.ylabel("尺度 (Scale)")
plt.colorbar(label='幅值')
plt.show()
