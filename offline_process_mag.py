import os
import pandas as pd
import matplotlib.pyplot as plt
import pywt
import numpy as np
import time

file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'saved_datasets')
data_path = os.listdir(file_path)
csv_path = [f for f in data_path if os.path.splitext(f)[1] == '.csv']
df = pd.read_csv(os.path.join(file_path, csv_path[1]))

# 2. 设置小波变换参数
# wavelet 选择 'sym5' (Symlets 小波)，它的数学形态和心电图的脉冲非常像，抓心跳极其好用
wavelet = 'sym5'
level = 7  # 我们把信号过 7 层筛子

if 'Aligned_ECG' in df.columns:
    signal = df['Aligned_ECG'].values
    algo_start = time.perf_counter()
    # 1. 准备信号：去除均值（消除直流偏置，防止波形飘在天上）
    s = signal
    s = s[100:]
    s = s - np.mean(s)

    # 3. 核心分解：把信号倒进筛子！
    # 返回的 coeffs 是一个列表，结构为: [cA7, cD7, cD6, cD5, cD4, cD3, cD2, cD1]
    coeffs = pywt.wavedec(s, wavelet, level=level)

    # ---------------------------------------------------------
    # 4. 提取心跳信号 (高频，剥离低频呼吸和超高频底噪)
    # ---------------------------------------------------------
    # 我们复制一份全空的筛子 (把所有层清零)
    coeffs_heart = [np.zeros_like(c) for c in coeffs]

    # 假设你的采样率是 100Hz，经过 7 层分解：
    # cD1, cD2, cD3 通常是高频环境噪声 (> 6Hz)
    # cD4, cD5, cD6 刚好对应大约 0.8Hz 到 6Hz 左右的频段，心跳就在这里！
    # coeffs_heart[2] = coeffs[2]  # 保留 cD6
    # coeffs_heart[3] = coeffs[3]  # 保留 cD5
    coeffs_heart[4] = coeffs[4]  # 保留 cD4
    coeffs_heart[5] = coeffs[5]  # 保留 cD3
    coeffs_heart[6] = coeffs[6]  # 保留 cD2
    # 逆小波变换：拿着保留的心跳系数，倒推回时间波形
    heartbeat_clean = pywt.waverec(coeffs_heart, wavelet)

    # ---------------------------------------------------------
    # 5. 提取呼吸信号 (低频，剥离高频心跳)
    # ---------------------------------------------------------
    coeffs_breath = [np.zeros_like(c) for c in coeffs]

    # 呼吸频率通常在 0.2Hz - 0.5Hz，属于非常低频的信号
    # 它们通常藏在 cA7 (最低频的近似分量) 和 cD7 (最低频的细节分量) 里
    coeffs_breath[0] = coeffs[0]  # 保留 cA7
    coeffs_breath[1] = coeffs[1]  # 保留 cD7

    # 逆小波变换：重构呼吸波形
    breath_clean = pywt.waverec(coeffs_breath, wavelet)

    algo_end = time.perf_counter()
    print(f"🚀 DWT 小波分解与重构耗时: {(algo_end - algo_start) * 1000:.3f} 毫秒")

    # 6. 画图看效果！
    plt.figure(figsize=(14, 8))

    # 原始信号
    plt.subplot(4, 1, 1)
    plt.plot(s, label='Original CSI (Subcarrier 40)', color='gray', alpha=0.7)
    plt.title("Step 1: Original Raw Signal")
    plt.legend()

    # 提取出的呼吸波
    plt.subplot(4, 1, 2)
    plt.plot(breath_clean, label='Extracted Respiration (Low Freq)', color='blue', linewidth=2)
    plt.title("Step 2: Clean Respiration Waveform")
    plt.legend()

    # 提取出的心跳波
    plt.subplot(4, 1, 3)
    plt.plot(heartbeat_clean, label='Extracted Heartbeat (Mid-High Freq)', color='red', linewidth=1.5)
    plt.title("Step 3: Clean Heartbeat Waveform")
    plt.legend()

    plt.subplot(4, 1, 4)
    ecg = df['Aligned_ECG']
    ecg = ecg[100:]
    plt.plot(ecg, label='Aligned ECG')
    plt.title("Step 4: Aligned ECG Waveform")
    plt.legend()

    plt.tight_layout()
    plt.show()