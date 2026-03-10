import os
import pandas as pd
import matplotlib.pyplot as plt
import pywt
import numpy as np
from scipy.signal import medfilt

file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'saved_datasets')
data_path = os.listdir(file_path)
csv_path = [f for f in data_path if os.path.splitext(f)[1] == '.csv']
df = pd.read_csv(os.path.join(file_path, csv_path[1]))
if 'Aligned_ECG' in df.columns:
    raw_phase_time = df['CSI_Phase_40'].values

    # 2. 时间维度解缠绕 (缝合时间轴上的 pi 或 2*pi 跳变)
    unwrapped_phase_time = np.unwrap(raw_phase_time)

    # 3. 中值滤波 (专杀孤立尖刺！kernel_size=5 表示每次看5个点，取中间值)
    # 这个操作会把那些突然飞出去的刺瞬间“拔”掉，而不破坏真实的波浪特征
    clean_phase_time = medfilt(unwrapped_phase_time, kernel_size=5)

    # 画出来对比一下
    plt.figure(figsize=(14, 8))
    plt.subplot(4, 1, 1)
    plt.plot(raw_phase_time, label='Raw Phase (Spikes)', color='gray', alpha=0.5)

    plt.subplot(4, 1, 2)
    plt.plot(unwrapped_phase_time, label='Cleaned Phase', color='green', linewidth=2)

    plt.subplot(4, 1, 3)
    plt.plot(clean_phase_time, label='Cleaned Phase', color='green', linewidth=2)

    plt.subplot(4, 1, 4)
    plt.plot(df['Aligned_ECG'], label='Aligned ECG', color='blue', linewidth=2)


    plt.title("Phase Signal: Before and After Temporal Cleaning")
    plt.legend()
    plt.show()
