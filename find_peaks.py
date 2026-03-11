import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, detrend
import os
import pandas as pd
from ecgdetectors import Detectors

fs = 125
detectors = Detectors(fs)
source_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'saved_datasets')
target_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'processed_datasets')
os.makedirs(target_path, exist_ok=True)

data_path = os.listdir(source_path)
csv_path = [f for f in data_path if os.path.splitext(f)[1] == '.csv']
process_path = csv_path[2]
df = pd.read_csv(os.path.join(source_path, process_path))

def generate_gaussian_mask(signal_length, peaks, sigma=3):
    """根据精准的 R 峰索引，生成 0-1 的高斯概率热力图"""
    mask = np.zeros(signal_length, dtype=float)
    x = np.arange(signal_length)
    for peak in peaks:
        bump = np.exp(-0.3 * ((x - peak) / sigma) ** 2)
        mask = np.maximum(mask, bump)
    return mask

if 'Aligned_ECG' in df.columns:
    df = df.iloc[300:].reset_index(drop=True)
    signal = df['Aligned_ECG'].values

    # 专业库处理
    # r_peaks = detectors.wqrs_detector(signal)
    # r_peaks1 = detectors.two_average_detector(signal)
    # print(r_peaks)
    # print(r_peaks1)

    # 自己写
    # fs=100Hz 时，0.4 秒 = 40 个点。所以 distance 设为 40，防止把 T 波误认为 R 峰。
    # prominence: 突起程度（根据你真实的 ECG 幅值调整，这里随便设个阈值，比如 50）
    r_peaks, _ = find_peaks(signal, distance=40, prominence=30)

    # 构建热力图
    probability_mask = generate_gaussian_mask(len(signal), r_peaks, sigma=4)
    df['Aligned_ECG'] = probability_mask
    df.rename(columns={'Aligned_ECG': 'ECG_Heatmap_Label'}, inplace=True)

    # 删除phase
    phase_cols = [col for col in df.columns if 'phase' in col.lower()]

    if len(phase_cols) > 0:
        df.drop(columns=phase_cols, inplace=True)
        print(f"🔪 成功砍掉 {len(phase_cols)} 列相位数据，只保留幅值！")
    else:
        print("⚠️ 没找到带有 phase 字眼的列，请检查你的原始 CSV 列名哦！")
    # 保存文件
    save_path = os.path.join(target_path, process_path)
    df.to_csv(save_path, index=False)

plt.figure(figsize=(16,10))

plt.subplot(2,1,1)
plt.plot(signal)
plt.scatter(r_peaks, signal[r_peaks], color='red')

plt.subplot(2,1,2)
plt.plot(probability_mask)
plt.show()

