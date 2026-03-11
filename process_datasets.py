import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, detrend
import os
import pandas as pd
from ecgdetectors import Detectors
import pywt

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

def del_df(df: pd.DataFrame):
    # 删除phase信息
    phase_cols = [col for col in df.columns if 'phase' in col.lower()]
    if len(phase_cols) > 0:
        df.drop(columns=phase_cols, inplace=True)
        print(f"🔪 成功砍掉 {len(phase_cols)} 列相位数据，只保留幅值！")
    else:
        print("⚠️ 没找到带有 phase 字眼的列，请检查你的原始 CSV 列名哦！")

    # 删除其他信息
    df.drop(columns=['WIFI_Timestamp'], inplace=True)
    df.drop(columns=['WIFI_Mean_Mag'], inplace=True)
    df.drop(columns=['Aligned_Breath'], inplace=True)

def process_df_label(df: pd.DataFrame):
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

    # 画图
    plt.figure(figsize=(16, 10))
    plt.subplot(2, 1, 1)
    plt.plot(signal)
    plt.scatter(r_peaks, signal[r_peaks], color='red')
    plt.subplot(2, 1, 2)
    plt.plot(probability_mask)
    plt.show()

    return df

def process_csi_df(df: pd.DataFrame):

    # 小波变换系数
    wavelet = 'sym5'
    level = 7  # 我们把信号过 7 层筛子

    mag_cols = [col for col in df.columns if 'mag' in col.lower()]
    print(mag_cols)
    for col in mag_cols:
        print(col)
        signal = df[col].values
        # 1. 准备信号：去除均值（消除直流偏置，防止波形飘在天上）
        s = signal - np.mean(signal)

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
        df[col] = heartbeat_clean

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
        ecg = df['ECG_Heatmap_Label']
        plt.plot(ecg, label='Aligned ECG_Heatmap_Label')
        plt.title("Step 4: Aligned ECG Waveform")
        plt.legend()

        plt.tight_layout()
        plt.show()
    return df



if 'Aligned_ECG' in df.columns:
    df = process_df_label(df)
    del_df(df)
    df = process_csi_df(df)
    # 保存文件
    save_path = os.path.join(target_path, process_path)
    df.to_csv(save_path, index=False)


