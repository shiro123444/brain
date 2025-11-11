#!/usr/bin/env python3
"""自动检测数据集中每个stimID的最优频率"""

import pandas as pd
import numpy as np
from scipy.fft import fft, fftfreq
from collections import defaultdict

def analyze_frequencies(csv_file, freq_range=(8, 30), top_n=3):
    """
    分析CSV文件中每个stimID对应的频率
    
    参数:
        csv_file: CSV文件路径
        freq_range: 频率范围 (min, max)
        top_n: 返回top_n个频率
    
    返回:
        freq_map: {stimID: primary_frequency}
    """
    data = pd.read_csv(csv_file)
    srate = 250
    
    stim_frequencies = defaultdict(lambda: defaultdict(int))
    
    # 对每个task进行FFT分析
    task_ids = sorted(data['taskID'].unique())
    
    for task_id in task_ids:
        mask = data['taskID'] == task_id
        task_data = data[mask]
        stim_id = int(task_data['stimID'].iloc[0])
        
        # 提取EEG数据（6个通道）
        eeg_signal = task_data.iloc[:, :6].values.T  # (6, samples)
        
        # 取前4秒数据
        samples = int(4.0 * srate)
        eeg_signal = eeg_signal[:, :samples]
        
        # 对每个通道进行FFT
        for ch in range(6):
            signal = eeg_signal[ch]
            fft_vals = np.abs(fft(signal))
            freqs = fftfreq(len(signal), 1/srate)
            
            # 只取正频率
            positive_idx = freqs > 0
            fft_vals = fft_vals[positive_idx]
            freqs = freqs[positive_idx]
            
            # 在指定频率范围内查找峰值
            mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
            local_freqs = freqs[mask]
            local_fft = fft_vals[mask]
            
            # 找到最强的频率分量
            if len(local_fft) > 0:
                peak_idx = np.argmax(local_fft)
                peak_freq = local_freqs[peak_idx]
                # 量化到0.5Hz
                peak_freq = round(peak_freq * 2) / 2
                stim_frequencies[stim_id][peak_freq] += 1
    
    # 从每个stimID中提取最常见的频率
    freq_map = {}
    for stim_id in sorted(stim_frequencies.keys()):
        freqs_dict = stim_frequencies[stim_id]
        # 按出现次数排序
        sorted_freqs = sorted(freqs_dict.items(), key=lambda x: x[1], reverse=True)
        
        print(f'\nstimID={stim_id}:')
        for freq, count in sorted_freqs[:top_n]:
            print(f'  {freq:6.2f} Hz (出现 {count} 次)')
        
        # 取最常见的频率作为主频率
        primary_freq = sorted_freqs[0][0]
        freq_map[stim_id] = primary_freq
    
    return freq_map

if __name__ == '__main__':
    print('='*60)
    print('【D1.csv 频率分析】')
    print('='*60)
    d1_freqs = analyze_frequencies('ExampleData/D1.csv')
    
    print('\n' + '='*60)
    print('【D2.csv 频率分析】')
    print('='*60)
    d2_freqs = analyze_frequencies('ExampleData/D2.csv')
    
    print('\n' + '='*60)
    print('【总结】')
    print('='*60)
    
    print('\nD1.csv 频率映射:')
    d1_list = [d1_freqs[i] for i in range(8)]
    print(f'FREQ_MAP_D1 = {d1_list}')
    
    print('\nD2.csv 频率映射:')
    d2_list = [d2_freqs[i] for i in range(8)]
    print(f'FREQ_MAP_D2 = {d2_list}')
