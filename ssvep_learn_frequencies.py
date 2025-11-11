#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SSVEP识别 - 从数据学习频率映射
"""

import pandas as pd
import numpy as np
from scipy.fft import fft, fftfreq
from ssvepdetect import ssvepDetect
from collections import defaultdict

def extract_segments_by_taskid(csv_file, srate=250):
    """根据taskID提取每个信号片段"""
    data = pd.read_csv(csv_file)
    
    channel_names = ['CP3', 'CPZ', 'CP4', 'PO3', 'POZ', 'PO4']
    task_ids = data['taskID'].values
    
    # 找出任务边界
    task_changes = np.concatenate(([0], np.where(np.diff(task_ids) != 0)[0] + 1, [len(task_ids)]))
    
    segments = []
    for i in range(len(task_changes) - 1):
        start_idx = task_changes[i]
        end_idx = task_changes[i + 1]
        
        segment_data = data.iloc[start_idx:end_idx][channel_names].values.T  # (6, samples)
        task_id = int(task_ids[start_idx])
        
        true_stim_id = None
        if 'stimID' in data.columns:
            true_stim_id = int(data.iloc[start_idx]['stimID'])
        
        segments.append({
            'taskID': task_id,
            'eeg_data': segment_data,
            'true_stimID': true_stim_id
        })
    
    return segments


def learn_frequency_mapping(csv_file, srate=250):
    """
    从示例数据学习每个stimID对应的频率
    使用stimID列（仅在示例数据中有）来学习频率映射
    """
    print("【从数据学习频率映射】")
    
    segments = extract_segments_by_taskid(csv_file, srate)
    
    # 统计每个stimID对应的频率
    stim_frequencies = defaultdict(list)
    
    for segment in segments:
        stim_id = segment['true_stimID']
        eeg_data = segment['eeg_data']
        
        if stim_id is None:
            continue
        
        # FFT分析
        fft_vals = np.abs(fft(eeg_data, axis=1))
        avg_fft = np.mean(fft_vals, axis=0)
        
        freqs = fftfreq(eeg_data.shape[1], 1/srate)
        
        # 只看正频率
        positive_mask = freqs > 0
        freqs = freqs[positive_mask]
        avg_fft = avg_fft[positive_mask]
        
        # 在6-30Hz范围内找最强峰值
        range_mask = (freqs >= 6) & (freqs <= 30)
        range_freqs = freqs[range_mask]
        range_fft = avg_fft[range_mask]
        
        if len(range_fft) > 0:
            peak_idx = np.argmax(range_fft)
            peak_freq = range_freqs[peak_idx]
            stim_frequencies[stim_id].append(peak_freq)
    
    # 对每个stimID，取所有频率的平均值
    freq_map = {}
    print()
    for stim_id in sorted(stim_frequencies.keys()):
        freqs = stim_frequencies[stim_id]
        avg_freq = np.median(freqs)  # 用中位数更稳健
        freq_map[stim_id] = avg_freq
        print(f"  stimID={stim_id}: {avg_freq:.2f} Hz (采样 {len(freqs)} 次)")
    
    # 转换为列表格式 [f0, f1, ..., f7]
    freq_list = [freq_map[i] for i in range(8)]
    print()
    return freq_list


def ssvep_recognize(csv_file, freq_list=None, srate=250, dataLen=4.0):
    """
    SSVEP识别
    """
    
    print("="*80)
    print("【SSVEP识别】")
    print("="*80)
    
    # 提取信号片段
    segments = extract_segments_by_taskid(csv_file, srate)
    print(f"✓ 提取了 {len(segments)} 个信号片段")
    
    # 如果没有提供频率，从数据学习
    if freq_list is None:
        freq_list = learn_frequency_mapping(csv_file, srate)
    
    print(f"使用频率: {[f'{f:.2f}' for f in freq_list]}")
    print()
    
    # 初始化检测器
    detector = ssvepDetect(srate, freq_list, dataLen)
    
    # 识别
    print("【识别结果】")
    print("-"*80)
    
    correct_count = 0
    results = []
    
    for segment in segments:
        task_id = segment['taskID']
        eeg_data = segment['eeg_data']
        true_stim_id = segment['true_stimID']
        
        # 取前dataLen秒的数据
        samples_needed = int(dataLen * srate)
        if eeg_data.shape[1] < samples_needed:
            eeg_use = eeg_data
        else:
            eeg_use = eeg_data[:, :samples_needed]
        
        # 运行CCA识别
        predicted_stim_id = detector.detect(eeg_use)
        
        # 验证
        if true_stim_id is not None:
            is_correct = (predicted_stim_id == true_stim_id)
            result = "✓" if is_correct else "✗"
            if is_correct:
                correct_count += 1
            
            print(f"{result} taskID={task_id:2d}: 预测={predicted_stim_id}, 真实={true_stim_id}")
        else:
            print(f"  taskID={task_id:2d}: 预测={predicted_stim_id}")
        
        results.append({
            'taskID': task_id,
            'predicted_stimID': predicted_stim_id,
            'true_stimID': true_stim_id
        })
    
    print("-"*80)
    print()
    
    # 统计
    if segments[0]['true_stimID'] is not None:
        accuracy = (correct_count / len(segments)) * 100
        print(f"【结果】准确率: {accuracy:.1f}% ({correct_count}/{len(segments)})")
    
    return results, freq_list


if __name__ == "__main__":
    print("\n" + "="*80)
    print("【SSVEP识别 - 从数据学习频率】")
    print("="*80 + "\n")
    
    # D1
    print("【D1.csv】\n")
    results_d1, freq_d1 = ssvep_recognize("ExampleData/D1.csv")
    
    # D2
    print("\n\n【D2.csv】\n")
    results_d2, freq_d2 = ssvep_recognize("ExampleData/D2.csv")
