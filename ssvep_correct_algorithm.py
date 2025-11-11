#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
正确的SSVEP识别算法：
1. 根据taskID提取信号片段
2. 对每个片段进行FFT频率分析
3. 用CCA进行识别
"""

import pandas as pd
import numpy as np
from scipy.fft import fft, fftfreq
from ssvepdetect import ssvepDetect

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


def detect_frequency_from_segment(eeg_segment, srate=250, freq_range=(6, 30)):
    """从FFT检测信号片段的主频率"""
    # FFT分析
    fft_vals = np.abs(fft(eeg_segment, axis=1))
    avg_fft = np.mean(fft_vals, axis=0)
    
    freqs = fftfreq(eeg_segment.shape[1], 1/srate)
    
    # 只看正频率
    positive_mask = freqs > 0
    freqs = freqs[positive_mask]
    avg_fft = avg_fft[positive_mask]
    
    # 在频率范围内找最强峰值
    range_mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
    range_freqs = freqs[range_mask]
    range_fft = avg_fft[range_mask]
    
    if len(range_fft) > 0:
        peak_idx = np.argmax(range_fft)
        detected_freq = range_freqs[peak_idx]
    else:
        detected_freq = 10.0
    
    return detected_freq


def ssvep_recognize(csv_file, srate=250, dataLen=4.0):
    """
    正确的SSVEP识别流程：
    1. 提取每个taskID的信号片段
    2. 从FFT检测该片段的频率
    3. 用8个标准频率与CCA匹配
    4. 输出predicted_stimID
    """
    
    print("="*80)
    print("【SSVEP正确识别 - 每个片段独立检测频率】")
    print("="*80)
    
    # 步骤1：提取信号片段
    segments = extract_segments_by_taskid(csv_file, srate)
    print(f"✓ 从 {csv_file} 提取了 {len(segments)} 个信号片段\n")
    
    # 8个标准频率（用于CCA匹配）
    standard_freqs = [16.0, 9.0, 10.0, 11.0, 12.0, 13.0, 10.58, 15.0]
    
    # 对每个片段进行识别
    print("【识别过程】")
    print("-"*80)
    
    correct_count = 0
    results = []
    
    for segment in segments:
        task_id = segment['taskID']
        eeg_data = segment['eeg_data']
        true_stim_id = segment['true_stimID']
        
        # 步骤2：检测该片段的频率
        detected_freq = detect_frequency_from_segment(eeg_data, srate)
        
        # 步骤3：用8个标准频率与CCA匹配
        # 初始化带有8个标准频率的检测器
        detector = ssvepDetect(srate, standard_freqs, dataLen)
        
        # 取前dataLen秒的数据
        samples_needed = int(dataLen * srate)
        if eeg_data.shape[1] < samples_needed:
            eeg_use = eeg_data
        else:
            eeg_use = eeg_data[:, :samples_needed]
        
        # 运行CCA识别
        predicted_stim_id = detector.detect(eeg_use)
        predicted_freq = standard_freqs[predicted_stim_id]
        
        # 验证（仅用于评估，识别时不用）
        if true_stim_id is not None:
            is_correct = (predicted_stim_id == true_stim_id)
            result = "✓" if is_correct else "✗"
            if is_correct:
                correct_count += 1
            
            print(f"{result} taskID={task_id:2d}: 检测频率={detected_freq:6.2f}Hz, "
                  f"预测stimID={predicted_stim_id}, 真实stimID={true_stim_id}")
        else:
            print(f"  taskID={task_id:2d}: 检测频率={detected_freq:6.2f}Hz, "
                  f"预测stimID={predicted_stim_id}")
        
        results.append({
            'taskID': task_id,
            'detected_freq': detected_freq,
            'predicted_stimID': predicted_stim_id,
            'true_stimID': true_stim_id
        })
    
    print("-"*80)
    print()
    
    # 统计结果
    if segments[0]['true_stimID'] is not None:
        accuracy = (correct_count / len(segments)) * 100
        print(f"【结果】")
        print(f"总片段数: {len(segments)}")
        print(f"正确数: {correct_count}")
        print(f"准确率: {accuracy:.1f}%")
    
    return results


if __name__ == "__main__":
    print("\n" + "="*80)
    print("【测试正确的SSVEP识别算法】")
    print("="*80 + "\n")
    
    # 测试D1
    print("【D1.csv 测试】")
    results_d1 = ssvep_recognize("ExampleData/D1.csv")
    
    # 测试D2
    print("\n\n【D2.csv 测试】")
    results_d2 = ssvep_recognize("ExampleData/D2.csv")
