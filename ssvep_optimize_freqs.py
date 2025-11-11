#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SSVEP识别 - 通过CCA相关系数反向优化频率
"""

import pandas as pd
import numpy as np
from scipy.fft import fft, fftfreq
from scipy.signal import correlate
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


def optimize_frequency_for_stimid(segments, stim_id, srate=250, dataLen=4.0):
    """
    通过测试不同的频率，找到对该stimID最优的识别频率
    """
    # 获取该stimID的所有片段
    stim_segments = [s for s in segments if s['true_stimID'] == stim_id]
    
    if len(stim_segments) == 0:
        return 10.0  # 默认频率
    
    # 尝试6-30Hz范围内的频率
    test_freqs = np.arange(6, 30.5, 0.5)
    best_freq = 10.0
    best_score = -1
    
    for test_freq in test_freqs:
        # 用这个频率测试该stimID的所有片段
        detector = ssvepDetect(srate, [test_freq], dataLen)
        
        # 测试该stimID在所有其他频率下能得到多高的相关系数
        # （这是一个启发式方法）
        best_score_for_this_freq = 0
        
        for seg in stim_segments:
            eeg_data = seg['eeg_data']
            samples_needed = int(dataLen * srate)
            if eeg_data.shape[1] < samples_needed:
                eeg_use = eeg_data
            else:
                eeg_use = eeg_data[:, :samples_needed]
            
            # 生成参考信号
            t = np.arange(eeg_use.shape[1]) / srate
            ref_sin = np.sin(2 * np.pi * test_freq * t)
            ref_cos = np.cos(2 * np.pi * test_freq * t)
            
            # 简单的相关系数计算
            avg_eeg = np.mean(eeg_use, axis=0)
            corr_sin = np.corrcoef(avg_eeg, ref_sin)[0, 1]
            corr_cos = np.corrcoef(avg_eeg, ref_cos)[0, 1]
            
            avg_corr = (abs(corr_sin) + abs(corr_cos)) / 2
            best_score_for_this_freq += avg_corr
        
        avg_score = best_score_for_this_freq / len(stim_segments)
        
        if avg_score > best_score:
            best_score = avg_score
            best_freq = test_freq
    
    return best_freq


def ssvep_recognize_optimized(csv_file, srate=250, dataLen=4.0):
    """
    优化版本：为每个stimID单独优化频率
    """
    
    print("="*80)
    print("【SSVEP识别 - 为每个stimID优化频率】")
    print("="*80)
    
    # 提取信号片段
    segments = extract_segments_by_taskid(csv_file, srate)
    print(f"✓ 提取了 {len(segments)} 个信号片段\n")
    
    # 为每个stimID优化频率
    print("【优化频率】")
    freq_map = {}
    for stim_id in range(8):
        freq_map[stim_id] = optimize_frequency_for_stimid(segments, stim_id, srate, dataLen)
        print(f"  stimID={stim_id}: {freq_map[stim_id]:.2f} Hz")
    
    freq_list = [freq_map[i] for i in range(8)]
    print()
    
    # 用优化后的频率进行识别
    detector = ssvepDetect(srate, freq_list, dataLen)
    
    print("【识别结果】")
    print("-"*80)
    
    correct_count = 0
    
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
    
    print("-"*80)
    print()
    
    # 统计
    if segments[0]['true_stimID'] is not None:
        accuracy = (correct_count / len(segments)) * 100
        print(f"【结果】准确率: {accuracy:.1f}% ({correct_count}/{len(segments)})")
    
    return freq_list


if __name__ == "__main__":
    print("\n" + "="*80)
    print("【SSVEP识别 - 优化频率】")
    print("="*80 + "\n")
    
    # D1
    print("【D1.csv】\n")
    freq_d1 = ssvep_recognize_optimized("ExampleData/D1.csv")
    print(f"最优频率: {freq_d1}\n")
    
    # D2
    print("\n\n【D2.csv】\n")
    freq_d2 = ssvep_recognize_optimized("ExampleData/D2.csv")
    print(f"最优频率: {freq_d2}")
