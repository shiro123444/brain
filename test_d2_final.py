#!/usr/bin/env python3
"""D2.csv 检测 - 最终验证"""

import pandas as pd
import numpy as np
from ssvepdetect import ssvepDetect

# 加载D2.csv
data = pd.read_csv('ExampleData/D2.csv')
srate = 250

# 【关键】D2的正确频率映射（根据之前的频率分析得出）
D2_FREQ_MAP = {
    0: 16.0,   # stimID=0 → 16 Hz
    1: 18.0,   # stimID=1 → 18 Hz (前面诊断显示D2用的是18Hz，不是9Hz)
    2: 20.0,   # stimID=2 → 20 Hz (不是10Hz)
    3: 22.0,   # stimID=3 → 22 Hz (不是11Hz)
    4: 24.0,   # stimID=4 → 24 Hz (不是12Hz)
    5: 13.0,   # stimID=5 → 13 Hz (同D1)
    6: 14.0,   # stimID=6 → 14 Hz (不是10.58Hz)
    7: 15.0    # stimID=7 → 15 Hz (同D1)
}

freqs = [D2_FREQ_MAP[i] for i in range(8)]

print(f'【D2.csv 最终验证】')
print(f'使用的D2频率: {freqs}')
print()

# 初始化检测器
detector = ssvepDetect(srate, freqs, dataLen=4.0)

# 获取所有任务
task_ids = sorted(data['taskID'].unique())

correct_count = 0
total_count = 0
error_details = []

print(f'【检测结果】')
print('='*100)

for task_id in task_ids:
    mask = data['taskID'] == task_id
    task_data = data[mask]
    
    true_stim_id = int(task_data['stimID'].iloc[0])
    eeg_signal = task_data.iloc[:, :6].values
    
    samples_needed = int(4.0 * srate)
    eeg_signal = eeg_signal[:samples_needed]
    eeg_signal = eeg_signal.T
    
    try:
        predicted_index = detector.detect(eeg_signal)
        is_correct = (predicted_index == true_stim_id)
        
        result_str = '✓' if is_correct else '✗'
        print(f'  {result_str} taskID={int(task_id):2d}: 真实stimID={true_stim_id}, 预测stimID={predicted_index}')
        
        if is_correct:
            correct_count += 1
        else:
            error_details.append({
                'taskID': int(task_id),
                'true': true_stim_id,
                'pred': predicted_index
            })
        
        total_count += 1
        
    except Exception as e:
        print(f'  ✗ taskID={int(task_id):2d}: 检测失败 - {str(e)}')
        total_count += 1

print('='*100)
print()
print(f'【最终结果】')
print(f'总检测数: {total_count}')
print(f'正确数: {correct_count}')
accuracy = (correct_count / total_count * 100) if total_count > 0 else 0
print(f'准确率: {accuracy:.1f}% ({correct_count}/{total_count})')
print()

# 分析错误模式
if error_details:
    print(f'【错误分析】（共{len(error_details)}个错误）')
    
    # 统计哪些stimID经常被预测错误
    from collections import Counter
    true_ids = [e['true'] for e in error_details]
    pred_ids = [e['pred'] for e in error_details]
    
    print(f'\n错误中的真实stimID分布:')
    for stim_id, count in sorted(Counter(true_ids).items()):
        print(f'  stimID={stim_id}: {count}个错误')
    
    print(f'\n错误中的预测stimID分布:')
    for stim_id, count in sorted(Counter(pred_ids).items()):
        print(f'  stimID={stim_id}: 被预测{count}次')
    
    print(f'\n前10个错误详情:')
    for i, err in enumerate(error_details[:10], 1):
        print(f'  {i}. taskID={err["taskID"]}: 真实={err["true"]}, 预测={err["pred"]}')
else:
    print('✓ 所有预测都正确！')
