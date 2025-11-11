#!/usr/bin/env python3
"""D2.csv 检测 - 使用D2的真实频率"""

import pandas as pd
import numpy as np
from ssvepdetect import ssvepDetect

# 加载D2.csv
data = pd.read_csv('ExampleData/D2.csv')
srate = 250

# D2.csv 的真实频率映射（根据频率分析得出）
D2_FREQ_MAP = {
    0: 16.0,   # 16 Hz
    1: 18.0,   # 18 Hz
    2: 20.0,   # 20 Hz
    3: 22.0,   # 22 Hz
    4: 24.0,   # 24 Hz
    5: 13.0,   # 13 Hz
    6: 14.0,   # 14 Hz
    7: 15.0    # 15 Hz
}

freqs = [D2_FREQ_MAP[i] for i in range(8)]

print(f'【D2.csv 检测 - 使用D2的真实频率映射】')
print(f'D2频率: {freqs}')
print()

# 初始化检测器
detector = ssvepDetect(srate, freqs, dataLen=4.0)

# 获取所有任务
task_ids = sorted(data['taskID'].unique())

correct_count = 0
total_count = 0
results = []

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
        
        total_count += 1
        results.append({
            'taskID': int(task_id),
            'true_stimID': true_stim_id,
            'predicted_stimID': predicted_index,
            'correct': is_correct
        })
        
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

errors = [r for r in results if not r['correct']]
if errors:
    print(f'【错误的预测】（共{len(errors)}个）')
    for err in errors:
        tid = err['taskID']
        true_id = err['true_stimID']
        pred_id = err['predicted_stimID']
        print(f'  taskID={tid}: 真实={true_id}, 预测={pred_id}')
else:
    print('✓ 所有预测都正确！')
