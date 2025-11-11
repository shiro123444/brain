#!/usr/bin/env python3
"""D2.csv 检测脚本 - 验证算法准确率"""

import pandas as pd
import numpy as np
from ssvepdetect import ssvepDetect

# 初始化检测器
srate = 250
freqs = [16.0, 9.0, 10.0, 11.0, 12.0, 13.0, 10.58, 15.0]
detector = ssvepDetect(srate, freqs, dataLen=4.0)

# 加载D2.csv
data = pd.read_csv('ExampleData/D2.csv')
print(f'D2.csv 数据形状: {data.shape}')
print(f'列名: {list(data.columns)}')
print()

# 按 taskID 分组，每个 taskID 是一个 SSVEP 实验
task_ids = sorted(data['taskID'].unique())
print(f'共有 {len(task_ids)} 个任务/实验片段')
print()

# 记录结果
correct_count = 0
total_count = 0
results = []

print(f'【D2.csv 检测结果】')
print('='*100)

for task_id in task_ids:
    # 获取该任务的所有数据行
    mask = data['taskID'] == task_id
    task_data = data[mask]
    
    # 获取真实的 stimID（所有行应该相同）
    true_stim_id = int(task_data['stimID'].iloc[0])
    
    # 提取 EEG 信号（前6列）
    eeg_signal = task_data.iloc[:, :6].values  # shape: (N, 6)
    
    # 只使用前4秒的数据
    samples_needed = int(4.0 * srate)
    eeg_signal = eeg_signal[:samples_needed]
    
    # 转置为 (通道, 样本) 格式
    eeg_signal = eeg_signal.T
    
    try:
        # 进行检测
        predicted_index = detector.detect(eeg_signal)
        predicted_stim_id = predicted_index
        
        # 检查是否正确
        is_correct = (predicted_stim_id == true_stim_id)
        
        # 显示结果
        result_str = '✓' if is_correct else '✗'
        print(f'  {result_str} taskID={int(task_id):2d}: 真实stimID={true_stim_id}, 预测stimID={predicted_stim_id}')
        
        if is_correct:
            correct_count += 1
        
        total_count += 1
        results.append({
            'taskID': int(task_id),
            'true_stimID': true_stim_id,
            'predicted_stimID': predicted_stim_id,
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

# 显示错误的预测
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
