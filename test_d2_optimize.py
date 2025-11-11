#!/usr/bin/env python3
"""测试不同窗口长度对D2.csv的影响"""

import pandas as pd
import numpy as np
from ssvepdetect import ssvepDetect

# 加载D2.csv
data = pd.read_csv('ExampleData/D2.csv')
srate = 250

# 获取所有任务
task_ids = sorted(data['taskID'].unique())

print(f'【D2.csv - 测试不同窗口长度】')
print('='*80)

# 测试不同的窗口长度
for dataLen in [2.0, 3.0, 4.0, 5.0, 6.0]:
    
    # 初始化检测器
    freqs = [16.0, 9.0, 10.0, 11.0, 12.0, 13.0, 10.58, 15.0]
    detector = ssvepDetect(srate, freqs, dataLen=dataLen)
    
    correct_count = 0
    
    for task_id in task_ids:
        mask = data['taskID'] == task_id
        task_data = data[mask]
        
        true_stim_id = int(task_data['stimID'].iloc[0])
        eeg_signal = task_data.iloc[:, :6].values
        
        # 使用指定长度的数据
        samples_needed = int(dataLen * srate)
        eeg_signal = eeg_signal[:samples_needed]
        eeg_signal = eeg_signal.T
        
        try:
            predicted_index = detector.detect(eeg_signal)
            if predicted_index == true_stim_id:
                correct_count += 1
        except:
            pass
    
    accuracy = (correct_count / len(task_ids)) * 100
    print(f'  dataLen={dataLen}秒: 准确率 {accuracy:.1f}% ({correct_count}/{len(task_ids)})')

print('='*80)
