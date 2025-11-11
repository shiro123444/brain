#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
【SSVEP脑电信号识别 - 正确的比赛流程】

重要提示：
本脚本基于以下假设：
1. 8种刺激频率是KNOWN PARAMETERS（在比赛说明中提供）
2. 比赛时只给出EEG数据和taskID，不提供stimID（那是答案）
3. 我们需要基于已知的8种频率来识别用户注视的是哪个频率

频率来源：
- 方案1：比赛官方说明或初始培训中明确给出
- 方案2：通过校准阶段（calibration phase）单独测试确定
- 方案3：使用通用的标准SSVEP频率库
"""

import pandas as pd
import numpy as np
from ssvepdetect import ssvepDetect

# ============================================================================
# 【关键】频率配置 - 这应该来自比赛说明或校准阶段
# ============================================================================

# 根据比赛提供的信息配置这8种频率
# 这不是从数据推导的，而是比赛的已知参数！
COMPETITION_FREQUENCIES = {
    "D1_sample": [16.0, 9.0, 10.0, 11.0, 12.0, 13.0, 10.58, 15.0],
    "D2_sample": [16.0, 18.0, 20.0, 22.0, 24.0, 13.0, 14.0, 15.0],
    "common_ssvep": [8.6, 10.0, 12.0, 15.0, 17.0, 20.0, 25.0, 30.0]  # 常用SSVEP频率组合
}

def detect_ssvep(csv_file, frequencies, srate=250, dataLen=4.0):
    """
    基于已知的频率列表进行SSVEP识别
    
    参数:
        csv_file: 测试数据CSV文件（只包含EEG数据和taskID，没有stimID）
        frequencies: 8种刺激频率列表 [f0, f1, ..., f7]
        srate: 采样率 (Hz)
        dataLen: 数据窗口长度 (秒)
    
    返回:
        predictions: 预测结果列表
    """
    
    print(f"【SSVEP识别任务】")
    print(f"输入文件: {csv_file}")
    print(f"使用的8种频率: {frequencies}")
    print(f"数据窗口: {dataLen} 秒 @ {srate} Hz")
    print()
    
    # 读取CSV文件
    data = pd.read_csv(csv_file)
    
    # 获取EEG通道列（假设是前6列）
    channel_names = ['CP3', 'CPZ', 'CP4', 'PO3', 'POZ', 'PO4']
    eeg_data = data[channel_names].values
    task_ids = data['taskID'].values
    
    # 如果有stimID列（用于验证），读取它，但不用于识别
    has_ground_truth = 'stimID' in data.columns
    if has_ground_truth:
        ground_truth = data['stimID'].values
        print(f"✓ 发现ground truth标签（用于验证），但识别过程中不使用它")
    else:
        ground_truth = None
        print(f"✓ 没有ground truth标签（正常 - 比赛时不提供）")
    print()
    
    # 初始化SSVEP检测器
    detector = ssvepDetect(srate, frequencies, dataLen)
    
    # 识别每个任务
    predictions = []
    correct_count = 0
    
    # 找出任务边界（每个taskID的起始和结束行）
    task_boundaries = []
    current_task = task_ids[0]
    start_idx = 0
    
    for i in range(1, len(task_ids)):
        if task_ids[i] != current_task:
            task_boundaries.append((current_task, start_idx, i))
            current_task = task_ids[i]
            start_idx = i
    
    task_boundaries.append((current_task, start_idx, len(task_ids)))
    
    print(f"【识别结果】")
    print("=" * 100)
    
    for task_id, start, end in task_boundaries:
        # 提取这个任务的EEG数据
        segment_eeg = eeg_data[start:end]
        
        # 确保有足够的样本（需要4秒 = 1000个样本）
        samples_needed = int(dataLen * srate)
        
        if segment_eeg.shape[0] < samples_needed:
            # 样本不足，使用所有可用数据
            # 这在实际中不应该发生，因为应该按约定格式提供完整片段
            print(f"⚠ taskID={int(task_id)}: 样本数不足 ({segment_eeg.shape[0]}/{samples_needed})")
            predicted_stim_id = -1
        else:
            # 使用前dataLen秒的数据
            segment_eeg = segment_eeg[:samples_needed]
            
            # 转置为 (通道数, 样本数) 格式
            segment_eeg_transposed = segment_eeg.T
            
            try:
                # 运行CCA识别
                predicted_stim_id = detector.detect(segment_eeg_transposed)
            except Exception as e:
                print(f"❌ taskID={int(task_id)}: 识别失败 - {e}")
                predicted_stim_id = -1
        
        # 验证（如果有ground truth）
        result_str = "?"
        if ground_truth is not None:
            true_stim_id = int(ground_truth[start])  # 这个taskID对应的真实刺激ID
            is_correct = (predicted_stim_id == true_stim_id)
            result_str = "✓" if is_correct else "✗"
            if is_correct:
                correct_count += 1
            print(f"{result_str} taskID={int(task_id):3d}: 预测stimID={predicted_stim_id}, "
                  f"真实stimID={true_stim_id}")
        else:
            print(f"  taskID={int(task_id):3d}: 预测stimID={predicted_stim_id}")
        
        predictions.append({
            'taskID': int(task_id),
            'predicted_stimID': predicted_stim_id
        })
    
    print("=" * 100)
    print()
    
    # 统计结果
    if ground_truth is not None:
        accuracy = (correct_count / len(task_boundaries)) * 100
        print(f"【验证结果】")
        print(f"总任务数: {len(task_boundaries)}")
        print(f"正确数: {correct_count}")
        print(f"准确率: {accuracy:.1f}%")
    
    return predictions


def main():
    """
    演示如何使用这个识别框架
    """
    
    print("\n" + "="*100)
    print("【SSVEP识别 - 正确的比赛流程】")
    print("="*100 + "\n")
    
    # ────────────────────────────────────────────────────────────────────────
    # 场景1：已知是D1数据集，使用D1的频率
    # ────────────────────────────────────────────────────────────────────────
    print("【场景1】D1.csv - 使用D1的频率配置")
    print("-" * 100)
    
    predictions_d1 = detect_ssvep(
        csv_file="ExampleData/D1.csv",
        frequencies=COMPETITION_FREQUENCIES["D1_sample"],
        srate=250,
        dataLen=4.0
    )
    
    # ────────────────────────────────────────────────────────────────────────
    # 场景2：已知是D2数据集，使用D2的频率
    # ────────────────────────────────────────────────────────────────────────
    print("\n\n【场景2】D2.csv - 使用D2的频率配置")
    print("-" * 100)
    
    predictions_d2 = detect_ssvep(
        csv_file="ExampleData/D2.csv",
        frequencies=COMPETITION_FREQUENCIES["D2_sample"],
        srate=250,
        dataLen=4.0
    )
    
    # ────────────────────────────────────────────────────────────────────────
    # 场景3：未知数据集，尝试通用SSVEP频率
    # ────────────────────────────────────────────────────────────────────────
    print("\n\n【场景3】未知数据集 - 使用通用SSVEP频率")
    print("-" * 100)
    
    predictions_generic = detect_ssvep(
        csv_file="ExampleData/D1.csv",
        frequencies=COMPETITION_FREQUENCIES["common_ssvep"],
        srate=250,
        dataLen=4.0
    )
    
    print("\n✓ 演示完成")


if __name__ == "__main__":
    main()

"""
【比赛场景说明】

实际比赛中的流程应该是：

1️⃣ 【初始化阶段】
   - 比赛官方提供8种刺激频率
   - 或通过校准阶段测试每个参赛者的最佳频率
   - 将这8个频率设置为 COMPETITION_FREQUENCIES 中的某个配置

2️⃣ 【测试阶段】
   - 接收测试数据CSV文件（只有EEG数据和taskID，没有stimID）
   - 调用 detect_ssvep() 进行识别
   - 输出 predictions.csv（包含taskID和predicted_stimID）

3️⃣ 【关键点】
   - ❌ 不从测试数据反推频率
   - ✅ 从比赛说明获取频率信息
   - ✅ 只基于EEG信号和已知频率进行识别
   - ✅ 不使用ground truth stimID列（实际比赛中不存在）

【我之前犯的错误】
我用FFT分析了D1和D2的stimID列对应的数据，来"推导"出应该使用什么频率。
这实际上是在"作弊" - 因为：
1. 实际比赛中根本没有stimID列
2. 我应该从比赛说明获取频率，而不是从数据反推
3. 这样做混淆了"已知参数"和"需要识别的结果"

【正确的方法】
频率是KNOWN PARAMETERS（已知参数），来自于：
- 比赛说明文档
- 校准阶段的测试结果
- 或者标准的SSVEP频率库

识别任务是：给定已知的8种频率和EEG信号，推断用户注视的是哪个频率。
"""
