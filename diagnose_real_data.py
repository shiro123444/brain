#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
正确的性能诊断：
- 原始版本 vs 优化版本在完整数据集上的真实性能对比
"""

import sys
import io
import pandas as pd
import numpy as np

if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from ssvep_optimization_framework import OptimizedSSVEPClassifier, DEFAULT_CONFIG


def extract_segments_properly(csv_file, srate=250):
    """
    正确地提取信号片段
    数据格式: 48000行，每个taskID是一个4秒的连续采样 (1000点)
    """
    df = pd.read_csv(csv_file)
    channel_cols = ['CP3', 'CPZ', 'CP4', 'PO3', 'POZ', 'PO4']
    
    eeg_data = df[channel_cols].values  # (N, 6)
    task_ids = df['taskID'].values
    stim_ids = df['stimID'].values
    
    # 按taskID分组
    X_list, y_list = [], []
    unique_tasks = sorted(set(task_ids))
    for task_id in unique_tasks:
        mask = task_ids == task_id
        segment = eeg_data[mask, :].T  # (6, n_samples)
        stim_id = stim_ids[mask][0]  # 同一task内的stimID相同
        
        X_list.append(segment)
        y_list.append(int(stim_id))
    
    return np.array(X_list), np.array(y_list)


print("\n" + "="*70)
print("  🔍 SSVEP数据规模分析与诊断")
print("="*70 + "\n")

# 正确地加载数据
print("📂 加载数据...")
print("  D1.csv: 48000行原始采样...")
X_d1, y_d1 = extract_segments_properly("ExampleData/D1.csv")
print(f"    ✓ 提取得到: {X_d1.shape} epochs")

print("  D2.csv: 48000行原始采样...")
X_d2, y_d2 = extract_segments_properly("ExampleData/D2.csv")
print(f"    ✓ 提取得到: {X_d2.shape} epochs")

# 合并
X_all = np.concatenate([X_d1, X_d2], axis=0)
y_all = np.concatenate([y_d1, y_d2], axis=0)

print(f"\n✓ 合并后数据规模:")
print(f"  - 总样本数: {X_all.shape[0]} epochs")
print(f"  - 每个样本: {X_all.shape[1]} channels × {X_all.shape[2]} samples")
print(f"  - 采样率: 250Hz")
print(f"  - 时长: {X_all.shape[2] / 250:.1f}s")
print(f"  - 类别数: {len(np.unique(y_all))}")
print(f"  - 每类样本: {X_all.shape[0] // 8} 个\n")

# 现在用完整数据测试两个版本
print("="*70)
print("测试1️⃣ : 原始算法 (仅基频+二次谐波CCA)")
print("="*70)

config_original = {
    'freq_map': DEFAULT_CONFIG['freq_map'],
    'use_fb_cca': False,
    'use_trca': False,
    'use_normalization': False,
    'use_stacking': False,
    'harmonics': 2,
}

clf_orig = OptimizedSSVEPClassifier(**config_original)
print("⏳ 训练...")
clf_orig.fit(X_all, y_all)
y_pred_orig = clf_orig.predict(X_all)
acc_orig = (y_pred_orig == y_all).mean()

print(f"✓ 在全部数据上的准确率: {acc_orig:.1%}")

# 按类别统计
print("✓ 按频率分布:")
freqs = [8, 9, 10, 11, 12, 13, 14, 15]
for i, freq in enumerate(freqs):
    mask = y_all == i
    acc_i = (y_pred_orig[mask] == y_all[mask]).mean()
    cnt = mask.sum()
    print(f"  - {freq}Hz: {acc_i:.1%} ({int(acc_i * cnt)}/{cnt})")

print("\n" + "="*70)
print("测试2️⃣ : 优化版本 (所有技术开启)")
print("="*70)

clf_opt = OptimizedSSVEPClassifier(**DEFAULT_CONFIG)
print("⏳ 训练...")
clf_opt.fit(X_all, y_all)
y_pred_opt = clf_opt.predict(X_all)
acc_opt = (y_pred_opt == y_all).mean()

print(f"✓ 在全部数据上的准确率: {acc_opt:.1%}")

# 按类别统计
print("✓ 按频率分布:")
for i, freq in enumerate(freqs):
    mask = y_all == i
    acc_i = (y_pred_opt[mask] == y_all[mask]).mean()
    cnt = mask.sum()
    print(f"  - {freq}Hz: {acc_i:.1%} ({int(acc_i * cnt)}/{cnt})")

# 对比
print("\n" + "="*70)
print("📊 性能对比")
print("="*70 + "\n")
print(f"原始算法:  {acc_orig:.1%}")
print(f"优化版本:  {acc_opt:.1%}")
print(f"差异:      {(acc_opt - acc_orig):+.1%}")

print("\n" + "="*70 + "\n")
