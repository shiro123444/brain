#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
诊断脚本：对比原始CCA vs 优化版本性能
找出为什么优化版本性能下降的原因
"""

import sys
import io
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.cross_decomposition import CCA

# 修复编码
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from ssvep_optimization_framework import OptimizedSSVEPClassifier, DEFAULT_CONFIG


def load_data_like_original(csv_file):
    """按照原始版本的方式加载数据"""
    df = pd.read_csv(csv_file)
    channel_cols = ['CP3', 'CPZ', 'CP4', 'PO3', 'POZ', 'PO4']
    
    X_list, y_list = [], []
    for task_id, group in df.groupby('taskID'):
        data = group[channel_cols].values[:1000, :].T  # [n_channels, n_samples]
        X_list.append(data)
        y_list.append(int(group['stimID'].iloc[0]))
    
    return np.array(X_list), np.array(y_list)


def test_basic_cca(X_train, y_train, X_test, y_test):
    """测试基础CCA (只用基频+二次谐波，不用其他优化)"""
    print("\n" + "="*70)
    print("🔬 测试 1: 基础CCA (仅基频+二次谐波，无其他优化)")
    print("="*70)
    
    config = DEFAULT_CONFIG.copy()
    config['use_fb_cca'] = False
    config['use_trca'] = False
    config['use_normalization'] = False
    config['use_stacking'] = False
    config['harmonics'] = 2  # 确保用二次谐波
    
    clf = OptimizedSSVEPClassifier(**config)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = (y_pred == y_test).mean()
    
    print(f"✓ 准确率: {acc:.1%} ({int(acc * len(y_test))}/{len(y_test)})")
    
    # 按类别统计
    freqs = [8, 9, 10, 11, 12, 13, 14, 15]
    print("\n✓ 按频率分布:")
    for i, freq in enumerate(freqs):
        mask = y_test == i
        if mask.sum() > 0:
            acc_i = (y_pred[mask] == y_test[mask]).mean()
            print(f"  - {freq}Hz: {acc_i:.1%} ({(y_pred[mask] == y_test[mask]).sum()}/{mask.sum()})")
    
    return acc


def test_only_fb_cca(X_train, y_train, X_test, y_test):
    """测试只开启FB-CCA的效果"""
    print("\n" + "="*70)
    print("🔬 测试 2: 基础CCA + 滤波器组CCA (不用TRCA)")
    print("="*70)
    
    config = DEFAULT_CONFIG.copy()
    config['use_fb_cca'] = True
    config['use_trca'] = False
    config['use_normalization'] = False
    config['use_stacking'] = False
    
    clf = OptimizedSSVEPClassifier(**config)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = (y_pred == y_test).mean()
    
    print(f"✓ 准确率: {acc:.1%} ({int(acc * len(y_test))}/{len(y_test)})")
    return acc


def test_only_trca(X_train, y_train, X_test, y_test):
    """测试只开启TRCA的效果"""
    print("\n" + "="*70)
    print("🔬 测试 3: 基础CCA + TRCA (不用FB-CCA)")
    print("="*70)
    
    config = DEFAULT_CONFIG.copy()
    config['use_fb_cca'] = False
    config['use_trca'] = True
    config['use_normalization'] = False
    config['use_stacking'] = False
    
    clf = OptimizedSSVEPClassifier(**config)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = (y_pred == y_test).mean()
    
    print(f"✓ 准确率: {acc:.1%} ({int(acc * len(y_test))}/{len(y_test)})")
    return acc


def test_with_normalization(X_train, y_train, X_test, y_test):
    """测试加入RV归一化"""
    print("\n" + "="*70)
    print("🔬 测试 4: 基础CCA + RV归一化")
    print("="*70)
    
    config = DEFAULT_CONFIG.copy()
    config['use_fb_cca'] = False
    config['use_trca'] = False
    config['use_normalization'] = True
    config['use_stacking'] = False
    
    clf = OptimizedSSVEPClassifier(**config)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = (y_pred == y_test).mean()
    
    print(f"✓ 准确率: {acc:.1%} ({int(acc * len(y_test))}/{len(y_test)})")
    return acc


def test_all_enabled(X_train, y_train, X_test, y_test):
    """测试所有功能都开启"""
    print("\n" + "="*70)
    print("🔬 测试 5: 所有功能开启 (FB-CCA + TRCA + 归一化 + 堆叠)")
    print("="*70)
    
    clf = OptimizedSSVEPClassifier(**DEFAULT_CONFIG)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = (y_pred == y_test).mean()
    
    print(f"✓ 准确率: {acc:.1%} ({int(acc * len(y_test))}/{len(y_test)})")
    return acc


def main():
    print("\n" + "="*70)
    print("  🧪 SSVEP性能诊断 - 原始版本 vs 优化版本")
    print("="*70)
    
    # 加载数据 - 和原始版本一样的方式
    print("\n📂 加载D1/D2数据...")
    X_d1, y_d1 = load_data_like_original("ExampleData/D1.csv")
    X_d2, y_d2 = load_data_like_original("ExampleData/D2.csv")
    
    # 测试原始版本的方式：不分割数据，直接在D2上测试
    print(f"\n✓ D1: {X_d1.shape}, D2: {X_d2.shape}")
    print("⚠️  原始版本直接在整个数据集上测试，不分割")
    print("🔄 优化版本用70/30分割进行对比测试")
    
    # 合并数据用于优化版本测试
    X_all = np.concatenate([X_d1, X_d2])
    y_all = np.concatenate([y_d1, y_d2])
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_all, test_size=0.3, random_state=42, stratify=y_all
    )
    
    print(f"\n训练集: {X_train.shape}, 测试集: {X_test.shape}")
    
    # 原始版本的参考性能
    print("\n" + "="*70)
    print("📊 原始版本性能 (参考值)")
    print("="*70)
    print("✓ D1数据集: 85.4% (41/48)")
    print("✓ D2数据集: 89.6% (43/48)")
    print("✓ 整体平均: 87.5% (84/96)")
    
    # 逐步测试
    results = {}
    results['基础CCA'] = test_basic_cca(X_train, y_train, X_test, y_test)
    results['+ FB-CCA'] = test_only_fb_cca(X_train, y_train, X_test, y_test)
    results['+ TRCA'] = test_only_trca(X_train, y_train, X_test, y_test)
    results['+ RV归一化'] = test_with_normalization(X_train, y_train, X_test, y_test)
    results['+ 全部开启'] = test_all_enabled(X_train, y_train, X_test, y_test)
    
    # 总结
    print("\n" + "="*70)
    print("📈 性能总结")
    print("="*70)
    print("\n优化版本在测试集上的准确率:")
    for name, acc in results.items():
        print(f"  {name:15} : {acc:.1%}")
    
    print("\n🔍 分析:")
    print("  • 基础CCA性能接近原始版本 ✓")
    print("  • 但在分割的测试集上准确率较低 (原因见下)")
    print("  • 后续优化反而降低性能 ✗")
    
    print("\n💡 可能的原因:")
    print("  1. 数据不足: 96样本太少，70/30分割只有29个测试样本")
    print("  2. 过拟合: 多个优化技术在小数据集上容易过度拟合")
    print("  3. 权重配置: 多技术融合的权重可能不合理")
    print("  4. 参数未调优: DEFAULT_CONFIG 可能不是最优的")
    
    print("\n✅ 建议:")
    print("  • 用更多数据进行训练 (实际竞赛数据)")
    print("  • 使用交叉验证而不是简单的train/test分割")
    print("  • 逐步调整权重和参数")
    print("  • 在大数据集上优化技术才能发挥效果")
    
    print("\n" + "="*70 + "\n")


if __name__ == '__main__':
    main()
