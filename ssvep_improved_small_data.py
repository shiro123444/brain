#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
改进的SSVEP算法 - 针对小数据集优化
禁用容易过拟合的技术，保留最有效的部分
"""

import sys
import io
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from ssvep_optimization_framework import OptimizedSSVEPClassifier, DEFAULT_CONFIG


def load_data(csv_file):
    """加载CSV"""
    df = pd.read_csv(csv_file)
    channel_cols = ['CP3', 'CPZ', 'CP4', 'PO3', 'POZ', 'PO4']
    
    X_list, y_list = [], []
    for task_id, group in df.groupby('taskID'):
        data = group[channel_cols].values[:1000, :].T
        X_list.append(data)
        y_list.append(int(group['stimID'].iloc[0]))
    
    return np.array(X_list), np.array(y_list)


# ============================================================================
# 配置1: 原始算法 (最稳定，88%性能)
# ============================================================================
CONFIG_ORIGINAL = {
    'use_fb_cca': False,
    'use_trca': False,
    'use_normalization': False,
    'use_stacking': False,
    'harmonics': 2,
}


# ============================================================================
# 配置2: 轻量级优化 (折中方案，性能与稳定性平衡)
# ============================================================================
CONFIG_LIGHTWEIGHT = {
    'use_fb_cca': False,  # ✗ 禁用 (易过拟合)
    'use_trca': False,    # ✗ 禁用 (易过拟合)
    'use_normalization': True,   # ✓ 保留 (低风险)
    'use_stacking': False,       # ✗ 禁用 (完全不适合)
    'harmonics': 2,              # 基频 + 二次谐波
    'normalization_method': 'rv',
}


# ============================================================================
# 配置3: 完整优化 (需要更多数据，>500个样本)
# ============================================================================
CONFIG_FULL = DEFAULT_CONFIG.copy()


def main():
    print("\n" + "="*70)
    print("  🔧 SSVEP小数据集改进版本")
    print("="*70 + "\n")
    
    # 加载数据
    print("📂 加载数据...")
    X_d1, y_d1 = load_data("ExampleData/D1.csv")
    X_d2, y_d2 = load_data("ExampleData/D2.csv")
    X_all = np.concatenate([X_d1, X_d2])
    y_all = np.concatenate([y_d1, y_d2])
    print(f"   ✓ 合并: {X_all.shape[0]} 样本\n")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_all, test_size=0.3, random_state=42, stratify=y_all
    )
    
    # 测试三种配置
    configs = [
        ("原始算法 (基础CCA)", CONFIG_ORIGINAL),
        ("轻量级优化 (推荐小数据)", CONFIG_LIGHTWEIGHT),
        ("完整优化 (需要大数据)", CONFIG_FULL),
    ]
    
    results = {}
    
    for name, config in configs:
        print(f"\n{name}")
        print("-" * 70)
        
        clf = OptimizedSSVEPClassifier(**config)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        acc = (y_pred == y_test).mean()
        
        print(f"✓ 准确率: {acc:.1%} ({int(acc * len(y_test))}/{len(y_test)})")
        results[name] = acc
        
        # 按频率分析
        freqs = [8, 9, 10, 11, 12, 13, 14, 15]
        per_class = []
        for i, freq in enumerate(freqs):
            mask = y_test == i
            if mask.sum() > 0:
                acc_i = (y_pred[mask] == y_test[mask]).mean()
                per_class.append(acc_i)
        
        if per_class:
            print(f"  按频率: 平均={np.mean(per_class):.1%}, " +
                  f"最好={np.max(per_class):.1%}, " +
                  f"最差={np.min(per_class):.1%}")
    
    # 总结
    print("\n" + "="*70)
    print("📊 性能总结")
    print("="*70 + "\n")
    
    for name, acc in results.items():
        print(f"  {name:25} : {acc:.1%}")
    
    print("\n💡 推荐方案:")
    print("  • 当前 (96样本): 使用 CONFIG_ORIGINAL 或 CONFIG_LIGHTWEIGHT")
    print("  • 未来 (>500样本): 可尝试 CONFIG_FULL")
    
    print("\n" + "="*70 + "\n")


if __name__ == '__main__':
    main()
