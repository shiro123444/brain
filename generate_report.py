#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成实验结果的可视化报告
"""

import numpy as np
import pandas as pd
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# 实验数据
results = {
    'DirectCCA': {
        'accuracy': 0.8958,
        'recall': 0.8958,
        'f1': 0.8968,
        'latency_ms': 5.77,
        'correct': 43,
        'total': 48,
        'per_class': [1.0, 0.8333, 0.8333, 1.0, 1.0, 0.6667, 1.0, 0.8333]
    },
    'OptimizedNoTRCA': {
        'accuracy': 0.9792,
        'recall': 0.9792,
        'f1': 0.9790,
        'latency_ms': 24.24,
        'correct': 47,
        'total': 48,
        'per_class': [0.8333, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    },
    'OptimizedFull': {
        'accuracy': 1.0000,
        'recall': 1.0000,
        'f1': 1.0000,
        'latency_ms': 24.61,
        'correct': 48,
        'total': 48,
        'per_class': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    }
}

print("\n")
print("╔" + "═" * 98 + "╗")
print("║" + " " * 98 + "║")
print("║" + "  SSVEP 三种算法对比 - 可视化结果报告".center(98) + "║")
print("║" + " " * 98 + "║")
print("╚" + "═" * 98 + "╝")

print("\n")
print("【1. 准确率对比 (ASCII条形图)】")
print("─" * 100)
print()

models = list(results.keys())
accuracies = [results[m]['accuracy'] for m in models]

for model, acc in zip(models, accuracies):
    bar = "█" * int(acc * 50)
    pct = acc * 100
    print(f"  {model:20s} │{bar:<50}│ {pct:6.2f}%")

print()
improvement_23 = (results['OptimizedNoTRCA']['accuracy'] - results['DirectCCA']['accuracy']) * 100
improvement_31 = (results['OptimizedFull']['accuracy'] - results['DirectCCA']['accuracy']) * 100
improvement_32 = (results['OptimizedFull']['accuracy'] - results['OptimizedNoTRCA']['accuracy']) * 100

print(f"  性能提升:")
print(f"    OptimizedNoTRCA vs DirectCCA:    +{improvement_23:.2f}%")
print(f"    OptimizedFull vs DirectCCA:       +{improvement_31:.2f}%")
print(f"    OptimizedFull vs OptimizedNoTRCA: +{improvement_32:.2f}%")

print("\n")
print("【2. 每类准确率热力图】")
print("─" * 100)
print()

frequencies = [8, 9, 10, 11, 12, 13, 14, 15]

# 创建热力图数据
per_class_data = {}
for model in models:
    per_class_data[model] = results[model]['per_class']

# 打印表头
print("  频率(Hz)  ", end="")
for model in models:
    print(f"│{model:18s}", end="")
print("│")
print("  " + "─" * 11, end="")
for _ in models:
    print("─" * 20, end="")
print()

# 打印数据
for i, freq in enumerate(frequencies):
    print(f"  {freq:3d}Hz     ", end="")
    for model in models:
        acc = per_class_data[model][i]
        # 用ASCII表示热度
        if acc >= 0.99:
            marker = "✓✓"
        elif acc >= 0.95:
            marker = "✓ "
        elif acc >= 0.80:
            marker = "◐ "
        else:
            marker = "✗ "
        print(f"│ {acc:.2%} {marker}", end="")
    print("│")

print("\n")
print("【3. 推理延迟对比】")
print("─" * 100)
print()

latencies = [results[m]['latency_ms'] for m in models]
max_latency = max(latencies)

for model, lat in zip(models, latencies):
    if lat < 10:
        rating = "⚡ 极快"
    elif lat < 30:
        rating = "✓ 可接受"
    else:
        rating = "△ 较慢"
    
    bar = "▪" * int((lat / max_latency) * 40)
    print(f"  {model:20s} │{bar:<40}│ {lat:5.2f}ms  {rating}")

print()
print(f"  说明: DirectCCA 是最快的 (5.77ms/样本)")
print(f"       OptimizedNoTRCA 和 OptimizedFull 都约 24ms/样本")
print(f"       24ms 延迟对于脑机接口应用仍在可接受范围 (<50ms)")

print("\n")
print("【4. 综合对比矩阵】")
print("─" * 100)
print()

# 创建综合评分
metrics_data = []
for model in models:
    accuracy_score = results[model]['accuracy'] * 100  # 0-100
    speed_score = (1 - (results[model]['latency_ms'] - 5.77) / 20) * 100  # 越快越好
    speed_score = max(0, min(100, speed_score))  # 限制在0-100
    
    # 综合评分 (60% 准确率 + 40% 速度)
    composite = accuracy_score * 0.6 + speed_score * 0.4
    
    metrics_data.append({
        '算法': model,
        '准确率': f"{results[model]['accuracy']:.2%}",
        '速度': f"{results[model]['latency_ms']:.2f}ms",
        '推理速度分': f"{speed_score:.1f}",
        '准确率分': f"{accuracy_score:.1f}",
        '综合评分': f"{composite:.1f}"
    })

df = pd.DataFrame(metrics_data)
print(df.to_string(index=False))

print("\n")
print("【5. 错误分析】")
print("─" * 100)
print()

for model in models:
    correct = results[model]['correct']
    total = results[model]['total']
    errors = total - correct
    
    print(f"  {model}:")
    print(f"    ✓ 正确: {correct}/{total} ({correct/total*100:.2f}%)")
    print(f"    ✗ 错误: {errors}/{total} ({errors/total*100:.2f}%)")
    
    if model == 'DirectCCA':
        print(f"    弱点类别: 13Hz (频率12Hz, 准确率66.67%) ← 低频识别困难")
    elif model == 'OptimizedNoTRCA':
        print(f"    弱点类别: 8Hz (一个边界样本) ← 仅1个错误")
    else:
        print(f"    完美分类: 所有8个频率准确率100% ✓✓✓")
    print()

print("\n")
print("【6. 主要改进对比】")
print("─" * 100)
print()

improvements = [
    ("DirectCCA → OptimizedNoTRCA", [
        ("Filter-Bank CCA", "4子带[4-8,8-12,12-20,20-35]Hz加权融合"),
        ("RV归一化", "修正频率间系统性偏差，特别是低频"),
        ("参数学习", "从训练集学习非目标得分的基线")
    ]),
    ("OptimizedNoTRCA → OptimizedFull", [
        ("TRCA模板学习", "学习每个频率的类特异空间投影"),
        ("CCA-TRCA融合", "0.6*CCA + 0.4*TRCA，融合时空特征"),
        ("主动学习", "2轮迭代，共标注20个高信息量样本")
    ])
]

for transition, improvements_list in improvements:
    print(f"  {transition}")
    for improvement, detail in improvements_list:
        print(f"    • {improvement}")
        print(f"      → {detail}")
    print()

print("\n")
print("【7. 竞赛推荐路线图】")
print("─" * 100)
print()

roadmap = [
    ("第1阶段 (Day1-2)", "DirectCCA", 89.58, "快速baseline, 验证数据质量", "✓"),
    ("第2阶段 (Day3-4)", "OptimizedNoTRCA", 97.92, "标准配置, 稳定可靠", "✓✓"),
    ("第3阶段 (Day5-6)", "OptimizedFull+AL", 100.00, "竞赛冠军级, 100%精度", "✓✓✓"),
]

for stage, model, acc, desc, rating in roadmap:
    bar = "▓" * int(acc / 4)
    print(f"  {stage:15s} │ {model:20s} │ {bar:<25} │ {acc:6.2f}% │ {rating:6s}")
    print(f"  {' '*15} └─ {desc}")
    print()

print("\n")
print("【8. 关键数字总结】")
print("─" * 100)
print()

key_metrics = [
    ("最高准确率", "100.00%", "OptimizedFull (完美分类)"),
    ("最快推理", "5.77ms/样本", "DirectCCA"),
    ("最佳平衡", "97.92%准确率, 24.24ms", "OptimizedNoTRCA"),
    ("性能提升范围", "+8.34% → +10.42%", "相对于DirectCCA"),
    ("主动学习工作量", "+20个样本", "原始48个样本的+42%"),
    ("推理延迟范围", "5.77ms - 24.61ms", "都在实时应用可接受范围"),
    ("频率覆盖", "8-15Hz (8个频率)", "均匀分布, 完全分类"),
]

for metric, value, note in key_metrics:
    print(f"  • {metric:20s} : {value:20s}  ({note})")

print("\n")
print("═" * 100)
print("生成时间: 2025年11月12日")
print("数据来源: experiment_comparison.py 实验结果")
print("═" * 100)

