#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
正确的主动学习竞赛方案
- 方案A: 标准train/test分割 (D1→D2)
- 方案B: 主动学习 (D1 + 部分D2→剩余D2)
- 方案C: 完全标注+交叉验证 (D1+D2的5折CV)
"""

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from improved_algorithms import OptimizedCCAClassifier
from full_comparison import load_and_prepare_data

class ActiveLearningStrategy:
    """
    现实的主动学习策略
    模拟竞赛中"有预算的标注"过程
    """
    
    def __init__(self, base_classifier_class):
        self.base_classifier_class = base_classifier_class
        self.classifier = None
        self.labeled_pool = None
        self.unlabeled_pool = None
        self.labeled_indices = []
        self.query_history = []
    
    def initialize(self, X_train, y_train, X_unlabeled):
        """
        初始化
        
        参数:
            X_train: 初始标注的训练数据 (D1)
            y_train: 初始标签
            X_unlabeled: 未标注的数据 (D2，初始都未标注)
        """
        self.classifier = self.base_classifier_class()
        self.classifier.fit(X_train, y_train)
        
        # 记录D2中所有数据都是未标注的（虽然我们有标签，但假装隐藏）
        self.unlabeled_indices = np.arange(len(X_unlabeled))
        self.X_unlabeled = X_unlabeled
        
        self.labeled_pool_X = X_train.copy()
        self.labeled_pool_y = y_train.copy()
        
        print(f"[AL初始化]")
        print(f"  初始训练集: {len(self.labeled_pool_X)} 个样本")
        print(f"  未标注集: {len(self.unlabeled_indices)} 个样本")
    
    def query_instances(self, n_queries, true_labels):
        """
        查询最不确定的样本
        
        参数:
            n_queries: 本轮查询数量
            true_labels: D2的真实标签（用于评估性能）
        
        返回:
            new_X, new_y: 新查询的样本及其标签
            uncertainty_scores: 不确定性分数
        """
        print(f"\n  【查询{n_queries}个样本】")
        
        # 计算未标注数据的不确定性
        if hasattr(self.classifier, 'predict_scores'):
            scores = self.classifier.predict_scores(self.X_unlabeled[self.unlabeled_indices])
        else:
            scores = self.classifier._compute_combined_scores(
                self.X_unlabeled[self.unlabeled_indices]
            )
        
        # Margin-based 不确定性: top1 - top2
        sorted_indices_in_scores = np.argsort(scores, axis=1)
        top1_score = scores[np.arange(len(scores)), sorted_indices_in_scores[:, -1]]
        top2_score = scores[np.arange(len(scores)), sorted_indices_in_scores[:, -2]]
        margins = top1_score - top2_score
        
        # 选择margin最小的样本（最不确定）
        uncertainty_rank = np.argsort(margins)[:n_queries]
        
        # 获取原始索引
        query_indices = self.unlabeled_indices[uncertainty_rank]
        
        # 获取这些样本及其真实标签
        new_X = self.X_unlabeled[query_indices]
        new_y = true_labels[query_indices]
        
        # 移除已查询的样本
        self.unlabeled_indices = np.delete(self.unlabeled_indices, uncertainty_rank)
        
        print(f"    查询样本的不确定性 (margin):")
        print(f"      最小: {margins[uncertainty_rank].min():.4f}")
        print(f"      平均: {margins[uncertainty_rank].mean():.4f}")
        print(f"      最大: {margins[uncertainty_rank].max():.4f}")
        
        return new_X, new_y, margins[uncertainty_rank]
    
    def learn_from_feedback(self, new_X, new_y):
        """用新查询的标签重新训练"""
        # 添加到标注集
        self.labeled_pool_X = np.vstack([self.labeled_pool_X, new_X])
        self.labeled_pool_y = np.concatenate([self.labeled_pool_y, new_y])
        
        # 重新训练
        print(f"    重新训练模型...")
        self.classifier.fit(self.labeled_pool_X, self.labeled_pool_y)
        print(f"    新的训练集大小: {len(self.labeled_pool_X)}")
    
    def predict_unlabeled(self):
        """预测所有剩余未标注的数据"""
        if len(self.unlabeled_indices) == 0:
            print("\n  所有数据已标注！")
            return np.array([])
        
        X_remaining = self.X_unlabeled[self.unlabeled_indices]
        return self.classifier.predict(X_remaining)


# ================================================================
# 完整竞赛模拟
# ================================================================

def run_competition_simulation():
    """
    模拟竞赛流程 - 正确版本
    1. 初始训练：D1 (48个样本)
    2. 主动学习：从D2中选择最不确定的20个样本标注
    3. 对比三种方案的性能
    """
    
    print("=" * 100)
    print("SSVEP竞赛 - 主动学习正确对比")
    print("=" * 100)
    
    # 加载数据
    print("\n【加载数据】")
    X_train, y_train = load_and_prepare_data('ExampleData/D1.csv')
    X_test_all, y_test_all = load_and_prepare_data('ExampleData/D2.csv')
    
    print(f"  D1 (可用训练): {X_train.shape} 个样本")
    print(f"  D2 (测试池): {X_test_all.shape} 个样本")
    
    # =====================================================================
    # 方案A: Baseline (D1训练 → D2测试)
    # =====================================================================
    print("\n" + "=" * 100)
    print("【方案A】Baseline - D1训练，D2全部测试")
    print("=" * 100)
    print("说明: 标准的train/test分割，没有主动学习")
    
    clf_baseline = OptimizedCCAClassifier()
    clf_baseline.fit(X_train, y_train)
    y_pred_baseline = clf_baseline.predict(X_test_all)
    
    acc_baseline = accuracy_score(y_test_all, y_pred_baseline)
    f1_baseline = f1_score(y_test_all, y_pred_baseline, average='macro')
    
    print(f"\n训练集: D1 (48个)")
    print(f"测试集: D2全部 (48个)")
    print(f"准确率: {acc_baseline:.4f} ({int(acc_baseline * len(y_test_all))}/{len(y_test_all)})")
    print(f"F1分数: {f1_baseline:.4f}")
    
    # =====================================================================
    # 方案B: 主动学习 (D1 + 部分D2标注)
    # =====================================================================
    print("\n" + "=" * 100)
    print("【方案B】主动学习 - D1 + 查询D2中20个样本")
    print("=" * 100)
    print("说明: 用D1训练，从D2中选最不确定的20个样本，剩余28个样本用作测试")
    
    # 初始用D1训练
    clf_al = OptimizedCCAClassifier()
    clf_al.fit(X_train, y_train)
    
    # 计算D2的不确定性分数
    scores_d2 = clf_al.predict_scores(X_test_all)
    
    # Margin-based不确定性: top1 - top2
    sorted_indices = np.argsort(scores_d2, axis=1)
    top1_score = scores_d2[np.arange(len(scores_d2)), sorted_indices[:, -1]]
    top2_score = scores_d2[np.arange(len(scores_d2)), sorted_indices[:, -2]]
    margins = top1_score - top2_score
    
    # 选择20个margin最小的样本（最不确定）
    query_indices = np.argsort(margins)[:20]
    test_indices = np.setdiff1d(np.arange(len(X_test_all)), query_indices)
    
    print(f"\n不确定性统计:")
    print(f"  最小margin: {margins[query_indices].min():.4f}")
    print(f"  平均margin: {margins[query_indices].mean():.4f}")
    print(f"  最大margin: {margins[query_indices].max():.4f}")
    
    # 合并D1和查询的D2
    X_train_al = np.vstack([X_train, X_test_all[query_indices]])
    y_train_al = np.concatenate([y_train, y_test_all[query_indices]])
    
    # 用剩余的D2作为测试集
    X_test_al = X_test_all[test_indices]
    y_test_al = y_test_all[test_indices]
    
    # 重新训练
    clf_al.fit(X_train_al, y_train_al)
    y_pred_al = clf_al.predict(X_test_al)
    
    acc_al = accuracy_score(y_test_al, y_pred_al)
    f1_al = f1_score(y_test_al, y_pred_al, average='macro')
    
    print(f"\n训练集: D1 (48个) + 查询的D2 (20个) = 68个")
    print(f"测试集: 剩余D2 (28个)")
    print(f"准确率: {acc_al:.4f} ({int(acc_al * len(y_test_al))}/{len(y_test_al)})")
    print(f"F1分数: {f1_al:.4f}")
    
    # =====================================================================
    # 方案C: 完全标注 + 交叉验证
    # =====================================================================
    print("\n" + "=" * 100)
    print("【方案C】完全标注 - D1+D2合并做5折交叉验证")
    print("=" * 100)
    print("说明: 由于没有独立的第三个测试集，用交叉验证估计D1+D2共同训练的性能上界")
    
    X_all = np.vstack([X_train, X_test_all])
    y_all = np.concatenate([y_train, y_test_all])
    
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    cv_scores = []
    cv_f1s = []
    fold_details = []
    
    for fold, (train_idx, test_idx) in enumerate(kfold.split(X_all, y_all), 1):
        X_train_fold = X_all[train_idx]
        y_train_fold = y_all[train_idx]
        X_test_fold = X_all[test_idx]
        y_test_fold = y_all[test_idx]
        
        clf = OptimizedCCAClassifier()
        clf.fit(X_train_fold, y_train_fold)
        y_pred = clf.predict(X_test_fold)
        
        acc = accuracy_score(y_test_fold, y_pred)
        f1 = f1_score(y_test_fold, y_pred, average='macro')
        
        cv_scores.append(acc)
        cv_f1s.append(f1)
        fold_details.append(f"  Fold {fold}: 准确率={acc:.4f}, F1={f1:.4f}")
    
    for detail in fold_details:
        print(detail)
    
    acc_cv = np.mean(cv_scores)
    f1_cv = np.mean(cv_f1s)
    acc_cv_std = np.std(cv_scores)
    
    print(f"\n平均准确率: {acc_cv:.4f} ± {acc_cv_std:.4f}")
    print(f"平均F1分数: {f1_cv:.4f}")
    
    # =====================================================================
    # 对比总结
    # =====================================================================
    print("\n" + "=" * 100)
    print("【对比总结】")
    print("=" * 100)
    
    print(f"\n{'方案':<30} {'训练集':<20} {'测试集':<20} {'准确率':<15}")
    print("-" * 85)
    print(f"{'A. Baseline':<30} {'D1 (48)':<20} {'D2 (48)':<20} {acc_baseline:.4f}")
    print(f"{'B. 主动学习':<30} {'D1+AL D2 (68)':<20} {'D2剩余 (28)':<20} {acc_al:.4f}")
    print(f"{'C. 完全标注 (5折CV)':<30} {'D1+D2 (96)':<20} {'交叉验证':<20} {acc_cv:.4f}±{acc_cv_std:.4f}")
    
    # =====================================================================
    # 关键洞察
    # =====================================================================
    print("\n" + "=" * 100)
    print("📊 关键洞察")
    print("=" * 100)
    
    print(f"\n✓ 方案解释")
    print(f"  方案A: 标准train/test分割")
    print(f"    - 用D1单独训练 → 预测D2全部")
    print(f"    - 最保守，最现实的评估方式")
    print(f"    - 准确率: {acc_baseline:.4f}")
    
    print(f"\n  方案B: 主动学习")
    print(f"    - 用D1 + 查询D2中20个最不确定样本 → 预测剩余28个")
    print(f"    - 模拟竞赛中的'有预算的标注'过程")
    print(f"    - 准确率: {acc_al:.4f} (测试集28个样本)")
    print(f"    - ⚠️  注意：测试集大小不同，直接比较不太公平")
    
    print(f"\n  方案C: 完全标注理论上界")
    print(f"    - D1+D2一起训练，用5折CV估计")
    print(f"    - 告诉你D1+D2共同使用的最好可能性")
    print(f"    - 准确率: {acc_cv:.4f} (5折交叉验证)")
    
    print(f"\n✓ 竞赛应用建议")
    print(f"  1. 基准线: {acc_baseline:.4f} (仅D1)")
    print(f"  2. 目标:  {acc_cv:.4f}±{acc_cv_std:.4f} (D1+D2，理论上界)")
    print(f"  3. 策略:  用主动学习逐步改进D1的性能")
    
    print(f"\n✓ 为什么需要三种方案？")
    print(f"  • Baseline(方案A): 最坏情况下的性能 → 保证安全")
    print(f"  • 完全标注(方案C): 最好情况下的性能 → 知道目标")
    print(f"  • 主动学习(方案B): 中间路线 → 用有限标注达到接近最好")
    
    print("\n" + "=" * 100)


if __name__ == '__main__':
    run_competition_simulation()
