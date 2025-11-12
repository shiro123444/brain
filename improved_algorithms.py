#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
改进的SSVEP识别算法 - 统一接口版本
基于config和core_modules的模块化设计
"""

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
from config import (
    COMPETITION_CONFIG, SIGNAL_PROCESSING_CONFIG, 
    FUSION_CONFIG, ACTIVE_LEARNING_CONFIG
)
from core_modules import (
    ImprovedSignalPreprocessor, ReferenceSignalGenerator,
    ImprovedCCACalculator, FilterBankAnalyzer
)


class DirectCCAClassifier:
    """
    改进的 DirectCCA
    - 基于标准化的CCA计算
    - 无需训练，跨被试泛化性最强
    """
    
    def __init__(self, config=None):
        self.config = config or COMPETITION_CONFIG
        self.fs = self.config['sampling_rate']
        self.freq_map = self.config['freq_map']
        
        # 初始化模块
        self.preprocessor = ImprovedSignalPreprocessor()
        self.ref_generator = ReferenceSignalGenerator(config)
        self.cca_calc = ImprovedCCACalculator()
        
        # 生成参考信号
        self.templates = self.ref_generator.generate_references(
            n_samples=1000, n_harmonics=2, harmonic_weights='uniform'
        )
        
        print("[DirectCCA] 初始化完成")
        print(f"  频率: {list(self.freq_map.values())}")
        print(f"  无需训练，直接使用固定参考信号")
    
    def fit(self, X_train, y_train):
        """DirectCCA不需要训练"""
        print("[DirectCCA.fit] 跳过（无需训练）")
        return self
    
    def predict(self, X_test):
        """预测"""
        if X_test.ndim == 2:
            X_test = X_test[np.newaxis, :, :]
        
        n_samples = X_test.shape[0]
        predictions = []
        
        for idx in range(n_samples):
            # 预处理
            segment = self.preprocessor.preprocess(X_test[idx])
            
            # 对齐长度
            template_len = self.templates[0].shape[1]
            if segment.shape[1] > template_len:
                segment = segment[:, :template_len]
            elif segment.shape[1] < template_len:
                pad_len = template_len - segment.shape[1]
                segment = np.pad(segment, ((0, 0), (0, pad_len)), mode='edge')
            
            # 计算所有类别的得分
            scores = []
            for class_id in range(self.config['n_classes']):
                score = self.cca_calc.compute_cca_score(
                    segment, self.templates[class_id]
                )
                scores.append(score)
            
            pred = int(np.argmax(scores))
            predictions.append(pred)
        
        return np.array(predictions)
    
    def predict_scores(self, X_test):
        """返回得分矩阵"""
        if X_test.ndim == 2:
            X_test = X_test[np.newaxis, :, :]
        
        n_samples = X_test.shape[0]
        n_classes = self.config['n_classes']
        scores_matrix = np.zeros((n_samples, n_classes))
        
        for idx in range(n_samples):
            segment = self.preprocessor.preprocess(X_test[idx])
            
            # 对齐长度
            template_len = self.templates[0].shape[1]
            if segment.shape[1] > template_len:
                segment = segment[:, :template_len]
            elif segment.shape[1] < template_len:
                pad_len = template_len - segment.shape[1]
                segment = np.pad(segment, ((0, 0), (0, pad_len)), mode='edge')
            
            for class_id in range(n_classes):
                score = self.cca_calc.compute_cca_score(
                    segment, self.templates[class_id]
                )
                scores_matrix[idx, class_id] = score
        
        return scores_matrix


class OptimizedCCAClassifier:
    """
    改进的优化CCA
    - Filter-Bank CCA
    - RV 标准化
    """
    
    def __init__(self, config=None):
        self.config = config or COMPETITION_CONFIG
        self.fs = self.config['sampling_rate']
        self.freq_map = self.config['freq_map']
        
        # 初始化模块
        self.preprocessor = ImprovedSignalPreprocessor()
        self.ref_generator = ReferenceSignalGenerator(config)
        self.cca_calc = ImprovedCCACalculator()
        self.fb_analyzer = FilterBankAnalyzer()
        
        # 生成参考信号
        self.templates = self.ref_generator.generate_references(
            n_samples=1000, n_harmonics=2, harmonic_weights='uniform'
        )
        
        self.fb_weights = self.fb_analyzer.get_weights()
        self.norm_stats = None  # 训练时学习
        
        print("[OptimizedCCA] 初始化完成")
        print(f"  Filter-Bank: {len(self.fb_analyzer.subbands)} 子带")
        print(f"  RV 标准化: 待训练")
    
    def fit(self, X_train, y_train):
        """训练 - 学习RV标准化参数"""
        print("[OptimizedCCA.fit] 开始训练...")
        
        # 计算训练集得分
        print("  计算训练集得分...")
        scores_train = self.predict_scores(X_train)
        
        # 学习RV标准化参数
        print("  学习RV标准化参数...")
        self.norm_stats = {}
        n_classes = self.config['n_classes']
        
        for class_id in range(n_classes):
            mask = y_train != class_id
            if mask.sum() > 0:
                mean_non_target = np.mean(scores_train[mask, class_id])
            else:
                mean_non_target = 0.5
            self.norm_stats[class_id] = mean_non_target
        
        print("[OptimizedCCA.fit] 训练完成")
        return self
    
    def predict(self, X_test):
        """预测"""
        scores = self.predict_scores(X_test)
        
        # 应用RV标准化
        if self.norm_stats:
            for class_id in range(self.config['n_classes']):
                mean_nt = self.norm_stats[class_id]
                scores[:, class_id] = (scores[:, class_id] - mean_nt) / (
                    scores[:, class_id] + mean_nt + 1e-8
                )
        
        return np.argmax(scores, axis=1)
    
    def predict_scores(self, X_test):
        """返回得分矩阵"""
        if X_test.ndim == 2:
            X_test = X_test[np.newaxis, :, :]
        
        n_samples = X_test.shape[0]
        n_classes = self.config['n_classes']
        n_bands = len(self.fb_analyzer.subbands)
        
        scores_matrix = np.zeros((n_samples, n_classes))
        
        for idx in range(n_samples):
            # 预处理
            segment = self.preprocessor.preprocess(X_test[idx])
            
            # 对齐长度
            template_len = self.templates[0].shape[1]
            if segment.shape[1] > template_len:
                segment = segment[:, :template_len]
            elif segment.shape[1] < template_len:
                pad_len = template_len - segment.shape[1]
                segment = np.pad(segment, ((0, 0), (0, pad_len)), mode='edge')
            
            # Filter-Bank CCA
            fb_scores = np.zeros(n_classes)
            
            subbands = self.fb_analyzer.apply_subbands(segment)
            
            for band_idx, (subband, weight) in enumerate(zip(subbands, self.fb_weights)):
                for class_id in range(n_classes):
                    score = self.cca_calc.compute_cca_score(
                        subband, self.templates[class_id]
                    )
                    fb_scores[class_id] += weight * score
            
            scores_matrix[idx] = fb_scores
        
        return scores_matrix


# ================================================================
# 快速测试
# ================================================================

if __name__ == '__main__':
    print("=" * 70)
    print("改进算法快速测试")
    print("=" * 70)
    
    # 生成虚拟数据
    from sklearn.datasets import make_classification
    
    np.random.seed(42)
    n_samples = 96
    X_dummy = np.random.randn(n_samples, 6, 1000)
    y_dummy = np.tile(np.arange(8), 12)
    
    # 分割
    X_train, X_test = X_dummy[:48], X_dummy[48:]
    y_train, y_test = y_dummy[:48], y_dummy[48:]
    
    # 测试 DirectCCA
    print("\n【DirectCCA】")
    clf1 = DirectCCAClassifier()
    clf1.fit(X_train, y_train)
    y_pred1 = clf1.predict(X_test)
    acc1 = accuracy_score(y_test, y_pred1)
    print(f"准确率: {acc1:.4f}")
    
    # 测试 OptimizedCCA
    print("\n【OptimizedCCA】")
    clf2 = OptimizedCCAClassifier()
    clf2.fit(X_train, y_train)
    y_pred2 = clf2.predict(X_test)
    acc2 = accuracy_score(y_test, y_pred2)
    print(f"准确率: {acc2:.4f}")
    
    print("\n" + "=" * 70)
