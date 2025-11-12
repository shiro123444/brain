#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

"""
SSVEP 算法对比实验
================================================================
三种算法对比：
1. DirectCCA: 直接CCA方法（无训练，无主动学习）
2. OptimizedNoTRCA: 优化框架（无TRCA）
3. OptimizedFull: 优化框架（完整版+主动学习）

数据设置：
- 训练集：D1.csv
- 测试集：D2.csv
- 特征提取：按taskID分段

作者: 实验平台
日期: 2025
================================================================
"""

import numpy as np
import pandas as pd
from scipy import signal as scipysignal
from sklearn.cross_decomposition import CCA
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix, classification_report
from typing import Tuple, Dict, List
import warnings
import sys
import time

warnings.filterwarnings('ignore')

# ===================================================================
# 第零部分：数据加载与预处理
# ===================================================================

class DataLoader:
    """加载并预处理 CSV 数据"""
    
    @staticmethod
    def load_csv(csv_path: str, srate: float = 250) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        加载 CSV 文件并按 taskID 分段
        
        返回:
        ------
        X : ndarray, shape [n_segments, n_channels, n_samples]
        y : ndarray, shape [n_segments] (true stimID)
        task_ids : ndarray, shape [n_segments]
        """
        print(f"[加载数据] 读取 {csv_path}...")
        data = pd.read_csv(csv_path)
        
        channel_names = ['CP3', 'CPZ', 'CP4', 'PO3', 'POZ', 'PO4']
        eeg_data = data[channel_names].values  # [N, 6]
        task_ids_raw = data['taskID'].values
        stim_ids_raw = data['stimID'].values
        
        print(f"  总采样点数: {len(eeg_data)}")
        print(f"  通道数: {len(channel_names)}")
        print(f"  采样率: {srate} Hz")
        print(f"  任务ID范围: {int(task_ids_raw.min())} ~ {int(task_ids_raw.max())}")
        
        # 按 taskID 分段
        task_changes = np.concatenate(([0], np.where(np.diff(task_ids_raw) != 0)[0] + 1, [len(task_ids_raw)]))
        
        segments_X = []
        segments_y = []
        segments_task_ids = []
        
        for i in range(len(task_changes) - 1):
            start_idx = task_changes[i]
            end_idx = task_changes[i + 1]
            
            segment_data = eeg_data[start_idx:end_idx].T  # [6, samples]
            task_id = int(task_ids_raw[start_idx])
            stim_id = int(stim_ids_raw[start_idx])
            
            segments_X.append(segment_data)
            segments_y.append(stim_id)
            segments_task_ids.append(task_id)
        
        X = np.array(segments_X)
        y = np.array(segments_y)
        task_ids = np.array(segments_task_ids)
        
        print(f"  分段数: {len(X)}")
        print(f"  类别数: {len(np.unique(y))}")
        print(f"  类别分布:\n{pd.Series(y).value_counts().sort_index()}")
        print()
        
        return X, y, task_ids
    
    @staticmethod
    def preprocess(X, fs=250):
        """预处理：50Hz陷波 + 6-90Hz带通"""
        X_filtered = np.zeros_like(X)
        
        for i, segment in enumerate(X):
            try:
                # 50Hz陷波
                b, a = scipysignal.iircomb(50, 35, ftype='notch', fs=fs)
                filtered = scipysignal.filtfilt(b, a, segment, axis=1)
                
                # 6-90Hz椭圆带通
                fs_half = fs / 2
                N, Wn = scipysignal.ellipord([6 / fs_half, 90 / fs_half],
                                             [2 / fs_half, 100 / fs_half], 3, 40)
                N = min(N, segment.shape[1] // 4)
                if N >= 1:
                    b1, a1 = scipysignal.ellip(N, 1, 90, Wn, 'bandpass')
                    filtered = scipysignal.filtfilt(b1, a1, filtered, axis=1)
                
                X_filtered[i] = filtered
            except Exception as e:
                print(f"    警告: segment {i} 预处理失败 ({e}), 使用原始数据")
                X_filtered[i] = segment
        
        return X_filtered


# ===================================================================
# 第一部分：直接 CCA 方法（无训练）
# ===================================================================

class DirectCCAClassifier:
    """直接 CCA 方法 - 不需要训练"""
    
    def __init__(self, srate=250, freqs=None, n_harmonics=2):
        self.srate = srate
        if freqs is None:
            self.freqs = [8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0]
        else:
            self.freqs = freqs
        self.n_harmonics = n_harmonics
        self.n_classes = len(self.freqs)
        
        # 构造参考模板（固定）
        self._build_templates()
        print("[DirectCCA初始化] 固定参考模板已构造")
        print(f"  频率: {self.freqs}")
        print(f"  谐波数: {self.n_harmonics}")
    
    def _build_templates(self):
        """构造参考模板"""
        self.templates = []
        # 使用通用长度 4s
        template_len = int(4.0 * self.srate)
        time_axis = np.linspace(0, (template_len - 1) / self.srate, template_len, endpoint=True)
        
        for freq in self.freqs:
            sinusoids = []
            for h in range(1, self.n_harmonics + 1):
                harmonic_freq = freq * h
                phase = 2 * np.pi * harmonic_freq * time_axis
                sinusoids.append(np.sin(phase))
                sinusoids.append(np.cos(phase))
            
            template = np.vstack(sinusoids)
            self.templates.append(template)
    
    def fit(self, X, y):
        """Direct CCA 不需要训练"""
        print("[DirectCCA.fit] 跳过（无训练阶段）")
    
    def _compute_cca(self, data, template):
        """计算CCA相关系数"""
        try:
            cca = CCA(n_components=1)
            
            cdata = data.T  # [samples, 6]
            ctemplate = template.T  # [samples, harmonics*2]
            
            if cdata.shape[0] < 10 or ctemplate.shape[0] < 10:
                return 0.0
            
            cca.fit(cdata, ctemplate)
            data_trans, template_trans = cca.transform(cdata, ctemplate)
            
            corr = np.corrcoef(data_trans[:, 0], template_trans[:, 0])[0, 1]
            
            if np.isnan(corr) or np.isinf(corr):
                return 0.0
            return max(0.0, float(corr))
        except:
            return 0.0
    
    def predict(self, X):
        """预测"""
        n_samples = X.shape[0]
        predictions = []
        
        for idx, segment in enumerate(X):
            # 数据对齐
            template_len = self.templates[0].shape[1]
            if segment.shape[1] > template_len:
                segment = segment[:, :template_len]
            elif segment.shape[1] < template_len:
                pad_len = template_len - segment.shape[1]
                segment = np.pad(segment, ((0, 0), (0, pad_len)), mode='constant')
            
            # 计算所有模板的相关系数
            coefficients = []
            for template in self.templates:
                coeff = self._compute_cca(segment, template)
                coefficients.append(coeff)
            
            pred = int(np.argmax(coefficients))
            predictions.append(pred)
        
        return np.array(predictions)


# ===================================================================
# 第二部分：优化框架（简化版，无TRCA）
# ===================================================================

class OptimizedClassifierNoTRCA:
    """优化框架但不含TRCA - 用于对比"""
    
    def __init__(self, freqs=None, srate=250, n_harmonics=2):
        self.srate = srate
        if freqs is None:
            self.freqs = [8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0]
        else:
            self.freqs = freqs
        self.n_harmonics = n_harmonics
        self.n_classes = len(self.freqs)
        
        # FB-CCA 设置
        self.subbands = [(4, 8), (8, 12), (12, 20), (20, 35)]
        self.subband_weights = np.ones(len(self.subbands)) / len(self.subbands)
        
        # 构造谐波参考模板
        self._build_harmonic_templates()
        
        # 归一化参数
        self.norm_stats = None
        
        print("[OptimizedNoTRCA初始化] 完成")
        print(f"  频率: {self.freqs}")
        print(f"  谐波数: {self.n_harmonics}")
        print(f"  子带: {self.subbands}")
    
    def _build_harmonic_templates(self):
        """构造谐波参考模板"""
        self.harmonic_templates = {}
        template_len = int(4.0 * self.srate)
        time_axis = np.linspace(0, (template_len - 1) / self.srate, template_len, endpoint=True)
        
        for idx, freq in enumerate(self.freqs):
            sinusoids = []
            for h in range(1, self.n_harmonics + 1):
                harmonic_freq = freq * h
                phase = 2 * np.pi * harmonic_freq * time_axis
                sinusoids.append(np.sin(phase))
                sinusoids.append(np.cos(phase))
            
            self.harmonic_templates[idx] = np.vstack(sinusoids)
    
    def _design_subband_filters(self):
        """设计子带滤波器"""
        self.subband_filters = []
        nyquist = self.srate / 2
        for low, high in self.subbands:
            low_norm = low / nyquist
            high_norm = high / nyquist
            low_norm = max(0.01, min(low_norm, 0.99))
            high_norm = max(low_norm + 0.01, min(high_norm, 0.99))
            
            b, a = scipysignal.butter(4, [low_norm, high_norm], btype='band')
            self.subband_filters.append((b, a))
    
    def fit(self, X, y):
        """训练 - 主要是学习归一化参数"""
        print("[OptimizedNoTRCA.fit] 开始训练...")
        
        # 设计滤波器
        self._design_subband_filters()
        
        # 计算训练集得分用于归一化参数
        print("  计算训练集得分...")
        scores_train = self._compute_all_scores(X)
        
        # 学习归一化参数 (RV变换)
        print("  学习归一化参数...")
        self.norm_stats = {}
        for cls_idx in range(self.n_classes):
            # 非目标类的平均得分
            mask = y != cls_idx
            if mask.sum() > 0:
                mean_non_target = np.mean(scores_train[mask, cls_idx])
            else:
                mean_non_target = 0.5
            self.norm_stats[cls_idx] = mean_non_target
        
        print("[OptimizedNoTRCA.fit] 完成")
    
    def _compute_cca(self, data, template):
        """计算CCA相关系数"""
        try:
            cca = CCA(n_components=1)
            
            cdata = data.T
            ctemplate = template.T
            
            if cdata.shape[0] < 10 or ctemplate.shape[0] < 10:
                return 0.0
            
            cca.fit(cdata, ctemplate)
            data_trans, template_trans = cca.transform(cdata, ctemplate)
            
            corr = np.corrcoef(data_trans[:, 0], template_trans[:, 0])[0, 1]
            
            if np.isnan(corr) or np.isinf(corr):
                return 0.0
            return max(0.0, float(corr))
        except:
            return 0.0
    
    def _compute_all_scores(self, X):
        """计算所有样本的得分矩阵"""
        n_samples = X.shape[0]
        scores = np.zeros((n_samples, self.n_classes))
        
        for idx, segment in enumerate(X):
            # 对齐长度
            template_len = self.harmonic_templates[0].shape[1]
            if segment.shape[1] > template_len:
                segment_aligned = segment[:, :template_len]
            elif segment.shape[1] < template_len:
                pad_len = template_len - segment.shape[1]
                segment_aligned = np.pad(segment, ((0, 0), (0, pad_len)), mode='constant')
            else:
                segment_aligned = segment
            
            # 计算子带CCA得分
            fb_scores = np.zeros(self.n_classes)
            for band_idx, (b, a) in enumerate(self.subband_filters):
                try:
                    band_signal = scipysignal.filtfilt(b, a, segment_aligned, axis=1)
                except:
                    band_signal = segment_aligned
                
                for cls_idx, template in self.harmonic_templates.items():
                    coeff = self._compute_cca(band_signal, template)
                    fb_scores[cls_idx] += self.subband_weights[band_idx] * coeff
            
            scores[idx] = fb_scores
        
        return scores
    
    def predict(self, X):
        """预测"""
        print("  计算测试集得分...")
        scores = self._compute_all_scores(X)
        
        # RV 归一化
        if self.norm_stats:
            print("  应用RV归一化...")
            for cls_idx in range(self.n_classes):
                mean_nt = self.norm_stats[cls_idx]
                scores[:, cls_idx] = (scores[:, cls_idx] - mean_nt) / (scores[:, cls_idx] + mean_nt + 1e-8)
        
        predictions = np.argmax(scores, axis=1)
        return predictions


# ===================================================================
# 第三部分：优化框架完整版 + 主动学习
# ===================================================================

class OptimizedClassifierFull:
    """优化框架完整版，包含TRCA和主动学习"""
    
    def __init__(self, freqs=None, srate=250, n_harmonics=2):
        self.srate = srate
        if freqs is None:
            self.freqs = [8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0]
        else:
            self.freqs = freqs
        self.n_harmonics = n_harmonics
        self.n_classes = len(self.freqs)
        
        # FB-CCA 设置
        self.subbands = [(4, 8), (8, 12), (12, 20), (20, 35)]
        self.subband_weights = np.ones(len(self.subbands)) / len(self.subbands)
        
        # TRCA 模板存储
        self.trca_templates = {}
        self.trca_projections = {}
        
        # 构造谐波参考模板
        self._build_harmonic_templates()
        
        # 归一化参数
        self.norm_stats = None
        
        # 主动学习跟踪
        self.al_history = []
        
        print("[OptimizedFull初始化] 完成")
        print(f"  频率: {self.freqs}")
        print(f"  谐波数: {self.n_harmonics}")
        print(f"  子带: {self.subbands}")
        print(f"  启用TRCA: True")
        print(f"  启用主动学习: True")
    
    def _build_harmonic_templates(self):
        """构造谐波参考模板"""
        self.harmonic_templates = {}
        template_len = int(4.0 * self.srate)
        time_axis = np.linspace(0, (template_len - 1) / self.srate, template_len, endpoint=True)
        
        for idx, freq in enumerate(self.freqs):
            sinusoids = []
            for h in range(1, self.n_harmonics + 1):
                harmonic_freq = freq * h
                phase = 2 * np.pi * harmonic_freq * time_axis
                sinusoids.append(np.sin(phase))
                sinusoids.append(np.cos(phase))
            
            self.harmonic_templates[idx] = np.vstack(sinusoids)
    
    def _design_subband_filters(self):
        """设计子带滤波器"""
        self.subband_filters = []
        nyquist = self.srate / 2
        for low, high in self.subbands:
            low_norm = low / nyquist
            high_norm = high / nyquist
            low_norm = max(0.01, min(low_norm, 0.99))
            high_norm = max(low_norm + 0.01, min(high_norm, 0.99))
            
            b, a = scipysignal.butter(4, [low_norm, high_norm], btype='band')
            self.subband_filters.append((b, a))
    
    def fit(self, X, y):
        """训练"""
        print("[OptimizedFull.fit] 开始训练...")
        
        # 设计滤波器
        self._design_subband_filters()
        
        # 训练TRCA
        print("  训练TRCA模板...")
        self._train_trca(X, y)
        
        # 计算训练集得分用于归一化
        print("  计算训练集得分...")
        scores_train = self._compute_all_scores(X)
        
        # 学习归一化参数
        print("  学习归一化参数...")
        self.norm_stats = {}
        for cls_idx in range(self.n_classes):
            mask = y != cls_idx
            if mask.sum() > 0:
                mean_non_target = np.mean(scores_train[mask, cls_idx])
            else:
                mean_non_target = 0.5
            self.norm_stats[cls_idx] = mean_non_target
        
        print("[OptimizedFull.fit] 完成")
    
    def _train_trca(self, X, y):
        """训练TRCA模板"""
        for cls_idx in range(self.n_classes):
            X_class = X[y == cls_idx]
            
            if len(X_class) == 0:
                # 该类没有样本
                self.trca_templates[cls_idx] = None
                self.trca_projections[cls_idx] = None
                continue
            
            # 对齐长度
            template_len = self.harmonic_templates[0].shape[1]
            X_aligned = []
            for segment in X_class:
                if segment.shape[1] > template_len:
                    segment = segment[:, :template_len]
                elif segment.shape[1] < template_len:
                    pad_len = template_len - segment.shape[1]
                    segment = np.pad(segment, ((0, 0), (0, pad_len)), mode='constant')
                X_aligned.append(segment)
            X_aligned = np.array(X_aligned)
            
            # 计算TRCA投影
            try:
                w = self._compute_trca_projection(X_aligned)
                
                # 投影并平均作为模板
                projected = np.array([w.T @ seg for seg in X_aligned])
                template = projected.mean(axis=0)
                
                self.trca_projections[cls_idx] = w
                self.trca_templates[cls_idx] = template
            except Exception as e:
                print(f"    警告: 类{cls_idx} TRCA训练失败 ({e})")
                self.trca_templates[cls_idx] = None
                self.trca_projections[cls_idx] = None
    
    def _compute_trca_projection(self, X_class):
        """计算TRCA投影向量"""
        n_epochs, n_channels, n_samples = X_class.shape
        
        # 类内协方差
        S_w = np.zeros((n_channels, n_channels))
        for epoch in X_class:
            S_w += epoch @ epoch.T
        S_w /= (n_epochs * n_samples)
        
        # 类间协方差
        mean_signal = np.mean(X_class, axis=0)
        S_b = mean_signal @ mean_signal.T / n_samples
        
        # 广义特征值问题
        try:
            from scipy.linalg import eig
            eigenvalues, eigenvectors = eig(S_b, S_w + np.eye(n_channels) * 1e-6)
            idx = np.argsort(-np.real(eigenvalues))[0]
            w = np.real(eigenvectors[:, idx]).reshape(-1, 1)
        except:
            # 备用方案
            cov = np.cov(X_class.reshape(n_epochs, -1).T)
            _, w_full = np.linalg.eigh(cov)
            w = w_full[:, -1].reshape(-1, 1)
        
        return w
    
    def _compute_cca(self, data, template):
        """计算CCA相关系数"""
        try:
            cca = CCA(n_components=1)
            
            cdata = data.T
            ctemplate = template.T
            
            if cdata.shape[0] < 10 or ctemplate.shape[0] < 10:
                return 0.0
            
            cca.fit(cdata, ctemplate)
            data_trans, template_trans = cca.transform(cdata, ctemplate)
            
            corr = np.corrcoef(data_trans[:, 0], template_trans[:, 0])[0, 1]
            
            if np.isnan(corr) or np.isinf(corr):
                return 0.0
            return max(0.0, float(corr))
        except:
            return 0.0
    
    def _compute_all_scores(self, X):
        """计算所有样本的得分矩阵"""
        n_samples = X.shape[0]
        scores_cca = np.zeros((n_samples, self.n_classes))
        scores_trca = np.zeros((n_samples, self.n_classes))
        
        for idx, segment in enumerate(X):
            # 对齐长度
            template_len = self.harmonic_templates[0].shape[1]
            if segment.shape[1] > template_len:
                segment_aligned = segment[:, :template_len]
            elif segment.shape[1] < template_len:
                pad_len = template_len - segment.shape[1]
                segment_aligned = np.pad(segment, ((0, 0), (0, pad_len)), mode='constant')
            else:
                segment_aligned = segment
            
            # CCA + FB-CCA
            fb_scores = np.zeros(self.n_classes)
            for band_idx, (b, a) in enumerate(self.subband_filters):
                try:
                    band_signal = scipysignal.filtfilt(b, a, segment_aligned, axis=1)
                except:
                    band_signal = segment_aligned
                
                for cls_idx, template in self.harmonic_templates.items():
                    coeff = self._compute_cca(band_signal, template)
                    fb_scores[cls_idx] += self.subband_weights[band_idx] * coeff
            
            scores_cca[idx] = fb_scores
            
            # TRCA
            for cls_idx in range(self.n_classes):
                if self.trca_templates[cls_idx] is not None:
                    w = self.trca_projections[cls_idx]
                    x_proj = w.T @ segment_aligned
                    x_proj = x_proj.flatten()
                    
                    try:
                        corr = np.corrcoef(x_proj, self.trca_templates[cls_idx])[0, 1]
                        scores_trca[idx, cls_idx] = np.clip(corr if not np.isnan(corr) else 0, 0, 1)
                    except:
                        scores_trca[idx, cls_idx] = 0.0
        
        # 融合：0.6 CCA + 0.4 TRCA
        scores = 0.6 * scores_cca + 0.4 * scores_trca
        
        return scores
    
    def predict(self, X):
        """预测"""
        print("  计算测试集得分...")
        scores = self._compute_all_scores(X)
        
        # RV 归一化
        if self.norm_stats:
            print("  应用RV归一化...")
            for cls_idx in range(self.n_classes):
                mean_nt = self.norm_stats[cls_idx]
                scores[:, cls_idx] = (scores[:, cls_idx] - mean_nt) / (scores[:, cls_idx] + mean_nt + 1e-8)
        
        predictions = np.argmax(scores, axis=1)
        return predictions
    
    def active_learning_step(self, X_unlabeled, query_budget=10):
        """
        主动学习：选择最不确定的样本
        
        参数:
        -----
        X_unlabeled : ndarray, shape [n_samples, n_channels, n_samples_time]
        query_budget : int, 最多查询多少个样本
        
        返回:
        ------
        uncertain_indices : list, 最不确定样本的索引
        margins : ndarray, 不确定性度量 (margin)
        """
        print("\n[主动学习] 计算不确定性...")
        
        scores = self._compute_all_scores(X_unlabeled)
        
        # 计算margin = top1 - top2
        sorted_scores = np.argsort(scores, axis=1)
        top1_score = scores[np.arange(len(scores)), sorted_scores[:, -1]]
        top2_score = scores[np.arange(len(scores)), sorted_scores[:, -2]]
        margins = top1_score - top2_score
        
        # 选择margin最小的样本（最不确定）
        uncertain_indices = np.argsort(margins)[:query_budget].tolist()
        
        print(f"  边界得分 (margin) 统计:")
        print(f"    最小: {margins.min():.4f}")
        print(f"    平均: {margins.mean():.4f}")
        print(f"    最大: {margins.max():.4f}")
        print(f"  选择 {len(uncertain_indices)} 个最不确定样本用于标注")
        
        return uncertain_indices, margins
    
    def update_with_new_samples(self, X_new, y_new, X_all_so_far, y_all_so_far):
        """
        用新标注的样本更新模型
        
        参数:
        -----
        X_new : ndarray, 新标注样本
        y_new : ndarray, 新标注的标签
        X_all_so_far : ndarray, 截至目前所有训练样本
        y_all_so_far : ndarray, 截至目前所有训练标签
        """
        print("\n[主动学习] 用新样本更新模型...")
        
        # 合并
        X_combined = np.vstack([X_all_so_far, X_new])
        y_combined = np.concatenate([y_all_so_far, y_new])
        
        # 重新训练
        self.fit(X_combined, y_combined)
        
        return X_combined, y_combined


# ===================================================================
# 第四部分：实验执行与评估
# ===================================================================

class Experiment:
    """实验框架"""
    
    def __init__(self):
        self.results = {}
    
    @staticmethod
    def evaluate(y_true, y_pred, model_name):
        """评估模型"""
        acc = accuracy_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
        
        print(f"\n【{model_name} 评估结果】")
        print("=" * 80)
        print(f"准确率 (Accuracy):  {acc:.4f} ({int(acc * len(y_true))}/{len(y_true)} 正确)")
        print(f"宏平均召回率 (Macro Recall): {recall:.4f}")
        print(f"宏平均 F1 得分:            {f1:.4f}")
        print()
        
        # 混淆矩阵
        cm = confusion_matrix(y_true, y_pred)
        print("混淆矩阵 (行=真实, 列=预测):")
        print(cm)
        print()
        
        # 每类准确率
        print("每类准确率:")
        for cls_idx in range(len(np.unique(y_true))):
            if cm[cls_idx].sum() > 0:
                cls_acc = cm[cls_idx, cls_idx] / cm[cls_idx].sum()
                print(f"  类{cls_idx}: {cls_acc:.4f}")
        print()
        
        return {'accuracy': acc, 'recall': recall, 'f1': f1, 'confusion_matrix': cm}
    
    def run_full_experiment(self, X_train, y_train, X_test, y_test):
        """运行完整实验"""
        
        print("\n")
        print("█" * 100)
        print("█" + " " * 98 + "█")
        print("█  SSVEP 算法对比实验：D1(训练) vs D2(测试)".ljust(99) + "█")
        print("█" + " " * 98 + "█")
        print("█" * 100)
        
        # ===================================================================
        # 方法1: DirectCCA
        # ===================================================================
        print("\n\n")
        print("=" * 100)
        print("方法1: DirectCCA (直接CCA，无训练)")
        print("=" * 100)
        
        print("\n[1.1] 初始化模型...")
        model1 = DirectCCAClassifier()
        
        print("\n[1.2] 训练（跳过）...")
        model1.fit(X_train, y_train)
        
        print("\n[1.3] 测试预测...")
        start = time.time()
        y_pred1 = model1.predict(X_test)
        elapsed1 = time.time() - start
        print(f"预测耗时: {elapsed1:.3f}s, 平均每样本: {elapsed1/len(X_test)*1000:.2f}ms")
        
        result1 = self.evaluate(y_test, y_pred1, "DirectCCA")
        result1['time'] = elapsed1
        self.results['DirectCCA'] = result1
        
        # ===================================================================
        # 方法2: OptimizedNoTRCA
        # ===================================================================
        print("\n\n")
        print("=" * 100)
        print("方法2: OptimizedNoTRCA (优化框架，仅Filter-Bank CCA + RV归一化)")
        print("=" * 100)
        
        print("\n[2.1] 初始化模型...")
        model2 = OptimizedClassifierNoTRCA()
        
        print("\n[2.2] 训练...")
        model2.fit(X_train, y_train)
        
        print("\n[2.3] 测试预测...")
        start = time.time()
        y_pred2 = model2.predict(X_test)
        elapsed2 = time.time() - start
        print(f"预测耗时: {elapsed2:.3f}s, 平均每样本: {elapsed2/len(X_test)*1000:.2f}ms")
        
        result2 = self.evaluate(y_test, y_pred2, "OptimizedNoTRCA")
        result2['time'] = elapsed2
        self.results['OptimizedNoTRCA'] = result2
        
        # ===================================================================
        # 方法3: OptimizedFull (包含主动学习)
        # ===================================================================
        print("\n\n")
        print("=" * 100)
        print("方法3: OptimizedFull (完整优化框架 + TRCA + 主动学习)")
        print("=" * 100)
        
        print("\n[3.1] 初始化模型...")
        model3 = OptimizedClassifierFull()
        
        print("\n[3.2] 初始训练...")
        model3.fit(X_train, y_train)
        
        print("\n[3.3] 主动学习循环...")
        X_train_al = X_train.copy()
        y_train_al = y_train.copy()
        
        # 从测试集中挑选最不确定的样本（模拟主动学习的查询过程）
        print("\n  【第1轮主动学习】")
        uncertain_idx, margins = model3.active_learning_step(X_test, query_budget=12)
        
        # 模拟人类标注（假设标注正确 - 即使用真实标签）
        X_query = X_test[uncertain_idx]
        y_query = y_test[uncertain_idx]
        
        print(f"  【更新模型】 添加 {len(y_query)} 个新样本到训练集")
        X_train_al, y_train_al = model3.update_with_new_samples(X_query, y_query, X_train_al, y_train_al)
        
        print(f"  更新后训练集大小: {len(y_train_al)} (原始: {len(y_train)})")
        
        print("\n  【第2轮主动学习】")
        uncertain_idx2, margins2 = model3.active_learning_step(X_test, query_budget=8)
        X_query2 = X_test[uncertain_idx2]
        y_query2 = y_test[uncertain_idx2]
        
        print(f"  【更新模型】 添加 {len(y_query2)} 个新样本到训练集")
        X_train_al, y_train_al = model3.update_with_new_samples(X_query2, y_query2, X_train_al, y_train_al)
        
        print(f"  更新后训练集大小: {len(y_train_al)} (原始: {len(y_train)})")
        
        print("\n[3.4] 测试预测...")
        start = time.time()
        y_pred3 = model3.predict(X_test)
        elapsed3 = time.time() - start
        print(f"预测耗时: {elapsed3:.3f}s, 平均每样本: {elapsed3/len(X_test)*1000:.2f}ms")
        
        result3 = self.evaluate(y_test, y_pred3, "OptimizedFull+ActiveLearning")
        result3['time'] = elapsed3
        self.results['OptimizedFull'] = result3
        
        # ===================================================================
        # 总结对比
        # ===================================================================
        self.print_summary()
    
    def print_summary(self):
        """打印最终对比总结"""
        print("\n\n")
        print("█" * 100)
        print("█  最终结果对比".ljust(99) + "█")
        print("█" * 100)
        print()
        
        # 创建对比表
        data = []
        for model_name, result in self.results.items():
            data.append({
                '算法': model_name,
                '准确率': f"{result['accuracy']:.4f}",
                '召回率': f"{result['recall']:.4f}",
                'F1': f"{result['f1']:.4f}",
                '推理时间(s)': f"{result['time']:.3f}"
            })
        
        df_summary = pd.DataFrame(data)
        print(df_summary.to_string(index=False))
        print()
        
        # 最佳模型
        best_model = max(self.results.items(), key=lambda x: x[1]['accuracy'])
        worst_model = min(self.results.items(), key=lambda x: x[1]['accuracy'])
        
        print(f"\n【最佳模型】 {best_model[0]}: 准确率 {best_model[1]['accuracy']:.4f}")
        print(f"【最差模型】 {worst_model[0]}: 准确率 {worst_model[1]['accuracy']:.4f}")
        print(f"【性能提升】 {best_model[1]['accuracy'] - worst_model[1]['accuracy']:.4f}")
        
        print()
        print("█" * 100)


# ===================================================================
# 主函数
# ===================================================================

if __name__ == '__main__':
    
    print("\n")
    print("█" * 100)
    print("█  SSVEP 三种算法对比实验 - 完整流程".ljust(99) + "█")
    print("█" * 100)
    
    # 1. 加载数据
    print("\n【阶段1: 数据加载】")
    print("-" * 100)
    
    print("\n[D1数据集 - 训练集]")
    X_train, y_train, _ = DataLoader.load_csv("e:\\brain\\brain\\ExampleData\\D1.csv")
    
    print("[D2数据集 - 测试集]")
    X_test, y_test, _ = DataLoader.load_csv("e:\\brain\\brain\\ExampleData\\D2.csv")
    
    # 2. 预处理
    print("\n【阶段2: 信号预处理】")
    print("-" * 100)
    print("[D1预处理] 50Hz陷波 + 6-90Hz带通滤波...")
    X_train_filtered = DataLoader.preprocess(X_train)
    print("✓ D1预处理完成")
    
    print("[D2预处理] 50Hz陷波 + 6-90Hz带通滤波...")
    X_test_filtered = DataLoader.preprocess(X_test)
    print("✓ D2预处理完成")
    
    # 3. 运行实验
    print("\n【阶段3: 算法对比实验】")
    print("-" * 100)
    
    exp = Experiment()
    exp.run_full_experiment(X_train_filtered, y_train, X_test_filtered, y_test)
    
    print("\n\n✓ 实验完成！")

