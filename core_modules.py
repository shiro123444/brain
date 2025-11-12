#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
改进的SSVEP识别核心模块
统一的预处理、参考信号、CCA计算
"""

import numpy as np
from scipy import signal
from sklearn.cross_decomposition import CCA
from config import COMPETITION_CONFIG, SIGNAL_PROCESSING_CONFIG

class ImprovedSignalPreprocessor:
    """改进的信号预处理模块"""
    
    def __init__(self, config=None):
        self.config = config or SIGNAL_PROCESSING_CONFIG
        self.fs = COMPETITION_CONFIG['sampling_rate']
    
    def preprocess(self, data):
        """
        应用完整的预处理管道
        
        参数:
            data: ndarray, shape [n_channels, n_samples] 或 [n_epochs, n_channels, n_samples]
        
        返回:
            filtered: ndarray, 同输入形状
        """
        if data.ndim == 2:
            return self._preprocess_2d(data)
        elif data.ndim == 3:
            n_epochs = data.shape[0]
            filtered = np.zeros_like(data)
            for i in range(n_epochs):
                filtered[i] = self._preprocess_2d(data[i])
            return filtered
        else:
            raise ValueError(f"期望2D或3D数组，得到{data.ndim}D")
    
    def _preprocess_2d(self, data):
        """处理单个2D数据 [n_channels, n_samples]"""
        config = self.config['preprocess']
        
        try:
            # 1. 50Hz 陷波滤波
            Q = config['notch_Q']
            b_notch, a_notch = signal.iircomb(
                config['notch_freq'], Q, ftype='notch', fs=self.fs
            )
            
            if data.shape[1] > len(a_notch) * 2:
                data_notch = signal.filtfilt(b_notch, a_notch, data, axis=1)
            else:
                data_notch = data.copy()
            
            # 2. 6-90Hz 带通滤波 (椭圆滤波器)
            nyquist = self.fs / 2
            low_norm = config['bandpass_low'] / nyquist
            high_norm = config['bandpass_high'] / nyquist
            
            # 计算必要阶数
            N, Wn = signal.ellipord(
                [low_norm, high_norm],
                [config['bandpass_low']/nyquist*0.8, 
                 config['bandpass_high']/nyquist*1.1],
                gpass=3,
                gstop=40
            )
            
            # 限制阶数
            N = min(N, data.shape[1] // 4)
            if N < 1:
                return data_notch
            
            b_bp, a_bp = signal.ellip(
                N, 1, 90, [low_norm, high_norm], 'bandpass'
            )
            
            if data_notch.shape[1] > len(a_bp) * 2:
                data_filtered = signal.filtfilt(b_bp, a_bp, data_notch, axis=1)
            else:
                data_filtered = data_notch
            
            return data_filtered
        
        except Exception as e:
            print(f"警告: 预处理失败 ({e})，使用原始数据")
            return data.copy()


class ReferenceSignalGenerator:
    """参考信号生成器"""
    
    def __init__(self, config=None):
        self.config = config or COMPETITION_CONFIG
        self.fs = self.config['sampling_rate']
        self.freq_map = self.config['freq_map']
    
    def generate_references(self, n_samples, n_harmonics=2, harmonic_weights='uniform'):
        """
        生成参考信号
        
        参数:
            n_samples: 采样点数
            n_harmonics: 谐波数
            harmonic_weights: 权重策略
        
        返回:
            templates: dict {class_id: ref_matrix}
        """
        t = np.arange(n_samples) / self.fs
        templates = {}
        
        # 计算谐波权重
        if harmonic_weights == 'uniform':
            weights = np.ones(n_harmonics)
        elif harmonic_weights == 'exp_decay':
            weights = np.exp(-0.5 * np.arange(1, n_harmonics + 1))
        elif harmonic_weights == 'reciprocal':
            weights = 1.0 / np.arange(1, n_harmonics + 1)
        else:
            weights = np.ones(n_harmonics)
        
        weights = weights / weights.sum()
        
        # 为每个频率生成参考信号
        for class_id, freq in self.freq_map.items():
            sinusoids = []
            for h in range(1, n_harmonics + 1):
                harmonic_freq = freq * h
                phase = 2 * np.pi * harmonic_freq * t
                w = weights[h-1]
                sinusoids.append(w * np.sin(phase))
                sinusoids.append(w * np.cos(phase))
            
            templates[class_id] = np.vstack(sinusoids)
        
        return templates


class ImprovedCCACalculator:
    """改进的CCA计算器（数值稳定版本）"""
    
    @staticmethod
    def compute_cca_score(data, template, n_components=1):
        """
        计算稳定的CCA相关系数
        
        参数:
            data: [n_channels, n_samples]
            template: [n_refs, n_samples]
            n_components: CCA分量数
        
        返回:
            score: float [0, 1]
        """
        try:
            # 确保数据足够
            if data.shape[1] < 10 or template.shape[1] < 10:
                return 0.0
            
            # 标准化
            data_norm = ImprovedCCACalculator._standardize(data)
            template_norm = ImprovedCCACalculator._standardize(template)
            
            # CCA
            cca = CCA(n_components=n_components)
            cca.fit(data_norm.T, template_norm.T)
            
            # 使用CCA的score方法而不是手动计算相关系数
            score = cca.score(data_norm.T, template_norm.T)
            
            # 确保分数有效
            if np.isnan(score) or np.isinf(score):
                return 0.0
            
            return float(np.clip(score, 0, 1))
        
        except Exception as e:
            # 异常时返回0
            return 0.0
    
    @staticmethod
    def _standardize(X):
        """标准化数据"""
        mean = X.mean(axis=1, keepdims=True)
        std = X.std(axis=1, keepdims=True)
        return (X - mean) / (std + 1e-10)


class FilterBankAnalyzer:
    """子带滤波分析器"""
    
    def __init__(self, config=None):
        self.config = config or SIGNAL_PROCESSING_CONFIG
        self.subbands = self.config['filter_bank']['subbands']
        self.fs = COMPETITION_CONFIG['sampling_rate']
        self.order = self.config['filter_bank']['butter_order']
        self._design_filters()
    
    def _design_filters(self):
        """设计子带滤波器"""
        self.filters = []
        nyquist = self.fs / 2
        
        for low, high in self.subbands:
            low_norm = max(0.01, min(low / nyquist, 0.99))
            high_norm = max(low_norm + 0.01, min(high / nyquist, 0.99))
            
            b, a = signal.butter(self.order, [low_norm, high_norm], btype='band')
            self.filters.append((b, a))
    
    def apply_subbands(self, data):
        """
        应用子带滤波
        
        参数:
            data: [n_channels, n_samples]
        
        返回:
            subbands_data: list of [n_channels, n_samples]
        """
        subbands_data = []
        
        for b, a in self.filters:
            try:
                filtered = signal.filtfilt(b, a, data, axis=-1)
                subbands_data.append(filtered)
            except:
                subbands_data.append(data.copy())
        
        return subbands_data
    
    def get_weights(self):
        """获取子带权重"""
        n_bands = len(self.subbands)
        weights = np.ones(n_bands) / n_bands
        return weights


# ================================================================
# 示例使用
# ================================================================

if __name__ == '__main__':
    print("导入改进的SSVEP核心模块...")
    
    # 初始化各模块
    preprocessor = ImprovedSignalPreprocessor()
    ref_gen = ReferenceSignalGenerator()
    cca_calc = ImprovedCCACalculator()
    fb_analyzer = FilterBankAnalyzer()
    
    print("✓ 所有模块初始化成功")
    
    # 生成虚拟数据测试
    X_test = np.random.randn(6, 1000)
    
    # 预处理
    X_processed = preprocessor.preprocess(X_test)
    print(f"✓ 预处理完成: {X_processed.shape}")
    
    # 生成参考信号
    templates = ref_gen.generate_references(1000, n_harmonics=2)
    print(f"✓ 生成{len(templates)}个参考信号")
    
    # CCA计算
    score = cca_calc.compute_cca_score(X_processed, templates[0])
    print(f"✓ CCA计算完成: 得分={score:.4f}")
    
    # 子带分析
    subbands = fb_analyzer.apply_subbands(X_processed)
    print(f"✓ 子带分析完成: {len(subbands)}个子带")
