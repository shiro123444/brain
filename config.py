#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SSVEP 竞赛配置文件
统一管理频率、采样率、任务配置等
"""

import numpy as np

# ================================================================
# 核心配置
# ================================================================

COMPETITION_CONFIG = {
    # 版本信息
    'version': '2024_official',
    'description': 'SSVEP 8分类脑机接口',
    
    # 采样率
    'sampling_rate': 250,  # Hz
    
    # 频率映射 (根据竞赛规则)
    # 注意：需根据实际数据验证，这里使用标准SSVEP频率
    'freq_map': {
        0: 8.0,    # 可根据实际调整 (8.18?)
        1: 9.0,    # 可根据实际调整 (8.97?)
        2: 10.0,   # 可根据实际调整 (9.81?)
        3: 11.0,   # 可根据实际调整 (10.70?)
        4: 12.0,   # 可根据实际调整 (11.64?)
        5: 13.0,   # 可根据实际调整 (12.62?)
        6: 14.0,   # 可根据实际调整 (13.65?)
        7: 15.0,   # 可根据实际调整 (14.71?)
    },
    
    # 数据配置
    'n_channels': 6,
    'channel_names': ['CP3', 'CPZ', 'CP4', 'PO3', 'POZ', 'PO4'],
    'n_classes': 8,
    
    # 数据特征（基于实际数据）
    'data_config': {
        'n_samples_per_segment': 1000,      # 根据实测数据：都是1000点
        'n_segments': 48,                    # 每个数据集的片段总数
        'duration_sec': 4.0,                 # 1000点 / 250Hz = 4秒
    },
}

# ================================================================
# 信号处理配置
# ================================================================

SIGNAL_PROCESSING_CONFIG = {
    # 预处理
    'preprocess': {
        'notch_freq': 50,           # 50Hz 工频陷波
        'notch_Q': 35,              # 品质因数
        'bandpass_low': 6,          # Hz
        'bandpass_high': 90,        # Hz
        'bandpass_order': 4,        # 椭圆滤波器阶数
    },
    
    # CCA配置
    'cca': {
        'n_components': 1,          # CCA分量数
        'n_harmonics': 2,           # 使用基频和2倍频
        'harmonic_weights': 'uniform',  # 'uniform', 'exp_decay', 'reciprocal'
    },
    
    # Filter-Bank CCA
    'filter_bank': {
        'enabled': True,
        'subbands': [
            (4, 8),      # Theta
            (8, 12),     # Alpha
            (12, 20),    # Beta-low
            (20, 35),    # Beta-high
        ],
        'subband_weights': 'uniform',  # 可选: 自定义权重
        'butter_order': 4,
    },
    
    # TRCA配置
    'trca': {
        'enabled': True,
        'n_components': 1,
        'use_shrinkage': False,     # Ledoit-Wolf 收缩
    },
    
    # 得分归一化
    'normalization': {
        'enabled': True,
        'method': 'rv',    # 'rv', 'zscore', 'quantile'
    },
    
    # 鲁棒处理
    'robust': {
        'detect_outliers': False,   # MAD 异常检测
        'outlier_threshold': 3.0,   # MAD倍数
        'channel_weighting': False, # 通道加权
    },
}

# ================================================================
# 模型融合配置
# ================================================================

FUSION_CONFIG = {
    'cca_fb_weight': 0.5,      # 标准CCA和FB-CCA融合权重
    'cca_trca_weight': 0.6,    # CCA和TRCA融合权重
    'trca_weight': 0.4,
    'use_stacking': False,     # Stacking集成
}

# ================================================================
# 主动学习配置
# ================================================================

ACTIVE_LEARNING_CONFIG = {
    'enabled': True,
    'method': 'margin',         # 'margin', 'entropy', 'bald'
    'query_budget': [12, 8],    # 每轮查询数量
    'n_rounds': 2,              # 迭代轮数
}

# ================================================================
# 交叉验证配置
# ================================================================

EVALUATION_CONFIG = {
    'cv_method': 'stratified_kfold',
    'cv_splits': 5,
    'random_state': 42,
    'metrics': ['accuracy', 'recall', 'f1', 'confusion_matrix'],
}

# ================================================================
# 验证函数
# ================================================================

def validate_config():
    """验证配置的有效性"""
    config = COMPETITION_CONFIG
    
    # 检查频率数量
    assert len(config['freq_map']) == config['n_classes'], \
        f"频率数量({len(config['freq_map'])}) != 类别数({config['n_classes']})"
    
    # 检查频率有效性
    freqs = list(config['freq_map'].values())
    assert all(f > 0 for f in freqs), "频率必须为正数"
    assert len(set(freqs)) == len(freqs), "频率必须互不相同"
    assert all(f <= config['sampling_rate']/2 for f in freqs), \
        f"频率必须小于采样率一半 ({config['sampling_rate']/2}Hz)"
    
    # 检查通道配置
    assert len(config['channel_names']) == config['n_channels'], \
        f"通道数量不匹配"
    
    print("✓ 配置验证通过")

if __name__ == '__main__':
    validate_config()
    
    print("\n配置信息:")
    print(f"  版本: {COMPETITION_CONFIG['version']}")
    print(f"  采样率: {COMPETITION_CONFIG['sampling_rate']} Hz")
    print(f"  通道: {COMPETITION_CONFIG['n_channels']} ({', '.join(COMPETITION_CONFIG['channel_names'])})")
    print(f"  类别: {COMPETITION_CONFIG['n_classes']}")
    print(f"  频率: {list(COMPETITION_CONFIG['freq_map'].values())}")
