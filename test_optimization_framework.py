"""
===================================================================
SSVEP 优化框架 - 真实数据测试与验证
===================================================================

本脚本演示如何用真实竞赛数据(D1, D2)测试优化算法
"""

import sys
import io
# 设置UTF-8输出编码
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import numpy as np
import pandas as pd
from pathlib import Path
import time
import warnings

warnings.filterwarnings('ignore')

from ssvep_optimization_framework import (
    OptimizedSSVEPClassifier,
    SSVEPEvaluator,
    ProductionSSVEPPipeline,
    DEFAULT_CONFIG,
    RobustPreprocessor,
)

# ===================================================================
# 配置
# ===================================================================

DATA_DIR = Path(__file__).parent / "ExampleData"
OUTPUT_DIR = Path(__file__).parent / "optimization_results"
OUTPUT_DIR.mkdir(exist_ok=True)

FREQ_MAP = {
    0: 8.18, 1: 8.97, 2: 9.81, 3: 10.70,
    4: 11.64, 5: 12.62, 6: 13.65, 7: 14.71
}

# ===================================================================
# 数据加载与预处理
# ===================================================================

class DataLoader:
    """加载竞赛D1/D2数据"""
    
    @staticmethod
    def load_csv(csv_file, fs=250):
        """
        加载CSV文件
        
        格式: [CP3, CPZ, CP4, PO3, POZ, PO4, taskID, stimID]
        共8列: 6个EEG通道 + taskID + stimID
        """
        df = pd.read_csv(csv_file)
        print(f"[Data] 加载 {csv_file.name}: shape={df.shape}")
        
        # 提取列
        eeg_cols = ['CP3', 'CPZ', 'CP4', 'PO3', 'POZ', 'PO4']
        eeg_data = df[eeg_cols].values  # [n_samples, 6_channels]
        
        # 使用 stimID (目标频率标签) 作为分段标签
        stim_ids = df['stimID'].values.astype(int)
        
        return eeg_data, stim_ids
    
    @staticmethod
    def segment_by_taskid(eeg_data, stim_ids, window_sec=4.0, fs=250):
        """
        按 stimID 分段
        
        参数:
        -----
        eeg_data : ndarray, [n_samples, n_channels]
        stim_ids : ndarray, [n_samples]
            刺激ID (对应的频率类别)
        window_sec : float
            窗口长度 (秒)
        fs : float
            采样率
        
        返回:
        ------
        X : ndarray, [n_epochs, n_channels, n_samples]
            分段的epoch数据
        y : ndarray, [n_epochs]
            对应的标签
        """
        window_samples = int(window_sec * fs)
        
        # 找出标签变化的地方 (新任务块的开始)
        stim_changes = np.where(np.diff(stim_ids) != 0)[0] + 1
        boundaries = np.concatenate([[0], stim_changes, [len(stim_ids)]])
        
        X_list = []
        y_list = []
        
        for i in range(len(boundaries) - 1):
            start_idx = boundaries[i]
            end_idx = boundaries[i + 1]
            
            # 如果该块足够长，可以提取多个窗口
            segment_data = eeg_data[start_idx:end_idx]  # [seg_len, 6]
            stim_id = stim_ids[start_idx]
            
            # 从该segment中提取窗口 (窗口不重叠)
            n_windows = len(segment_data) // window_samples
            
            for j in range(n_windows):
                start = j * window_samples
                end = start + window_samples
                epoch = segment_data[start:end].T  # [6, window_samples]
                X_list.append(epoch)
                y_list.append(stim_id)
        
        X = np.array(X_list)
        y = np.array(y_list)
        
        print(f"[Data] 分段完成: {len(X)} epochs, 形状 {X.shape}")
        print(f"[Data] 标签分布: {dict(zip(*np.unique(y, return_counts=True)))}")
        
        return X, y
    
    @staticmethod
    def load_and_segment(csv_file, window_sec=4.0, fs=250):
        """一站式加载与分段"""
        eeg_data, task_ids = DataLoader.load_csv(csv_file, fs)
        X, y = DataLoader.segment_by_taskid(eeg_data, task_ids, window_sec, fs)
        return X, y


# ===================================================================
# 测试场景 1: 基线对比
# ===================================================================

def test_baseline_vs_optimized(X_train, y_train, X_test, y_test):
    """
    对比: 基线CCA vs 优化算法
    """
    print("\n" + "="*70)
    print("场景1: 基线CCA vs 优化算法对比")
    print("="*70)
    
    # 配置1: 基线 (仅CCA)
    config_baseline = DEFAULT_CONFIG.copy()
    config_baseline['use_fb_cca'] = False
    config_baseline['use_trca'] = False
    config_baseline['use_normalization'] = False
    
    # 配置2: 优化 (FB-CCA + TRCA + 归一化)
    config_optimized = DEFAULT_CONFIG.copy()
    
    results = {}
    
    for name, config in [("基线CCA", config_baseline), ("优化版本", config_optimized)]:
        print(f"\n📊 测试: {name}")
        print("-" * 70)
        
        # 训练
        start = time.time()
        model = OptimizedSSVEPClassifier(**config)
        model.fit(X_train, y_train)
        train_time = time.time() - start
        
        # 预测
        start = time.time()
        y_pred = model.predict(X_test)
        pred_time = time.time() - start
        
        # 评估
        acc = (y_pred == y_test).mean()
        from sklearn.metrics import confusion_matrix, recall_score, f1_score
        
        cm = confusion_matrix(y_test, y_pred)
        recall_per_class = cm.diagonal() / cm.sum(axis=1)
        recall_macro = recall_score(y_test, y_pred, average='macro')
        f1_macro = f1_score(y_test, y_pred, average='macro')
        
        results[name] = {
            'accuracy': acc,
            'recall_macro': recall_macro,
            'f1_macro': f1_macro,
            'recall_per_class': recall_per_class,
            'train_time': train_time,
            'pred_time_ms': pred_time * 1000 / len(X_test),
        }
        
        print(f"✓ 准确率: {acc:.4f} ({int(acc*len(y_test))}/{len(y_test)})")
        print(f"✓ 宏平均召回率: {recall_macro:.4f}")
        print(f"✓ 宏平均F1: {f1_macro:.4f}")
        print(f"✓ 每类召回率: {', '.join([f'{r:.2%}' for r in recall_per_class])}")
        print(f"✓ 训练耗时: {train_time:.3f}s")
        print(f"✓ 预测延迟: {results[name]['pred_time_ms']:.2f}ms/epoch")
    
    # 改进量
    acc_gain = (results["优化版本"]["accuracy"] - results["基线CCA"]["accuracy"]) * 100
    print(f"\n🎯 准确率改进: +{acc_gain:.2f}pp (相对提升 {acc_gain/results['基线CCA']['accuracy']*100:.1f}%)")
    
    return results


# ===================================================================
# 测试场景 2: 交叉验证
# ===================================================================

def test_cross_validation(X, y, k=5):
    """
    K折交叉验证评估
    """
    print("\n" + "="*70)
    print(f"场景2: {k}折交叉验证评估")
    print("="*70)
    
    config = DEFAULT_CONFIG.copy()
    
    results = SSVEPEvaluator.kfold_cv(X, y, OptimizedSSVEPClassifier, config, k=k)
    
    print("\n📈 交叉验证结果汇总:")
    print(f"  准确率:    {results['accuracy_mean']:.4f} ± {results['accuracy_std']:.4f}")
    print(f"  召回率:    {results['recall_mean']:.4f} ± {results['recall_std']:.4f}")
    print(f"  F1-Score:  {results['f1_mean']:.4f} ± {results['f1_std']:.4f}")
    
    return results


# ===================================================================
# 测试场景 3: 消融实验
# ===================================================================

def test_ablation_study(X, y, k=5):
    """
    消融实验: 测试各组件的贡献
    """
    print("\n" + "="*70)
    print("场景3: 消融实验 (各组件贡献度)")
    print("="*70)
    
    ablation_configs = {
        '1. 基线 (仅CCA)': {
            'use_fb_cca': False, 'use_trca': False, 'use_normalization': False,
            'harmonics': 2
        },
        '2. + 规范化': {
            'use_fb_cca': False, 'use_trca': False, 'use_normalization': True,
            'harmonics': 2
        },
        '3. + FB-CCA': {
            'use_fb_cca': True, 'use_trca': False, 'use_normalization': True,
            'harmonics': 2
        },
        '4. + TRCA': {
            'use_fb_cca': True, 'use_trca': True, 'use_normalization': True,
            'harmonics': 2
        },
        '5. + 增强谐波': {
            'use_fb_cca': True, 'use_trca': True, 'use_normalization': True,
            'harmonics': 3
        },
    }
    
    ablation_results = {}
    
    print(f"\n运行 {len(ablation_configs)} 种配置的 {k} 折交叉验证...\n")
    
    for config_name, config_params in ablation_configs.items():
        # 合并配置
        config = DEFAULT_CONFIG.copy()
        config.update(config_params)
        
        # 运行CV
        results = SSVEPEvaluator.kfold_cv(X, y, OptimizedSSVEPClassifier, config, k=k)
        ablation_results[config_name] = results
        
        acc = results['accuracy_mean']
        f1 = results['f1_mean']
        print(f"{config_name:25s} | Acc={acc:.4f} | F1={f1:.4f}")
    
    # 计算增量
    baseline_acc = ablation_results['1. 基线 (仅CCA)']['accuracy_mean']
    print(f"\n相对基线的改进:")
    for config_name, results in ablation_results.items():
        acc = results['accuracy_mean']
        improvement = (acc - baseline_acc) * 100
        print(f"  {config_name:25s}: +{improvement:5.2f}pp")
    
    return ablation_results


# ===================================================================
# 测试场景 4: 生产级部署
# ===================================================================

def test_production_pipeline(X_train, y_train, X_test, y_test):
    """
    生产级管道测试
    """
    print("\n" + "="*70)
    print("场景4: 生产级部署管道")
    print("="*70)
    
    # 创建管道
    pipeline = ProductionSSVEPPipeline(DEFAULT_CONFIG, latency_budget_ms=20)
    
    # 训练
    print("\n[1] 训练管道...")
    pipeline.fit(X_train, y_train, validate=True)
    
    # 预测
    print("\n[2] 批量预测...")
    y_pred, latencies = pipeline.predict_batch(X_test, return_latency=True)
    
    # 评估
    acc = (y_pred == y_test).mean()
    print(f"\n[3] 评估结果:")
    print(f"  ✓ 准确率: {acc:.4f}")
    
    # 性能报告
    perf = pipeline.get_performance_report()
    print(f"\n[4] 延迟性能:")
    if perf:
        print(f"  Mean:   {perf['mean_latency_ms']:.2f}ms")
        print(f"  Std:    {perf['std_latency_ms']:.2f}ms")
        print(f"  Min:    {perf['min_latency_ms']:.2f}ms")
        print(f"  Max:    {perf['max_latency_ms']:.2f}ms")
        print(f"  P95:    {perf['p95_latency_ms']:.2f}ms")
        print(f"  P99:    {perf['p99_latency_ms']:.2f}ms")
        print(f"  ✓ 满足预算: {perf['meets_budget']}")
    else:
        print(f"  无延迟数据")
    
    return pipeline, perf


# ===================================================================
# 测试场景 5: 异常值与鲁棒性
# ===================================================================

def test_robustness(X, y):
    """
    测试鲁棒性: 异常值检测, 通道加权
    """
    print("\n" + "="*70)
    print("场景5: 鲁棒性测试")
    print("="*70)
    
    # 异常值检测
    print("\n[1] 异常epoch检测...")
    outlier_mask = RobustPreprocessor.detect_outliers_mad(X, threshold=3.0)
    n_outliers = outlier_mask.sum()
    print(f"  检测到 {n_outliers} 个异常样本 ({n_outliers/len(X)*100:.1f}%)")
    
    # 清理
    X_clean = X[~outlier_mask]
    y_clean = y[~outlier_mask]
    
    # 对比
    print(f"\n[2] 清理前后对比 (3折CV):")
    
    results_dirty = SSVEPEvaluator.kfold_cv(X, y, OptimizedSSVEPClassifier, DEFAULT_CONFIG, k=3)
    results_clean = SSVEPEvaluator.kfold_cv(X_clean, y_clean, OptimizedSSVEPClassifier, DEFAULT_CONFIG, k=3)
    
    print(f"  清理前准确率: {results_dirty['accuracy_mean']:.4f}")
    print(f"  清理后准确率: {results_clean['accuracy_mean']:.4f}")
    improvement = (results_clean['accuracy_mean'] - results_dirty['accuracy_mean']) * 100
    print(f"  改进: +{improvement:.2f}pp")
    
    # 通道加权
    print(f"\n[3] 通道重要性分析...")
    ch_weights = RobustPreprocessor.channel_weights_by_correlation(X_clean, y_clean)
    print(f"  通道权重: {', '.join([f'Ch{i}={w:.2f}' for i, w in enumerate(ch_weights)])}")
    print(f"  权重最高的通道: Ch{np.argmax(ch_weights)}")
    print(f"  权重最低的通道: Ch{np.argmin(ch_weights)}")


# ===================================================================
# 主函数
# ===================================================================

def main():
    """完整测试流程"""
    
    print("\n")
    print("╔" + "="*68 + "╗")
    print("║" + " "*15 + "SSVEP 优化框架 - 完整测试" + " "*27 + "║")
    print("╚" + "="*68 + "╝")
    
    # ─────────────────────────────────────────────────────────────
    # 加载数据
    # ─────────────────────────────────────────────────────────────
    
    print("\n📥 加载数据...")
    
    d1_file = DATA_DIR / "D1.csv"
    d2_file = DATA_DIR / "D2.csv"
    
    if not d1_file.exists() or not d2_file.exists():
        print(f"⚠️ 数据文件未找到! 请确保在 {DATA_DIR}")
        print("   预期文件: D1.csv, D2.csv")
        return
    
    X_d1, y_d1 = DataLoader.load_and_segment(d1_file)
    X_d2, y_d2 = DataLoader.load_and_segment(d2_file)
    
    # 合并数据
    X_all = np.vstack([X_d1, X_d2])
    y_all = np.hstack([y_d1, y_d2])
    
    print(f"\n✓ 总数据: {X_all.shape[0]} epochs, 形状 {X_all.shape}")
    print(f"✓ 类别分布:\n{pd.Series(y_all).value_counts().sort_index()}")
    
    # 分割 train/test
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_all, test_size=0.2, stratify=y_all, random_state=42
    )
    
    print(f"\n✓ Train: {len(X_train)} epochs")
    print(f"✓ Test:  {len(X_test)} epochs")
    
    # ─────────────────────────────────────────────────────────────
    # 运行测试场景
    # ─────────────────────────────────────────────────────────────
    
    # 场景1: 基线对比
    baseline_results = test_baseline_vs_optimized(X_train, y_train, X_test, y_test)
    
    # 场景2: 交叉验证
    cv_results = test_cross_validation(X_all, y_all, k=5)
    
    # 场景3: 消融实验
    ablation_results = test_ablation_study(X_all, y_all, k=3)
    
    # 场景4: 生产级管道
    pipeline, perf = test_production_pipeline(X_train, y_train, X_test, y_test)
    
    # 场景5: 鲁棒性
    test_robustness(X_all, y_all)
    
    # ─────────────────────────────────────────────────────────────
    # 总结
    # ─────────────────────────────────────────────────────────────
    
    print("\n" + "="*70)
    print("✅ 所有测试完成")
    print("="*70)
    
    print("\n📊 关键指标汇总:")
    print(f"  • 基线准确率:        {baseline_results['基线CCA']['accuracy']:.4f}")
    print(f"  • 优化准确率:        {baseline_results['优化版本']['accuracy']:.4f}")
    print(f"  • 改进:             +{(baseline_results['优化版本']['accuracy']-baseline_results['基线CCA']['accuracy'])*100:.2f}pp")
    print(f"  • CV验证准确率:      {cv_results['accuracy_mean']:.4f} ± {cv_results['accuracy_std']:.4f}")
    if perf:
        print(f"  • 平均延迟:         {perf['mean_latency_ms']:.2f}ms")
        print(f"  • 是否满足预算(20ms): {perf['meets_budget']}")
    
    print("\n✨ 框架特性:")
    print("  ✓ Filter-Bank CCA (4子带)")
    print("  ✓ TRCA模板法")
    print("  ✓ RV得分归一化")
    print("  ✓ 协方差收缩 (备选)")
    print("  ✓ K折交叉验证")
    print("  ✓ 生产级部署")


if __name__ == '__main__':
    main()
