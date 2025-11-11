#!/usr/bin/env python3
"""
【SSVEP脑电信号识别 - 比赛运行指南 v2】
支持自动频率检测功能

这个脚本展示了如何使用CCA算法进行SSVEP识别，
适用于比赛场景（输入测试数据，输出预测结果）。
"""

import numpy as np
import pandas as pd
from ssvepdetect import ssvepDetect
import os
from pathlib import Path
from collections import defaultdict
from scipy.fft import fft, fftfreq

# ============================================================================
# 【第1步】配置参数 - 根据你的比赛数据调整
# ============================================================================

class SSVEPCompetitionRunner:
    """SSVEP比赛运行器 - 支持自动频率检测"""
    
    def __init__(self, srate=250, dataLen=4.0, freq_map=None):
        """
        初始化比赛运行器
        
        参数:
            srate: 采样率 (Hz)，通常是250Hz
            dataLen: 数据窗口长度 (秒)，建议4-5秒
            freq_map: 频率映射字典或CSV文件路径
                     如果是None，使用默认D1频率
                     如果是dict，使用自定义频率
                     如果是str路径，自动检测该CSV的频率
        """
        self.srate = srate
        self.dataLen = dataLen
        
        # 处理频率映射
        if freq_map is None:
            # 默认D1频率
            print("📌 使用默认频率 (D1.csv频率)")
            self.FREQ_MAP = {
                0: 16.0,     # stimID=0 对应 16.0 Hz
                1: 9.0,      # stimID=1 对应 9.0 Hz
                2: 10.0,     # stimID=2 对应 10.0 Hz
                3: 11.0,     # stimID=3 对应 11.0 Hz
                4: 12.0,     # stimID=4 对应 12.0 Hz
                5: 13.0,     # stimID=5 对应 13.0 Hz
                6: 10.58,    # stimID=6 对应 10.58 Hz
                7: 15.0      # stimID=7 对应 15.0 Hz
            }
        elif isinstance(freq_map, str):
            # 从CSV文件自动检测频率
            print(f"📌 从 {freq_map} 自动检测频率...")
            self.FREQ_MAP = self._auto_detect_frequencies(freq_map)
        elif isinstance(freq_map, dict):
            # 使用自定义频率
            print("📌 使用自定义频率映射")
            self.FREQ_MAP = freq_map
        else:
            raise ValueError("freq_map 必须是None、字典或CSV文件路径")
        
        # 创建频率列表（顺序很重要！）
        self.freqs = [self.FREQ_MAP[i] for i in range(8)]
        
        # 创建检测器
        self.detector = ssvepDetect(srate, self.freqs, dataLen)
        
        print(f"✓ 已初始化SSVEP检测器")
        print(f"  采样率: {srate} Hz")
        print(f"  数据窗口: {dataLen} 秒")
        print(f"  刺激频率: {self.freqs}")
    
    def _auto_detect_frequencies(self, csv_file, freq_range=(8, 30)):
        """
        自动检测CSV文件中每个stimID对应的频率
        
        参数:
            csv_file: CSV文件路径
            freq_range: 频率范围 (min, max)
        
        返回:
            freq_map: {stimID: frequency}
        """
        if not os.path.exists(csv_file):
            print(f"❌ 错误: 文件不存在 {csv_file}")
            raise FileNotFoundError(csv_file)
        
        data = pd.read_csv(csv_file)
        stim_frequencies = defaultdict(lambda: defaultdict(int))
        
        task_ids = sorted(data['taskID'].unique())
        print(f"  分析 {len(task_ids)} 个任务中的频率分布...")
        
        for task_id in task_ids:
            mask = data['taskID'] == task_id
            task_data = data[mask]
            stim_id = int(task_data['stimID'].iloc[0])
            
            # 提取EEG数据
            eeg_signal = task_data.iloc[:, :6].values.T  # (6, samples)
            
            # 取前4秒数据
            samples = int(4.0 * self.srate)
            eeg_signal = eeg_signal[:, :samples]
            
            # 对每个通道进行FFT
            for ch in range(6):
                signal = eeg_signal[ch]
                fft_vals = np.abs(fft(signal))
                freqs = fftfreq(len(signal), 1/self.srate)
                
                # 只取正频率
                positive_idx = freqs > 0
                fft_vals = fft_vals[positive_idx]
                freqs = freqs[positive_idx]
                
                # 在指定频率范围内查找峰值
                mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
                local_freqs = freqs[mask]
                local_fft = fft_vals[mask]
                
                if len(local_fft) > 0:
                    peak_idx = np.argmax(local_fft)
                    peak_freq = local_freqs[peak_idx]
                    # 量化到0.5Hz
                    peak_freq = round(peak_freq * 2) / 2
                    stim_frequencies[stim_id][peak_freq] += 1
        
        # 从每个stimID中提取最常见的频率
        freq_map = {}
        print(f"\n  检测结果:")
        for stim_id in sorted(stim_frequencies.keys()):
            freqs_dict = stim_frequencies[stim_id]
            sorted_freqs = sorted(freqs_dict.items(), key=lambda x: x[1], reverse=True)
            
            primary_freq = sorted_freqs[0][0]
            count = sorted_freqs[0][1]
            freq_map[stim_id] = primary_freq
            print(f"    stimID={stim_id}: {primary_freq:.2f} Hz (检测 {count} 次)")
        
        print()
        return freq_map
    
    # ========================================================================
    # 【比赛场景1】从CSV文件加载单个数据片段并识别
    # ========================================================================
    
    def predict_from_csv(self, csv_file, stim_id=None):
        """
        从CSV文件中提取特定刺激的数据并进行预测
        
        参数:
            csv_file: CSV文件路径 (e.g., "test_data.csv")
            stim_id: 要处理的stimID (0-7)，如果为None则处理所有
            
        返回:
            predictions: 预测的stimID列表
        """
        print(f"\n{'='*80}")
        print(f"【从CSV文件进行预测】")
        print(f"{'='*80}")
        print(f"文件: {csv_file}")
        
        if not os.path.exists(csv_file):
            print(f"❌ 错误: 文件不存在 {csv_file}")
            return None
        
        # 读取CSV文件
        data = pd.read_csv(csv_file)
        print(f"✓ 已加载数据，形状: {data.shape}")
        
        if stim_id is None:
            # 处理所有的刺激ID
            unique_stim_ids = sorted(data['stimID'].unique())
        else:
            unique_stim_ids = [stim_id]
        
        predictions = []
        
        for target_stim_id in unique_stim_ids:
            # 提取该刺激ID的数据
            mask = data['stimID'] == target_stim_id
            segment_data = data.loc[mask, :'PO4'].values  # 只取前6列通道数据
            
            # 确保有足够的数据
            samples_needed = int(self.dataLen * self.srate)
            if segment_data.shape[0] < samples_needed:
                print(f"  ⚠ stimID={int(target_stim_id)}: 数据不足，跳过")
                continue
            
            # 只使用前dataLen秒的数据
            segment_data = segment_data[:samples_needed]
            
            # 转置为 (通道数, 样本数) 格式
            data_transposed = segment_data.T
            
            try:
                # 进行预测
                predicted_index = self.detector.detect(data_transposed)
                predicted_freq = self.freqs[predicted_index]
                true_freq = self.FREQ_MAP[int(target_stim_id)]
                
                # 检查是否正确
                is_correct = abs(predicted_freq - true_freq) < 0.5
                result = "✓" if is_correct else "✗"
                
                print(f"  {result} stimID={int(target_stim_id)}: "
                      f"真实频率={true_freq:.2f}Hz, 预测频率={predicted_freq:.2f}Hz, "
                      f"预测ID={predicted_index}")
                
                predictions.append({
                    'true_stim_id': int(target_stim_id),
                    'predicted_stim_id': predicted_index,
                    'true_freq': true_freq,
                    'predicted_freq': predicted_freq,
                    'correct': is_correct
                })
                
            except Exception as e:
                print(f"  ❌ stimID={int(target_stim_id)}: 检测失败 - {str(e)}")
        
        return predictions
    
    # ========================================================================
    # 【比赛场景2】从原始数据阵列进行预测（不通过CSV）
    # ========================================================================
    
    def predict_from_array(self, eeg_data):
        """
        直接从EEG数据数组进行预测
        
        参数:
            eeg_data: EEG信号数组
                      形状: (通道数, 样本数)
                      例如: (6, 1000) 表示6个通道，1000个样本
        
        返回:
            predicted_stim_id: 预测的刺激ID (0-7)
            predicted_freq: 预测的频率 (Hz)
            coefficients: 与各频率的相关系数列表
        """
        print(f"\n{'='*80}")
        print(f"【从EEG数据阵列进行预测】")
        print(f"{'='*80}")
        print(f"输入数据形状: {eeg_data.shape}")
        
        # 数据形状检查
        if eeg_data.shape[0] != 6:
            print(f"❌ 错误: 需要6个通道，但输入了{eeg_data.shape[0]}个")
            return None, None, None
        
        samples_needed = int(self.dataLen * self.srate)
        if eeg_data.shape[1] < samples_needed:
            print(f"❌ 错误: 需要{samples_needed}个样本，但只有{eeg_data.shape[1]}个")
            return None, None, None
        
        # 只使用前dataLen秒的数据
        eeg_data = eeg_data[:, :samples_needed]
        
        try:
            # 进行预测
            predicted_index = self.detector.detect(eeg_data)
            predicted_freq = self.freqs[predicted_index]
            
            print(f"✓ 预测成功!")
            print(f"  预测的stimID: {predicted_index}")
            print(f"  预测的频率: {predicted_freq:.2f} Hz")
            
            return predicted_index, predicted_freq, None
            
        except Exception as e:
            print(f"❌ 预测失败: {str(e)}")
            return None, None, None
    
    # ========================================================================
    # 【比赛场景3】批量预测并生成提交文件
    # ========================================================================
    
    def batch_predict_and_submit(self, test_csv, output_csv="predictions.csv"):
        """
        批量预测所有测试数据并生成提交文件
        
        参数:
            test_csv: 测试数据CSV文件路径
            output_csv: 输出预测结果的CSV文件路径
        """
        print(f"\n{'='*80}")
        print(f"【批量预测 - 生成比赛提交文件】")
        print(f"{'='*80}")
        print(f"输入文件: {test_csv}")
        print(f"输出文件: {output_csv}")
        
        if not os.path.exists(test_csv):
            print(f"❌ 错误: 文件不存在 {test_csv}")
            return False
        
        # 读取测试数据
        data = pd.read_csv(test_csv)
        
        # 获取所有的taskID（每个taskID对应一个待预测的片段）
        task_ids = data['taskID'].unique()
        print(f"✓ 共有 {len(task_ids)} 个任务需要预测")
        
        results = []
        correct_count = 0
        
        for task_id in sorted(task_ids):
            # 提取该taskID的数据
            mask = data['taskID'] == task_id
            segment_data = data.loc[mask, :'PO4'].values
            
            # 获取该segment的真实刺激ID（如果有的话）
            true_stim_id = data.loc[mask, 'stimID'].iloc[0] if 'stimID' in data.columns else None
            
            # 确保有足够的数据
            samples_needed = int(self.dataLen * self.srate)
            if segment_data.shape[0] < samples_needed:
                print(f"  ⚠ taskID={int(task_id)}: 数据不足，跳过")
                continue
            
            segment_data = segment_data[:samples_needed]
            data_transposed = segment_data.T
            
            try:
                # 预测
                predicted_index = self.detector.detect(data_transposed)
                
                # 计算准确性（如果有真实标签）
                is_correct = False
                if true_stim_id is not None:
                    is_correct = (predicted_index == int(true_stim_id))
                    if is_correct:
                        correct_count += 1
                
                results.append({
                    'taskID': int(task_id),
                    'predicted_stimID': predicted_index,
                    'confidence': 0.9  # 可以根据相关系数调整置信度
                })
                
                if (len(results) % 10 == 0):
                    print(f"  ✓ 已完成 {len(results)} 个预测...")
                
            except Exception as e:
                print(f"  ❌ taskID={int(task_id)}: 预测失败 - {str(e)}")
        
        # 生成输出文件
        results_df = pd.DataFrame(results)
        results_df.to_csv(output_csv, index=False)
        
        print(f"\n✓ 预测完成!")
        print(f"  总预测数: {len(results)}")
        if correct_count > 0:
            accuracy = (correct_count / len(results)) * 100
            print(f"  准确率: {accuracy:.1f}% ({correct_count}/{len(results)})")
        print(f"  输出文件: {output_csv}")
        
        return True


# ============================================================================
# 【使用示例】
# ============================================================================

def main():
    """主函数 - 展示各种使用场景"""
    
    print("\n" + "="*80)
    print("【SSVEP脑电信号识别 - 比赛运行示例 v2】")
    print("="*80)
    
    # ────────────────────────────────────────────────────────────────────────
    # 【方案1】使用默认D1频率
    # ────────────────────────────────────────────────────────────────────────
    
    print("\n\n【方案1】使用默认D1频率")
    print("-" * 80)
    runner_d1 = SSVEPCompetitionRunner(srate=250, dataLen=4.0)
    
    test_file = "ExampleData/D1.csv"
    if os.path.exists(test_file):
        predictions = runner_d1.predict_from_csv(test_file, stim_id=0)
        if predictions:
            correct = sum(1 for p in predictions if p['correct'])
            accuracy = (correct / len(predictions)) * 100
            print(f"\n📊 D1.csv 准确率: {accuracy:.1f}%")
    
    # ────────────────────────────────────────────────────────────────────────
    # 【方案2】自动检测D2频率
    # ────────────────────────────────────────────────────────────────────────
    
    print("\n\n【方案2】自动检测D2频率")
    print("-" * 80)
    
    test_file_d2 = "ExampleData/D2.csv"
    if os.path.exists(test_file_d2):
        runner_d2 = SSVEPCompetitionRunner(
            srate=250, 
            dataLen=4.0,
            freq_map="ExampleData/D2.csv"  # 自动检测D2频率
        )
        
        runner_d2.batch_predict_and_submit(
            test_csv=test_file_d2,
            output_csv="predictions_d2.csv"
        )


if __name__ == "__main__":
    main()
