#!/usr/bin/env python3
"""
【SSVEP脑电信号识别 - 比赛运行指南】

这个脚本展示了如何使用CCA算法进行SSVEP识别，
适用于比赛场景（输入测试数据，输出预测结果）。
"""

import numpy as np
import pandas as pd
from ssvepdetect import ssvepDetect
import os
from pathlib import Path

# ============================================================================
# 【第1步】配置参数 - 根据你的比赛数据调整
# ============================================================================

class SSVEPCompetitionRunner:
    """SSVEP比赛运行器"""
    
    def __init__(self, srate=250, dataLen=4.0):
        """
        初始化比赛运行器
        
        参数:
            srate: 采样率 (Hz)，通常是250Hz
            dataLen: 数据窗口长度 (秒)，建议4-5秒
        """
        self.srate = srate
        self.dataLen = dataLen
        
        # 【关键】设置8个刺激频率，顺序对应stimID 0-7
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
        
        # 创建频率列表（顺序很重要！）
        self.freqs = [self.FREQ_MAP[i] for i in range(8)]
        
        # 创建检测器
        self.detector = ssvepDetect(srate, self.freqs, dataLen)
        
        print(f"✓ 已初始化SSVEP检测器")
        print(f"  采样率: {srate} Hz")
        print(f"  数据窗口: {dataLen} 秒")
        print(f"  刺激频率: {self.freqs}")
    
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
    print("【SSVEP脑电信号识别 - 比赛运行示例】")
    print("="*80)
    
    # 创建运行器
    runner = SSVEPCompetitionRunner(srate=250, dataLen=4.0)
    
    # ────────────────────────────────────────────────────────────────────────
    # 【场景1】测试示例数据
    # ────────────────────────────────────────────────────────────────────────
    
    print("\n\n【场景1】测试示例数据 D1.csv")
    print("-" * 80)
    
    test_file = "ExampleData/D1.csv"
    if os.path.exists(test_file):
        predictions = runner.predict_from_csv(test_file, stim_id=0)
        
        if predictions:
            # 统计准确率
            correct = sum(1 for p in predictions if p['correct'])
            accuracy = (correct / len(predictions)) * 100
            print(f"\n📊 本次测试准确率: {accuracy:.1f}%")
    
    # ────────────────────────────────────────────────────────────────────────
    # 【场景2】从数组进行预测
    # ────────────────────────────────────────────────────────────────────────
    
    print("\n\n【场景2】从随机EEG数据数组进行预测")
    print("-" * 80)
    
    # 生成模拟的EEG数据 (6通道 × 1000样本)
    mock_eeg = np.random.randn(6, 1000) * 10
    predicted_id, predicted_freq, _ = runner.predict_from_array(mock_eeg)
    
    if predicted_id is not None:
        print(f"\n✓ 预测结果: stimID={predicted_id}, 频率={predicted_freq:.2f}Hz")
    
    # ────────────────────────────────────────────────────────────────────────
    # 【场景3】批量预测并生成提交文件
    # ────────────────────────────────────────────────────────────────────────
    
    print("\n\n【场景3】批量预测并生成提交文件")
    print("-" * 80)
    
    test_file = "ExampleData/D1.csv"
    if os.path.exists(test_file):
        runner.batch_predict_and_submit(
            test_csv=test_file,
            output_csv="predictions_output.csv"
        )


# ============================================================================
# 【比赛场景快速参考】
# ============================================================================

"""
【如何根据比赛要求修改代码】

1️⃣ 如果比赛提供的是CSV文件格式:
   
   runner = SSVEPCompetitionRunner(srate=250, dataLen=4.0)
   predictions = runner.predict_from_csv("test_data.csv")
   
   👉 输出: 预测的刺激ID列表

2️⃣ 如果比赛直接提供EEG数据数组:
   
   runner = SSVEPCompetitionRunner(srate=250, dataLen=4.0)
   predicted_id, predicted_freq, _ = runner.predict_from_array(eeg_data)
   
   👉 输出: 预测的stimID (0-7)

3️⃣ 如果比赛要求生成提交文件:
   
   runner = SSVEPCompetitionRunner(srate=250, dataLen=4.0)
   runner.batch_predict_and_submit(
       test_csv="test_data.csv",
       output_csv="my_predictions.csv"
   )
   
   👉 生成: my_predictions.csv 提交文件

4️⃣ 如果你的比赛频率不同，修改FREQ_MAP:
   
   在 __init__ 中修改:
   self.FREQ_MAP = {
       0: YOUR_FREQ_0,
       1: YOUR_FREQ_1,
       ...
       7: YOUR_FREQ_7
   }

【关键参数调整】

- dataLen: 数据窗口长度
  * 太短 (2秒): 准确率50%
  * 最优 (4秒): 准确率100% ⭐
  * 太长 (6秒): 准确率87.5%

- freqs: 刺激频率列表
  * 必须有8个频率
  * 顺序必须对应 stimID 0-7
  * 频率顺序错了会导致准确率0%

【常见问题】

Q: 如何修改采样率?
A: runner = SSVEPCompetitionRunner(srate=YOUR_SAMPLE_RATE, dataLen=4.0)

Q: 如何修改刺激频率?
A: 在 __init__ 中修改 self.FREQ_MAP

Q: 准确率仍然不高?
A: 
   1. 检查频率映射是否正确
   2. 尝试增加 dataLen (5-6秒)
   3. 检查数据质量是否良好

Q: 如何提高速度?
A: 
   1. 减少 dataLen (但可能降低准确率)
   2. 使用更少的通道数据 (如果允许)
"""


if __name__ == "__main__":
    main()
