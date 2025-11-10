import numpy as np
import pandas as pd
from ssvepdetect import ssvepDetect
import os

def load_single_stimulus(file_path, stim_id):
    """加载单个刺激的数据"""
    data = pd.read_csv(file_path)
    
    # 提取该刺激ID对应的数据
    mask = data['stimID'] == stim_id
    channels_data = data.loc[mask, :'PO4'].values  # 只取前6列通道数据
    
    return channels_data

def test_with_different_window_lengths():
    """用不同的时间窗口长度测试"""
    
    FREQ_MAP = {
        0: 16.0, 1: 9.0, 2: 10.0, 3: 11.0,
        4: 12.0, 5: 13.0, 6: 10.58, 7: 15.0
    }
    
    freqs = [FREQ_MAP[i] for i in range(8)]
    srate = 250
    
    # 测试不同的时间窗口长度
    window_lengths = [2.0, 3.0, 4.0, 5.0, 6.0]
    
    print("=" * 100)
    print("【测试不同的数据窗口长度对检测准确率的影响】")
    print("=" * 100)
    
    results = {}
    
    for dataLen in window_lengths:
        print(f"\n→ 测试窗口长度: {dataLen} 秒")
        print("-" * 100)
        
        # 创建检测器
        detector = ssvepDetect(srate, freqs, dataLen)
        
        correct_count = 0
        total_count = 0
        
        for file_path in ["ExampleData/D1.csv", "ExampleData/D2.csv"]:
            if not os.path.exists(file_path):
                continue
            
            data = pd.read_csv(file_path)
            unique_stim_ids = sorted(data['stimID'].unique())
            
            for stim_id in unique_stim_ids:
                stim_id = int(stim_id)
                
                # 加载该刺激的数据
                segment_data = load_single_stimulus(file_path, stim_id)
                
                # 只使用指定长度的数据
                samples_needed = int(dataLen * srate)
                if segment_data.shape[0] < samples_needed:
                    continue
                
                segment_data = segment_data[:samples_needed]
                data_transposed = segment_data.T
                
                try:
                    detected_index = detector.detect(data_transposed)
                    detected_freq = freqs[detected_index]
                    true_freq = FREQ_MAP[stim_id]
                    
                    is_correct = abs(detected_freq - true_freq) < 0.5
                    
                    if is_correct:
                        correct_count += 1
                    total_count += 1
                    
                except Exception as e:
                    pass
        
        accuracy = (correct_count / total_count * 100) if total_count > 0 else 0
        results[dataLen] = accuracy
        print(f"  识别准确率: {correct_count}/{total_count} = {accuracy:.1f}%")
    
    # 找出最好的窗口长度
    best_length = max(results, key=results.get)
    best_accuracy = results[best_length]
    
    print(f"\n{'=' * 100}")
    print(f"【最优结果】")
    print(f"最佳窗口长度: {best_length} 秒，准确率: {best_accuracy:.1f}%")
    print(f"{'=' * 100}")
    
    return best_length, best_accuracy

def analyze_cca_coefficients(file_path="ExampleData/D1.csv"):
    """分析CCA相关系数，找出问题"""
    
    print("\n" + "=" * 100)
    print("【分析CCA相关系数的问题】")
    print("=" * 100)
    
    FREQ_MAP = {
        0: 16.0, 1: 9.0, 2: 10.0, 3: 11.0,
        4: 12.0, 5: 13.0, 6: 10.58, 7: 15.0
    }
    
    freqs = [FREQ_MAP[i] for i in range(8)]
    srate = 250
    dataLen = 4.0
    
    detector = ssvepDetect(srate, freqs, dataLen)
    
    data = pd.read_csv(file_path)
    
    # 对每个刺激进行详细分析
    for stim_id in [1, 4, 7]:  # 重点检查失败的案例
        print(f"\n→ 分析 stimID={stim_id} (真实频率: {FREQ_MAP[stim_id]:.2f} Hz)")
        print("-" * 100)
        
        # 加载数据
        segment_data = load_single_stimulus(file_path, stim_id)
        samples_needed = int(dataLen * srate)
        segment_data = segment_data[:samples_needed]
        data_transposed = segment_data.T
        
        # 预处理
        filtered_data = detector.pre_filter(data_transposed)
        cdata = filtered_data.transpose()
        
        # 对每个模板计算相关系数
        coefficients = []
        for idx, template in enumerate(detector.TemplateSet):
            ctemplate = template.transpose()
            
            try:
                detector.cca.fit(cdata, ctemplate)
                datatran, templatetran = detector.cca.transform(cdata, ctemplate)
                coe = np.corrcoef(datatran[:, 0], templatetran[:, 0])[0, 1]
                coefficients.append(coe)
            except:
                coefficients.append(np.nan)
        
        # 打印结果
        for idx, (freq, coef) in enumerate(zip(freqs, coefficients)):
            marker = "→" if idx == np.nanargmax(coefficients) else " "
            print(f"  {marker} 频率 {freq:.2f} Hz: 相关系数 = {coef:.4f}")
        
        max_idx = np.nanargmax(coefficients)
        print(f"  最高相关系数: {freqs[max_idx]:.2f} Hz (索引={max_idx})")

if __name__ == "__main__":
    # 测试不同窗口长度
    best_length, best_accuracy = test_with_different_window_lengths()
    
    # 分析系数分布
    analyze_cca_coefficients()
