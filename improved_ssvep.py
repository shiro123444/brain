import numpy as np
import pandas as pd
import os
from ssvepdetect import ssvepDetect
import matplotlib.pyplot as plt

def improved_ssvep_detection():
    """改进的SSVEP检测"""
    
    # 参数设置
    srate = 250  # 采样率 (Hz)
    freqs = [1.0, 7.0, 10.0, 12.0, 15.0, 8.57]  # 测试频率 (Hz)
    dataLen = 2.0  # 分析时间窗口长度 (秒)
    min_corr_threshold = 0.3  # 最小相关系数阈值
    
    print("=" * 60)
    print("改进的SSVEP检测")
    print("=" * 60)
    print(f"采样率: {srate} Hz")
    print(f"测试频率: {freqs} Hz")
    print(f"分析窗口长度: {dataLen} 秒")
    print(f"最小相关系数阈值: {min_corr_threshold}")
    print()
    
    # 测试两个文件
    test_files = [
        ("ExampleData/D1.csv", 1.0),
        ("ExampleData/D2.csv", 7.0)
    ]
    
    results = []
    
    for file_path, expected_freq in test_files:
        print(f"处理文件: {file_path}")
        print("-" * 40)
        
        if not os.path.exists(file_path):
            print(f"文件不存在: {file_path}")
            continue
        
        # 加载数据
        data = pd.read_csv(file_path)
        channels_data = data.iloc[:, :-2].values
        stim_id = data.iloc[:, -1].iloc[0]
        
        print(f"数据形状: {channels_data.shape}")
        print(f"刺激ID: {stim_id}")
        
        # 转置数据
        data_transposed = channels_data.T
        
        # 创建检测器
        detector = ssvepDetect(srate, freqs, dataLen)
        
        # 截断数据
        template_length = detector.TemplateSet[0].shape[1]
        data_used = data_transposed[:, :template_length]
        
        # 预处理
        filtered_data = detector.pre_filter(data_used)
        
        # 计算相关系数
        correlations = []
        cdata = filtered_data.transpose()
        
        print(f"\n相关系数分析:")
        for i, template in enumerate(detector.TemplateSet):
            ctemplate = template.transpose()
            try:
                detector.cca.fit(cdata, ctemplate)
                datatran, templatetran = detector.cca.transform(cdata, ctemplate)
                coe = np.corrcoef(datatran[:,0], templatetran[:,0])[0,1]
                correlations.append(coe)
                
                print(f"  {freqs[i]:5.2f} Hz: {coe:.6f}")
                
            except Exception as e:
                print(f"  {freqs[i]:5.2f} Hz: 错误 - {str(e)}")
                correlations.append(0.0)
        
        # 分析结果
        max_corr = max(correlations)
        max_index = correlations.index(max_corr)
        detected_freq = freqs[max_index]
        
        print(f"\n分析结果:")
        print(f"  最高相关系数: {max_corr:.6f} at {detected_freq} Hz")
        print(f"  真实频率: {stim_id} Hz")
        
        # 判断检测是否可信
        is_confident = max_corr > min_corr_threshold
        is_correct = abs(detected_freq - stim_id) < 0.1  # 考虑浮点精度
        
        print(f"  检测可信度: {'高' if is_confident else '低'} (阈值: {min_corr_threshold})")
        print(f"  检测正确性: {'正确' if is_correct else '错误'}")
        
        # 如果相关系数太低，可能是没有检测到有效的SSVEP
        if not is_confident:
            print(f"  警告: 相关系数过低，可能没有检测到有效的SSVEP响应")
            detected_freq = None
        
        results.append({
            'file': file_path,
            'expected_freq': expected_freq,
            'true_stim_id': stim_id,
            'detected_freq': detected_freq,
            'max_correlation': max_corr,
            'is_confident': is_confident,
            'is_correct': is_correct,
            'all_correlations': correlations.copy()
        })
        
        print()
    
    # 总结
    print("=" * 60)
    print("检测总结")
    print("=" * 60)
    
    confident_detections = [r for r in results if r['is_confident'] and r['detected_freq'] is not None]
    correct_detections = [r for r in results if r['is_correct']]
    
    print(f"总测试数: {len(results)}")
    print(f"可信检测数: {len(confident_detections)}")
    print(f"正确检测数: {len(correct_detections)}")
    
    if confident_detections:
        confidence_rate = len(confident_detections) / len(results) * 100
        accuracy_rate = len(correct_detections) / len(confident_detections) * 100
        print(f"检测置信度: {confidence_rate:.1f}%")
        print(f"检测准确率: {accuracy_rate:.1f}%")
    else:
        print(f"检测置信度: 0.0%")
        print(f"检测准确率: 0.0%")
    
    print()
    
    for i, result in enumerate(results):
        print(f"测试 {i+1}: {result['file']}")
        print(f"  真实频率: {result['true_stim_id']} Hz")
        if result['detected_freq'] is not None:
            print(f"  检测频率: {result['detected_freq']} Hz")
            print(f"  相关系数: {result['max_correlation']:.6f}")
        else:
            print(f"  检测结果: 无有效检测")
        print(f"  结果: {'✓ 正确' if result['is_correct'] else '✗ 错误' if result['is_confident'] else '⚠ 不可信'}")
        print()

def analyze_signal_characteristics():
    """分析信号特征"""
    print("=" * 60)
    print("信号特征分析")
    print("=" * 60)
    
    # 加载数据
    data1 = pd.read_csv("ExampleData/D1.csv")
    data2 = pd.read_csv("ExampleData/D2.csv")
    
    for i, (data, filename) in enumerate([(data1, "D1.csv"), (data2, "D2.csv")]):
        print(f"\n文件: {filename}")
        print("-" * 30)
        
        # 提取通道数据
        channels_data = data.iloc[:, :-2].values
        stim_id = data.iloc[:, -1].iloc[0]
        
        print(f"刺激频率: {stim_id} Hz")
        print(f"数据形状: {channels_data.shape}")
        print(f"采样率: 250 Hz")
        print(f"总时长: {channels_data.shape[0] / 250:.1f} 秒")
        
        # 计算基本统计量
        print(f"数据统计:")
        print(f"  均值: {np.mean(channels_data):.3f}")
        print(f"  标准差: {np.std(channels_data):.3f}")
        print(f"  最大值: {np.max(channels_data):.3f}")
        print(f"  最小值: {np.min(channels_data):.3f}")
        
        # 分析前几秒的数据
        first_2_seconds = channels_data[:500, :]  # 前2秒
        print(f"前2秒数据统计:")
        print(f"  均值: {np.mean(first_2_seconds):.3f}")
        print(f"  标准差: {np.std(first_2_seconds):.3f}")

if __name__ == "__main__":
    # 运行改进的检测
    improved_ssvep_detection()
    
    # 分析信号特征
    analyze_signal_characteristics()
