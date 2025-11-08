import numpy as np
import pandas as pd
from ssvepdetect import ssvepDetect
import os

def load_data(file_path):
    """加载CSV数据"""
    print(f"正在加载数据: {file_path}")
    
    # 读取CSV文件
    data = pd.read_csv(file_path)
    
    # 打印基本信息
    print(f"数据形状: {data.shape}")
    print(f"列名: {list(data.columns)}")
    print(f"前5行数据:")
    print(data.head())
    
    # 提取通道数据（去掉最后两列的taskID和stimID）
    channels_data = data.iloc[:, :-2].values  # shape: (samples, channels)
    stim_id = data.iloc[:, -1].iloc[0]  # 获取刺激ID
    
    print(f"通道数据形状: {channels_data.shape}")
    print(f"刺激ID: {stim_id}")
    
    return channels_data, stim_id

def test_ssvep_detection():
    """测试SSVEP检测"""
    
    # 参数设置
    srate = 250  # 采样率 (Hz)
    freqs = [1.0, 7.0, 10.0, 12.0, 15.0, 8.57]  # 测试频率 (Hz) - 包含D1和D2的真实频率
    dataLen = 2.0  # 分析时间窗口长度 (秒)
    
    print("=" * 60)
    print("SSVEP检测测试")
    print("=" * 60)
    print(f"采样率: {srate} Hz")
    print(f"测试频率: {freqs} Hz")
    print(f"分析窗口长度: {dataLen} 秒")
    print()
    
    # 创建检测器
    detector = ssvepDetect(srate, freqs, dataLen)
    
    # 测试文件列表
    test_files = [
        "ExampleData/D1.csv",
        "ExampleData/D2.csv"
    ]
    
    results = []
    
    for file_path in test_files:
        if not os.path.exists(file_path):
            print(f"警告: 文件不存在 {file_path}")
            continue
            
        print(f"处理文件: {file_path}")
        print("-" * 40)
        
        try:
            # 加载数据
            data, true_stim_id = load_data(file_path)
            
            # 转置数据以匹配算法期望的格式 (channels x samples)
            data_transposed = data.T
            print(f"转置后数据形状: {data_transposed.shape}")
            
            # 进行检测
            print("正在进行SSVEP检测...")
            detected_index = detector.detect(data_transposed)
            detected_freq = freqs[detected_index]
            
            print(f"检测结果:")
            print(f"  检测到的频率索引: {detected_index}")
            print(f"  检测到的频率: {detected_freq} Hz")
            print(f"  真实刺激频率: {true_stim_id} Hz")
            
            # 计算是否正确
            is_correct = (detected_freq == true_stim_id)
            print(f"  检测是否正确: {is_correct}")
            
            results.append({
                'file': file_path,
                'true_freq': true_stim_id,
                'detected_freq': detected_freq,
                'detected_index': detected_index,
                'correct': is_correct
            })
            
        except Exception as e:
            print(f"处理文件 {file_path} 时出错: {str(e)}")
            import traceback
            traceback.print_exc()
        
        print()
    
    # 打印总结
    print("=" * 60)
    print("测试结果总结")
    print("=" * 60)
    
    if results:
        correct_count = sum(1 for r in results if r['correct'])
        total_count = len(results)
        
        print(f"总测试数: {total_count}")
        print(f"正确检测数: {correct_count}")
        print(f"准确率: {correct_count/total_count*100:.1f}%")
        print()
        
        for i, result in enumerate(results):
            print(f"测试 {i+1}: {result['file']}")
            print(f"  真实频率: {result['true_freq']} Hz")
            print(f"  检测频率: {result['detected_freq']} Hz")
            print(f"  结果: {'✓ 正确' if result['correct'] else '✗ 错误'}")
            print()
    else:
        print("没有完成任何测试")

def test_algorithm_parameters():
    """测试不同参数设置"""
    print("=" * 60)
    print("参数测试")
    print("=" * 60)
    
    # 测试不同的分析窗口长度
    window_lengths = [1.0, 2.0, 3.0, 4.0]
    
    for data_len in window_lengths:
        print(f"\n测试窗口长度: {data_len} 秒")
        print("-" * 30)
        
        try:
            # 加载一个小数据样本进行测试
            if os.path.exists("ExampleData/D1.csv"):
                data = pd.read_csv("ExampleData/D1.csv").iloc[:, :-2].values
                data_transposed = data.T
                
                # 限制数据长度以避免内存问题
                max_samples = int(data_len * 250)  # 250Hz采样率
                if data_transposed.shape[1] > max_samples:
                    data_transposed = data_transposed[:, :max_samples]
                
                detector = ssvepDetect(250, [12.0, 15.0, 10.0, 8.57], data_len)
                detected_index = detector.detect(data_transposed)
                
                print(f"检测到的频率索引: {detected_index}")
                print(f"检测到的频率: {detector.freqs[detected_index]} Hz")
                
        except Exception as e:
            print(f"参数测试出错: {str(e)}")

if __name__ == "__main__":
    print("SSVEP检测算法测试程序")
    print("作者: Cline")
    print("日期: 2025/11/8")
    print()
    
    # 运行主要测试
    test_ssvep_detection()
    
    # 可选: 运行参数测试
    # test_algorithm_parameters()
