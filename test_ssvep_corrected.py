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

def test_ssvep_with_correct_frequencies():
    """使用正确的频率进行SSVEP检测测试"""
    
    # 【关键】根据频率分析的结果，设置正确的频率映射
    # D1.csv和D2.csv都包含stimID 0-7对应的8种频率
    # 为了兼容两个文件，使用D1的频率作为基准
    FREQ_MAP = {
        0: 16.0,   # stimID=0
        1: 9.0,    # stimID=1
        2: 10.0,   # stimID=2
        3: 11.0,   # stimID=3
        4: 12.0,   # stimID=4
        5: 13.0,   # stimID=5
        6: 10.58,  # stimID=6 (近似10.6Hz或11Hz)
        7: 15.0    # stimID=7
    }
    
    # 根据stimID创建频率列表（顺序很重要！）
    freqs = [FREQ_MAP[i] for i in range(8)]
    
    srate = 250  # 采样率
    dataLen = 2.0  # 使用2秒的数据窗口
    
    print("=" * 80)
    print("【改正版】SSVEP检测测试 - 使用正确的频率映射")
    print("=" * 80)
    print(f"采样率: {srate} Hz")
    print(f"分析窗口长度: {dataLen} 秒")
    print(f"\n【stimID到频率的正确映射】")
    for stim_id, freq in FREQ_MAP.items():
        print(f"  stimID={stim_id} -> {freq:.2f} Hz")
    print(f"\n频率列表: {freqs}")
    print()
    
    # 创建检测器
    detector = ssvepDetect(srate, freqs, dataLen)
    
    # 测试文件
    test_files = ["ExampleData/D1.csv", "ExampleData/D2.csv"]
    
    correct_count = 0
    total_count = 0
    
    for file_path in test_files:
        if not os.path.exists(file_path):
            print(f"警告: 文件不存在 {file_path}")
            continue
        
        print(f"\n{'=' * 80}")
        print(f"处理文件: {file_path}")
        print(f"{'=' * 80}")
        
        data = pd.read_csv(file_path)
        unique_stim_ids = sorted(data['stimID'].unique())
        
        file_correct = 0
        
        for stim_id in unique_stim_ids:
            stim_id = int(stim_id)
            
            # 加载该刺激的数据
            segment_data = load_single_stimulus(file_path, stim_id)
            
            # 只使用前2秒的数据（对应dataLen=2.0秒）
            samples_needed = int(dataLen * srate)
            if segment_data.shape[0] < samples_needed:
                print(f"  stimID={stim_id}: 数据不足 ({segment_data.shape[0]} < {samples_needed}), 跳过")
                continue
            
            segment_data = segment_data[:samples_needed]
            
            # 转置为 (通道数, 样本数) 格式
            data_transposed = segment_data.T
            
            try:
                # 进行检测
                detected_index = detector.detect(data_transposed)
                detected_freq = freqs[detected_index]
                true_freq = FREQ_MAP[stim_id]
                
                # 检查是否正确（允许±0.5Hz的误差范围）
                is_correct = abs(detected_freq - true_freq) < 0.5
                
                result_str = "✓ 正确" if is_correct else "✗ 错误"
                print(f"  stimID={stim_id}: 真实频率={true_freq:.2f}Hz, "
                      f"检测频率={detected_freq:.2f}Hz {result_str}")
                
                if is_correct:
                    file_correct += 1
                    correct_count += 1
                total_count += 1
                
            except Exception as e:
                print(f"  stimID={stim_id}: 检测失败 - {str(e)}")
        
        file_accuracy = (file_correct / 8 * 100) if 8 > 0 else 0
        print(f"\n{file_path} 的识别准确率: {file_correct}/8 = {file_accuracy:.1f}%")
    
    # 总体结果
    print(f"\n{'=' * 80}")
    print(f"【总体结果】")
    print(f"{'=' * 80}")
    overall_accuracy = (correct_count / total_count * 100) if total_count > 0 else 0
    print(f"总体识别准确率: {correct_count}/{total_count} = {overall_accuracy:.1f}%")
    
    if overall_accuracy > 90:
        print("✓ 检测效果良好！")
    elif overall_accuracy > 50:
        print("△ 检测效果一般，建议调整滤波参数或增加数据长度")
    else:
        print("✗ 检测效果不佳，需要进一步调试")

if __name__ == "__main__":
    test_ssvep_with_correct_frequencies()
