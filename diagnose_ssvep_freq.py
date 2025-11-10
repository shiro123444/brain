import numpy as np
import pandas as pd
from scipy import signal as scipysignal
import matplotlib.pyplot as plt

def analyze_data_structure():
    """分析CSV数据结构和刺激编号"""
    print("=" * 80)
    print("【分析数据结构】")
    print("=" * 80)
    
    for file_name in ["ExampleData/D1.csv", "ExampleData/D2.csv"]:
        print(f"\n分析文件: {file_name}")
        print("-" * 80)
        
        data = pd.read_csv(file_name)
        
        # 查看唯一的stimID
        stim_ids = data['stimID'].unique()
        print(f"文件中包含的刺激ID: {sorted(stim_ids)}")
        print(f"刺激ID数量: {len(stim_ids)}")
        
        # 统计每个stimID的样本数
        for stim_id in sorted(stim_ids):
            count = (data['stimID'] == stim_id).sum()
            print(f"  stimID={int(stim_id)}: {count} 行数据")
        
        # 查看taskID的变化规律
        print(f"\ntaskID的变化规律:")
        task_ids = np.array(data['taskID'].values)
        print(f"  前20个taskID: {task_ids[:20]}")
        
        # 找出taskID每次变化的位置
        task_changes = np.where(np.diff(task_ids) != 0)[0] + 1
        if len(task_changes) > 0:
            # 计算每个片段的长度
            segment_lengths = np.diff(np.concatenate(([0], task_changes, [len(task_ids)])))
            print(f"  信号片段长度: {segment_lengths[:10]}... (前10个)")
            print(f"  每个片段长度: {segment_lengths[0]} 行 (对应 {segment_lengths[0]/250:.2f} 秒)")

def extract_freq_from_signal(signal_data, srate=250):
    """通过FFT分析信号的主要频率成分"""
    # 计算FFT
    fft = np.fft.rfft(signal_data, axis=0)
    freqs = np.fft.rfftfreq(signal_data.shape[0], 1/srate)
    magnitude = np.abs(fft)
    
    # 取平均幅值（多通道）
    avg_magnitude = np.mean(magnitude, axis=1)
    
    # 找最强的频率成分
    peak_idx = np.argmax(avg_magnitude)
    peak_freq = freqs[peak_idx]
    
    return peak_freq, avg_magnitude, freqs

def analyze_frequency_content():
    """分析每个刺激ID对应的频率"""
    print("\n" + "=" * 80)
    print("【分析每个刺激的频率成分】")
    print("=" * 80)
    
    srate = 250
    
    for file_name in ["ExampleData/D1.csv", "ExampleData/D2.csv"]:
        print(f"\n分析文件: {file_name}")
        print("-" * 80)
        
        data = pd.read_csv(file_name)
        
        # 获取通道数据
        channel_data = data.iloc[:, :-2].values
        stim_ids = data['stimID'].values
        
        # 对每个stimID进行分析
        stim_freq_mapping = {}
        
        for stim_id in sorted(data['stimID'].unique()):
            # 获取这个stimID的数据
            mask = stim_ids == stim_id
            segment_data = channel_data[mask]
            
            # 分析频率
            peak_freq, magnitude, freqs = extract_freq_from_signal(segment_data, srate)
            
            # 查找6-30Hz范围内的前3个频率
            freq_range_mask = (freqs >= 6) & (freqs <= 30)
            freq_range_indices = np.where(freq_range_mask)[0]
            top_3_indices = freq_range_indices[np.argsort(magnitude[freq_range_indices])[-3:]][::-1]
            
            print(f"\nstimID={int(stim_id)}:")
            print(f"  样本数: {segment_data.shape[0]} (对应 {segment_data.shape[0]/srate:.2f} 秒)")
            print(f"  最强频率: {peak_freq:.2f} Hz")
            print(f"  6-30Hz范围内的TOP3频率:")
            for i, idx in enumerate(top_3_indices, 1):
                print(f"    {i}. {freqs[idx]:.2f} Hz (幅值: {magnitude[idx]:.2f})")
            
            stim_freq_mapping[int(stim_id)] = peak_freq

        print(f"\n【stimID到频率的映射】")
        print(stim_freq_mapping)

def main():
    analyze_data_structure()
    analyze_frequency_content()
    
    print("\n" + "=" * 80)
    print("【建议】")
    print("=" * 80)
    print("""
基于以上分析结果，建议：
1. 确定每个stimID (0-7) 对应的真实刺激频率
2. 更新ssvepdetect.py中的频率列表，确保与真实刺激频率对应
3. 确保频率列表的顺序与stimID的顺序一致

例如，如果分析结果显示：
  stimID=0 -> 12.0 Hz
  stimID=1 -> 15.0 Hz
  ...
  
那么在test_ssvep.py中应该这样设置：
  freqs = [12.0, 15.0, ...]  # 按stimID顺序排列
""")

if __name__ == "__main__":
    main()
