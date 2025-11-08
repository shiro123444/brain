import numpy as np
import pandas as pd
from ssvepdetect import ssvepDetect
import matplotlib.pyplot as plt

def debug_ssvep_detection():
    """调试SSVEP检测"""
    
    # 参数设置
    srate = 250  # 采样率 (Hz)
    freqs = [1.0, 7.0, 10.0, 12.0, 15.0, 8.57]  # 测试频率 (Hz)
    dataLen = 2.0  # 分析时间窗口长度 (秒)
    
    print("=" * 60)
    print("SSVEP检测调试")
    print("=" * 60)
    print(f"采样率: {srate} Hz")
    print(f"测试频率: {freqs} Hz")
    print(f"分析窗口长度: {dataLen} 秒")
    print()
    
    # 创建检测器
    detector = ssvepDetect(srate, freqs, dataLen)
    
    # 加载数据
    file_path = "ExampleData/D1.csv"
    print(f"加载数据: {file_path}")
    
    data = pd.read_csv(file_path)
    print(f"数据形状: {data.shape}")
    
    # 提取通道数据
    channels_data = data.iloc[:, :-2].values  # shape: (samples, channels)
    stim_id = data.iloc[:, -1].iloc[0]  # 获取刺激ID
    
    print(f"通道数据形状: {channels_data.shape}")
    print(f"刺激ID: {stim_id}")
    
    # 转置数据
    data_transposed = channels_data.T
    print(f"转置后数据形状: {data_transposed.shape}")
    
    # 检查模板长度
    template_length = detector.TemplateSet[0].shape[1]
    print(f"模板长度: {template_length} 样本")
    
    # 确保数据长度与模板长度匹配
    if data_transposed.shape[1] > template_length:
        data_used = data_transposed[:, :template_length]
        print(f"数据被截断到前 {template_length} 个样本")
    else:
        data_used = data_transposed
        print(f"使用全部 {data_used.shape[1]} 个样本")
    
    print(f"用于检测的数据形状: {data_used.shape}")
    
    # 显示每个模板的信息
    print("\n模板信息:")
    for i, template in enumerate(detector.TemplateSet):
        print(f"  频率 {freqs[i]} Hz: 模板形状 {template.shape}")
        # 显示模板的前几个值
        print(f"    正弦部分前5个值: {template[0, :5]}")
        print(f"    余弦部分前5个值: {template[1, :5]}")
    
    # 进行预处理
    print("\n进行预处理...")
    filtered_data = detector.pre_filter(data_used)
    print(f"预处理后数据形状: {filtered_data.shape}")
    
    # 计算相关系数
    print("\n计算相关系数...")
    p = []
    cdata = filtered_data.transpose()
    
    for i, template in enumerate(detector.TemplateSet):
        ctemplate = template.transpose()
        print(f"\n处理频率 {freqs[i]} Hz:")
        print(f"  数据形状: {cdata.shape}")
        print(f"  模板形状: {ctemplate.shape}")
        
        # 计算相关系数
        try:
            detector.cca.fit(cdata, ctemplate)
            datatran, templatetran = detector.cca.transform(cdata, ctemplate)
            coe = np.corrcoef(datatran[:,0], templatetran[:,0])[0,1]
            p.append(coe)
            print(f"  相关系数: {coe}")
            
            # 显示变换后的数据
            print(f"  变换后数据形状: {datatran.shape}")
            print(f"  变换后模板形状: {templatetran.shape}")
            
        except Exception as e:
            print(f"  错误: {str(e)}")
            p.append(0.0)
    
    # 显示所有相关系数
    print(f"\n所有相关系数: {p}")
    print(f"最大相关系数: {max(p)} at index {p.index(max(p))}")
    
    # 检测结果
    detected_index = p.index(max(p))
    detected_freq = freqs[detected_index]
    
    print(f"\n检测结果:")
    print(f"  检测到的频率索引: {detected_index}")
    print(f"  检测到的频率: {detected_freq} Hz")
    print(f"  真实刺激频率: {stim_id} Hz")
    print(f"  检测是否正确: {detected_freq == stim_id}")

if __name__ == "__main__":
    debug_ssvep_detection()
