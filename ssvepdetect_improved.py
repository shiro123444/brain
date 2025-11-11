#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
改进的SSVEP检测器 - 增加稳健性和调试信息
"""

import numpy as np
from typing import cast, Tuple, List
from scipy import signal as scipysignal
from sklearn.cross_decomposition import CCA

class ssvepDetectImproved:
    def __init__(self, srate, freqs, dataLen):
        self.cca = CCA(n_components=1)
        self.srate = srate
        self.freqs = freqs
        templLen = int(dataLen * srate)
        self.TemplateSet = []
        sample = np.linspace(0, (templLen - 1) / srate, templLen, endpoint=True)

        for freq in freqs:
            phase = 2 * np.pi * freq * sample
            sintemp = np.sin(phase)
            costemp = np.cos(phase)
            tempset = np.vstack((sintemp, costemp))
            self.TemplateSet.append(tempset)

    def detect(self, data, return_coefficients=False):
        """识别信号"""
        data = self.pre_filter(data)
        
        template_length = self.TemplateSet[0].shape[1]
        
        # 确保数据长度与模板长度匹配
        if data.shape[1] > template_length:
            data = data[:, :template_length]
        elif data.shape[1] < template_length:
            pad_length = template_length - data.shape[1]
            data = np.pad(data, ((0, 0), (0, pad_length)), mode='constant', constant_values=0)
        
        cdata = data.transpose()  # (samples, channels)
        coefficients = []
        
        for template in self.TemplateSet:
            ctemplate = template.transpose()  # (samples, 2)
            try:
                # CCA分析
                self.cca.fit(cdata, ctemplate)
                datatran, templatetran = self.cca.transform(cdata, ctemplate)
                
                # 计算相关系数
                coe = np.corrcoef(datatran[:, 0], templatetran[:, 0])[0, 1]
                
                # 处理NaN
                if np.isnan(coe) or np.isinf(coe):
                    coe = 0.0
                
                coefficients.append(max(0.0, float(coe)))
            except Exception as e:
                coefficients.append(0.0)
        
        # 选择最大相关系数对应的频率
        best_idx = np.argmax(coefficients)
        
        if return_coefficients:
            return best_idx, coefficients
        return best_idx

    def pre_filter(self, data):
        """预处理 - 陷波和带通滤波"""
        # 50Hz陷波滤波
        b, a = scipysignal.iircomb(50, 35, ftype='notch', fs=self.srate)
        
        # 带通滤波 6-90Hz
        fs = self.srate / 2
        N, Wn = scipysignal.ellipord([6 / fs, 90 / fs], [2 / fs, 100 / fs], 3, 40)
        b1, a1 = cast(Tuple[np.ndarray, np.ndarray], 
                     scipysignal.ellip(N, 1, 90, Wn, 'bandpass'))
        
        # 应用滤波 (axis=1用于样本轴)
        filter_data = scipysignal.filtfilt(b1, a1, 
                     scipysignal.filtfilt(b, a, data, axis=1), axis=1)
        return filter_data
