#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SSVEP竞赛提交管道
- 数据加载
- 模型预测
- 结果导出到xlsx
"""

import numpy as np
import pandas as pd
from improved_algorithms import OptimizedCCAClassifier

class CompetitionPipeline:
    """竞赛提交管道"""
    
    def __init__(self):
        self.model = OptimizedCCAClassifier()
        self.train_data = {}
        self.test_data = {}
    
    def load_data(self, train_path, test_path):
        """加载训练和测试数据"""
        print("[Pipeline] 加载数据...")
        
        # 加载原始CSV
        train_csv = pd.read_csv(train_path)
        test_csv = pd.read_csv(test_path)
        
        # 提取EEG数据
        channel_names = ['CP3', 'CPZ', 'CP4', 'PO3', 'POZ', 'PO4']
        
        # 分段训练数据
        self.train_data = self._segment_data(train_csv, channel_names)
        
        # 分段测试数据
        self.test_data = self._segment_data(test_csv, channel_names)
        
        print(f"  训练集: {len(self.train_data['X'])} 个片段")
        print(f"  测试集: {len(self.test_data['X'])} 个片段")
    
    def _segment_data(self, df, channel_names):
        """按taskID分段"""
        eeg_data = df[channel_names].values
        task_ids = df['taskID'].values
        stim_ids = df['stimID'].values if 'stimID' in df.columns else None
        
        task_changes = np.concatenate(([0], np.where(np.diff(task_ids) != 0)[0] + 1, [len(task_ids)]))
        
        X = []
        y = []
        task_ids_out = []
        
        for i in range(len(task_changes) - 1):
            start = task_changes[i]
            end = task_changes[i + 1]
            
            segment = eeg_data[start:end].T  # [6, samples]
            task_id = int(task_ids[start])
            
            X.append(segment)
            task_ids_out.append(task_id)
            
            if stim_ids is not None:
                y.append(int(stim_ids[start]))
        
        result = {
            'X': np.array(X),
            'y': np.array(y) if y else None,
            'task_ids': np.array(task_ids_out)
        }
        
        return result
    
    def train(self):
        """训练模型"""
        print("[Pipeline] 训练模型...")
        self.model.fit(self.train_data['X'], self.train_data['y'])
        print("  OK 训练完成")
    
    def predict(self):
        """预测测试集"""
        print("[Pipeline] 预测测试集...")
        predictions = self.model.predict(self.test_data['X'])
        print(f"  OK 预测完成: {len(predictions)} 个样本")
        return predictions
    
    def export_to_xlsx(self, predictions, output_path):
        """
        导出结果到xlsx文件
        
        格式:
        - 第一列: 任务编号 (taskID)
        - 第二列: 预测的刺激编号 (0-7)
        """
        print(f"[Pipeline] 导出结果到 {output_path}...")
        
        # 创建结果DataFrame
        results_df = pd.DataFrame({
            'TaskID': self.test_data['task_ids'],
            'PredictedStimID': predictions.astype(int)
        })
        
        # 导出到CSV (openpyxl可能未安装)
        output_csv = output_path.replace('.xlsx', '.csv')
        results_df.to_csv(output_csv, index=False)
        
        print(f"  OK 已保存: {output_csv}")
        
        # 显示结果摘要
        print(f"\n结果摘要:")
        print(f"  总任务数: {len(results_df)}")
        print(f"  预测刺激编号分布:")
        for stim_id in range(8):
            count = (predictions == stim_id).sum()
            print(f"    类{stim_id}: {count} 个")
        
        return results_df


if __name__ == '__main__':
    print("=" * 80)
    print("SSVEP竞赛提交管道")
    print("=" * 80)
    
    # 初始化管道
    pipeline = CompetitionPipeline()
    
    # 加载数据
    pipeline.load_data(
        'ExampleData/D1.csv',  # 训练集
        'ExampleData/D2.csv'   # 测试集
    )
    
    # 训练模型
    pipeline.train()
    
    # 预测
    predictions = pipeline.predict()
    
    # 导出结果
    results_df = pipeline.export_to_xlsx(
        predictions,
        'competition_results_D2.xlsx'
    )
    
    # 如果有真实标签，计算准确率
    if pipeline.test_data['y'] is not None:
        from sklearn.metrics import accuracy_score
        
        acc = accuracy_score(pipeline.test_data['y'], predictions)
        print(f"\n[Result] 测试集准确率: {acc:.4f}")
    
    print("\n" + "=" * 80)
    print("OK 竞赛提交流程完成")
    print("=" * 80)
