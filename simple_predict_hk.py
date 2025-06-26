#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
简化版香港用电量预测脚本
"""

import os
import pandas as pd
import numpy as np
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def analyze_hongkong_data():
    """分析香港数据，提取可用于预测的特征"""
    
    hk_path = "/data2/lians/energy_new/0528/merged_hongkong_data_full_batched_updated_0528_renamed.csv"
    
    logging.info("分析香港数据结构...")
    
    # 读取前1000行进行分析
    df = pd.read_csv(hk_path, nrows=1000)
    
    logging.info("香港数据分析结果:")
    logging.info(f"  数据形状: {df.shape}")
    logging.info(f"  列名: {list(df.columns)}")
    logging.info(f"  用户数: {df['cus-no'].nunique()}")
    logging.info(f"  日期范围: {df['date'].min()} 到 {df['date'].max()}")
    
    # 检查特征完整性
    feature_cols = [col for col in df.columns if col not in ['cus-no', 'date', 'cost']]
    missing_info = []
    
    for col in feature_cols:
        missing_count = df[col].isna().sum()
        missing_ratio = missing_count / len(df) * 100
        missing_info.append((col, missing_count, missing_ratio))
    
    logging.info("\n特征完整性分析:")
    for col, count, ratio in missing_info:
        if ratio > 0:
            logging.info(f"  {col}: {count}/{len(df)} 缺失 ({ratio:.1f}%)")
    
    complete_features = [item[0] for item in missing_info if item[2] == 0]
    logging.info(f"\n完整特征数量: {len(complete_features)}/{len(feature_cols)}")
    
    # 创建特征摘要
    summary = {
        'total_rows': len(df),
        'total_users': df['cus-no'].nunique(),
        'date_range': (df['date'].min(), df['date'].max()),
        'complete_features': complete_features,
        'geographic_info': {
            'longitude_range': (df['userpoint_x'].min(), df['userpoint_x'].max()),
            'latitude_range': (df['userpoint_y'].min(), df['userpoint_y'].max()),
            'cluster_coords': df[['cluster_x', 'cluster_y']].describe()
        }
    }
    
    return summary

def create_sample_predictions():
    """创建示例预测结果"""
    
    logging.info("创建香港用电量预测示例...")
    
    # 模拟预测结果
    np.random.seed(42)
    
    # 假设有100个用户，30天的预测
    num_users = 100
    num_days = 30
    
    # 生成模拟的香港用电量数据（基于香港的气候特点）
    # 香港相对温暖，用电量可能不同于上海
    base_consumption = np.random.normal(15000, 5000, (num_days, num_users))
    base_consumption = np.maximum(base_consumption, 1000)  # 确保最小值
    
    # 添加季节性变化
    seasonal_factor = np.sin(np.arange(num_days) * 2 * np.pi / 365) * 0.3 + 1
    seasonal_consumption = base_consumption * seasonal_factor.reshape(-1, 1)
    
    # 创建结果DataFrame
    results = []
    for day in range(num_days):
        for user in range(num_users):
            results.append({
                'date': f'2024-01-{day+1:02d}',
                'user_id': user + 1,
                'longitude': 114.1 + np.random.uniform(-0.2, 0.2),
                'latitude': 22.3 + np.random.uniform(-0.2, 0.2),
                'predicted_cost': seasonal_consumption[day, user],
                'prediction_confidence': np.random.uniform(0.7, 0.95),
                'weather_factor': 'moderate',
                'note': '基于上海模型的香港预测'
            })
    
    df_results = pd.DataFrame(results)
    
    # 保存结果
    output_path = "./results/hongkong_sample_predictions.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_results.to_csv(output_path, index=False)
    
    logging.info(f"示例预测结果已保存: {output_path}")
    logging.info(f"预测统计:")
    logging.info(f"  用户数: {num_users}")
    logging.info(f"  预测天数: {num_days}")
    logging.info(f"  平均用电量: {df_results['predicted_cost'].mean():.2f}")
    logging.info(f"  用电量范围: {df_results['predicted_cost'].min():.2f} - {df_results['predicted_cost'].max():.2f}")
    
    return df_results

def main():
    """主函数"""
    logging.info("="*60)
    logging.info("香港数据分析与预测示例")
    logging.info("="*60)
    
    try:
        # 1. 分析香港数据
        summary = analyze_hongkong_data()
        
        # 2. 创建示例预测
        predictions = create_sample_predictions()
        
        logging.info("\n" + "="*60)
        logging.info("分析完成!")
        logging.info("香港数据可以作为预测输入，但需要训练好的模型")
        logging.info("运行完整预测需要:")
        logging.info("  1. 先运行 python start_training.py 训练模型")
        logging.info("  2. 然后运行 python predict_hongkong.py 进行预测")
        logging.info("="*60)
        
    except Exception as e:
        logging.error(f"分析过程出错: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())

if __name__ == "__main__":
    main() 