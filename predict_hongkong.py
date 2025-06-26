#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
使用训练好的模型为香港生成用电量预测数据
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
import logging
import time
from datetime import datetime

from config import CONFIG
from data_utils import preprocess_data
from energy_prediction_model import Generator, Discriminator, ClusterEncoder

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_trained_model(checkpoint_path, device):
    """加载训练好的模型"""
    logging.info(f"加载训练好的模型: {checkpoint_path}")
    
    if not os.path.exists(checkpoint_path):
        logging.error(f"模型文件不存在: {checkpoint_path}")
        return None, None, None
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 获取模型配置
    model_config = checkpoint.get('config', CONFIG)
    
    # 创建模型实例
    generator = Generator(
        input_dim=model_config['feature_dim'],
        hidden_dim=model_config['hidden_dim'],
        flow_input_dim=model_config['flow_input_dim'],
        flow_hidden_dim=model_config['flow_hidden_dim'],
        num_nodes=model_config['num_users'],
        time_steps=model_config['time_steps'],
        num_flows=model_config['num_flows']
    ).to(device)
    
    discriminator = Discriminator(
        input_dim=1,
        hidden_dim=model_config['hidden_dim'],
        num_nodes=model_config['num_users'],
        time_steps=model_config['time_steps']
    ).to(device)
    
    cluster_encoder = ClusterEncoder(
        input_dim=model_config['sequence_length'],
        hidden_dim=model_config['cluster_hidden_dim'],
        output_dim=2
    ).to(device)
    
    # 加载模型参数
    generator.load_state_dict(checkpoint['generator'])
    discriminator.load_state_dict(checkpoint['discriminator'])
    cluster_encoder.load_state_dict(checkpoint['cluster_encoder'])
    
    logging.info("模型加载成功!")
    return generator, discriminator, cluster_encoder

def preprocess_hongkong_data(hk_data_path, shanghai_config):
    """预处理香港数据，使其格式与上海数据一致"""
    logging.info(f"预处理香港数据: {hk_data_path}")
    
    # 读取香港数据
    df = pd.read_csv(hk_data_path)
    logging.info(f"香港数据行数: {len(df)}")
    
    # 由于cost列为空，我们将其设为0（后续会被模型预测替代）
    df['cost'] = 0.0
    
    # 使用相同的预处理管道，但仅提取特征
    temp_config = shanghai_config.copy()
    temp_config['test_size'] = 0.0  # 不分割测试集
    
    # 创建临时文件
    temp_file = "/tmp/hongkong_temp.csv"
    df.to_csv(temp_file, index=False)
    
    try:
        # 预处理数据
        processed_data = preprocess_data(
            temp_file,
            time_steps=temp_config['time_steps'],
            test_size=temp_config['test_size'],
            sequence_length=temp_config['sequence_length'],
            use_cache=False  # 不使用缓存，避免冲突
        )
        
        # 清理临时文件
        os.remove(temp_file)
        
        logging.info("香港数据预处理完成")
        return processed_data
        
    except Exception as e:
        if os.path.exists(temp_file):
            os.remove(temp_file)
        raise e

def generate_hongkong_predictions(generator, hk_data, device, output_path):
    """为香港数据生成用电量预测"""
    logging.info("开始为香港生成用电量预测...")
    
    generator.eval()
    
    # 获取数据
    features = hk_data['train_dataset']  # 使用全部数据
    coordinates = torch.FloatTensor(hk_data['coordinates']).to(device)
    
    # 批量生成预测
    all_predictions = []
    batch_size = 32  # 使用较小的批次大小
    
    with torch.no_grad():
        for i in range(0, len(features), batch_size):
            logging.info(f"处理批次 {i//batch_size + 1}/{(len(features)-1)//batch_size + 1}")
            
            # 获取批次数据
            batch_data = []
            batch_coords = []
            
            for j in range(i, min(i + batch_size, len(features))):
                sample = features[j]
                batch_data.append(sample['features'])
                batch_coords.append(sample['coordinates'])
            
            # 转换为张量
            batch_features = torch.stack(batch_data).to(device)
            batch_coordinates = torch.stack(batch_coords).to(device)
            
            # 生成预测
            predictions = generator(batch_features, batch_coordinates, reverse=True)
            
            # 保存预测结果
            all_predictions.append(predictions.cpu())
    
    # 合并所有预测
    all_predictions = torch.cat(all_predictions, dim=0)
    
    # 逆标准化
    predictions_flat = all_predictions.reshape(-1, 1)
    predictions_unscaled = hk_data['target_scaler'].inverse_transform(predictions_flat.numpy())
    predictions = predictions_unscaled.reshape(all_predictions.shape)
    
    logging.info(f"生成预测完成，形状: {predictions.shape}")
    
    # 保存结果
    save_predictions(predictions, hk_data, output_path)
    
    return predictions

def save_predictions(predictions, hk_data, output_path):
    """保存预测结果为CSV文件"""
    logging.info(f"保存预测结果到: {output_path}")
    
    # 创建结果DataFrame
    results = []
    
    num_samples, num_users = predictions.shape
    user_coords = hk_data['coordinates']
    
    for sample_idx in range(num_samples):
        for user_idx in range(num_users):
            results.append({
                'time_step': sample_idx,
                'user_id': user_idx,
                'longitude': user_coords[user_idx][0],
                'latitude': user_coords[user_idx][1], 
                'predicted_cost': predictions[sample_idx, user_idx],
                'generation_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            })
    
    df_results = pd.DataFrame(results)
    
    # 保存到CSV
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_results.to_csv(output_path, index=False)
    
    logging.info(f"预测结果已保存: {output_path}")
    logging.info(f"预测数据统计:")
    logging.info(f"  样本数: {num_samples}")
    logging.info(f"  用户数: {num_users}")
    logging.info(f"  预测值范围: {predictions.min():.2f} - {predictions.max():.2f}")
    logging.info(f"  预测值平均: {predictions.mean():.2f}")

def main():
    """主函数"""
    logging.info("="*80)
    logging.info("香港用电量预测 - 使用上海训练的模型")
    logging.info("="*80)
    
    # 设备设置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"使用设备: {device}")
    
    # 文件路径
    hk_data_path = "/data2/lians/energy_new/0528/merged_hongkong_data_full_batched_updated_0528_renamed.csv"
    model_checkpoint = "./checkpoints/final_model.pth"
    output_path = "./results/hongkong_predictions.csv"
    
    # 检查文件
    if not os.path.exists(hk_data_path):
        logging.error(f"香港数据文件不存在: {hk_data_path}")
        sys.exit(1)
    
    if not os.path.exists(model_checkpoint):
        logging.error(f"模型检查点不存在: {model_checkpoint}")
        logging.error("请先运行 python start_training.py 训练模型")
        sys.exit(1)
    
    try:
        # 1. 加载训练好的模型
        generator, discriminator, cluster_encoder = load_trained_model(model_checkpoint, device)
        if generator is None:
            sys.exit(1)
        
        # 2. 预处理香港数据
        hk_data = preprocess_hongkong_data(hk_data_path, CONFIG)
        
        # 3. 生成预测
        predictions = generate_hongkong_predictions(generator, hk_data, device, output_path)
        
        logging.info("="*80)
        logging.info("香港用电量预测完成!")
        logging.info(f"预测结果保存在: {output_path}")
        logging.info("="*80)
        
    except Exception as e:
        logging.error(f"预测过程出错: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main() 