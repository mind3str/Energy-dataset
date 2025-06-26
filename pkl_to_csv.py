import os
import pickle
import pandas as pd
import numpy as np
import argparse
from pathlib import Path
import logging
import json

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def convert_pkl_to_csv(pkl_path, output_dir, max_rows=1000):
    """
    将预处理的PKL数据转换为CSV和JSON格式
    
    参数:
        pkl_path: pickle文件路径
        output_dir: 输出目录
        max_rows: 每个CSV文件的最大行数（避免CSV文件过大）
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    logging.info(f"正在加载pickle文件: {pkl_path}")
    try:
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        logging.info("加载成功")
    except Exception as e:
        logging.error(f"加载失败: {e}")
        return
    
    # 提取主要数据组件
    logging.info("正在提取数据组件")
    
    # 1. 保存数据集结构说明
    data_structure = {
        "num_users": data.get('num_users', "未知"),
        "feature_dim": data.get('feature_dim', "未知"),
        "available_keys": list(data.keys()),
        "data_types": {k: str(type(v)) for k, v in data.items()},
        "shapes": {},
        "description": {
            "train_dataset": "训练数据集 - ElectricityDataset类",
            "test_dataset": "测试数据集 - ElectricityDataset类",
            "feature_scaler": "特征标准化器 - StandardScaler",
            "target_scaler": "目标标准化器 - StandardScaler",
            "cluster_data": "聚类数据 - 每个用户的用电序列",
            "coordinates": "用户坐标 - [num_users, 2]",
            "all_dates": "所有日期 - numpy日期数组",
            "num_users": "用户数量",
            "feature_dim": "特征维度"
        }
    }
    
    # 保存形状信息
    for key in data.keys():
        if hasattr(data[key], 'shape'):
            data_structure["shapes"][key] = str(data[key].shape)
        elif isinstance(data[key], list) or isinstance(data[key], tuple):
            data_structure["shapes"][key] = f"Length: {len(data[key])}"
    
    # 将结构信息保存为JSON
    structure_path = os.path.join(output_dir, "data_structure.json")
    with open(structure_path, 'w', encoding='utf-8') as f:
        json.dump(data_structure, f, ensure_ascii=False, indent=4)
    logging.info(f"数据结构信息已保存至: {structure_path}")
    
    # 2. 保存坐标数据
    if 'coordinates' in data:
        coords = data['coordinates']
        coords_df = pd.DataFrame(coords, columns=['longitude', 'latitude'])
        coords_df['user_id'] = range(len(coords_df))
        coords_path = os.path.join(output_dir, "user_coordinates.csv")
        coords_df.to_csv(coords_path, index=False)
        logging.info(f"用户坐标已保存至: {coords_path}")
    
    # 3. 保存日期数据
    if 'all_dates' in data:
        dates = data['all_dates']
        dates_df = pd.DataFrame({'date': dates})
        dates_path = os.path.join(output_dir, "all_dates.csv")
        dates_df.to_csv(dates_path, index=False)
        logging.info(f"日期信息已保存至: {dates_path}")
    
    # 4. 保存聚类数据
    if 'cluster_data' in data:
        cluster_data = data['cluster_data']
        
        # 为列创建名称（基于索引或日期）
        if 'all_dates' in data and len(data['all_dates']) >= cluster_data.shape[1]:
            # 使用日期作为列名
            date_columns = [str(date)[:10] for date in data['all_dates'][:cluster_data.shape[1]]]
        else:
            # 使用索引作为列名
            date_columns = [f'day_{i}' for i in range(cluster_data.shape[1])]
        
        # 创建DataFrame
        cluster_df = pd.DataFrame(cluster_data, columns=date_columns)
        cluster_df['user_id'] = range(len(cluster_df))
        
        # 如果行数过多，分块保存
        if len(cluster_df) > max_rows:
            chunks = np.array_split(cluster_df, np.ceil(len(cluster_df) / max_rows))
            for i, chunk in enumerate(chunks):
                chunk_path = os.path.join(output_dir, f"cluster_data_part{i+1}.csv")
                chunk.to_csv(chunk_path, index=False)
            logging.info(f"聚类数据已分块保存至: {output_dir}/cluster_data_part*.csv")
        else:
            cluster_path = os.path.join(output_dir, "cluster_data.csv")
            cluster_df.to_csv(cluster_path, index=False)
            logging.info(f"聚类数据已保存至: {cluster_path}")
    
    # 5. 保存训练数据样本
    if 'train_dataset' in data and hasattr(data['train_dataset'], 'features'):
        try:
            train_features = data['train_dataset'].features
            train_targets = data['train_dataset'].targets
            
            # 保存特征样本（取前几天的数据作为样本）
            sample_days = min(5, train_features.shape[0])
            sample_users = min(100, train_features.shape[1])
            
            for day in range(sample_days):
                # 保存这一天的特征
                feature_sample = train_features[day, :sample_users, :]
                
                # 获取特征列名
                feature_cols = [f'feature_{i}' for i in range(feature_sample.shape[1])]
                
                # 创建DataFrame
                feature_df = pd.DataFrame(feature_sample, columns=feature_cols)
                feature_df['user_id'] = range(sample_users)
                
                # 保存
                feature_path = os.path.join(output_dir, f"train_features_day{day+1}.csv")
                feature_df.to_csv(feature_path, index=False)
            
            # 保存目标样本
            target_sample = train_targets[:sample_days, :sample_users]
            
            # 转置为用户为行，日期为列的格式
            target_df = pd.DataFrame(target_sample.T)
            
            # 设置列名
            if 'all_dates' in data and len(data['all_dates']) >= sample_days:
                date_columns = [str(date)[:10] for date in data['all_dates'][:sample_days]]
                target_df.columns = date_columns
            else:
                target_df.columns = [f'day_{i+1}' for i in range(sample_days)]
                
            target_df['user_id'] = range(sample_users)
            
            # 保存
            target_path = os.path.join(output_dir, f"train_targets_sample.csv")
            target_df.to_csv(target_path, index=False)
            
            logging.info(f"训练数据样本已保存至: {output_dir}/train_features_day*.csv 和 {target_path}")
        except Exception as e:
            logging.error(f"保存训练数据样本时出错: {e}")
    
    # 6. 样本数据反归一化（如果有scaler）
    if 'target_scaler' in data and 'train_dataset' in data and hasattr(data['train_dataset'], 'targets'):
        try:
            # 获取反归一化函数
            target_scaler = data['target_scaler']
            
            # 获取部分目标样本
            sample_days = min(5, data['train_dataset'].targets.shape[0])
            sample_users = min(100, data['train_dataset'].targets.shape[1])
            target_sample = data['train_dataset'].targets[:sample_days, :sample_users]
            
            # 反归一化
            target_sample_flat = target_sample.reshape(-1, 1)
            target_sample_original = target_scaler.inverse_transform(target_sample_flat)
            target_sample_original = target_sample_original.reshape(target_sample.shape)
            
            # 转置为用户为行，日期为列的格式
            target_df_original = pd.DataFrame(target_sample_original.T)
            
            # 设置列名
            if 'all_dates' in data and len(data['all_dates']) >= sample_days:
                date_columns = [str(date)[:10] for date in data['all_dates'][:sample_days]]
                target_df_original.columns = date_columns
            else:
                target_df_original.columns = [f'day_{i+1}' for i in range(sample_days)]
                
            target_df_original['user_id'] = range(sample_users)
            
            # 保存
            target_original_path = os.path.join(output_dir, f"train_targets_original_sample.csv")
            target_df_original.to_csv(target_original_path, index=False)
            
            logging.info(f"原始尺度的目标数据样本已保存至: {target_original_path}")
        except Exception as e:
            logging.error(f"保存反归一化数据时出错: {e}")
    
    logging.info("转换完成！数据已保存到 " + output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="将pkl格式的数据转换为csv格式")
    parser.add_argument("--pkl_path", type=str, default="data_cache/preprocessed_new_cleaned_data_7_0.2_365.pkl", 
                        help="pickle文件路径")
    parser.add_argument("--output_dir", type=str, default="data_exported", 
                        help="输出目录")
    parser.add_argument("--max_rows", type=int, default=1000, 
                        help="每个CSV文件的最大行数")
    
    args = parser.parse_args()
    convert_pkl_to_csv(args.pkl_path, args.output_dir, args.max_rows) 