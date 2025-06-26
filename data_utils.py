import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import datetime
import logging
import time
import os
import pickle
from pathlib import Path

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ElectricityDataset(Dataset):
    def __init__(self, features, targets, coordinates, time_steps=7):
        """
        features: [num_samples, num_nodes, feature_dim] - 外部特征
        targets: [num_samples, num_nodes] - 用电量数据
        coordinates: [num_nodes, 2] - GPS坐标 (x, y)
        time_steps: 时间窗口大小
        """
        self.features = features
        self.targets = targets
        self.coordinates = coordinates
        self.time_steps = time_steps
        self.num_samples = features.shape[0] - time_steps + 1
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # 获取时间窗口特征
        features_window = self.features[idx:idx+self.time_steps]  # [time_steps, num_nodes, feature_dim]
        
        # 获取目标 (仅最后一天的用电数据)
        target = self.targets[idx+self.time_steps-1]  # [num_nodes]
        
        return {
            'features': torch.FloatTensor(features_window),
            'target': torch.FloatTensor(target),
            'coordinates': torch.FloatTensor(self.coordinates)
        }

def preprocess_data(csv_path, time_steps=7, test_size=0.2, sequence_length=365, use_cache=True, cache_dir="./data_cache"):
    """
    预处理CSV数据，支持数据缓存加速
    csv_path: CSV文件路径
    time_steps: 时间窗口大小
    test_size: 测试集比例
    sequence_length: 用于聚类的序列长度 (通常为一年)
    use_cache: 是否使用缓存加速
    cache_dir: 缓存目录
    """
    # 创建缓存目录
    if use_cache:
        os.makedirs(cache_dir, exist_ok=True)
        
        # 生成缓存文件路径（基于输入参数的哈希值）
        cache_path = Path(cache_dir) / f"preprocessed_{Path(csv_path).stem}_{time_steps}_{test_size}_{sequence_length}.pkl"
        
        # 检查缓存是否存在
        if cache_path.exists():
            logging.info(f"发现预处理数据缓存，正在加载: {cache_path}")
            start_time = time.time()
            
            try:
                with open(cache_path, 'rb') as f:
                    data = pickle.load(f)
                logging.info(f"加载缓存数据成功，耗时: {time.time() - start_time:.2f}秒")
                return data
            except Exception as e:
                logging.warning(f"加载缓存失败: {e}，将重新处理数据")
    
    # 如果没有缓存或不使用缓存，则进行正常处理
    start_time = time.time()
    logging.info(f"开始预处理数据: {csv_path}")
    
    # 读取数据
    logging.info("正在读取CSV文件...")
    df = pd.read_csv(csv_path)
    logging.info(f"CSV文件读取完成，共 {len(df)} 行")
    
    # 转换日期列
    logging.info("正在转换日期列...")
    df['date'] = pd.to_datetime(df['date'])
    
    # 定义新的特征列（27个特征）
    base_feature_cols = [
        # 1. 用户身份性格品质聚类特征
        'cluster_x', 'cluster_y',
        # 2. 天气特征
        'rain_sum', 'temperature_2m_max', 'temperature_2m_min', 'temperature_2m_mean',
        'soil_temperature_0_to_7cm_mean', 'soil_temperature_7_to_28cm_mean',
        'apparent_temperature_mean', 'apparent_temperature_max', 'apparent_temperature_min',
        'relative_humidity_2m_mean', 'wind_speed_10m_mean', 'precipitation_hours',
        'snowfall_sum', 'sunshine_duration',
        # 3. 时间特征
        'is_holiday', 'holiday_length', 'weekend', 'week_of_year', 'month',
        'day_of_week', 'day_of_month',
        # 4. 分类特征（独热编码前）
        'season', 'weather_code',
        # 5. 社会经济特征
        'population_density', 'house_price'
    ]
    
    # 处理缺失值和特殊值
    logging.info("正在处理缺失值和特殊值...")
    
    # 替换特殊值为NaN
    for col in base_feature_cols:
        if col in df.columns:
            logging.info(f"处理列: {col}")
            df[col] = df[col].replace(['--', 'NA', 'N/A', ''], np.nan)
            # 确保列是数值类型（除了分类特征）
            if col not in ['season', 'weather_code']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # 填充缺失值
    logging.info("正在填充缺失值...")
    
    # 对于天气数据，使用前后日期的平均值填充
    weather_cols = ['rain_sum', 'temperature_2m_max', 'temperature_2m_min', 'temperature_2m_mean',
                   'soil_temperature_0_to_7cm_mean', 'soil_temperature_7_to_28cm_mean',
                   'apparent_temperature_mean', 'apparent_temperature_max', 'apparent_temperature_min',
                   'relative_humidity_2m_mean', 'wind_speed_10m_mean', 'precipitation_hours',
                   'snowfall_sum', 'sunshine_duration']
    
    for col in weather_cols:
        if col in df.columns:
            logging.info(f"填充天气数据: {col}")
            df[col] = df.groupby('cus-no')[col].transform(lambda x: x.fillna(method='ffill').fillna(method='bfill'))
    
    # 对于聚类坐标，使用相同用户的值填充，若无法填充则使用平均值
    logging.info("填充聚类坐标数据")
    for col in ['cluster_x', 'cluster_y']:
        if col in df.columns:
            df[col] = df.groupby('cus-no')[col].transform(lambda x: x.fillna(x.mean()))
            # 若仍有缺失，使用全局平均值
            df[col] = df[col].fillna(df[col].mean())
    
    # 对于社会经济特征，使用相同用户的值填充，若无法填充则使用平均值
    logging.info("填充社会经济特征")
    for col in ['population_density', 'house_price']:
        if col in df.columns:
            df[col] = df.groupby('cus-no')[col].transform(lambda x: x.fillna(x.mean()))
            # 若仍有缺失，使用全局平均值
            df[col] = df[col].fillna(df[col].mean())
    
    # 对于时间相关特征，使用常数填充
    logging.info("填充时间相关特征")
    time_cols = ['is_holiday', 'holiday_length', 'weekend', 'week_of_year', 'month', 'day_of_week', 'day_of_month']
    for col in time_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)
    
    # 对于分类特征，使用众数填充
    logging.info("填充分类特征")
    for col in ['season', 'weather_code']:
        if col in df.columns:
            mode_value = df[col].mode().iloc[0] if not df[col].mode().empty else 'unknown'
            df[col] = df[col].fillna(mode_value)
    
    # 独热编码处理分类特征
    logging.info("处理分类特征的独热编码...")
    
    # 处理季节特征
    if 'season' in df.columns:
        season_dummies = pd.get_dummies(df['season'], prefix='season')
        df = pd.concat([df, season_dummies], axis=1)
        df = df.drop('season', axis=1)
        logging.info(f"季节特征独热编码完成，新增列: {list(season_dummies.columns)}")
    
    # 处理天气代码特征
    if 'weather_code' in df.columns:
        weather_dummies = pd.get_dummies(df['weather_code'], prefix='weather')
        df = pd.concat([df, weather_dummies], axis=1)
        df = df.drop('weather_code', axis=1)
        logging.info(f"天气代码独热编码完成，新增列: {list(weather_dummies.columns)}")
    
    # 更新特征列列表（包含独热编码后的特征）
    feature_cols = [col for col in df.columns if col not in ['cus-no', 'date', 'cost'] and 
                   not col.startswith('userpoint_')]
    
    logging.info(f"最终特征列数量: {len(feature_cols)}")
    logging.info(f"特征列: {feature_cols}")
    
    # 提取所有唯一用户
    logging.info("提取唯一用户ID...")
    user_ids = df['cus-no'].unique()
    num_users = len(user_ids)
    logging.info(f"共有 {num_users} 个唯一用户")
    
    # 基于日期排序
    logging.info("排序数据...")
    df = df.sort_values(by=['cus-no', 'date'])
    
    # 创建用户ID映射
    logging.info("创建用户ID映射...")
    user_id_map = {user_id: idx for idx, user_id in enumerate(user_ids)}
    df['user_idx'] = df['cus-no'].map(user_id_map)
    
    # 获取所有日期并排序
    logging.info("获取唯一日期...")
    all_dates = df['date'].unique()
    # 使用numpy的sort方法对日期数组进行排序
    all_dates = np.sort(all_dates)
    num_days = len(all_dates)
    logging.info(f"共有 {num_days} 天的数据")
    
    # 用户坐标（使用聚类坐标替代GPS坐标）
    logging.info("提取用户聚类坐标...")
    coordinates = df[['cluster_x', 'cluster_y', 'user_idx']].drop_duplicates('user_idx')
    coordinates = coordinates.sort_values('user_idx')
    coordinates = coordinates[['cluster_x', 'cluster_y']].values
    
    # 初始化数据数组
    logging.info(f"初始化数据数组: 形状为 [{num_days}, {num_users}, {len(feature_cols)}]")
    features = np.zeros((num_days, num_users, len(feature_cols)))
    targets = np.zeros((num_days, num_users))
    
    # 优化：通过MultiIndex更快速地填充数据
    logging.info("使用MultiIndex优化数据填充...")
    
    # 创建日期映射
    date_to_idx = {date: idx for idx, date in enumerate(all_dates)}
    
    # 为每行数据添加日期索引
    df['date_idx'] = df['date'].map(date_to_idx)
    
    # 提取所需的所有列的数据
    logging.info("批量提取数据...")
    user_indices = df['user_idx'].values
    date_indices = df['date_idx'].values
    feature_values = df[feature_cols].values
    target_values = df['cost'].values
    
    # 批量填充
    logging.info("批量填充数据数组...")
    for i in range(len(df)):
        user_idx = user_indices[i]
        date_idx = date_indices[i]
        features[date_idx, user_idx, :] = feature_values[i]
        targets[date_idx, user_idx] = target_values[i]
    
    # 标准化特征
    logging.info("标准化特征...")
    feature_scaler = StandardScaler()
    features_flat = features.reshape(-1, features.shape[-1])
    features_scaled = feature_scaler.fit_transform(features_flat)
    features = features_scaled.reshape(features.shape)
    
    # 标准化目标
    logging.info("标准化目标...")
    target_scaler = StandardScaler()
    targets_flat = targets.reshape(-1, 1)
    targets_scaled = target_scaler.fit_transform(targets_flat)
    targets = targets_scaled.reshape(targets.shape)
    
    # 划分训练集和测试集 (按时间)
    logging.info("划分训练集和测试集...")
    train_size = int(num_days * (1 - test_size))
    
    train_features = features[:train_size]
    train_targets = targets[:train_size]
    
    test_features = features[train_size:]
    test_targets = targets[train_size:]
    
    # 创建数据集
    logging.info("创建pytorch数据集...")
    train_dataset = ElectricityDataset(train_features, train_targets, coordinates, time_steps)
    test_dataset = ElectricityDataset(test_features, test_targets, coordinates, time_steps)
    
    # 为聚类准备完整序列数据 (每个用户一整年的数据)
    logging.info("准备聚类数据...")
    # 通常会取与sequence_length一致的天数
    cluster_end_idx = min(sequence_length, train_size)
    cluster_data = targets[:cluster_end_idx].transpose(1, 0)  # [num_users, sequence_length]
    
    elapsed_time = time.time() - start_time
    logging.info(f"数据预处理完成，耗时 {elapsed_time:.2f} 秒")
    
    # 准备返回数据
    data = {
        'train_dataset': train_dataset,
        'test_dataset': test_dataset,
        'feature_scaler': feature_scaler,
        'target_scaler': target_scaler,
        'cluster_data': cluster_data,
        'coordinates': coordinates,
        'all_dates': all_dates,
        'num_users': num_users,
        'feature_dim': len(feature_cols),
        'feature_cols': feature_cols  # 添加特征列名信息
    }
    
    # 如果使用缓存，保存处理后的数据
    if use_cache:
        logging.info(f"保存预处理数据到缓存: {cache_path}")
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(data, f)
            logging.info("数据缓存保存成功")
        except Exception as e:
            logging.warning(f"保存缓存失败: {e}")
    
    return data

def generate_batch_adj_matrix(coordinates, batch_size, sigma=0.1, epsilon=0.5, device='cuda', use_cache=True, cache_dir="./data_cache"):
    """
    为批次生成邻接矩阵，支持邻接矩阵缓存
    coordinates: [num_nodes, 2] - GPS坐标
    batch_size: 批次大小
    """
    # 创建缓存
    if use_cache:
        os.makedirs(cache_dir, exist_ok=True)
        
        # 生成缓存文件路径
        # 使用坐标形状和参数作为标识
        coord_hash = hash(str(coordinates.shape) + str(sigma) + str(epsilon))
        cache_path = Path(cache_dir) / f"adj_matrix_{coord_hash}_{batch_size}.pt"
        
        # 检查缓存是否存在
        if cache_path.exists():
            logging.info(f"发现邻接矩阵缓存，正在加载: {cache_path}")
            try:
                adj_batch = torch.load(cache_path, map_location=device)
                logging.info(f"邻接矩阵加载成功，形状: {adj_batch.shape}")
                return adj_batch
            except Exception as e:
                logging.warning(f"加载邻接矩阵缓存失败: {e}，将重新计算")
    
    # 如果没有缓存或不使用缓存，则重新计算
    logging.info(f"生成邻接矩阵，节点数: {coordinates.shape[0]}, 批次大小: {batch_size}")
    start_time = time.time()
    
    num_nodes = coordinates.shape[0]
    
    # 计算欧氏距离
    logging.info("计算节点间欧氏距离...")
    coord_i = torch.tensor(coordinates, device=device).unsqueeze(0).unsqueeze(1)  # [1, 1, num_nodes, 2]
    coord_j = torch.tensor(coordinates, device=device).unsqueeze(0).unsqueeze(2)  # [1, num_nodes, 1, 2]
    dist = torch.sqrt(torch.sum((coord_i - coord_j) ** 2, dim=3) + 1e-8)  # [1, num_nodes, num_nodes]
    
    # 通过高斯核计算相似度
    logging.info("使用高斯核计算相似度...")
    adj = torch.exp(-dist**2 / sigma**2)
    
    # epsilon-近邻图
    logging.info("创建epsilon-近邻图...")
    mask = (dist <= epsilon).float()
    adj = adj * mask
    
    # 归一化邻接矩阵
    logging.info("归一化邻接矩阵...")
    D = torch.sum(adj, dim=2, keepdim=True)  # 度矩阵
    D_sqrt_inv = torch.pow(D + 1e-10, -0.5)  # 添加小值以避免除零
    
    adj_normalized = D_sqrt_inv * adj * D_sqrt_inv.transpose(1, 2)
    
    # 添加自连接
    logging.info("添加自连接...")
    identity = torch.eye(num_nodes, device=device).unsqueeze(0)
    adj_normalized = adj_normalized + identity
    
    # 复制扩展为批次
    logging.info("扩展为批次...")
    adj_batch = adj_normalized.repeat(batch_size, 1, 1)
    
    elapsed_time = time.time() - start_time
    logging.info(f"邻接矩阵生成完成，耗时 {elapsed_time:.2f} 秒")
    
    # 保存缓存
    if use_cache:
        logging.info(f"保存邻接矩阵到缓存: {cache_path}")
        try:
            torch.save(adj_batch, cache_path)
            logging.info("邻接矩阵缓存保存成功")
        except Exception as e:
            logging.warning(f"保存邻接矩阵缓存失败: {e}")
    
    return adj_batch
    
def prepare_tsne_data(real_data, gen_data, sequence_length=365):
    """
    准备用于TSNE的数据
    real_data: [num_users, sequence_length] - 真实用电数据
    gen_data: [num_users, generated_length] - 生成的用电数据
    """
    logging.info("准备TSNE数据...")
    num_users, gen_length = gen_data.shape
    logging.info(f"真实数据形状: [{num_users}, {real_data.shape[1]}], 生成数据形状: [{num_users}, {gen_length}]")
    
    # 如果生成数据少于序列长度，需要拼接历史数据
    if gen_length < sequence_length:
        logging.info(f"生成数据长度 {gen_length} 小于序列长度 {sequence_length}，进行数据填充...")
        padding_length = sequence_length - gen_length
        # 使用真实数据的最后padding_length天作为填充
        padding_data = real_data[:, -padding_length:]
        combined_gen_data = np.concatenate([padding_data, gen_data], axis=1)
        logging.info(f"填充后的数据形状: {combined_gen_data.shape}")
    else:
        # 如果生成数据足够长，取最后的sequence_length天
        logging.info(f"生成数据长度 {gen_length} 大于等于序列长度 {sequence_length}，截取数据...")
        combined_gen_data = gen_data[:, -sequence_length:]
        logging.info(f"截取后的数据形状: {combined_gen_data.shape}")
    
    logging.info("TSNE数据准备完成")
    return real_data, combined_gen_data 