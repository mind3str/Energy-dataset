import torch
import numpy as np
import pandas as pd
import argparse
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import os

from energy_prediction_model import Generator, ClusterEncoder
from data_utils import preprocess_data, prepare_tsne_data
from train import generate_new_city_data

def main(args):
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载预处理数据 (用于加载数据集信息和标准化器)
    data = preprocess_data(
        args.original_data_path,
        time_steps=args.time_steps,
        test_size=0.2,
        sequence_length=args.sequence_length
    )
    
    # 加载新城市数据
    new_city_df = pd.read_csv(args.new_city_data_path)
    
    # 使用与训练时相同的特征列定义
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
    # 替换特殊值为NaN
    for col in base_feature_cols:
        if col in new_city_df.columns:
            new_city_df[col] = new_city_df[col].replace(['--', 'NA', 'N/A', ''], np.nan)
            # 确保列是数值类型（除了分类特征）
            if col not in ['season', 'weather_code']:
                new_city_df[col] = pd.to_numeric(new_city_df[col], errors='coerce')
    
    # 填充缺失值
    # 对于天气数据，使用前后日期的平均值填充
    weather_cols = ['rain_sum', 'temperature_2m_max', 'temperature_2m_min', 'temperature_2m_mean',
                   'soil_temperature_0_to_7cm_mean', 'soil_temperature_7_to_28cm_mean',
                   'apparent_temperature_mean', 'apparent_temperature_max', 'apparent_temperature_min',
                   'relative_humidity_2m_mean', 'wind_speed_10m_mean', 'precipitation_hours',
                   'snowfall_sum', 'sunshine_duration']
    
    for col in weather_cols:
        if col in new_city_df.columns:
            new_city_df[col] = new_city_df.groupby('cus-no')[col].transform(lambda x: x.fillna(method='ffill').fillna(method='bfill'))
    
    # 对于聚类坐标，使用相同用户的值填充，若无法填充则使用平均值
    for col in ['cluster_x', 'cluster_y']:
        if col in new_city_df.columns:
            new_city_df[col] = new_city_df.groupby('cus-no')[col].transform(lambda x: x.fillna(x.mean()))
            # 若仍有缺失，使用全局平均值
            new_city_df[col] = new_city_df[col].fillna(new_city_df[col].mean())
    
    # 对于社会经济特征，使用相同用户的值填充，若无法填充则使用平均值
    for col in ['population_density', 'house_price']:
        if col in new_city_df.columns:
            new_city_df[col] = new_city_df.groupby('cus-no')[col].transform(lambda x: x.fillna(x.mean()))
            # 若仍有缺失，使用全局平均值
            new_city_df[col] = new_city_df[col].fillna(new_city_df[col].mean())
    
    # 对于时间相关特征，使用常数填充
    time_cols = ['is_holiday', 'holiday_length', 'weekend', 'week_of_year', 'month', 'day_of_week', 'day_of_month']
    for col in time_cols:
        if col in new_city_df.columns:
            new_city_df[col] = new_city_df[col].fillna(0)
    
    # 对于分类特征，使用众数填充
    for col in ['season', 'weather_code']:
        if col in new_city_df.columns:
            mode_value = new_city_df[col].mode().iloc[0] if not new_city_df[col].mode().empty else 'unknown'
            new_city_df[col] = new_city_df[col].fillna(mode_value)
    
    # 独热编码处理分类特征
    # 处理季节特征
    if 'season' in new_city_df.columns:
        season_dummies = pd.get_dummies(new_city_df['season'], prefix='season')
        new_city_df = pd.concat([new_city_df, season_dummies], axis=1)
        new_city_df = new_city_df.drop('season', axis=1)
    
    # 处理天气代码特征
    if 'weather_code' in new_city_df.columns:
        weather_dummies = pd.get_dummies(new_city_df['weather_code'], prefix='weather')
        new_city_df = pd.concat([new_city_df, weather_dummies], axis=1)
        new_city_df = new_city_df.drop('weather_code', axis=1)
    
    # 更新特征列列表（包含独热编码后的特征）
    feature_cols = [col for col in new_city_df.columns if col not in ['cus-no', 'date', 'cost']]
    
    # 确保与训练数据的特征列一致
    if 'feature_cols' in data:
        training_feature_cols = data['feature_cols']
        # 检查是否有缺失的特征列
        missing_cols = set(training_feature_cols) - set(feature_cols)
        if missing_cols:
            print(f"警告：新城市数据缺少以下特征列: {missing_cols}")
            # 为缺失的特征列添加零值
            for col in missing_cols:
                new_city_df[col] = 0
        
        # 使用与训练时相同的特征列顺序
        feature_cols = training_feature_cols
    
    # 提取新城市用户
    user_ids = new_city_df['cus-no'].unique()
    num_users = len(user_ids)
    
    # 基于日期排序
    new_city_df = new_city_df.sort_values(by=['cus-no', 'date'])
    
    # 创建用户ID映射
    user_id_map = {user_id: idx for idx, user_id in enumerate(user_ids)}
    new_city_df['user_idx'] = new_city_df['cus-no'].map(user_id_map)
    
    # 获取所有日期
    all_dates = pd.to_datetime(new_city_df['date'].unique())
    # 使用numpy的sort方法对日期数组进行排序
    all_dates = np.sort(all_dates)
    num_days = len(all_dates)
    
    # 用户坐标（使用聚类坐标替代GPS坐标）
    coordinates = new_city_df[['cluster_x', 'cluster_y', 'user_idx']].drop_duplicates('user_idx')
    coordinates = coordinates.sort_values('user_idx')
    coordinates = coordinates[['cluster_x', 'cluster_y']].values
    
    # 初始化数据数组
    features = np.zeros((num_days, num_users, len(feature_cols)))
    
    # 填充数据
    for day_idx, date in enumerate(all_dates):
        day_data = new_city_df[new_city_df['date'] == date]
        
        for _, row in day_data.iterrows():
            user_idx = int(row['user_idx'])
            
            # 外部特征
            features[day_idx, user_idx, :] = row[feature_cols].values
    
    # 标准化特征
    feature_scaler = data['feature_scaler']
    features_flat = features.reshape(-1, features.shape[-1])
    features_scaled = feature_scaler.transform(features_flat)
    features = features_scaled.reshape(features.shape)
    
    # 加载模型
    checkpoint = torch.load(args.model_path, map_location=device)
    
    # 创建模型实例
    generator = Generator(
        input_dim=len(feature_cols),
        hidden_dim=args.hidden_dim,
        flow_input_dim=args.flow_input_dim,
        flow_hidden_dim=args.flow_hidden_dim,
        num_nodes=num_users,
        time_steps=args.time_steps,
        num_flows=args.num_flows
    ).to(device)
    
    generator.load_state_dict(checkpoint['generator_state_dict'])
    generator.eval()
    
    # 如果需要比较聚类，加载聚类编码器
    if args.compare_clusters:
        cluster_encoder = ClusterEncoder(
            input_dim=args.sequence_length,
            hidden_dim=args.cluster_hidden_dim,
            output_dim=2
        ).to(device)
        
        cluster_encoder.load_state_dict(checkpoint['cluster_encoder_state_dict'])
        cluster_encoder.eval()
    
    # 生成数据
    generated_data = []
    
    # 按时间窗口生成数据
    for t in range(0, num_days, args.time_steps):
        end_t = min(t + args.time_steps, num_days)
        curr_features = features[t:end_t]
        
        # 添加批次维度
        curr_features = np.expand_dims(curr_features, axis=0)
        
        # 生成当前窗口的数据
        curr_generated = generate_new_city_data(
            generator, 
            data['feature_scaler'], 
            data['target_scaler'], 
            curr_features, 
            coordinates, 
            device
        )
        
        generated_data.append(curr_generated)
    
    # 合并所有生成的数据
    generated_data = np.concatenate(generated_data, axis=1)
    
    # 整形为 [num_users, num_days]
    generated_data = generated_data.reshape(num_users, num_days)
    
    # 保存生成的数据
    save_results(generated_data, user_ids, all_dates, args.output_path)
    
    # 如果需要比较聚类
    if args.compare_clusters and args.original_cluster_data is not None:
        # 加载原始城市的聚类数据
        original_cluster_data = np.load(args.original_cluster_data)
        
        # 准备用于TSNE的数据
        real_data, gen_data = prepare_tsne_data(
            original_cluster_data, 
            generated_data, 
            sequence_length=args.sequence_length
        )
        
        # 可视化聚类
        visualize_clusters(
            real_data, 
            gen_data, 
            cluster_encoder, 
            device, 
            args.output_dir
        )

def save_results(generated_data, user_ids, dates, output_path):
    """
    保存生成的数据
    """
    # 创建DataFrame
    results = []
    
    for user_idx, user_id in enumerate(user_ids):
        for day_idx, date in enumerate(dates):
            results.append({
                'cus-no': user_id,
                'date': date,
                'predicted_cost': generated_data[user_idx, day_idx]
            })
    
    # 创建DataFrame并保存
    results_df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    results_df.to_csv(output_path, index=False)
    print(f"生成的数据已保存到 {output_path}")

def visualize_clusters(real_data, gen_data, encoder, device, output_dir):
    """
    可视化聚类结果
    """
    encoder.eval()
    
    with torch.no_grad():
        # 转换为张量
        real_tensor = torch.FloatTensor(real_data).to(device)
        gen_tensor = torch.FloatTensor(gen_data).to(device)
        
        # 使用编码器降维
        encoded_real = encoder(real_tensor).cpu().numpy()
        encoded_gen = encoder(gen_tensor).cpu().numpy()
    
    # 绘制聚类结果
    plt.figure(figsize=(12, 10))
    
    # 真实数据聚类
    plt.subplot(2, 1, 1)
    plt.scatter(encoded_real[:, 0], encoded_real[:, 1], c='blue', alpha=0.5, label='Original City Data')
    plt.title('TSNE Clusters of Original City Data')
    plt.legend()
    
    # 生成数据聚类
    plt.subplot(2, 1, 2)
    plt.scatter(encoded_gen[:, 0], encoded_gen[:, 1], c='red', alpha=0.5, label='Generated New City Data')
    plt.title('TSNE Clusters of Generated New City Data')
    plt.legend()
    
    # 保存图像
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'clusters_comparison.png'))
    plt.close()
    
    print(f"聚类比较可视化已保存到 {os.path.join(output_dir, 'clusters_comparison.png')}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate electricity consumption data for a new city')
    
    parser.add_argument('--original_data_path', type=str, required=True,
                        help='Path to original city data (for preprocessing)')
    parser.add_argument('--new_city_data_path', type=str, required=True,
                        help='Path to new city data (features only)')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--output_path', type=str, default='./results/new_city_predictions.csv',
                        help='Path to save generated data')
    parser.add_argument('--output_dir', type=str, default='./results',
                        help='Directory to save visualization results')
    parser.add_argument('--time_steps', type=int, default=7,
                        help='Number of time steps')
    parser.add_argument('--hidden_dim', type=int, default=128,
                        help='Hidden dimension')
    parser.add_argument('--flow_input_dim', type=int, default=1,
                        help='Flow input dimension')
    parser.add_argument('--flow_hidden_dim', type=int, default=64,
                        help='Flow hidden dimension')
    parser.add_argument('--num_flows', type=int, default=5,
                        help='Number of flow layers')
    parser.add_argument('--sequence_length', type=int, default=365,
                        help='Sequence length for clustering')
    parser.add_argument('--cluster_hidden_dim', type=int, default=128,
                        help='Cluster encoder hidden dimension')
    parser.add_argument('--compare_clusters', action='store_true',
                        help='Whether to compare clusters with original city')
    parser.add_argument('--original_cluster_data', type=str, default=None,
                        help='Path to original city cluster data')
    
    args = parser.parse_args()
    main(args) 