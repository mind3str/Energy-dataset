"""
快速入门脚本，演示电力消耗预测和生成模型的基本用法
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
import time
from tqdm import tqdm
from datetime import datetime
from config import CONFIG
from data_utils import preprocess_data
from energy_prediction_model import Generator, Discriminator, ClusterEncoder

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def quickstart():
    # 欢迎信息
    logging.info("="*80)
    logging.info("电力消耗预测与生成模型 - 快速入门演示")
    logging.info("="*80)
    
    # 检查设备
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logging.info(f"使用CUDA设备: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        logging.info("使用CPU设备")
    
    # 创建输出目录
    for directory in [CONFIG['checkpoint_dir'], CONFIG['vis_dir'], CONFIG['output_dir'], CONFIG['cache_dir']]:
        os.makedirs(directory, exist_ok=True)
        logging.info(f"创建目录: {directory}")
    
    # 1. 数据预处理
    start_time = time.time()
    logging.info("\n步骤1: 数据预处理")
    logging.info(f"正在处理数据文件: {CONFIG['data_path']}")
    
    try:
        # 检查文件是否存在
        if not os.path.exists(CONFIG['data_path']):
            logging.error(f"错误: 数据文件 {CONFIG['data_path']} 不存在!")
            sys.exit(1)
            
        # 读取数据文件的前几行以检查格式
        logging.info("检查数据文件格式...")
        sample_data = pd.read_csv(CONFIG['data_path'], nrows=5)
        logging.info(f"数据样例:\n{sample_data.head()}")
        logging.info(f"数据列: {', '.join(sample_data.columns)}")
        
        # 进行预处理，启用缓存
        logging.info("开始数据预处理...")
        data = preprocess_data(
            CONFIG['data_path'],
            time_steps=CONFIG['time_steps'],
            test_size=CONFIG['test_size'],
            sequence_length=CONFIG['sequence_length'],
            use_cache=CONFIG.get('use_cache', True),
            cache_dir=CONFIG.get('cache_dir', "./data_cache")
        )
        
        logging.info(f"数据预处理完成，耗时: {time.time() - start_time:.2f}秒")
        
        num_users = data['num_users']
        feature_dim = data['feature_dim']
        logging.info(f"用户数量: {num_users}，特征维度: {feature_dim}")
        
        # 显示数据统计信息
        logging.info("数据统计信息:")
        logging.info(f"训练集大小: {len(data['train_dataset'])}")
        logging.info(f"测试集大小: {len(data['test_dataset'])}")
        
        # 可视化样本数据
        sample_idx = np.random.randint(0, len(data['train_dataset']))
        sample = data['train_dataset'][sample_idx]
        logging.info(f"随机样本形状: 特征={sample['features'].shape}, 目标={sample['target'].shape}")
        
    except Exception as e:
        logging.error(f"数据预处理出错: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
        sys.exit(1)
    
    # 2. 模型训练
    logging.info("\n步骤2: 模型训练")
    logging.info("注: 示例仅训练1轮，完整训练请使用train.py")
    
    try:
        # 创建演示配置（只训练1轮）
        demo_config = CONFIG.copy()
        demo_config['num_epochs'] = 1
        demo_config['batch_size'] = 8  # 降低批次大小以减少内存使用
        
        # 初始化模型
        logging.info("初始化模型...")
        
        logging.info("创建生成器...")
        generator = Generator(
            input_dim=feature_dim,
            hidden_dim=demo_config['hidden_dim'],
            flow_input_dim=demo_config['flow_input_dim'],
            flow_hidden_dim=demo_config['flow_hidden_dim'],
            num_nodes=num_users,
            time_steps=demo_config['time_steps'],
            num_flows=demo_config['num_flows']
        ).to(device)
        
        logging.info("创建判别器...")
        discriminator = Discriminator(
            input_dim=1,  # 用电量
            hidden_dim=demo_config['hidden_dim'],
            num_nodes=num_users,
            time_steps=demo_config['time_steps']
        ).to(device)
        
        logging.info("创建聚类编码器...")
        cluster_encoder = ClusterEncoder(
            input_dim=demo_config['sequence_length'],
            hidden_dim=demo_config['cluster_hidden_dim'],
            output_dim=2  # 降到2维
        ).to(device)
        
        # 跳过实际训练过程
        logging.info("演示中跳过实际训练过程...")
        logging.info("在实际应用中，使用 python train.py 进行完整训练")
        
    except Exception as e:
        logging.error(f"模型初始化出错: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
        sys.exit(1)
    
    # 3. 生成新城市数据
    logging.info("\n步骤3: 生成新城市数据")
    
    try:
        # 选择部分样本作为"新城市"数据
        logging.info("从原始数据集中选择样本作为新城市数据...")
        
        # 限制样本数量，避免内存问题
        max_samples = min(10, len(data['train_dataset']))
        sample_indices = np.random.choice(len(data['train_dataset']), max_samples, replace=False)
        
        # 准备特征和坐标
        demo_features = []
        demo_coords = []
        
        logging.info(f"选择 {max_samples} 个样本作为新城市数据")
        for i in sample_indices:
            sample = data['train_dataset'][i]
            demo_features.append(sample['features'])
            demo_coords.append(sample['coordinates'])
        
        demo_features = torch.stack(demo_features)
        demo_coords = torch.stack(demo_coords)
        
        logging.info(f"新城市特征形状: {demo_features.shape}")
        logging.info(f"新城市坐标形状: {demo_coords.shape}")
        
        # 假设新城市数据已准备好
        logging.info("演示中跳过实际数据生成...")
        logging.info("在实际应用中，使用 python generate_new_city.py --new_city_data_path YOUR_PATH")
        
    except Exception as e:
        logging.error(f"新城市数据准备出错: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
        sys.exit(1)
    
    # 4. 可视化聚类结果
    logging.info("\n步骤4: 可视化聚类结果")
    logging.info("演示中跳过实际可视化过程...")
    logging.info("在实际应用中，训练过程中会自动生成可视化结果")
    
    # 总结
    logging.info("\n总结")
    logging.info("=" * 80)
    logging.info("快速入门演示完成，您已了解了基本工作流程：")
    logging.info("1. 数据预处理：从CSV加载数据并进行预处理")
    logging.info("2. 模型训练：训练生成器、判别器和聚类编码器")
    logging.info("3. 生成新城市数据：基于条件特征生成新数据")
    logging.info("4. 可视化聚类结果：验证生成数据的分布特性")
    logging.info("=" * 80)
    logging.info("使用建议:")
    logging.info("- 完整训练: python train.py")
    logging.info("- 生成新城市数据: python generate_new_city.py --new_city_data_path YOUR_PATH")
    logging.info("- 聚类分析: 查看 ./visualizations 目录下的可视化结果")
    logging.info("=" * 80)

if __name__ == "__main__":
    # 捕获整个过程的异常，避免无提示崩溃
    try:
        start_time = time.time()
        quickstart()
        logging.info(f"快速入门演示完成，总耗时: {time.time() - start_time:.2f}秒")
    except Exception as e:
        logging.error(f"快速入门演示出错: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
        sys.exit(1) 