"""
电力消耗预测/生成模型的配置文件
"""

# 数据配置
DATA_CONFIG = {
    'data_path': '/data2/lians/energy_new/0528/merged_shanghai_data_full_batched_updated_0528_renamed.csv',  # 修改为上海数据文件路径
    'test_size': 0.2,                     # 测试集比例
    'time_steps': 7,                      # 时间窗口大小
    'sequence_length': 365,               # 用于聚类的序列长度 (一年)
    'use_cache': True,                    # 启用数据缓存
    'cache_dir': './data_cache',          # 数据缓存目录
}

# 模型配置
MODEL_CONFIG = {
    'hidden_dim': 128,                    # 隐藏层维度
    'flow_input_dim': 1,                  # 流模型输入维度（用电量）
    'flow_hidden_dim': 64,                # 流模型隐藏层维度
    'num_flows': 5,                       # 归一化流层数
    'cluster_hidden_dim': 128,            # 聚类编码器隐藏层维度
    'n_clusters': 4,                      # 聚类数量
}

# 训练配置
TRAIN_CONFIG = {
    'batch_size': 64,                     # 批次大小
    'num_epochs': 100,                    # 训练轮数
    'g_lr': 1e-4,                         # 生成器学习率
    'd_lr': 1e-4,                         # 判别器学习率
    'c_lr': 1e-4,                         # 聚类编码器学习率
    
    # 损失权重
    'mse_weight': 1.0,                    # 重建损失权重
    'flow_weight': 0.1,                   # 流模型损失权重
    'adv_weight': 0.5,                    # 对抗损失权重
    'cluster_weight_initial': 0.0,        # 聚类损失初始权重
    'cluster_weight_max': 1.0,            # 聚类损失最大权重
    'mode_ratio_weight': 0.5,             # 模式比例损失权重
    'cluster_start_epoch': 10,            # 开始使用聚类损失的轮次
    
    # 其他训练参数
    'test_interval': 5,                   # 测试间隔
    'vis_interval': 10,                   # 可视化间隔
    'save_interval': 20,                  # 保存模型间隔
    'checkpoint_dir': './checkpoints',    # 检查点保存目录
    'vis_dir': './visualizations',        # 可视化结果保存目录
    
    # 内存管理参数
    'auto_batch_size': True,              # 是否自动调整批大小
    'num_workers': 4,                     # 数据加载器工作线程数
    'pin_memory': True,                   # 是否使用pin_memory加速
}

# 邻接矩阵构造参数
GRAPH_CONFIG = {
    'sigma': 0.1,                         # 高斯核参数
    'epsilon': 0.5,                       # 邻域阈值
}

# 生成配置
GENERATE_CONFIG = {
    'output_dir': './results',            # 输出目录
}

# 合并所有配置
CONFIG = {
    **DATA_CONFIG,
    **MODEL_CONFIG,
    **TRAIN_CONFIG,
    **GRAPH_CONFIG,
    **GENERATE_CONFIG
} 