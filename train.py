import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import logging
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

from energy_prediction_model import Generator, Discriminator, ClusterEncoder
from energy_prediction_model import sliced_wasserstein_distance, mode_ratio_loss
from data_utils import preprocess_data, prepare_tsne_data

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 训练函数
def train(config):
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"使用设备: {device}")
    
    # 错误计数和学习率退火
    consecutive_errors = 0
    error_threshold = 5  # 连续错误阈值
    original_lr = {
        'g_lr': config['g_lr'],
        'd_lr': config['d_lr'],
        'c_lr': config['c_lr']
    }
    
    # 检查GPU内存
    if torch.cuda.is_available():
        free_memory = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()
        logging.info(f"GPU总内存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        logging.info(f"当前可用GPU内存: {free_memory / 1e9:.2f} GB")
        
        # 根据可用内存动态调整批大小
        if config['auto_batch_size'] and free_memory < 40 * 1e9:  # 如果可用内存小于40GB
            original_batch_size = config['batch_size']
            # 根据内存大小缩减批大小
            adjusted_batch_size = max(1, min(original_batch_size, int(original_batch_size * free_memory / (80 * 1e9))))
            logging.info(f"由于内存限制，将批大小从 {original_batch_size} 调整到 {adjusted_batch_size}")
            config['batch_size'] = adjusted_batch_size
    
    # 数据预处理，启用缓存
    logging.info(f"开始处理数据: {config['data_path']}")
    data = preprocess_data(
        config['data_path'],
        time_steps=config['time_steps'],
        test_size=config['test_size'],
        sequence_length=config['sequence_length'],
        use_cache=config.get('use_cache', True),
        cache_dir=config.get('cache_dir', "./data_cache")
    )
    logging.info("数据处理完成")
    
    # 检查节点数，对于大规模节点进行特殊处理
    num_users = data['num_users']
    logging.info(f"数据集中用户(节点)数量: {num_users}")
    
    # 如果节点数超过阈值，强制使用较小的批大小
    if num_users > 5000:  # 节点数阈值
        max_allowed_batch = max(1, min(config['batch_size'], 4))  # 限制最大批大小为4
        if config['batch_size'] > max_allowed_batch:
            logging.warning(f"由于节点数过多({num_users} > 5000)，强制将批大小从 {config['batch_size']} 减少到 {max_allowed_batch}")
            config['batch_size'] = max_allowed_batch
    
    logging.info(f"创建数据加载器, 批次大小: {config['batch_size']}")
    train_loader = DataLoader(
        data['train_dataset'],
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config.get('num_workers', 4),
        pin_memory=config.get('pin_memory', True)
    )
    
    test_loader = DataLoader(
        data['test_dataset'],
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config.get('num_workers', 4),
        pin_memory=config.get('pin_memory', True)
    )
    logging.info(f"训练批次数: {len(train_loader)}, 测试批次数: {len(test_loader)}")
    
    # 在创建大模型前清理缓存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logging.info(f"清理GPU缓存后可用内存: {(torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()) / 1e9:.2f} GB")
    
    # 创建模型
    logging.info("创建模型...")
    
    logging.info("初始化生成器...")
    generator = Generator(
        input_dim=data['feature_dim'],
        hidden_dim=config['hidden_dim'],
        flow_input_dim=config['flow_input_dim'],
        flow_hidden_dim=config['flow_hidden_dim'],
        num_nodes=data['num_users'],
        time_steps=config['time_steps'],
        num_flows=config['num_flows']
    ).to(device)
    
    logging.info("初始化判别器...")
    discriminator = Discriminator(
        input_dim=1,  # 用电量
        hidden_dim=config['hidden_dim'],
        num_nodes=data['num_users'],
        time_steps=config['time_steps']
    ).to(device)
    
    logging.info("初始化聚类编码器...")
    cluster_encoder = ClusterEncoder(
        input_dim=config['sequence_length'],
        hidden_dim=config['cluster_hidden_dim'],
        output_dim=2  # 降到2维
    ).to(device)
    
    # 优化器
    logging.info("设置优化器...")
    g_optimizer = optim.Adam(generator.parameters(), lr=config['g_lr'])
    d_optimizer = optim.Adam(discriminator.parameters(), lr=config['d_lr'])
    c_optimizer = optim.Adam(cluster_encoder.parameters(), lr=config['c_lr'])
    
    # 损失函数
    mse_loss = nn.MSELoss()
    bce_loss = nn.BCELoss()
    
    # 准备聚类数据
    logging.info("准备聚类数据...")
    cluster_data_real = torch.FloatTensor(data['cluster_data']).to(device)
    logging.info(f"聚类数据形状: {cluster_data_real.shape}")
    
    # 设置工具函数
    def safe_check_tensor(tensor, name, min_val=0.0, max_val=1.0):
        """检查并修复张量中的无效值"""
        has_nan = torch.isnan(tensor).any()
        has_inf = torch.isinf(tensor).any()
        out_of_bounds = (tensor < min_val).any() or (tensor > max_val).any()
        
        if has_nan or has_inf or out_of_bounds:
            logging.warning(f"{name} 含有无效值: NaN={has_nan}, Inf={has_inf}, 越界={out_of_bounds}")
            # 替换无效值
            tensor = torch.where(torch.isnan(tensor), torch.zeros_like(tensor) + 0.5, tensor)
            tensor = torch.where(torch.isinf(tensor), torch.zeros_like(tensor) + 0.5, tensor)
            tensor = torch.clamp(tensor, min_val, max_val)
        
        return tensor
    
    # 训练循环
    logging.info(f"开始训练，共 {config['num_epochs']} 轮...")
    for epoch in range(config['num_epochs']):
        start_time = time.time()
        
        # 动态调整聚类损失权重 (课程学习)
        cluster_weight = min(config['cluster_weight_max'], 
                             config['cluster_weight_initial'] + 
                             epoch * (config['cluster_weight_max'] - config['cluster_weight_initial']) / (config['num_epochs'] * 0.8))
        
        generator.train()
        discriminator.train()
        cluster_encoder.train()
        
        epoch_g_loss = 0.0
        epoch_d_loss = 0.0
        epoch_mse_loss = 0.0
        epoch_flow_loss = 0.0
        epoch_cluster_loss = 0.0
        
        logging.info(f"Epoch {epoch+1}/{config['num_epochs']} - 开始批次训练")
        
        # 使用tqdm显示进度条
        train_iter = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['num_epochs']}")
        
        for batch_idx, batch in enumerate(train_iter):
            if batch_idx % 50 == 0:
                logging.info(f"处理批次 {batch_idx+1}/{len(train_loader)}")
                
            features = batch['features'].to(device)  # [batch_size, time_steps, num_nodes, feature_dim]
            targets = batch['target'].to(device)  # [batch_size, num_nodes]
            coords = batch['coordinates'].to(device)  # [batch_size, num_nodes, 2]
            
            batch_size = features.size(0)
            
            # ----------------------
            # 训练判别器
            # ----------------------
            d_optimizer.zero_grad()
            
            # 真实数据
            real_targets = targets.unsqueeze(-1)  # [batch_size, num_nodes, 1]
            # 确保输入数据在合适范围内
            real_targets = torch.sigmoid(real_targets)  # 映射到(0,1)范围
            real_sequence = torch.cat([real_targets.unsqueeze(1).repeat(1, config['time_steps'], 1, 1)], dim=-1)
            
            # 生成数据
            with torch.no_grad():
                fake_targets = generator(features, coords, reverse=True)  # [batch_size, num_nodes, 1]
                # 确保生成的数据也在合适范围内
                fake_targets = torch.sigmoid(fake_targets)  # 映射到(0,1)范围
            
            fake_sequence = torch.cat([fake_targets.unsqueeze(1).repeat(1, config['time_steps'], 1, 1)], dim=-1)
            
            # 判别器输出
            d_real = discriminator(real_sequence, coords)
            d_fake = discriminator(fake_sequence, coords)
            
            # 安全检查并修复
            d_real = safe_check_tensor(d_real, "判别器真实输出")
            d_fake = safe_check_tensor(d_fake, "判别器虚假输出")
            
            # 判别器损失
            d_loss_real = bce_loss(d_real, torch.ones_like(d_real))
            d_loss_fake = bce_loss(d_fake, torch.zeros_like(d_fake))
            d_loss = d_loss_real + d_loss_fake
            
            # 检查损失是否为NaN
            if torch.isnan(d_loss).any():
                if batch_idx % 20 == 0:  # 只每20批次显示一次警告
                    logging.warning(f"检测到NaN损失值! d_loss_real={d_loss_real.item():.4f}, d_loss_fake={d_loss_fake.item():.4f}")
                # 使用小的有效损失而非跳过批次
                d_loss = torch.tensor(0.1, device=device, requires_grad=True)
            
            d_loss.backward()
            
            # 梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=1.0)
            
            d_optimizer.step()
            
            # ----------------------
            # 训练生成器
            # ----------------------
            g_optimizer.zero_grad()
            c_optimizer.zero_grad()
            
            # 前向传播 (数据 -> 噪声)
            z, log_det = generator(features, coords, reverse=False)
            
            # 反向传播 (噪声 -> 数据)
            fake_targets = generator(features, coords, z=z, reverse=True)
            
            # 确保生成的数据在合适范围内
            fake_targets_sigmoid = torch.sigmoid(fake_targets)
            
            # 计算重建损失 - 对比归一化后的值
            real_targets_sigmoid = torch.sigmoid(targets.unsqueeze(-1))
            mse = mse_loss(fake_targets_sigmoid, real_targets_sigmoid)
            
            # 流模型的负对数似然损失
            flow_loss = -torch.mean(log_det - 0.5 * torch.sum(z ** 2, dim=2))
            
            # 对抗损失
            fake_sequence = torch.cat([fake_targets_sigmoid.unsqueeze(1).repeat(1, config['time_steps'], 1, 1)], dim=-1)
            d_fake = discriminator(fake_sequence, coords)
            
            # 安全检查并修复
            d_fake = safe_check_tensor(d_fake, "生成器对抗损失输入")
            adv_loss = bce_loss(d_fake, torch.ones_like(d_fake))
            
            # 计算聚类损失
            if epoch >= config['cluster_start_epoch']:
                try:
                    # 减少内存使用：只在部分批次计算聚类损失
                    if batch_idx % 5 == 0:  # 每5个批次计算一次
                        # 使用生成器生成一批数据 (不是当前批次的重建)
                        random_indices = torch.randperm(min(len(data['train_dataset']), 1000))[:min(config['batch_size'], 32)]  # 限制样本数量
                        random_samples = [data['train_dataset'][i] for i in random_indices]
                        random_features = torch.stack([sample['features'] for sample in random_samples]).to(device)
                        random_coords = torch.stack([sample['coordinates'] for sample in random_samples]).to(device)
                        
                        # 生成完整序列的数据
                        logging.debug("生成用于聚类的序列数据...")
                        gen_sequence = []
                        for t in range(0, config['sequence_length'], config['time_steps']):
                            end_t = min(t + config['time_steps'], config['sequence_length'])
                            steps = end_t - t
                            
                            if steps < config['time_steps']:
                                # 如果不足一个完整的时间窗口，使用最后一个窗口
                                curr_features = random_features[:, -steps:, :, :]
                            else:
                                curr_features = random_features
                            
                            with torch.no_grad():
                                gen_targets = generator(curr_features, random_coords, reverse=True)
                                # 应用Sigmoid确保值在合理范围
                                gen_targets = torch.sigmoid(gen_targets)
                                gen_sequence.append(gen_targets)
                        
                        # 拼接生成的序列
                        gen_sequence = torch.cat(gen_sequence, dim=2) if len(gen_sequence) > 1 else gen_sequence[0]
                        
                        # 使用聚类编码器降维
                        logging.debug("使用聚类编码器降维...")
                        # 仅使用一部分数据进行聚类计算，减少内存使用
                        sample_size = min(cluster_data_real.shape[0], 2000)  # 最多使用2000个样本
                        indices = torch.randperm(cluster_data_real.shape[0])[:sample_size]
                        sampled_real = cluster_data_real[indices]
                        encoded_real = cluster_encoder(sampled_real)
                        
                        # 同样对生成数据进行采样
                        indices = torch.randperm(gen_sequence.shape[0])[:min(gen_sequence.shape[0], sample_size)]
                        sampled_gen = gen_sequence[indices]
                        encoded_gen = cluster_encoder(sampled_gen)
                        
                        # 安全检查编码结果
                        has_issues_real = torch.isnan(encoded_real).any() or torch.isinf(encoded_real).any() or \
                                        (encoded_real < -10.0).any() or (encoded_real > 10.0).any()
                        has_issues_gen = torch.isnan(encoded_gen).any() or torch.isinf(encoded_gen).any() or \
                                      (encoded_gen < -10.0).any() or (encoded_gen > 10.0).any()
                        
                        if has_issues_real or has_issues_gen:
                            # 检测到问题时，使用默认值而不是每次都警告
                            if batch_idx % 20 == 0:  # 只打印部分警告，避免日志过多
                                logging.warning(f"编码结果存在问题，使用默认聚类损失。批次 {batch_idx}")
                            # 使用小的默认损失值
                            cluster_loss = torch.tensor(0.1, device=device, requires_grad=True)
                        else:
                            # 正常情况下计算损失
                            encoded_real = safe_check_tensor(encoded_real, "聚类真实编码", min_val=-10.0, max_val=10.0)
                            encoded_gen = safe_check_tensor(encoded_gen, "聚类生成编码", min_val=-10.0, max_val=10.0)
                            
                            # 计算Sliced Wasserstein Distance
                            logging.debug("计算Sliced Wasserstein Distance...")
                            swd_loss = sliced_wasserstein_distance(encoded_real, encoded_gen, 
                                                                    num_projections=20,  # 减少投影数量
                                                                    device=device)
                            
                            # 计算Mode比例损失
                            logging.debug("计算Mode比例损失...")
                            mr_loss = mode_ratio_loss(encoded_real, encoded_gen, 
                                                    n_clusters=config['n_clusters'], 
                                                    device=device)
                            
                            # 聚类总损失
                            cluster_loss = swd_loss + config['mode_ratio_weight'] * mr_loss
                    else:
                        # 对于大多数批次，使用上一次计算的聚类损失
                        cluster_loss = epoch_cluster_loss / max(1, batch_idx)
                except Exception as e:
                    logging.warning(f"计算聚类损失出错: {str(e)}")
                    # 使用较小的默认损失值，避免跳过过多批次
                    cluster_loss = torch.tensor(0.1, device=device, requires_grad=True)
            else:
                cluster_loss = torch.tensor(0.0, device=device)
            
            # 总损失
            g_loss = (
                config['mse_weight'] * mse + 
                config['flow_weight'] * flow_loss + 
                config['adv_weight'] * adv_loss + 
                cluster_weight * cluster_loss
            )
            
            # 检查损失是否为NaN
            if torch.isnan(g_loss).any() or torch.isinf(g_loss).any():
                if batch_idx % 20 == 0:  # 只每20批次显示一次警告
                    logging.warning(f"检测到无效的生成器损失值! mse={mse.item():.4f}, flow_loss={flow_loss.item():.4f}, adv_loss={adv_loss.item():.4f}, cluster_loss={cluster_loss.item():.4f}")
                # 尝试使用较小的有效损失值而非跳过批次
                g_loss = torch.tensor(0.1, device=device, requires_grad=True)
                consecutive_errors += 1
                
                # 如果连续错误过多，降低学习率
                if consecutive_errors >= error_threshold:
                    for param_group in g_optimizer.param_groups:
                        param_group['lr'] *= 0.5
                    for param_group in d_optimizer.param_groups:
                        param_group['lr'] *= 0.5
                    logging.warning(f"连续检测到 {consecutive_errors} 个错误，将学习率降低为 g_lr={g_optimizer.param_groups[0]['lr']:.6f}, d_lr={d_optimizer.param_groups[0]['lr']:.6f}")
                    consecutive_errors = 0  # 重置计数器
            else:
                consecutive_errors = 0  # 如果没有错误，重置计数器
            
            # 更新梯度
            g_loss.backward()
            
            # 梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(cluster_encoder.parameters(), max_norm=1.0)
            
            # 优化器步进
            g_optimizer.step()
            c_optimizer.step()
            
            # 记录损失
            epoch_g_loss += g_loss.item()
            epoch_d_loss += d_loss.item()
            epoch_mse_loss += mse.item()
            epoch_flow_loss += flow_loss.item()
            epoch_cluster_loss += cluster_loss.item() if epoch >= config['cluster_start_epoch'] else 0
            
            # 更新进度条
            train_iter.set_postfix({
                'g_loss': g_loss.item(),
                'd_loss': d_loss.item(),
                'mse': mse.item()
            })
            
            # 清理缓存
            if batch_idx % 10 == 0:
                torch.cuda.empty_cache()
        
        # 计算平均损失
        epoch_g_loss /= len(train_loader)
        epoch_d_loss /= len(train_loader)
        epoch_mse_loss /= len(train_loader)
        epoch_flow_loss /= len(train_loader)
        epoch_cluster_loss /= max(1, len(train_loader) * (1 if epoch >= config['cluster_start_epoch'] else 0))
        
        # 测试
        if (epoch + 1) % config['test_interval'] == 0:
            logging.info("开始评估模型...")
            test_loss = evaluate(
                generator, 
                test_loader, 
                device, 
                use_cache=config.get('use_cache', True),
                cache_dir=config.get('cache_dir', "./data_cache")
            )
            
            # 可视化
            if (epoch + 1) % config['vis_interval'] == 0:
                logging.info("开始可视化聚类结果...")
                visualize_clusters(generator, cluster_encoder, data, config, epoch, device)
        
        # 清理缓存
        torch.cuda.empty_cache()
        
        # 打印统计信息
        time_per_epoch = time.time() - start_time
        logging.info(f"Epoch {epoch+1}/{config['num_epochs']} 完成，耗时 {time_per_epoch:.2f}秒")
        logging.info(f"G Loss: {epoch_g_loss:.4f}, D Loss: {epoch_d_loss:.4f}, MSE: {epoch_mse_loss:.4f}")
        logging.info(f"Flow Loss: {epoch_flow_loss:.4f}, Cluster Loss: {epoch_cluster_loss:.4f}")
        
        # 每10个epoch尝试恢复学习率到原始值的一半(如果之前被降低过)
        if (epoch + 1) % 10 == 0:
            current_g_lr = g_optimizer.param_groups[0]['lr']
            if current_g_lr < original_lr['g_lr']:
                # 恢复学习率到原始值和当前值的中间
                new_g_lr = min(original_lr['g_lr'], current_g_lr * 2)
                new_d_lr = min(original_lr['d_lr'], d_optimizer.param_groups[0]['lr'] * 2)
                
                for param_group in g_optimizer.param_groups:
                    param_group['lr'] = new_g_lr
                for param_group in d_optimizer.param_groups:
                    param_group['lr'] = new_d_lr
                
                logging.info(f"恢复学习率: g_lr={new_g_lr:.6f}, d_lr={new_d_lr:.6f}")
        
        # 保存模型
        if (epoch + 1) % config['save_interval'] == 0:
            logging.info(f"保存检查点: Epoch {epoch+1}")
            save_checkpoint({
                'epoch': epoch + 1,
                'generator_state_dict': generator.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'cluster_encoder_state_dict': cluster_encoder.state_dict(),
                'g_optimizer_state_dict': g_optimizer.state_dict(),
                'd_optimizer_state_dict': d_optimizer.state_dict(),
                'c_optimizer_state_dict': c_optimizer.state_dict(),
            }, config['checkpoint_dir'], f"checkpoint_epoch_{epoch+1}.pth")
    
    logging.info("训练完成")
    return generator, discriminator, cluster_encoder

# 评估函数
def evaluate(model, data_loader, device, use_cache=True, cache_dir="./data_cache"):
    model.eval()
    total_loss = 0.0
    criterion = nn.MSELoss()
    
    logging.info("开始评估...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(data_loader, desc="Evaluating")):
            if batch_idx % 20 == 0:
                logging.info(f"评估批次 {batch_idx+1}/{len(data_loader)}")
                
            features = batch['features'].to(device)
            targets = batch['target'].to(device)
            coords = batch['coordinates'].to(device)
            
            # 生成预测
            predictions = model(features, coords, reverse=True)
            
            # 确保预测和目标数据在同一范围内
            predictions_sigmoid = torch.sigmoid(predictions)
            targets_sigmoid = torch.sigmoid(targets.unsqueeze(-1))
            
            # 计算损失
            loss = criterion(predictions_sigmoid, targets_sigmoid)
            total_loss += loss.item()
    
    avg_loss = total_loss / len(data_loader)
    logging.info(f"测试损失: {avg_loss:.4f}")
    
    return avg_loss

# 可视化聚类结果
def visualize_clusters(generator, encoder, data, config, epoch, device):
    logging.info("开始可视化聚类...")
    generator.eval()
    encoder.eval()
    
    with torch.no_grad():
        # 真实数据
        logging.info("准备真实数据...")
        real_data = torch.FloatTensor(data['cluster_data']).to(device)
        
        # 为减少内存使用，限制样本数量
        sample_size = min(real_data.shape[0], 2000)  # 最多使用2000个样本
        if sample_size < real_data.shape[0]:
            logging.info(f"对真实数据进行采样, 从 {real_data.shape[0]} 减少到 {sample_size}")
            indices = torch.randperm(real_data.shape[0])[:sample_size]
            real_data = real_data[indices]
        
        encoded_real = encoder(real_data).cpu().numpy()
        logging.info(f"真实数据编码完成，形状: {encoded_real.shape}")
        
        # 生成数据
        # 使用训练集中的条件特征生成数据
        logging.info("准备生成数据...")
        features_list = []
        coords_list = []
        
        # 从数据集中收集特征
        logging.info("收集特征数据...")
        # 限制收集的样本数量
        max_samples = min(len(data['train_dataset']), sample_size)
        sample_indices = np.random.choice(len(data['train_dataset']), max_samples, replace=False)
        
        for i in sample_indices:
            sample = data['train_dataset'][i]
            features_list.append(sample['features'])
            coords_list.append(sample['coordinates'])
        
        # 根据收集到的数据数量确定批次大小
        batch_size = min(config['batch_size'], len(features_list))
        
        # 批量处理
        logging.info(f"批量处理特征数据，批次大小: {batch_size}")
        batched_features = torch.stack(features_list[:batch_size]).to(device)
        batched_coords = torch.stack(coords_list[:batch_size]).to(device)
        
        # 生成序列数据
        logging.info("生成序列数据...")
        gen_sequences = []
        for t in range(0, config['sequence_length'], config['time_steps']):
            t_start = time.time()
            end_t = min(t + config['time_steps'], config['sequence_length'])
            steps = end_t - t
            
            gen_batch = generator(batched_features[:, :steps], batched_coords, reverse=True)
            gen_sequences.append(gen_batch)
            logging.info(f"生成时间步 {t}-{end_t} 完成，耗时 {time.time()-t_start:.2f}秒")
        
        # 合并生成的序列
        logging.info("合并生成的序列...")
        gen_data = torch.cat(gen_sequences, dim=2) if len(gen_sequences) > 1 else gen_sequences[0]
        
        # 如果生成的序列长度不等于预期序列长度，填充或截断
        if gen_data.shape[2] != config['sequence_length']:
            if gen_data.shape[2] < config['sequence_length']:
                # 填充
                logging.info(f"序列长度不足，进行填充: {gen_data.shape[2]} -> {config['sequence_length']}")
                padding = torch.zeros((gen_data.shape[0], gen_data.shape[1], 
                                      config['sequence_length'] - gen_data.shape[2]), device=device)
                gen_data = torch.cat([gen_data, padding], dim=2)
            else:
                # 截断
                logging.info(f"序列过长，进行截断: {gen_data.shape[2]} -> {config['sequence_length']}")
                gen_data = gen_data[:, :, :config['sequence_length']]
        
        # 编码生成的数据
        logging.info("编码生成的数据...")
        # Reshape gen_data before passing to the encoder
        batch_size_gen, num_nodes_gen, seq_len_gen = gen_data.shape
        gen_data_reshaped = gen_data.reshape(batch_size_gen * num_nodes_gen, seq_len_gen)
        encoded_gen = encoder(gen_data_reshaped).cpu().numpy()
        logging.info(f"生成数据编码完成，形状: {encoded_gen.shape}")
    
    # 绘制TSNE结果
    logging.info("绘制TSNE结果...")
    plt.figure(figsize=(12, 10))
    
    # 真实数据聚类
    plt.subplot(2, 1, 1)
    plt.scatter(encoded_real[:, 0], encoded_real[:, 1], c='blue', alpha=0.5, label='Real Data')
    plt.title('TSNE Clusters of Real Data')
    plt.legend()
    
    # 生成数据聚类
    plt.subplot(2, 1, 2)
    plt.scatter(encoded_gen[:, 0], encoded_gen[:, 1], c='red', alpha=0.5, label='Generated Data')
    plt.title('TSNE Clusters of Generated Data')
    plt.legend()
    
    # 保存图像
    os.makedirs(config['vis_dir'], exist_ok=True)
    plt.savefig(os.path.join(config['vis_dir'], f'clusters_epoch_{epoch+1}.png'))
    plt.close()
    logging.info(f"TSNE可视化已保存: clusters_epoch_{epoch+1}.png")
    
    # 计算KMeans聚类并比较模式比例
    logging.info(f"进行KMeans聚类分析，聚类数: {config['n_clusters']}")
    kmeans = KMeans(n_clusters=config['n_clusters'], random_state=0)
    kmeans.fit(encoded_real)
    
    real_labels = kmeans.predict(encoded_real)
    gen_labels = kmeans.predict(encoded_gen)
    
    real_counts = np.bincount(real_labels, minlength=config['n_clusters'])
    gen_counts = np.bincount(gen_labels, minlength=config['n_clusters'])
    
    real_ratio = real_counts / real_counts.sum()
    gen_ratio = gen_counts / gen_counts.sum()
    
    # 绘制模式比例
    logging.info("绘制模式比例图...")
    plt.figure(figsize=(10, 6))
    x = np.arange(config['n_clusters'])
    width = 0.35
    
    plt.bar(x - width/2, real_ratio, width, label='Real Data')
    plt.bar(x + width/2, gen_ratio, width, label='Generated Data')
    
    plt.xlabel('Cluster')
    plt.ylabel('Ratio')
    plt.title('Cluster Mode Ratio Comparison')
    plt.xticks(x)
    plt.legend()
    
    plt.savefig(os.path.join(config['vis_dir'], f'mode_ratio_epoch_{epoch+1}.png'))
    plt.close()
    logging.info(f"模式比例图已保存: mode_ratio_epoch_{epoch+1}.png")

# 保存检查点
def save_checkpoint(state, checkpoint_dir, filename):
    os.makedirs(checkpoint_dir, exist_ok=True)
    torch.save(state, os.path.join(checkpoint_dir, filename))
    logging.info(f"检查点已保存: {filename}")

# 生成新城市数据
def generate_new_city_data(generator, feature_scaler, target_scaler, features, coordinates, device):
    """
    为新城市生成电力消耗数据
    features: [time_steps, num_nodes, feature_dim] - 新城市的条件特征
    coordinates: [num_nodes, 2] - 新城市的GPS坐标
    """
    logging.info(f"开始生成新城市数据，特征形状: {features.shape}, 坐标形状: {coordinates.shape}")
    generator.eval()
    
    # 转换为张量
    features = torch.FloatTensor(features).to(device)
    coordinates = torch.FloatTensor(coordinates).to(device)
    
    # 添加批次维度
    if features.dim() == 3:
        features = features.unsqueeze(0)
        logging.info(f"添加批次维度，新形状: {features.shape}")
    if coordinates.dim() == 2:
        coordinates = coordinates.unsqueeze(0)
        logging.info(f"添加批次维度，新形状: {coordinates.shape}")
    
    with torch.no_grad():
        # 生成数据
        logging.info("使用生成器生成数据...")
        generated_data = generator(features, coordinates, reverse=True)
        
        # 转换回CPU和NumPy
        generated_data = generated_data.cpu().numpy()
        
        # 逆标准化
        logging.info("进行逆标准化...")
        generated_data_flat = generated_data.reshape(-1, 1)
        generated_data_unscaled = target_scaler.inverse_transform(generated_data_flat)
        generated_data = generated_data_unscaled.reshape(generated_data.shape)
    
    logging.info(f"数据生成完成，形状: {generated_data.shape}")
    return generated_data

if __name__ == "__main__":
    # 配置
    config = {
        'data_path': 'new_cleaned_data_0422.csv',  # 更新数据路径为新数据集
        'batch_size': 1,  # 彻底解决内存问题：使用最小批大小
        'time_steps': 7,  # 时间窗口大小
        'test_size': 0.2,
        'sequence_length': 365,  # 用于聚类的序列长度
        'hidden_dim': 64,  # 减小隐藏层维度
        'flow_input_dim': 1,  # 用电量维度
        'flow_hidden_dim': 32,  # 减小流模型隐藏层维度
        'num_flows': 3,  # 减少归一化流层数
        'cluster_hidden_dim': 64,  # 减小聚类编码器隐藏层维度
        'n_clusters': 4,  # 聚类数量
        'num_epochs': 5,  # 减少训练轮数
        'g_lr': 1e-4,
        'd_lr': 1e-4,
        'c_lr': 1e-4,
        'mse_weight': 1.0,
        'flow_weight': 0.1,
        'adv_weight': 0.5,
        'cluster_weight_initial': 0.0,
        'cluster_weight_max': 0.5,  # 减小聚类损失权重
        'mode_ratio_weight': 0.5,
        'cluster_start_epoch': 10,  # 开始使用聚类损失的轮次
        'test_interval': 5,
        'vis_interval': 5,
        'save_interval': 5,  # 更频繁地保存检查点
        'checkpoint_dir': './checkpoints',
        'vis_dir': './visualizations',
        # 添加数据缓存相关配置
        'use_cache': True,  # 启用数据缓存
        'cache_dir': './data_cache',  # 缓存目录
        'auto_batch_size': True,  # 启用自动调整批大小
        'num_workers': 4,  # 数据加载工作线程数
        'pin_memory': True,  # 使用pin_memory加速数据传输
    }
    
    # 创建输出目录
    os.makedirs(config['checkpoint_dir'], exist_ok=True)
    os.makedirs(config['vis_dir'], exist_ok=True)
    os.makedirs(config['cache_dir'], exist_ok=True)  # 确保缓存目录存在
    
    logging.info("开始训练模型...")
    
    # 训练模型
    try:
        generator, discriminator, cluster_encoder = train(config)
        logging.info("训练成功完成")
    except Exception as e:
        logging.error(f"训练过程出错: {str(e)}")
        import traceback
        logging.error(traceback.format_exc()) 