import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import logging
import time
import os
import gc
import sys
from torch.nn.utils import weight_norm
from torch.utils.checkpoint import checkpoint
from functools import wraps
from datetime import datetime

# 配置环境变量以优化PyTorch性能
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'  # 限制单个内存块大小，减少碎片

# 设置日志配置
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

# 创建当前日期的日志文件名
current_date = datetime.now().strftime("%Y-%m-%d")
log_file = os.path.join(log_dir, f"energy_model_{current_date}.log")

# 高级日志配置
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("EnergyModel")

# 配置性能检测级别
PERFORMANCE_DEBUG = False  # 设置为True启用详细性能日志

if PERFORMANCE_DEBUG:
    logger.setLevel(logging.DEBUG)
    logger.info("启用性能调试模式")
else:
    logger.setLevel(logging.INFO)
    
# 检测CUDA可用性并输出设备信息
if torch.cuda.is_available():
    device_count = torch.cuda.device_count()
    current_device = torch.cuda.current_device()
    device_name = torch.cuda.get_device_name(current_device)
    logger.info(f"CUDA可用，设备数量: {device_count}，当前设备: {current_device} ({device_name})")
    
    # 获取设备属性
    device_props = torch.cuda.get_device_properties(current_device)
    logger.info(f"GPU内存: {device_props.total_memory / 1e9:.2f} GB")
    logger.info(f"CUDA能力: {device_props.major}.{device_props.minor}")
    
    # 设置PyTorch使用确定性算法（可能会降低性能）
    # torch.backends.cudnn.deterministic = True  # 启用确定性计算
    torch.backends.cudnn.benchmark = True  # 启用cudnn自动调优
else:
    logger.warning("CUDA不可用，将使用CPU进行计算，性能可能受限")

# 高级性能监控装饰器
def timing_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        # 只在性能调试模式下记录详细信息
        if PERFORMANCE_DEBUG:
            # 记录起始内存状态
            if torch.cuda.is_available():
                torch.cuda.synchronize()  # 确保之前的CUDA操作已完成
                mem_before = torch.cuda.memory_allocated() / 1e9
                
            start_time = time.time()
            
            # 添加NVTX标记用于nvprof/nsight性能分析
            if torch.cuda.is_available() and hasattr(torch.cuda, 'nvtx'):
                torch.cuda.nvtx.range_push(f"{func.__name__}")
                
            # 执行函数
            result = func(*args, **kwargs)
            
            # 结束NVTX标记
            if torch.cuda.is_available() and hasattr(torch.cuda, 'nvtx'):
                torch.cuda.nvtx.range_pop()
                
            # 计算执行时间
            end_time = time.time()
            execution_time = end_time - start_time
            
            # 记录内存使用
            if torch.cuda.is_available():
                torch.cuda.synchronize()  # 确保CUDA操作完成
                mem_after = torch.cuda.memory_allocated() / 1e9
                mem_diff = mem_after - mem_before
                
                # 详细记录
                logger.debug(f"{func.__name__} 执行时间: {execution_time:.4f}秒, " 
                           f"内存变化: {mem_diff:.4f} GB, 当前内存: {mem_after:.4f} GB")
            else:
                logger.debug(f"{func.__name__} 执行时间: {execution_time:.4f}秒")
        else:
            # 简单执行模式
            result = func(*args, **kwargs)
            
        return result
    return wrapper

# 主动内存管理函数
def manage_memory():
    """主动释放未使用的内存并减少碎片"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        # 记录当前内存状态
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        logger.debug(f"内存管理: 已分配 {allocated:.2f} GB, 已保留 {reserved:.2f} GB")

# 图卷积模块
class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        logger.debug(f"初始化GraphConvolution: in_features={in_features}, out_features={out_features}")

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    @timing_decorator
    def forward(self, input, adj):
        """
        input: [batch_size, num_nodes, in_features]
        adj: [batch_size, num_nodes, num_nodes] - 邻接矩阵
        """
        batch_size, num_nodes, _ = input.size()
        
        # 转换形状以进行批量图卷积
        x = input.transpose(1, 2).contiguous()  # [batch_size, in_features, num_nodes]
        x = torch.bmm(x, adj)  # [batch_size, in_features, num_nodes]
        x = x.transpose(1, 2).contiguous()  # [batch_size, num_nodes, in_features]
        
        output = torch.matmul(x, self.weight)
        
        if self.bias is not None:
            return output + self.bias
        else:
            return output

# 时间卷积模块
class TemporalConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1):
        super(TemporalConvBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(in_channels, out_channels, kernel_size, 
                                          padding=dilation*(kernel_size-1)//2, dilation=dilation))
        self.conv2 = weight_norm(nn.Conv1d(out_channels, out_channels, kernel_size, 
                                          padding=dilation*(kernel_size-1)//2, dilation=dilation))
        self.net = nn.Sequential(
            self.conv1,
            nn.ReLU(),
            self.conv2,
            nn.ReLU()
        )
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.relu = nn.ReLU()
        self.init_weights()
        logger.debug(f"初始化TemporalConvBlock: in_channels={in_channels}, out_channels={out_channels}")
        
    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    @timing_decorator
    def forward(self, x):
        """
        x: [batch_size, time_steps, num_nodes, in_channels]
        """
        batch_size, time_steps, num_nodes, in_channels = x.size()
        
        # 安全检查：确保没有NaN值
        if torch.isnan(x).any():
            nan_count = torch.isnan(x).sum().item()
            nan_percentage = 100.0 * nan_count / x.numel()
            logger.warning(f"TemporalConvBlock输入包含{nan_count}个NaN值 ({nan_percentage:.2f}%)")
            
            # 检查每个特征维度的NaN值
            for i in range(in_channels):
                dim_nan_count = torch.isnan(x[:,:,:,i]).sum().item()
                if dim_nan_count > 0:
                    dim_percentage = 100.0 * dim_nan_count / (batch_size * time_steps * num_nodes)
                    logger.warning(f"  特征维度[{i}]有{dim_nan_count}个NaN值 ({dim_percentage:.2f}%)")
                    # 找出第一个NaN值的位置
                    nan_indices = torch.where(torch.isnan(x[:,:,:,i]))
                    if len(nan_indices[0]) > 0:
                        first_idx = (nan_indices[0][0].item(), nan_indices[1][0].item(), nan_indices[2][0].item())
                        logger.warning(f"  第一个NaN值位置: batch={first_idx[0]}, time={first_idx[1]}, node={first_idx[2]}")
            
            x = torch.nan_to_num(x, nan=0.0)
        
        # 调整维度用于时间卷积
        x_temp = x.permute(0, 3, 1, 2).contiguous()  # [batch, in_channels, time_steps, num_nodes]
        x_temp = x_temp.view(batch_size*num_nodes, in_channels, time_steps)  # [batch*num_nodes, in_channels, time_steps]
        
        # 使用checkpoint减少内存使用
        if self.training and x_temp.requires_grad and x_temp.numel() > 1e6:
            out = checkpoint(self.net, x_temp, use_reentrant=False)
        else:
            out = self.net(x_temp)
        
        # 检查卷积输出是否有NaN值
        if torch.isnan(out).any():
            nan_count = torch.isnan(out).sum().item()
            nan_percentage = 100.0 * nan_count / out.numel()
            logger.warning(f"时间卷积网络输出包含{nan_count}个NaN值 ({nan_percentage:.2f}%)")
            out = torch.nan_to_num(out, nan=0.0)
        
        # 残差连接
        if self.downsample is not None:
            residual = self.downsample(x_temp)
            if torch.isnan(residual).any():
                nan_count = torch.isnan(residual).sum().item()
                nan_percentage = 100.0 * nan_count / residual.numel()
                logger.warning(f"下采样残差连接包含{nan_count}个NaN值 ({nan_percentage:.2f}%)")
                residual = torch.nan_to_num(residual, nan=0.0)
            out = out + residual
        else:
            out = out + x_temp
            
        # 调整回原始形状
        out = out.view(batch_size, num_nodes, -1, time_steps).permute(0, 3, 1, 2)  # [batch, time_steps, num_nodes, out_channels]
        
        result = self.relu(out)
        
        # 最终检查
        if torch.isnan(result).any():
            nan_count = torch.isnan(result).sum().item()
            nan_percentage = 100.0 * nan_count / result.numel()
            logger.warning(f"TemporalConvBlock最终输出包含{nan_count}个NaN值 ({nan_percentage:.2f}%)")
            result = torch.nan_to_num(result, nan=0.0)
            
        return result

# STGCN模块
class STGCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, spatial_channels, time_steps, num_nodes):
        super(STGCNBlock, self).__init__()
        self.temporal1 = TemporalConvBlock(in_channels, out_channels)
        self.spatial = GraphConvolution(out_channels, spatial_channels)
        self.temporal2 = TemporalConvBlock(spatial_channels, out_channels)
        self.batch_norm = nn.BatchNorm2d(num_nodes)
        logger.debug(f"初始化STGCNBlock: in_channels={in_channels}, out_channels={out_channels}, spatial_channels={spatial_channels}")
        
    @timing_decorator
    def forward(self, x, adj):
        """
        x: [batch_size, time_steps, num_nodes, in_channels]
        adj: [batch_size, num_nodes, num_nodes]
        """
        # 记录开始时间
        start_time = time.time()
        
        # 使用CUDA性能分析标记（如果可用）
        if torch.cuda.is_available() and hasattr(torch.cuda, 'nvtx'):
            torch.cuda.nvtx.range_push("STGCNBlock.forward")
            
        # 安全检查：确保没有NaN值
        if torch.isnan(x).any():
            logger.warning("STGCN输入包含NaN值，进行修正")
            x = torch.nan_to_num(x, nan=0.0)
            
        if torch.isnan(adj).any():
            logger.warning("邻接矩阵包含NaN值，进行修正")
            adj = torch.nan_to_num(adj, nan=0.0)
            
        # 时间卷积
        temporal1_start = time.time()
        # 如果输入非常大，使用checkpointing技术节省内存
        if self.training and x.requires_grad and x.numel() > 10000000:  # 约10M元素
            logger.debug("第一次时间卷积使用checkpointing")
            temporal1_out = checkpoint(self.temporal1, x, use_reentrant=False)
        else:
            temporal1_out = self.temporal1(x)
            
        logger.debug(f"第一次时间卷积耗时: {time.time() - temporal1_start:.4f}秒")
        batch_size, time_steps, num_nodes, out_channels = temporal1_out.size()
        
        # 检查时间卷积输出是否有NaN值
        if torch.isnan(temporal1_out).any():
            logger.warning("时间卷积输出包含NaN值，进行修正")
            temporal1_out = torch.nan_to_num(temporal1_out, nan=0.0)
        
        # 图卷积 - 需要逐个时间步骤处理
        spatial_start = time.time()
        
        # 检查内存使用情况
        if torch.cuda.is_available():
            logger.debug(f"空间卷积前GPU内存: 已分配 {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        
        # 启用混合精度计算以加速（在FP16支持的GPU上）
        use_amp = torch.cuda.is_available() and hasattr(torch.cuda, 'amp') and torch.cuda.get_device_capability()[0] >= 7
        
        # 使用时间步分批处理，避免大量内存分配
        spatial_out_list = []
        
        # 计算每批处理的时间步数（自动调整）
        # 设定目标内存使用量：每个节点每个特征消耗4字节，乘以安全系数3
        target_memory_bytes = 3e9  # 目标3GB内存使用
        if torch.cuda.is_available():
            # 基于可用GPU内存调整批大小
            free_memory = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()
            # 使用80%的可用内存
            target_memory_bytes = min(target_memory_bytes, free_memory * 0.8)
            
        # 估计每个时间步需要的内存
        bytes_per_step = batch_size * num_nodes * num_nodes * out_channels * 4  # 估计每个时间步需要的字节数
        max_steps_per_batch = max(1, min(time_steps, int(target_memory_bytes / bytes_per_step)))
        
        logger.debug(f"空间卷积每批处理的时间步数: {max_steps_per_batch}/{time_steps}")
        
        # 判断是否需要分批处理
        large_batch = time_steps > 1 and (bytes_per_step * time_steps > target_memory_bytes)
        
        if large_batch:
            # 分批处理时间步
            for t_start in range(0, time_steps, max_steps_per_batch):
                t_end = min(t_start + max_steps_per_batch, time_steps)
                steps_this_batch = t_end - t_start
                logger.debug(f"处理时间步 {t_start} 到 {t_end-1}, 共 {steps_this_batch} 步")
                
                # 仅处理当前批次的时间步
                current_temporal = temporal1_out[:, t_start:t_end].reshape(batch_size * steps_this_batch, num_nodes, out_channels)
                
                # 为当前批次的时间步复制邻接矩阵
                current_adj = adj.repeat(steps_this_batch, 1, 1)
                
                # 判断是否使用混合精度
                if use_amp:
                    with torch.cuda.amp.autocast():
                        # 应用图卷积
                        current_spatial = self.spatial(current_temporal, current_adj)
                else:
                    # 标准精度计算
                    current_spatial = self.spatial(current_temporal, current_adj)
                
                # 检查图卷积输出是否有NaN值
                if torch.isnan(current_spatial).any():
                    nan_count = torch.isnan(current_spatial).sum().item()
                    nan_percentage = 100.0 * nan_count / current_spatial.numel()
                    logger.warning(f"图卷积输出（批次{t_start}/{t_end-1}）包含{nan_count}个NaN值 ({nan_percentage:.2f}%)")
                    current_spatial = torch.nan_to_num(current_spatial, nan=0.0)
                
                # 调整形状并存储结果
                current_spatial = current_spatial.view(batch_size, steps_this_batch, num_nodes, -1)
                spatial_out_list.append(current_spatial)
                
                # 手动释放不再需要的张量
                del current_temporal, current_adj, current_spatial
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        else:
            # 一次性处理所有时间步
            # 将邻接矩阵复制到每个时间步
            batch_temporal = temporal1_out.reshape(batch_size * time_steps, num_nodes, out_channels)
            batch_adj = adj.repeat(time_steps, 1, 1)
            
            # 判断是否使用混合精度
            if use_amp:
                with torch.cuda.amp.autocast():
                    spatial_out = self.spatial(batch_temporal, batch_adj)
            else:
                spatial_out = self.spatial(batch_temporal, batch_adj)
            
            # 检查图卷积输出是否有NaN值
            if torch.isnan(spatial_out).any():
                nan_count = torch.isnan(spatial_out).sum().item()
                nan_percentage = 100.0 * nan_count / spatial_out.numel()
                logger.warning(f"一次性图卷积输出包含{nan_count}个NaN值 ({nan_percentage:.2f}%)")
                
                # 检查每个批次的NaN值分布
                for b in range(batch_size):
                    batch_nan = torch.isnan(spatial_out.view(batch_size, time_steps, num_nodes, -1)[b]).sum().item()
                    if batch_nan > 0:
                        batch_percentage = 100.0 * batch_nan / (time_steps * num_nodes * spatial_out.size(-1))
                        logger.warning(f"  批次[{b}]有{batch_nan}个NaN值 ({batch_percentage:.2f}%)")
                
                spatial_out = torch.nan_to_num(spatial_out, nan=0.0)
                
            # 重塑为正确的维度
            spatial_out = spatial_out.view(batch_size, time_steps, num_nodes, -1)
            spatial_out_list = [spatial_out]
            
            # 清理内存
            del batch_temporal, batch_adj
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # 合并所有批次的结果
        spatial_out = torch.cat(spatial_out_list, dim=1) if len(spatial_out_list) > 1 else spatial_out_list[0]
        logger.debug(f"空间卷积耗时: {time.time() - spatial_start:.4f}秒")
        
        # 释放不再需要的内存
        del temporal1_out, spatial_out_list
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # 批归一化
        bn_start = time.time()
        spatial_out = spatial_out.permute(0, 2, 1, 3).contiguous()  # [batch_size, num_nodes, time_steps, channels]
        spatial_out = self.batch_norm(spatial_out)
        spatial_out = spatial_out.permute(0, 2, 1, 3).contiguous()  # [batch_size, time_steps, num_nodes, channels]
        logger.debug(f"批归一化耗时: {time.time() - bn_start:.4f}秒")
        
        # 检查批归一化输出是否有NaN值
        if torch.isnan(spatial_out).any():
            nan_count = torch.isnan(spatial_out).sum().item()
            nan_percentage = 100.0 * nan_count / spatial_out.numel()
            logger.warning(f"批归一化输出包含{nan_count}个NaN值 ({nan_percentage:.2f}%)")
            
            # 检查每个通道的NaN值分布
            num_channels = spatial_out.size(-1)
            for c in range(num_channels):
                channel_nan = torch.isnan(spatial_out[:,:,:,c]).sum().item()
                if channel_nan > 0:
                    channel_percentage = 100.0 * channel_nan / (batch_size * time_steps * num_nodes)
                    logger.warning(f"  通道[{c}]有{channel_nan}个NaN值 ({channel_percentage:.2f}%)")
                    
            spatial_out = torch.nan_to_num(spatial_out, nan=0.0)
        
        # 第二次时间卷积
        temporal2_start = time.time()
        # 再次检查是否使用checkpointing
        if self.training and spatial_out.requires_grad and spatial_out.numel() > 10000000:
            logger.debug("第二次时间卷积使用checkpointing")
            temporal2_out = checkpoint(self.temporal2, spatial_out, use_reentrant=False)
        else:
            temporal2_out = self.temporal2(spatial_out)
            
        logger.debug(f"第二次时间卷积耗时: {time.time() - temporal2_start:.4f}秒")
        
        # 检查最终输出是否有NaN值
        if torch.isnan(temporal2_out).any():
            nan_count = torch.isnan(temporal2_out).sum().item()
            nan_percentage = 100.0 * nan_count / temporal2_out.numel()
            logger.warning(f"最终时间卷积输出包含{nan_count}个NaN值 ({nan_percentage:.2f}%)")
            
            # 检查时间维度的NaN值分布
            for t in range(time_steps):
                time_nan = torch.isnan(temporal2_out[:,t]).sum().item()
                if time_nan > 0:
                    time_percentage = 100.0 * time_nan / (batch_size * num_nodes * temporal2_out.size(-1))
                    logger.warning(f"  时间步[{t}]有{time_nan}个NaN值 ({time_percentage:.2f}%)")
            
            temporal2_out = torch.nan_to_num(temporal2_out, nan=0.0)
        
        # 总耗时
        total_time = time.time() - start_time
        logger.debug(f"STGCNBlock.forward总耗时: {total_time:.4f}秒")
        
        if torch.cuda.is_available() and hasattr(torch.cuda, 'nvtx'):
            torch.cuda.nvtx.range_pop()
        
        return temporal2_out

# 条件归一化流模型
class ConditionalNormalizingFlow(nn.Module):
    def __init__(self, input_dim, hidden_dim, cond_dim, num_flows=5):
        super(ConditionalNormalizingFlow, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.cond_dim = cond_dim
        self.num_flows = num_flows
        
        # 网络定义
        self.fc_mu = nn.ModuleList([nn.Sequential(
            nn.Linear(input_dim + cond_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        ) for _ in range(num_flows)])
        
        self.fc_sigma = nn.ModuleList([nn.Sequential(
            nn.Linear(input_dim + cond_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Softplus()
        ) for _ in range(num_flows)])
        
        logger.debug(f"初始化ConditionalNormalizingFlow: input_dim={input_dim}, hidden_dim={hidden_dim}, num_flows={num_flows}")
    
    @timing_decorator
    def forward(self, x, condition, reverse=False):
        """
        x: [batch_size, input_dim]
        condition: [batch_size, cond_dim]
        reverse: 是否反向变换 (从噪声生成数据)
        """
        if not reverse:
            # 前向变换: 数据 -> 噪声
            log_det_sum = 0
            
            for i in range(self.num_flows):
                # 拼接条件
                x_cond = torch.cat([x, condition], dim=1)
                
                # 计算仿射变换参数
                mu = self.fc_mu[i](x_cond)
                sigma = self.fc_sigma[i](x_cond) + 1e-5
                
                # 仿射变换
                z = (x - mu) / sigma
                log_det = -torch.sum(torch.log(sigma), dim=1)
                log_det_sum += log_det
                
                x = z
            
            return x, log_det_sum
        
        else:
            # 反向变换: 噪声 -> 数据
            z = x
            
            for i in range(self.num_flows-1, -1, -1):
                # 拼接条件
                z_cond = torch.cat([z, condition], dim=1)
                
                # 计算仿射变换参数
                mu = self.fc_mu[i](z_cond)
                sigma = self.fc_sigma[i](z_cond) + 1e-5
                
                # 逆仿射变换
                x = z * sigma + mu
                z = x
            
            return x, None

# 生成器模型
class Generator(nn.Module):
    def __init__(self, input_dim, hidden_dim, flow_input_dim, flow_hidden_dim, num_nodes, 
                 time_steps=7, num_flows=5):
        super(Generator, self).__init__()
        
        # STGCN部分处理空间条件
        self.stgcn1 = STGCNBlock(input_dim, 64, 64, time_steps, num_nodes)
        self.stgcn2 = STGCNBlock(64, 128, 128, time_steps, num_nodes)
        
        # 全局池化层，从空间特征获取全局条件
        self.global_pool = nn.AdaptiveAvgPool2d((1, 128))
        
        # 条件处理MLP
        self.condition_mlp = nn.Sequential(
            nn.Linear(128, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, flow_hidden_dim)
        )
        
        # 条件归一化流
        self.flow = ConditionalNormalizingFlow(flow_input_dim, flow_hidden_dim, flow_hidden_dim, num_flows)
        
        # 邻接矩阵缓存
        self.adj_cache = {}
        
        logger.info(f"初始化Generator: input_dim={input_dim}, hidden_dim={hidden_dim}, flow_input_dim={flow_input_dim}, num_nodes={num_nodes}")
        
    @timing_decorator
    def construct_adjacency(self, coordinates, sigma=0.1, epsilon=0.5):
        """
        根据GPS坐标构造邻接矩阵
        coordinates: [batch_size, num_nodes, 2] - (x, y)坐标
        """
        batch_size, num_nodes, _ = coordinates.size()
        
        # 内存优化：检查节点数量并输出警告
        if num_nodes > 5000:
            logger.warning(f"构造邻接矩阵时节点数量过多: {num_nodes}，可能导致内存不足")
            
        # 内存优化：对于大型邻接矩阵，分批处理
        if num_nodes > 2048:
            logger.info(f"节点数 {num_nodes} 过多，使用分批处理计算邻接矩阵")
            return self.construct_adjacency_chunked(coordinates, sigma, epsilon)
        
        # 计算欧氏距离 - 使用高效的矩阵运算
        # 计算距离矩阵的平方 ||x_i - x_j||^2 = ||x_i||^2 + ||x_j||^2 - 2<x_i, x_j>
        x_norm = torch.sum(coordinates ** 2, dim=2, keepdim=True)
        y_norm = x_norm.transpose(1, 2)
        xy = torch.bmm(coordinates, coordinates.transpose(1, 2))
        dist_squared = x_norm + y_norm - 2 * xy
        # 确保所有值非负
        dist_squared = torch.clamp(dist_squared, min=0.0)
        dist = torch.sqrt(dist_squared + 1e-8)
        
        # 通过高斯核计算相似度
        adj = torch.exp(-dist**2 / sigma**2)
        
        # epsilon-近邻图
        mask = (dist <= epsilon).float()
        adj = adj * mask
        
        # 归一化邻接矩阵
        D = torch.sum(adj, dim=2, keepdim=True)  # 度矩阵
        D_sqrt_inv = torch.pow(D + 1e-10, -0.5)  # 添加小值以避免除零
        
        adj_normalized = D_sqrt_inv * adj * D_sqrt_inv.transpose(1, 2)
        
        # 添加自连接
        identity = torch.eye(num_nodes, device=adj.device).unsqueeze(0).expand(batch_size, -1, -1)
        adj_normalized = adj_normalized + identity
        
        return adj_normalized
    
    def construct_adjacency_chunked(self, coordinates, sigma=0.1, epsilon=0.5):
        """
        对大规模节点分块构造邻接矩阵，降低内存使用
        coordinates: [batch_size, num_nodes, 2] - (x, y)坐标
        """
        batch_size, num_nodes, _ = coordinates.size()
        device = coordinates.device
        
        # 记录起始时间
        start_time = time.time()
        
        # 在CUDA环境中添加分析标记
        if torch.cuda.is_available() and hasattr(torch.cuda, 'nvtx'):
            torch.cuda.nvtx.range_push("construct_adjacency_chunked")
        
        # 创建稀疏邻接矩阵（对于非常大的图）
        use_sparse = num_nodes > 5000
        if use_sparse:
            logger.info(f"节点数 {num_nodes} 过多，使用稀疏矩阵表示")
            # 为稀疏表示创建存储
            indices_list = []
            values_list = []
        else:
            # 创建常规的密集邻接矩阵
            adj_normalized = torch.zeros(batch_size, num_nodes, num_nodes, device=device)
        
        # 基于可用内存自动调整分块大小
        available_memory = 0
        if torch.cuda.is_available():
            # 估计可用显存
            available_memory = torch.cuda.get_device_properties(device).total_memory
            current_memory = torch.cuda.memory_allocated(device) + torch.cuda.memory_reserved(device)
            available_memory = (available_memory - current_memory) / 2  # 保守估计，只使用一半
        
        # 计算每个块大约需要多少内存（粗略估计）
        # 假设每个浮点数占用4字节
        bytes_per_element = 4
        memory_per_node = num_nodes * bytes_per_element  # 每个节点需要的邻接矩阵行内存
        
        # 最理想的分块大小（如果内存足够）
        ideal_chunk_size = min(1024, num_nodes // 2)
        
        if available_memory > 0:
            # 基于可用内存估计安全的分块大小
            safe_nodes_count = int(available_memory / (memory_per_node * 3))  # 留出3倍内存余量
            chunk_size = max(1, min(ideal_chunk_size, safe_nodes_count))
        else:
            # 如果无法估计内存，使用保守值
            chunk_size = max(1, min(512, num_nodes // 8))
        
        logger.info(f"分块构造邻接矩阵: 节点数={num_nodes}, 块大小={chunk_size}, 预计块数={math.ceil(num_nodes/chunk_size)**2}")
        
        # 添加自连接用于后续合并
        identity = torch.eye(num_nodes, device=device).unsqueeze(0).expand(batch_size, -1, -1)
        
        # 使用tqdm创建进度条
        total_chunks = (math.ceil(num_nodes / chunk_size))**2
        current_chunk = 0
        
        # 分块处理
        for i in range(0, num_nodes, chunk_size):
            end_i = min(i + chunk_size, num_nodes)
            chunk_i = coordinates[:, i:end_i, :]
            
            # 计算这一块的x_norm
            x_norm_i = torch.sum(chunk_i ** 2, dim=2, keepdim=True)  # [batch, chunk_i, 1]
            
            for j in range(0, num_nodes, chunk_size):
                current_chunk += 1
                if current_chunk % 10 == 0:
                    logger.debug(f"处理分块 {current_chunk}/{total_chunks} ({current_chunk/total_chunks*100:.1f}%)")
                
                end_j = min(j + chunk_size, num_nodes)
                chunk_j = coordinates[:, j:end_j, :]
                
                # 计算这一块的y_norm
                y_norm_j = torch.sum(chunk_j ** 2, dim=2, keepdim=True).transpose(1, 2)  # [batch, 1, chunk_j]
                
                # 计算xy内积
                xy = torch.bmm(chunk_i, chunk_j.transpose(1, 2))  # [batch, chunk_i, chunk_j]
                
                # 计算距离
                dist_squared = x_norm_i + y_norm_j - 2 * xy
                dist_squared = torch.clamp(dist_squared, min=0.0)
                dist = torch.sqrt(dist_squared + 1e-8)
                
                # 高斯核
                curr_adj = torch.exp(-dist**2 / sigma**2)
                
                # epsilon-近邻图 - 只保留距离小于epsilon的边
                mask = (dist <= epsilon).float()
                curr_adj = curr_adj * mask
                
                # 减少稀疏性：移除非常小的值
                if use_sparse:
                    # 为稀疏表示找到非零元素
                    for b in range(batch_size):
                        # 找到此批次中的非零元素
                        batch_adj = curr_adj[b]  # [chunk_i, chunk_j]
                        # 只保留大于阈值的值，提高稀疏度
                        threshold = 1e-4
                        nonzero_mask = batch_adj > threshold
                        
                        if nonzero_mask.any():
                            # 获取非零元素的坐标和值
                            nonzero_coords = nonzero_mask.nonzero(as_tuple=True)
                            row_indices = nonzero_coords[0] + i  # 添加块偏移
                            col_indices = nonzero_coords[1] + j  # 添加块偏移
                            
                            # 将坐标转换为COO格式
                            batch_indices = torch.full_like(row_indices, b)
                            curr_indices = torch.stack([batch_indices, row_indices, col_indices])
                            
                            # 收集值和索引
                            indices_list.append(curr_indices)
                            values_list.append(batch_adj[nonzero_mask])
                else:
                    # 常规密集矩阵存储
                    adj_normalized[:, i:end_i, j:end_j] = curr_adj
                
                # 清理内存
                del chunk_j, y_norm_j, xy, dist_squared, dist, curr_adj, mask
                if torch.cuda.is_available():
                    # 偶尔清理缓存
                    if current_chunk % 5 == 0:
                        torch.cuda.empty_cache()
            
            # 清理这个块的内存
            del chunk_i, x_norm_i
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # 对稀疏矩阵进行度归一化处理
        if use_sparse:
            # 合并所有稀疏索引和值
            if indices_list:
                all_indices = torch.cat(indices_list, dim=1)
                all_values = torch.cat(values_list)
                
                # 构建稀疏张量
                sparse_adj = torch.sparse_coo_tensor(
                    all_indices, all_values, 
                    size=(batch_size, num_nodes, num_nodes),
                    device=device
                )
                
                # 计算度矩阵（对于稀疏矩阵）
                # 将稀疏矩阵转为密集矩阵进行处理
                adj_normalized = sparse_adj.to_dense()
                
                # 清理稀疏表示内存
                del sparse_adj, all_indices, all_values, indices_list, values_list
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        # 计算度矩阵并归一化
        D = torch.sum(adj_normalized, dim=2, keepdim=True)
        D_sqrt_inv = torch.pow(D + 1e-10, -0.5)
        adj_normalized = D_sqrt_inv * adj_normalized * D_sqrt_inv.transpose(1, 2)
        
        # 添加自连接
        adj_normalized = adj_normalized + identity
        
        # 计算总耗时
        total_time = time.time() - start_time
        logger.info(f"邻接矩阵分块构建完成，耗时: {total_time:.2f}秒")
        
        if torch.cuda.is_available() and hasattr(torch.cuda, 'nvtx'):
            torch.cuda.nvtx.range_pop()
            
        return adj_normalized
    
    @timing_decorator
    def forward(self, features, coordinates, z=None, reverse=False):
        """
        features: [batch_size, time_steps, num_nodes, input_dim] - 条件特征
        coordinates: [batch_size, num_nodes, 2] - GPS坐标
        z: [batch_size, num_nodes, flow_input_dim] - 噪声输入 (仅在反向时使用)
        reverse: 是否为反向变换 (生成数据)
        """
        batch_size, time_steps, num_nodes, _ = features.size()
        # 记录设备以便后续使用
        device = features.device
        
        # 记录开始时间，用于计算各阶段耗时
        start_time = time.time()
        
        # 使用torch.cuda.nvtx.range_push/pop进行性能分析标记（仅在CUDA环境）
        if torch.cuda.is_available() and hasattr(torch.cuda, 'nvtx'):
            torch.cuda.nvtx.range_push("Generator.forward")
        
        # 安全检查：确保没有NaN值
        if torch.isnan(features).any():
            # 检测每个特征维度中的NaN
            feature_dim = features.size(-1)
            nan_count_total = torch.isnan(features).sum().item()
            # logger.warning(f"输入特征包含 {nan_count_total} 个NaN值")
            
            # 检查每个特征维度
            for dim in range(feature_dim):
                nan_count = torch.isnan(features[:, :, :, dim]).sum().item()
                if nan_count > 0:
                    percentage = nan_count / (batch_size * time_steps * num_nodes) * 100
                    # logger.warning(f"特征维度 {dim} 包含 {nan_count} 个NaN值 ({percentage:.2f}%)")
                    
                    # 找到第一个NaN值的位置作为示例
                    nan_indices = torch.where(torch.isnan(features[:, :, :, dim]))
                    if len(nan_indices[0]) > 0:
                        b, t, n = nan_indices[0][0].item(), nan_indices[1][0].item(), nan_indices[2][0].item()
                        # logger.warning(f"NaN示例位置: 批次={b}, 时间步={t}, 节点={n}, 特征维度={dim}")
            
            # 修正NaN值
            features = torch.nan_to_num(features, nan=0.0)
            
        if torch.isnan(coordinates).any():
            # 检测坐标中的NaN
            coord_dim = coordinates.size(-1)
            nan_count_total = torch.isnan(coordinates).sum().item()
            logger.warning(f"坐标包含 {nan_count_total} 个NaN值")
            
            # 检查每个坐标维度
            for dim in range(coord_dim):
                nan_count = torch.isnan(coordinates[:, :, dim]).sum().item()
                if nan_count > 0:
                    percentage = nan_count / (batch_size * num_nodes) * 100
                    logger.warning(f"坐标维度 {dim} 包含 {nan_count} 个NaN值 ({percentage:.2f}%)")
            
            # 修正NaN值
            coordinates = torch.nan_to_num(coordinates, nan=0.0)
        
        # 构造邻接矩阵（使用缓存）
        adj_start = time.time()
        
        # 生成缓存键（使用节点数和坐标的哈希值）
        cache_key = f"{num_nodes}_{coordinates.shape}"
        
        # 检查缓存中是否有可用的邻接矩阵
        if cache_key in self.adj_cache:
            logger.debug(f"Generator使用缓存的邻接矩阵，键: {cache_key}")
            adj = self.adj_cache[cache_key]
        else:
            logger.debug("构造邻接矩阵")
            adj = self.construct_adjacency(coordinates)
            # 存储邻接矩阵到缓存中
            self.adj_cache[cache_key] = adj
            # 如果缓存太大，清除旧条目
            if len(self.adj_cache) > 5:  # 最多保存5个不同大小的邻接矩阵
                oldest_key = next(iter(self.adj_cache))
                del self.adj_cache[oldest_key]
                
        adj_time = time.time() - adj_start
        logger.debug(f"邻接矩阵构建/获取耗时: {adj_time:.4f}秒")
        
        # 使用STGCN提取空间特征
        stgcn_start = time.time()
        # 检查输入是否可以使用检查点机制节省内存
        if self.training and features.requires_grad and num_nodes > 1000:
            logger.debug("使用检查点机制执行STGCN1以节省内存")
            x = checkpoint(lambda f, a: self.stgcn1(f, a), features, adj, use_reentrant=False)
        else:
            x = self.stgcn1(features, adj)
            
        # 检查是否需要释放特征内存
        features_copy = None
        if num_nodes > 5000 and features.shape != x.shape:
            # 特征已经转换，可以释放原始特征内存，但先保存设备信息
            features_copy = features.device
            del features
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # 检查输出中是否有NaN值
        if torch.isnan(x).any():
            logger.warning("STGCN1输出包含NaN值，进行修正")
            x = torch.nan_to_num(x, nan=0.0)
        
        # 继续执行第二个STGCN
        if self.training and x.requires_grad and num_nodes > 1000:
            logger.debug("使用检查点机制执行STGCN2以节省内存")
            x = checkpoint(lambda x, a: self.stgcn2(x, a), x, adj, use_reentrant=False)
        else:
            x = self.stgcn2(x, adj)
            
        stgcn_time = time.time() - stgcn_start
        logger.debug(f"STGCN特征提取耗时: {stgcn_time:.4f}秒")
        
        # 检查STGCN2输出中是否有NaN值
        if torch.isnan(x).any():
            logger.warning("STGCN2输出包含NaN值，进行修正")
            x = torch.nan_to_num(x, nan=0.0)
        
        # 提取全局条件
        cond_start = time.time()
        # 取最后一个时间步的特征
        x_last = x[:, -1, :, :]
        x_last = x_last.transpose(1, 2).contiguous()  # [batch_size, channels, num_nodes]
        
        # 全局池化和维度处理
        logger.debug(f"全局池化前形状: {x_last.shape}")
        global_condition = self.global_pool(x_last)  # [batch_size, channels, 1, 1]
        
        # 扁平化全局条件
        global_condition = global_condition.view(batch_size, -1)
        logger.debug(f"调整后全局条件形状: {global_condition.shape}")
        
        # 检查全局条件中是否有NaN值
        if torch.isnan(global_condition).any():
            logger.warning("全局条件包含NaN值，进行修正")
            global_condition = torch.nan_to_num(global_condition, nan=0.0)
        
        # 通过MLP处理
        global_condition = self.condition_mlp(global_condition)  # [batch_size, flow_hidden_dim]
        cond_time = time.time() - cond_start
        logger.debug(f"条件处理耗时: {cond_time:.4f}秒")
        
        flow_start = time.time()
        if not reverse:
            # 前向变换: 数据 -> 噪声
            logger.debug("执行前向变换: 数据 -> 噪声")
            # 只使用最后一个时间步的第一个特征维度作为流输入
            flow_in = x[:, -1, :, 0].reshape(batch_size*num_nodes, -1)
            
            # 检查流输入中是否有NaN值
            if torch.isnan(flow_in).any():
                logger.warning("流输入包含NaN值，进行修正")
                flow_in = torch.nan_to_num(flow_in, nan=0.0)
            
            # 预计算条件重复，使用expand比repeat更高效
            condition_repeated = global_condition.unsqueeze(1).expand(batch_size, num_nodes, -1)
            condition_repeated = condition_repeated.reshape(batch_size*num_nodes, -1)
            
            # 使用流模型
            z, log_det = self.flow(flow_in, condition_repeated, reverse=False)
            
            # 检查流输出是否有NaN值
            if torch.isnan(z).any() or (log_det is not None and torch.isnan(log_det).any()):
                logger.warning("流模型输出包含NaN值，进行修正")
                z = torch.nan_to_num(z, nan=0.0)
                if log_det is not None:
                    log_det = torch.nan_to_num(log_det, nan=0.0)
            
            z = z.reshape(batch_size, num_nodes, -1)
            if log_det is not None:
                log_det = log_det.reshape(batch_size, num_nodes)
            
            flow_time = time.time() - flow_start
            logger.debug(f"流模型前向传递耗时: {flow_time:.4f}秒")
            
            # 清理中间变量
            del flow_in, condition_repeated, x, x_last
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            if torch.cuda.is_available() and hasattr(torch.cuda, 'nvtx'):
                torch.cuda.nvtx.range_pop()
                
            return z, log_det
        
        else:
            # 反向变换: 噪声 -> 数据
            logger.debug("执行反向变换: 噪声 -> 数据")
            # 如果没有提供噪声，则采样标准正态分布
            if z is None:
                # 使用之前保存的设备信息或当前device变量
                target_device = device
                z = torch.randn(batch_size, num_nodes, self.flow.input_dim, device=target_device)
            
            # 检查噪声中是否有NaN值
            if torch.isnan(z).any():
                logger.warning("噪声输入包含NaN值，进行修正")
                z = torch.nan_to_num(z, nan=0.0)
            
            # 扁平化噪声和条件
            z_flat = z.reshape(batch_size*num_nodes, -1)
            condition_repeated = global_condition.unsqueeze(1).expand(batch_size, num_nodes, -1)
            condition_repeated = condition_repeated.reshape(batch_size*num_nodes, -1)
            
            # 使用流模型生成数据
            x_gen, _ = self.flow(z_flat, condition_repeated, reverse=True)
            
            # 检查生成的数据中是否有NaN值
            if torch.isnan(x_gen).any():
                logger.warning("生成的数据包含NaN值，进行修正")
                x_gen = torch.nan_to_num(x_gen, nan=0.0)
                
            x_gen = x_gen.reshape(batch_size, num_nodes, -1)
            
            flow_time = time.time() - flow_start
            logger.debug(f"流模型反向传递耗时: {flow_time:.4f}秒")
            
            # 清理中间变量
            del z_flat, condition_repeated, x, x_last
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            if torch.cuda.is_available() and hasattr(torch.cuda, 'nvtx'):
                torch.cuda.nvtx.range_pop()
                
            return x_gen

# 判别器模型
class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_nodes, time_steps):
        super(Discriminator, self).__init__()
        
        # STGCN层
        self.stgcn1 = STGCNBlock(input_dim, 64, 64, time_steps, num_nodes)
        self.stgcn2 = STGCNBlock(64, 128, 128, time_steps, num_nodes)
        
        # LSTM层
        self.lstm = nn.LSTM(128, hidden_dim, batch_first=True)
        
        # 全连接层
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()  # 确保输出在[0,1]范围内
        )
        
        # 邻接矩阵缓存
        self.adj_cache = {}
        
        # 初始化坐标属性
        self.coords = None
        
        logger.info(f"初始化Discriminator: input_dim={input_dim}, hidden_dim={hidden_dim}, num_nodes={num_nodes}")
    
    @timing_decorator
    def forward(self, x, coordinates=None):
        """
        x: [batch_size, time_steps, num_nodes, input_dim]
        coordinates: [batch_size, num_nodes, 2] 坐标数据
        """
        # 测量原始数据的内存使用
        logger.debug(f"Discriminator: 输入张量大小: {x.shape}, 内存: {x.numel() * x.element_size() / 1024 / 1024:.2f} MB")
        
        # 安全检查：确保没有NaN值
        if torch.isnan(x).any():
            # 检测每个特征维度中的NaN
            batch_size, time_steps, num_nodes, feature_dim = x.size()
            nan_count_total = torch.isnan(x).sum().item()
            # logger.warning(f"判别器输入包含 {nan_count_total} 个NaN值")
            
            # 检查每个特征维度
            for dim in range(feature_dim):
                nan_count = torch.isnan(x[:, :, :, dim]).sum().item()
                if nan_count > 0:
                    percentage = nan_count / (batch_size * time_steps * num_nodes) * 100
                    # logger.warning(f"判别器特征维度 {dim} 包含 {nan_count} 个NaN值 ({percentage:.2f}%)")
                    
                    # 找到第一个NaN值的位置作为示例
                    nan_indices = torch.where(torch.isnan(x[:, :, :, dim]))
                    if len(nan_indices[0]) > 0:
                        b, t, n = nan_indices[0][0].item(), nan_indices[1][0].item(), nan_indices[2][0].item()
                        # logger.warning(f"判别器NaN示例位置: 批次={b}, 时间步={t}, 节点={n}, 特征维度={dim}")
            
            # 修正NaN值
            x = torch.nan_to_num(x, nan=0.0)
        
        # 更新坐标数据
        if coordinates is not None:
            self.coords = coordinates
            
        # 确保坐标已设置
        if self.coords is None:
            logger.warning("Discriminator未设置坐标数据，使用随机坐标")
            batch_size, time_steps, num_nodes, _ = x.size()
            self.coords = torch.randn(batch_size, num_nodes, 2, device=x.device)
            
        batch_size, time_steps, num_nodes, input_dim = x.size()
        
        # 获取坐标并构建邻接矩阵（优化为使用缓存）
        if not hasattr(self, 'adj_cache'):
            self.adj_cache = {}
            
        # 构建缓存键
        cache_key = f"{num_nodes}_{self.coords.shape[0]}"
        
        # 使用缓存的邻接矩阵（如果有）
        if cache_key in self.adj_cache:
            adj = self.adj_cache[cache_key]
            logger.debug(f"使用缓存的邻接矩阵，大小: {adj.shape}")
        else:
            start_time = time.time()
            # 计算邻接矩阵
            if self.coords is not None:
                if torch.isnan(self.coords).any():
                    logger.warning("坐标包含NaN值，进行修正")
                    coords_cleaned = torch.nan_to_num(self.coords, nan=0.0)
                    adj = self.construct_adjacency(coords_cleaned)
                else:
                    adj = self.construct_adjacency(self.coords)
            else:
                logger.warning("没有坐标信息，使用全连接的邻接矩阵")
                adj = torch.ones((batch_size, num_nodes, num_nodes), device=x.device)
                
            # 存储到缓存
            self.adj_cache[cache_key] = adj
            logger.debug(f"构建邻接矩阵用时: {time.time() - start_time:.4f}秒")
        
        # 通过STGCN处理
        start_time = time.time()
        x = self.stgcn1(x, adj)
        x = self.stgcn2(x, adj)
        logger.debug(f"STGCN处理用时: {time.time() - start_time:.4f}秒")
        
        # 内存优化：对于大规模节点，清理不再需要的变量
        if num_nodes > 5000:
            del adj
            torch.cuda.empty_cache()
            logger.debug("清理大型邻接矩阵内存")
        
        # LSTM处理时间特征
        start_time = time.time()
        x = x.permute(0, 2, 1, 3).contiguous()  # [batch, num_nodes, time_steps, hidden_dim]
        x = x.view(batch_size*num_nodes, time_steps, -1)  # [batch*num_nodes, time_steps, hidden_dim]
        
        # 应用LSTM
        x, _ = self.lstm(x)
        
        # 检查LSTM输出
        if torch.isnan(x).any():
            logger.warning("LSTM输出包含NaN值，进行修正")
            x = torch.nan_to_num(x, nan=0.0)
            
        # 获取最后一个时间步
        x = x[:, -1, :]  # [batch*num_nodes, hidden_dim]
        
        # 重塑张量
        x = x.view(batch_size, num_nodes, -1)  # [batch, num_nodes, hidden_dim]
        
        # 全局池化
        x = torch.mean(x, dim=1)  # [batch, hidden_dim]
        
        logger.debug(f"LSTM和池化用时: {time.time() - start_time:.4f}秒")
        
        # 最终MLP分类
        start_time = time.time()
        x = self.fc(x)
        logger.debug(f"MLP处理用时: {time.time() - start_time:.4f}秒")
        
        # 确保输出在[0,1]范围内
        output = torch.clamp(x, min=1e-7, max=1-1e-7)
        
        # 最终检查
        if torch.isnan(output).any():
            logger.warning("Discriminator最终输出包含NaN值，进行修正")
            output = torch.nan_to_num(output, nan=0.5)  # 对于二分类问题，用0.5替换NaN
            
        return output
    
    @timing_decorator
    def construct_adjacency(self, coordinates, sigma=0.1, epsilon=0.5):
        """
        根据GPS坐标构造邻接矩阵
        coordinates: [batch_size, num_nodes, 2] - (x, y)坐标
        """
        batch_size, num_nodes, _ = coordinates.size()
        
        # 内存优化：检查节点数量并输出警告
        if num_nodes > 5000:
            logger.warning(f"构造邻接矩阵时节点数量过多: {num_nodes}，可能导致内存不足")
            
        # 内存优化：对于大型邻接矩阵，分批处理
        if num_nodes > 2048:
            logger.info(f"节点数 {num_nodes} 过多，使用分批处理计算邻接矩阵")
            return self.construct_adjacency_chunked(coordinates, sigma, epsilon)
        
        # 计算欧氏距离 - 使用高效的矩阵运算
        # 计算距离矩阵的平方 ||x_i - x_j||^2 = ||x_i||^2 + ||x_j||^2 - 2<x_i, x_j>
        x_norm = torch.sum(coordinates ** 2, dim=2, keepdim=True)
        y_norm = x_norm.transpose(1, 2)
        xy = torch.bmm(coordinates, coordinates.transpose(1, 2))
        dist_squared = x_norm + y_norm - 2 * xy
        # 确保所有值非负
        dist_squared = torch.clamp(dist_squared, min=0.0)
        dist = torch.sqrt(dist_squared + 1e-8)
        
        # 通过高斯核计算相似度
        adj = torch.exp(-dist**2 / sigma**2)
        
        # epsilon-近邻图
        mask = (dist <= epsilon).float()
        adj = adj * mask
        
        # 归一化邻接矩阵
        D = torch.sum(adj, dim=2, keepdim=True)  # 度矩阵
        D_sqrt_inv = torch.pow(D + 1e-10, -0.5)  # 添加小值以避免除零
        
        adj_normalized = D_sqrt_inv * adj * D_sqrt_inv.transpose(1, 2)
        
        # 添加自连接
        identity = torch.eye(num_nodes, device=adj.device).unsqueeze(0).expand(batch_size, -1, -1)
        adj_normalized = adj_normalized + identity
        
        return adj_normalized
    
    def construct_adjacency_chunked(self, coordinates, sigma=0.1, epsilon=0.5):
        """
        对大规模节点分块构造邻接矩阵，降低内存使用
        coordinates: [batch_size, num_nodes, 2] - (x, y)坐标
        """
        batch_size, num_nodes, _ = coordinates.size()
        device = coordinates.device
        
        # 记录起始时间
        start_time = time.time()
        
        # 在CUDA环境中添加分析标记
        if torch.cuda.is_available() and hasattr(torch.cuda, 'nvtx'):
            torch.cuda.nvtx.range_push("construct_adjacency_chunked")
        
        # 创建稀疏邻接矩阵（对于非常大的图）
        use_sparse = num_nodes > 5000
        if use_sparse:
            logger.info(f"节点数 {num_nodes} 过多，使用稀疏矩阵表示")
            # 为稀疏表示创建存储
            indices_list = []
            values_list = []
        else:
            # 创建常规的密集邻接矩阵
            adj_normalized = torch.zeros(batch_size, num_nodes, num_nodes, device=device)
        
        # 基于可用内存自动调整分块大小
        available_memory = 0
        if torch.cuda.is_available():
            # 估计可用显存
            available_memory = torch.cuda.get_device_properties(device).total_memory
            current_memory = torch.cuda.memory_allocated(device) + torch.cuda.memory_reserved(device)
            available_memory = (available_memory - current_memory) / 2  # 保守估计，只使用一半
        
        # 计算每个块大约需要多少内存（粗略估计）
        # 假设每个浮点数占用4字节
        bytes_per_element = 4
        memory_per_node = num_nodes * bytes_per_element  # 每个节点需要的邻接矩阵行内存
        
        # 最理想的分块大小（如果内存足够）
        ideal_chunk_size = min(1024, num_nodes // 2)
        
        if available_memory > 0:
            # 基于可用内存估计安全的分块大小
            safe_nodes_count = int(available_memory / (memory_per_node * 3))  # 留出3倍内存余量
            chunk_size = max(1, min(ideal_chunk_size, safe_nodes_count))
        else:
            # 如果无法估计内存，使用保守值
            chunk_size = max(1, min(512, num_nodes // 8))
        
        logger.info(f"分块构造邻接矩阵: 节点数={num_nodes}, 块大小={chunk_size}, 预计块数={math.ceil(num_nodes/chunk_size)**2}")
        
        # 添加自连接用于后续合并
        identity = torch.eye(num_nodes, device=device).unsqueeze(0).expand(batch_size, -1, -1)
        
        # 使用tqdm创建进度条
        total_chunks = (math.ceil(num_nodes / chunk_size))**2
        current_chunk = 0
        
        # 分块处理
        for i in range(0, num_nodes, chunk_size):
            end_i = min(i + chunk_size, num_nodes)
            chunk_i = coordinates[:, i:end_i, :]
            
            # 计算这一块的x_norm
            x_norm_i = torch.sum(chunk_i ** 2, dim=2, keepdim=True)  # [batch, chunk_i, 1]
            
            for j in range(0, num_nodes, chunk_size):
                current_chunk += 1
                if current_chunk % 10 == 0:
                    logger.debug(f"处理分块 {current_chunk}/{total_chunks} ({current_chunk/total_chunks*100:.1f}%)")
                
                end_j = min(j + chunk_size, num_nodes)
                chunk_j = coordinates[:, j:end_j, :]
                
                # 计算这一块的y_norm
                y_norm_j = torch.sum(chunk_j ** 2, dim=2, keepdim=True).transpose(1, 2)  # [batch, 1, chunk_j]
                
                # 计算xy内积
                xy = torch.bmm(chunk_i, chunk_j.transpose(1, 2))  # [batch, chunk_i, chunk_j]
                
                # 计算距离
                dist_squared = x_norm_i + y_norm_j - 2 * xy
                dist_squared = torch.clamp(dist_squared, min=0.0)
                dist = torch.sqrt(dist_squared + 1e-8)
                
                # 高斯核
                curr_adj = torch.exp(-dist**2 / sigma**2)
                
                # epsilon-近邻图 - 只保留距离小于epsilon的边
                mask = (dist <= epsilon).float()
                curr_adj = curr_adj * mask
                
                # 减少稀疏性：移除非常小的值
                if use_sparse:
                    # 为稀疏表示找到非零元素
                    for b in range(batch_size):
                        # 找到此批次中的非零元素
                        batch_adj = curr_adj[b]  # [chunk_i, chunk_j]
                        # 只保留大于阈值的值，提高稀疏度
                        threshold = 1e-4
                        nonzero_mask = batch_adj > threshold
                        
                        if nonzero_mask.any():
                            # 获取非零元素的坐标和值
                            nonzero_coords = nonzero_mask.nonzero(as_tuple=True)
                            row_indices = nonzero_coords[0] + i  # 添加块偏移
                            col_indices = nonzero_coords[1] + j  # 添加块偏移
                            
                            # 将坐标转换为COO格式
                            batch_indices = torch.full_like(row_indices, b)
                            curr_indices = torch.stack([batch_indices, row_indices, col_indices])
                            
                            # 收集值和索引
                            indices_list.append(curr_indices)
                            values_list.append(batch_adj[nonzero_mask])
                else:
                    # 常规密集矩阵存储
                    adj_normalized[:, i:end_i, j:end_j] = curr_adj
                
                # 清理内存
                del chunk_j, y_norm_j, xy, dist_squared, dist, curr_adj, mask
                if torch.cuda.is_available():
                    # 偶尔清理缓存
                    if current_chunk % 5 == 0:
                        torch.cuda.empty_cache()
            
            # 清理这个块的内存
            del chunk_i, x_norm_i
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # 对稀疏矩阵进行度归一化处理
        if use_sparse:
            # 合并所有稀疏索引和值
            if indices_list:
                all_indices = torch.cat(indices_list, dim=1)
                all_values = torch.cat(values_list)
                
                # 构建稀疏张量
                sparse_adj = torch.sparse_coo_tensor(
                    all_indices, all_values, 
                    size=(batch_size, num_nodes, num_nodes),
                    device=device
                )
                
                # 计算度矩阵（对于稀疏矩阵）
                # 将稀疏矩阵转为密集矩阵进行处理
                adj_normalized = sparse_adj.to_dense()
                
                # 清理稀疏表示内存
                del sparse_adj, all_indices, all_values, indices_list, values_list
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        # 计算度矩阵并归一化
        D = torch.sum(adj_normalized, dim=2, keepdim=True)
        D_sqrt_inv = torch.pow(D + 1e-10, -0.5)
        adj_normalized = D_sqrt_inv * adj_normalized * D_sqrt_inv.transpose(1, 2)
        
        # 添加自连接
        adj_normalized = adj_normalized + identity
        
        # 计算总耗时
        total_time = time.time() - start_time
        logger.info(f"邻接矩阵分块构建完成，耗时: {total_time:.2f}秒")
        
        if torch.cuda.is_available() and hasattr(torch.cuda, 'nvtx'):
            torch.cuda.nvtx.range_pop()
            
        return adj_normalized

def mode_ratio_loss(encoded_real, encoded_gen, n_clusters=4, device='cuda'):
    """
    计算Mode比例损失 (KL散度)
    encoded_real: [batch_size, num_nodes, 2] - 真实数据的降维表示
    encoded_gen: [batch_size, num_nodes, 2] - 生成数据的降维表示
    """
    try:
        # 记录开始时间
        start_time = time.time()
        
        # 使用CUDA性能分析标记（如果可用）
        if torch.cuda.is_available() and hasattr(torch.cuda, 'nvtx'):
            torch.cuda.nvtx.range_push("mode_ratio_loss")
        
        # 尝试按需导入sklearn
        try:
            from sklearn.cluster import KMeans
            import torch.nn.functional as F
        except ImportError as e:
            logger.error(f"导入所需库失败: {str(e)}")
            return torch.tensor(1.0, device=device, requires_grad=True)
        
        # 安全检查：确保没有NaN值
        if torch.isnan(encoded_real).any() or torch.isnan(encoded_gen).any():
            logger.warning("Mode Ratio Loss计算中检测到NaN值，执行修复")
            encoded_real = torch.nan_to_num(encoded_real, nan=0.0)
            encoded_gen = torch.nan_to_num(encoded_gen, nan=0.0)
        
        # 获取输入维度
        batch_size, num_nodes, dim = encoded_real.size()
        total_samples = batch_size * num_nodes
        
        # 优化: 自动调整聚类数量
        if total_samples < n_clusters * 10:
            # 确保每个聚类至少有10个样本
            adjusted_clusters = max(2, min(n_clusters, total_samples // 10))
            if adjusted_clusters != n_clusters:
                logger.debug(f"样本数 {total_samples} 较少，将聚类数从 {n_clusters} 调整为 {adjusted_clusters}")
                n_clusters = adjusted_clusters
        
        # 确定采样策略，降低计算开销
        max_samples = 10000
        sample_indices = None
        
        # 如果样本数过多，进行采样
        if total_samples > max_samples:
            logger.info(f"样本数 {total_samples} 过多，采样到 {max_samples}")
            
            # 使用随机采样选择索引
            sample_indices = torch.randperm(total_samples, device='cpu')[:max_samples]
            
            # 平坦化然后采样
            encoded_real_flat = encoded_real.reshape(-1, dim).cpu()
            encoded_gen_flat = encoded_gen.reshape(-1, dim).cpu()
            
            encoded_real_np = encoded_real_flat[sample_indices].numpy()
            encoded_gen_np = encoded_gen_flat[sample_indices].numpy()
            
            # 清理不再需要的临时变量
            del encoded_real_flat, encoded_gen_flat
        else:
            # 样本数量适中，可以全部处理
            encoded_real_np = encoded_real.reshape(-1, dim).detach().cpu().numpy()
            encoded_gen_np = encoded_gen.reshape(-1, dim).detach().cpu().numpy()
        
        # 安全检查：确保没有NaN值(numpy版本)
        encoded_real_np = np.nan_to_num(encoded_real_np)
        encoded_gen_np = np.nan_to_num(encoded_gen_np)
        
        # 记录数据准备时间
        prep_time = time.time() - start_time
        logger.debug(f"数据准备耗时: {prep_time:.4f}秒")
        
        kmeans_start = time.time()
        try:
            # 设置KMeans参数，优化速度
            # n_init='auto'让scikit-learn自动选择合适的初始化次数
            # max_iter降低最大迭代次数，加快收敛
            # tol增加容差，加快收敛
            kmeans = KMeans(
                n_clusters=n_clusters, 
                random_state=0, 
                n_init='auto',
                max_iter=100,  # 减少最大迭代次数
                tol=1e-3       # 增加容差
            )
            
            # 使用真实数据拟合KMeans
            kmeans.fit(encoded_real_np)
            
            # 获取聚类中心
            centers = torch.tensor(kmeans.cluster_centers_, device=device, dtype=torch.float)
            
            # 对真实数据和生成数据进行预测
            real_labels_np = kmeans.predict(encoded_real_np)
            gen_labels_np = kmeans.predict(encoded_gen_np)
            
            # 记录KMeans处理时间
            kmeans_time = time.time() - kmeans_start
            logger.debug(f"KMeans聚类耗时: {kmeans_time:.4f}秒")
            
            # 清理不再需要的数据
            del encoded_real_np, encoded_gen_np
            
            # 计算各个模式的比例
            count_start = time.time()
            real_mode_counts = np.bincount(real_labels_np, minlength=n_clusters)
            gen_mode_counts = np.bincount(gen_labels_np, minlength=n_clusters)
            
            # 确保没有零计数，避免除零错误
            real_mode_counts = np.maximum(real_mode_counts, 1)
            gen_mode_counts = np.maximum(gen_mode_counts, 1)
            
            real_mode_ratio = real_mode_counts / real_mode_counts.sum()
            gen_mode_ratio = gen_mode_counts / gen_mode_counts.sum()
            
            # 将比例转换为PyTorch张量
            real_mode_ratio = torch.tensor(real_mode_ratio, device=device, dtype=torch.float)
            gen_mode_ratio = torch.tensor(gen_mode_ratio, device=device, dtype=torch.float)
            
            # 为避免对数为零或无穷大，添加一个小值
            epsilon = 1e-6
            real_mode_ratio = real_mode_ratio + epsilon
            gen_mode_ratio = gen_mode_ratio + epsilon
            
            # 重新归一化
            real_mode_ratio = real_mode_ratio / real_mode_ratio.sum()
            gen_mode_ratio = gen_mode_ratio / gen_mode_ratio.sum()
            
            # 计算KL散度
            kl_div = F.kl_div(
                F.log_softmax(gen_mode_ratio, dim=0),
                F.softmax(real_mode_ratio, dim=0),
                reduction='batchmean'
            )
            
            # 记录比例计算时间
            count_time = time.time() - count_start
            logger.debug(f"比例计算耗时: {count_time:.4f}秒")
            
            # 安全检查：处理无效结果
            if torch.isnan(kl_div) or torch.isinf(kl_div):
                logger.warning(f"Mode Ratio Loss结果无效: {kl_div.item()}")
                return torch.tensor(1.0, device=device, requires_grad=True)
                
            # 记录总耗时
            total_time = time.time() - start_time
            logger.debug(f"Mode Ratio Loss总耗时: {total_time:.4f}秒")
            
            if torch.cuda.is_available() and hasattr(torch.cuda, 'nvtx'):
                torch.cuda.nvtx.range_pop()
                
            return kl_div
            
        except Exception as e:
            logger.error(f"KMeans聚类出错: {str(e)}")
            if torch.cuda.is_available() and hasattr(torch.cuda, 'nvtx'):
                torch.cuda.nvtx.range_pop()
            return torch.tensor(1.0, device=device, requires_grad=True)
            
    except Exception as e:
        logger.error(f"计算mode_ratio_loss出错: {str(e)}")
        # 返回一个默认损失值以防止训练中断
        if torch.cuda.is_available() and hasattr(torch.cuda, 'nvtx'):
            torch.cuda.nvtx.range_pop()
        return torch.tensor(1.0, device=device, requires_grad=True)

@timing_decorator
def sliced_wasserstein_distance(x, y, num_projections=50, device='cuda'):
    """
    计算两个分布之间的切片Wasserstein距离
    
    x: [batch_size, dim] - 第一个分布的样本
    y: [batch_size, dim] - 第二个分布的样本
    num_projections: 用于近似的投影数量
    
    返回：近似的Wasserstein距离
    """
    # 记录开始时间
    start_time = time.time()
    
    # 在CUDA环境中添加性能分析标记
    if torch.cuda.is_available() and hasattr(torch.cuda, 'nvtx'):
        torch.cuda.nvtx.range_push("sliced_wasserstein_distance")
    
    # 安全检查：确保没有NaN值
    if torch.isnan(x).any() or torch.isnan(y).any():
        logger.warning("Wasserstein距离计算中检测到NaN值，执行修复")
        x = torch.nan_to_num(x, nan=0.0)
        y = torch.nan_to_num(y, nan=0.0)
    
    # 获取维度信息
    batch_size_x = x.size(0)
    batch_size_y = y.size(0)
    dim = x.size(1)
    
    # 优化：根据输入大小动态调整投影数量
    total_elements = batch_size_x * batch_size_y * dim
    
    # 如果数据量很大，减少投影数量以节省计算资源
    if total_elements > 10000000:  # 1千万元素
        adjusted_projections = max(10, num_projections // 4)
        logger.debug(f"数据量大 ({total_elements} 元素)，将投影数从 {num_projections} 减少到 {adjusted_projections}")
        num_projections = adjusted_projections
        
    # 生成随机投影向量 - 使用标准正态分布生成
    # 尝试减少内存使用，优先使用float32而非float64
    projections = torch.randn((num_projections, dim), dtype=torch.float32, device=device)
    
    # 归一化投影向量
    projections = projections / (torch.norm(projections, dim=1, keepdim=True) + 1e-8)
    
    # 计算投影后的值
    x_projections = torch.matmul(x, projections.t())  # [batch_size_x, num_projections]
    y_projections = torch.matmul(y, projections.t())  # [batch_size_y, num_projections]
    
    # 对每个投影方向上的值排序
    x_sorted, _ = torch.sort(x_projections, dim=0)  # 沿batch维度排序
    y_sorted, _ = torch.sort(y_projections, dim=0)  # 沿batch维度排序
    
    # 处理批大小不一致的情况
    if batch_size_x == batch_size_y:
        # 如果批大小相同，直接计算
        wasserstein_distances = torch.mean(torch.abs(x_sorted - y_sorted), dim=0)
    else:
        # 如果批大小不同，需要插值处理
        logger.debug(f"批大小不同: x={batch_size_x}, y={batch_size_y}，执行插值")
        
        if batch_size_x < batch_size_y:
            # 对x进行上采样
            indices = torch.linspace(0, batch_size_x-1, batch_size_y).long()
            x_sorted = x_sorted[indices]
        else:
            # 对y进行上采样
            indices = torch.linspace(0, batch_size_y-1, batch_size_x).long()
            y_sorted = y_sorted[indices]
        
        wasserstein_distances = torch.mean(torch.abs(x_sorted - y_sorted), dim=0)
    
    # 计算平均距离
    swd = torch.mean(wasserstein_distances)
    
    # 安全检查：处理无效结果
    if torch.isnan(swd) or torch.isinf(swd):
        logger.warning(f"Wasserstein距离计算结果无效: {swd.item()}")
        swd = torch.tensor(1.0, device=device, requires_grad=True)
    
    # 记录总耗时
    total_time = time.time() - start_time
    logger.debug(f"切片Wasserstein距离计算耗时: {total_time:.4f}秒")
    
    if torch.cuda.is_available() and hasattr(torch.cuda, 'nvtx'):
        torch.cuda.nvtx.range_pop()
    
    return swd

# 聚类编码器，用于数据降维和可视化
class ClusterEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=2):
        super(ClusterEncoder, self).__init__()
        
        # 渐进降维网络结构
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim // 4),
            nn.Linear(hidden_dim // 4, output_dim)
        )
        
        logger.info(f"初始化ClusterEncoder: input_dim={input_dim}, hidden_dim={hidden_dim}, output_dim={output_dim}")
        
    @timing_decorator
    def forward(self, x):
        """
        x: [batch_size, input_dim] - 输入序列
        返回降维后的表示: [batch_size, output_dim]
        """
        # 安全检查：确保没有NaN值
        if torch.isnan(x).any():
            logger.warning("ClusterEncoder输入包含NaN值，进行修正")
            x = torch.nan_to_num(x, nan=0.0)
            
        # 应用编码器获取低维表示
        encoded = self.encoder(x)
        
        # 检查输出中是否有NaN值
        if torch.isnan(encoded).any():
            logger.warning("编码器输出包含NaN值，进行修正")
            encoded = torch.nan_to_num(encoded, nan=0.0)
            
        return encoded