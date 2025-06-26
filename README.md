# 城市电力消耗数据生成模型

本项目实现了一个基于条件归一化流（Conditional Normalizing Flow）和时空图卷积网络（STGCN）的模型，用于在给定环境特征（气象数据、人口密度、节假日等）的基础上，预测/生成城市居民的用电数据。该模型不仅能生成符合特定条件的点状预测，还能保证生成数据的整体分布与真实数据一致。

## 项目结构

### 核心文件
- `energy_prediction_model.py`: 定义模型架构，包括STGCN、条件归一化流和聚类编码器
- `data_utils.py`: 数据预处理、加载和缓存工具
- `train.py`: 模型训练和评估脚本
- `generate_new_city.py`: 用于生成新城市电力消耗数据的脚本
- `handle_new_data.py`: 新数据处理和特征映射工具
- `config.py`: 全局配置参数管理

### 辅助工具
- `pkl_to_csv.py`: 数据格式转换工具
- `quickstart.py`: 快速启动演示脚本
- `__init__.py`: 包初始化文件
- `requirements.txt`: 依赖库列表

### 支持文件夹

- `checkpoints/`: 保存的模型检查点
- `data_cache/`: 缓存处理后的数据，加速训练
- `logs/`: 训练和运行日志
- `visualizations/`: 生成的可视化结果
- `data_exported/`: 导出的数据文件

## 模型架构

本项目结合了多种先进技术构建一个复杂的生成模型：

1. **时空图卷积网络 (STGCN)**: 用于捕获用户间的空间依赖关系和时间序列模式
2. **条件归一化流 (Conditional Normalizing Flow)**: 用于生成符合特定条件的电力消耗数据
3. **聚类编码器 (Cluster Encoder)**: 用于约束生成数据的整体分布，确保与真实数据具有相似的聚类特征

### 模型工作流程

1. 使用用户GPS坐标构造图结构，反映用户间的空间关系
2. 利用STGCN提取空间条件信息，并通过全局池化获取条件向量
3. 条件归一化流将噪声分布转换为符合条件的电力消耗数据
4. 判别器通过对抗训练提高生成数据的时序合理性
5. 聚类编码器将用电数据映射到低维空间，通过损失函数约束生成数据与真实数据的聚类分布一致

## 性能优化

模型集成了多项性能优化技术：

1. **动态内存管理**: 自动监控和管理GPU内存使用，根据可用资源动态调整批处理大小
2. **数据缓存机制**: 使用磁盘缓存预处理后的数据，避免重复计算
3. **渐进式批处理**: 对大规模节点图采用分块邻接矩阵计算，解决内存瓶颈
4. **梯度检查点**: 对大型模型使用梯度检查点技术，降低内存需求
5. **自适应学习率调整**: 根据训练过程中的错误自动调整学习率
6. **性能分析装饰器**: 提供详细的函数执行时间和内存使用统计

## 训练过程

训练过程涉及多个损失函数：

1. **重建损失**: 确保生成的数据与真实数据一致
2. **流模型损失**: 优化归一化流的对数似然
3. **对抗损失**: 通过判别器提高生成数据的真实性
4. **聚类损失**: 包括切片Wasserstein距离和模式比例损失，确保生成数据的整体分布与真实数据一致

采用动态权重调整策略，在训练初期弱化聚类损失，随着训练进行逐步增强其影响，实现由粗到精的学习过程。

## 特征设计

本模型采用27个精心设计的特征，涵盖多个维度以实现高精度预测：

**核心优势：**
- 使用聚类坐标进行高效空间建模
- 包含14个详细天气特征，捕获气候影响
- 集成7个时间特征，识别时序模式
- 支持2个分类特征的自动独热编码
- 融合2个社会经济特征，增强预测准确性

## 使用说明

### 环境要求

```
Python 3.8+
PyTorch 1.8+
numpy>=1.19.0
pandas>=1.1.0
scikit-learn>=0.24.0
matplotlib>=3.3.0
tqdm>=4.50.0
```

安装依赖：
```bash
pip install -r requirements.txt
```

### 数据格式

输入数据应为CSV格式，包含以下字段：

**必需字段：**
- `cus-no`: 用户ID（标识符）
- `date`: 日期
- `cost`: 用电量（目标变量）

**模型训练特征（27个主要特征）：**

1. **用户身份性格品质聚类特征**
   - `cluster_x`: 聚类坐标X（替代GPS坐标用于空间建模）
   - `cluster_y`: 聚类坐标Y（替代GPS坐标用于空间建模）

2. **天气特征**
   - `rain_sum`: 降雨量总和
   - `temperature_2m_max`: 2米高度最高温度
   - `temperature_2m_min`: 2米高度最低温度
   - `temperature_2m_mean`: 2米高度平均温度
   - `soil_temperature_0_to_7cm_mean`: 0-7cm土壤平均温度
   - `soil_temperature_7_to_28cm_mean`: 7-28cm土壤平均温度
   - `apparent_temperature_mean`: 体感平均温度
   - `apparent_temperature_max`: 体感最高温度
   - `apparent_temperature_min`: 体感最低温度
   - `relative_humidity_2m_mean`: 2米高度平均相对湿度
   - `wind_speed_10m_mean`: 10米高度平均风速
   - `precipitation_hours`: 降水时长
   - `snowfall_sum`: 降雪量总和
   - `sunshine_duration`: 日照时长

3. **时间特征**
   - `is_holiday`: 是否节假日（0/1）
   - `holiday_length`: 节假日长度
   - `weekend`: 是否周末（0/1）
   - `week_of_year`: 一年中的第几周
   - `month`: 月份
   - `day_of_week`: 星期几
   - `day_of_month`: 月份中的第几天

4. **分类特征（会进行独热编码）**
   - `season`: 季节（spring/summer/autumn/winter）
   - `weather_code`: 天气代码

5. **社会经济特征**
   - `population_density`: 人口密度
   - `house_price`: 房价

### 快速开始

```bash
python quickstart.py --data_path path/to/data.csv
```

### 训练模型

运行以下命令训练模型：

```bash
python train.py --data_path path/to/data.csv --batch_size 32 --num_epochs 100
```

可以在`config.py`或`train.py`的`config`字典中修改训练参数。

### 处理新数据

对于新的原始数据，使用数据处理工具进行特征映射和清理：

```bash
python handle_new_data.py
```

该工具会：
- 自动映射中英文列名
- 处理重复列问题
- 验证特征完整性
- 输出标准化的数据格式

### 生成新城市数据

使用训练好的模型，基于新城市的环境特征生成用电数据：

```bash
python generate_new_city.py \
  --original_data_path path/to/original_data.csv \
  --new_city_data_path path/to/new_city_data.csv \
  --model_path path/to/checkpoint.pth \
  --output_path results/new_city_predictions.csv \
  --compare_clusters \
  --original_cluster_data path/to/original_cluster_data.npy
```

## 数据处理优化

项目对大规模数据处理进行了多项优化：

1. **多索引批量处理**: 使用pandas的MultiIndex加速数据填充
2. **并行数据加载**: 使用多线程DataLoader优化I/O操作
3. **增量式处理**: 支持对超大数据集的增量式处理
4. **自动缓存**: 基于输入参数哈希的智能缓存系统
5. **数据验证**: 自动检测和处理无效值（NaN, Inf）

## 聚类效果

训练成功后，可视化真实数据和生成数据的TSNE聚类结果，应呈现类似的"橄榄型"分布：
- 橄榄两端分别对应高能耗和低能耗用户
- 橄榄中间的左右区域分别对应夏季和冬季模式用户

## 调试工具

项目内置了完善的调试功能：

- 自动NaN值检测和修复机制
- 详细的训练日志记录
- 内存使用监控和优化
- 数据质量自动验证


## 注意事项

### 数据处理
1. **特征完整性**：确保数据包含所有27个必需特征字段
2. **数据标准化**：所有特征在预处理时会自动标准化
3. **列名映射**：支持中英文列名自动映射，使用`handle_new_data.py`处理原始数据
4. **分类特征**：`season`和`weather_code`会自动进行独热编码

### 模型训练
5. **聚类坐标**：使用`cluster_x`和`cluster_y`替代GPS坐标进行空间建模
6. **内存优化**：大规模节点（>5000）会自动降低批处理大小
7. **缓存机制**：首次运行数据处理较慢，后续运行利用缓存加速

### 数据生成
8. **标准化器**：新城市数据必须使用与训练数据相同的标准化器
9. **节点限制**：确保新城市用户数量不超过模型设计的最大节点数
10. **聚类比较**：建议使用完整365天数据，否则需要填充或截断 