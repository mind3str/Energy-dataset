import pandas as pd

def standardize_csv_columns(input_filepath: str, output_filepath: str):
    """
    读取CSV文件，标准化列名，处理重复列问题，
    并将"cost"列移动到末尾，然后保存到新的CSV文件。

    参数:
    input_filepath (str): 输入CSV文件的路径。
    output_filepath (str): 输出CSV文件的路径。
    """
    # 定义列名映射关系（适应新的27个特征）
    column_mapping = {
        'cluster-x': 'cluster_x',
        'cluster-y': 'cluster_y',
        'prep': 'rain_sum',
        'max_temp': 'temperature_2m_max',
        'min_temp': 'temperature_2m_min',
        '房价': 'house_price'  # 将"房价"重命名为英文
    }

    try:
        # 读取CSV文件
        # 尝试使用不同的编码格式，如果utf-8失败，则尝试gbk
        try:
            df = pd.read_csv(input_filepath, encoding='utf-8')
        except UnicodeDecodeError:
            print(f"使用UTF-8编码读取文件 '{input_filepath}' 失败，尝试GBK编码...")
            df = pd.read_csv(input_filepath, encoding='gbk')
        
        print(f"成功读取文件: {input_filepath}")
        print("原始列名:")
        print(df.columns.tolist())

        # 检查是否同时存在"房价"和"house_price"列
        has_chinese_house_price = '房价' in df.columns
        has_english_house_price = 'house_price' in df.columns
        
        if has_chinese_house_price and has_english_house_price:
            print("\n警告：发现同时存在'房价'和'house_price'列")
            # 保留中文列"房价"，删除英文列"house_price"
            print("保留中文列'房价'，删除英文列'house_price'")
            df = df.drop(columns=['house_price'])
            # 然后将中文列重命名为英文
            df.rename(columns={'房价': 'house_price'}, inplace=True)
            print("将'房价'列重命名为'house_price'")
        elif has_chinese_house_price:
            # 只有中文列，正常重命名
            df.rename(columns={'房价': 'house_price'}, inplace=True)
            print("将'房价'列重命名为'house_price'")
        elif has_english_house_price:
            # 只有英文列，保持不变
            print("只有英文列'house_price'，保持不变")

        # 应用其他列名映射（排除房价相关的映射）
        other_mapping = {k: v for k, v in column_mapping.items() if k != '房价'}
        df.rename(columns=other_mapping, inplace=True)

        print("\n重命名后的列名:")
        print(df.columns.tolist())

        # 验证是否包含所需的特征列
        expected_features = [
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
            # 4. 分类特征
            'season', 'weather_code',
            # 5. 社会经济特征
            'population_density', 'house_price'
        ]
        
        missing_features = [col for col in expected_features if col not in df.columns]
        if missing_features:
            print(f"\n警告：缺少以下期望的特征列: {missing_features}")
        
        present_features = [col for col in expected_features if col in df.columns]
        print(f"\n已包含的特征列 ({len(present_features)}/{len(expected_features)}): {present_features}")

        # 将 'cost' 列移动到最后
        if 'cost' in df.columns:
            cost_column = df.pop('cost') # 使用pop移除列并获取它
            df['cost'] = cost_column    # 将 'cost' 列添加到DataFrame的末尾
            print("\n已将 'cost' 列移动到最后。")
        else:
            print("\n警告: 'cost' 列未在文件中找到，无法移动。")

        print("\n最终列名顺序:")
        print(df.columns.tolist())

        # 将标准化后的DataFrame保存到新的CSV文件
        # 使用 utf-8-sig 编码以便在 Excel 中正确显示中文字符（如果还有的话）
        df.to_csv(output_filepath, index=False, encoding='utf-8-sig')

        print(f"\n已将处理后的数据保存到文件: {output_filepath}")
        return True

    except FileNotFoundError:
        print(f"错误：文件 '{input_filepath}' 未找到。请确保文件路径正确。")
        return False
    except Exception as e:
        print(f"处理过程中发生错误: {e}")
        return False

# --- 如何使用这个函数 ---
if __name__ == "__main__":
    # 设置输入和输出文件名/路径
    input_csv_file = 'sum_new_final.csv'  # 您的原始文件名
    output_csv_file = 'standardized_data_function_output.csv' # 处理后保存的文件名

    # 调用函数进行处理
    success = standardize_csv_columns(input_csv_file, output_csv_file)

    if success:
        print(f"CSV文件处理成功。结果保存在 {output_csv_file}")
    else:
        print("CSV文件处理失败。")

