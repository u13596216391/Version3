import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib
import traceback
import re

def merge_microseismic_data(ms_folder):
    """
    从文件夹及其所有子文件夹中递归地加载并合并所有微震数据文件。
    """
    print(f"\n--- 正在递归搜索并合并微震数据: {ms_folder} ---")
    if not os.path.isdir(ms_folder):
        raise FileNotFoundError(f"微震数据文件夹不存在: {ms_folder}")
    
    all_data = []
    date_pattern = re.compile(r'(\d{4}-\d{2}-\d{2})')

    # --- 核心修改：使用 os.walk 进行递归遍历 ---
    for root, dirs, files in os.walk(ms_folder):
        for file in files:
            if not file.endswith('.csv'):
                continue

            # 从文件名提取日期
            match = date_pattern.search(file)
            if not match:
                print(f"警告: 文件名 '{file}' 中未找到 'YYYY-MM-DD' 格式的日期，已跳过。")
                continue
            date_str = match.group(1)
            
            file_path = os.path.join(root, file)
            try:
                df = pd.read_csv(file_path)
                
                time_col = 'Event_Time' if 'Event_Time' in df.columns else 'time'
                energy_col = 'Event_Energy' if 'Event_Energy' in df.columns else 'energy'

                if time_col not in df.columns or energy_col not in df.columns:
                    print(f"警告: 文件 {file} 缺少必需的时间或能量列，已跳过。")
                    continue

                df['time'] = pd.to_datetime(date_str + ' ' + df[time_col].astype(str), errors='coerce')
                df['energy'] = pd.to_numeric(df[energy_col], errors='coerce')
                
                df.dropna(subset=['time', 'energy'], inplace=True)
                all_data.append(df[['time', 'energy']])

            except Exception as e:
                print(f"处理文件 {file} 时出错: {e}")

    if not all_data:
        # 这个错误就是您之前遇到的
        raise FileNotFoundError(f"在文件夹及其子文件夹中均未找到有效的CSV文件: {ms_folder}")

    merged_df = pd.concat(all_data, ignore_index=True)
    merged_df.sort_values(by='time', inplace=True)
    merged_df.reset_index(drop=True, inplace=True)
    
    print(f"成功合并并排序 {len(merged_df)} 条有效微震记录。")
    return merged_df

def process_data(config, output_dir):
    # ... (此函数其余部分保持不变)
    print("--- 开始简化版微震数据预处理 ---")
    try:
        data_paths = config['data_paths']
        df = merge_microseismic_data(data_paths['ms_folder'])
        if df is None or df.empty:
            raise ValueError("微震数据加载或合并失败，没有生成数据。")
        df.set_index('time', inplace=True)
        df_resampled = df[['energy']].resample('H').sum()
        df_resampled.fillna(0, inplace=True)
        scaler = MinMaxScaler()
        df_resampled['energy_scaled'] = scaler.fit_transform(df_resampled[['energy']])
        time_steps = config.get('time_steps', 24)
        num_timesteps_output = config.get('num_timesteps_output', 1)
        X_samples, y_samples = [], []
        scaled_values = df_resampled['energy_scaled'].values
        for i in range(len(scaled_values) - time_steps - num_timesteps_output + 1):
            X_samples.append(scaled_values[i : i + time_steps])
            y_samples.append(scaled_values[i + time_steps : i + time_steps + num_timesteps_output])
        X = np.array(X_samples).reshape(-1, time_steps, 1)
        y = np.array(y_samples)
        if len(X) == 0:
            raise ValueError("数据不足，无法创建任何训练样本。")
        total_samples = len(X)
        test_ratio = config.get('test_ratio', 0.2)
        val_ratio = config.get('val_ratio', 0.1)
        train_end = int(total_samples * (1 - test_ratio - val_ratio))
        val_end = train_end + int(total_samples * val_ratio)
        data_dict = {
            'train': (X[:train_end], y[:train_end]),
            'val': (X[train_end:val_end], y[train_end:val_end]),
            'test': (X[val_end:], y[val_end:]),
        }
        data_path = os.path.join(output_dir, "data.joblib")
        joblib.dump(data_dict, data_path)
        scaler_path = os.path.join(output_dir, "scaler.joblib")
        joblib.dump(scaler, scaler_path)
        print(f"分割后的数据集已保存到: {data_path}")
        print("--- 数据预处理完成 ---\n")
    except Exception as e:
        print(f"预处理过程中发生严重错误: {e}")
        traceback.print_exc()
        raise
