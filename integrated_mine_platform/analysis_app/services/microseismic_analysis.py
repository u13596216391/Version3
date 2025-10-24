"""
微震数据分析服务
包括: 散点图、核密度分析、频次分析、能量分析
"""
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from scipy.interpolate import griddata
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
from datetime import datetime
from django.utils import timezone
from ..models import MicroseismicEvent
from data_app.models import MicroseismicData


# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def calculate_frequency_density(x_coords, y_coords, grid_size=100):
    """
    计算微震频次核密度
    
    Args:
        x_coords: X坐标数组
        y_coords: Y坐标数组
        grid_size: 网格大小
        
    Returns:
        x_grid, y_grid, density: 网格坐标和密度值
    """
    if len(x_coords) < 2:
        return None, None, None
    
    # 创建网格
    x_min, x_max = x_coords.min() - 100, x_coords.max() + 100
    y_min, y_max = y_coords.min() - 50, y_coords.max() + 50
    
    x_grid = np.linspace(x_min, x_max, grid_size)
    y_grid = np.linspace(y_min, y_max, grid_size)
    X, Y = np.meshgrid(x_grid, y_grid)
    
    # 核密度估计
    try:
        positions = np.vstack([X.ravel(), Y.ravel()])
        values = np.vstack([x_coords, y_coords])
        kernel = gaussian_kde(values)
        density = np.reshape(kernel(positions).T, X.shape)
        return X, Y, density
    except Exception as e:
        print(f"核密度计算错误: {e}")
        return X, Y, np.zeros_like(X)


def calculate_energy_density(x_coords, y_coords, energy_values, grid_size=100):
    """
    计算微震能量核密度
    
    Args:
        x_coords: X坐标数组
        y_coords: Y坐标数组
        energy_values: 能量值数组
        grid_size: 网格大小
        
    Returns:
        x_grid, y_grid, density: 网格坐标和能量密度值
    """
    if len(x_coords) < 3:
        return None, None, None
    
    # 创建网格
    x_min, x_max = x_coords.min() - 100, x_coords.max() + 100
    y_min, y_max = y_coords.min() - 50, y_coords.max() + 50
    
    x_grid = np.linspace(x_min, x_max, grid_size)
    y_grid = np.linspace(y_min, y_max, grid_size)
    X, Y = np.meshgrid(x_grid, y_grid)
    
    # 使用griddata进行插值
    try:
        points = np.column_stack([x_coords, y_coords])
        density = griddata(points, energy_values, (X, Y), method='cubic', fill_value=0)
        return X, Y, density
    except Exception as e:
        print(f"能量密度计算错误: {e}")
        return X, Y, np.zeros_like(X)


def generate_scatter_plot(event_df, auxiliary_lines=None, title='Microseismic Event Scatter Plot'):
    """
    生成微震散点图
    
    Args:
        event_df: 微震事件DataFrame (需包含x_coord, y_coord列)
        auxiliary_lines: 辅助线列表 [{coords: [[x1,y1], [x2,y2]], color: 'red', name: 'Working Face'}]
        title: 图表标题
        
    Returns:
        base64编码的PNG图片
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    
    try:
        # 绘制散点
        if not event_df.empty:
            scatter = ax.scatter(
                event_df['x_coord'], 
                event_df['y_coord'],
                c='blue', 
                s=20, 
                alpha=0.6,
                label='Microseismic Events'
            )
        
        # 绘制辅助线
        if auxiliary_lines:
            for line in auxiliary_lines:
                coords = line["coords"]
                x_coords = [coords[0][0], coords[1][0]]
                y_coords = [coords[0][1], coords[1][1]]
                label = line.get("name")
                ax.plot(
                    x_coords, y_coords,
                    linestyle='--',
                    color=line.get("color", "red"),
                    linewidth=2.5,
                    label=label
                )
        
        # 设置图表样式
        ax.set_xlabel("X Coordinate (m)", fontsize=12)
        ax.set_ylabel("Y Coordinate (m)", fontsize=12)
        ax.set_title(title, fontsize=16, weight='bold')
        ax.legend(loc="upper left")
        ax.grid(True, linestyle='--', alpha=0.6)
        
        # 保存到内存
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=120, bbox_inches='tight')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        return f"data:image/png;base64,{img_base64}"
    finally:
        plt.close(fig)


def generate_density_plot(x_grid, y_grid, density, event_df, auxiliary_lines=None, title='Microseismic Density Map'):
    """
    生成核密度图
    
    Args:
        x_grid, y_grid: 网格坐标
        density: 密度值
        event_df: 微震事件DataFrame
        auxiliary_lines: 辅助线列表
        title: 图表标题
        
    Returns:
        base64编码的PNG图片
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    
    try:
        # 绘制核密度热力图
        if density is not None and density.any():
            contour = ax.contourf(x_grid, y_grid, density, levels=100, cmap='jet', alpha=0.8)
            plt.colorbar(contour, ax=ax, label='Density')
        
        # 绘制散点
        if not event_df.empty:
            ax.scatter(
                event_df['x_coord'],
                event_df['y_coord'],
                c='black',
                s=10,
                alpha=0.5,
                label='Microseismic Events'
            )
        
        # 绘制辅助线
        if auxiliary_lines:
            for line in auxiliary_lines:
                coords = line["coords"]
                x_coords = [coords[0][0], coords[1][0]]
                y_coords = [coords[0][1], coords[1][1]]
                label = line.get("name")
                ax.plot(
                    x_coords, y_coords,
                    linestyle='--',
                    color=line.get("color", "white"),
                    linewidth=2.5,
                    label=label
                )
        
        # 设置图表样式
        ax.set_xlabel("X Coordinate (m)", fontsize=12)
        ax.set_ylabel("Y Coordinate (m)", fontsize=12)
        ax.set_title(title, fontsize=16, weight='bold')
        ax.legend(loc="upper left")
        ax.grid(True, linestyle='--', alpha=0.6)
        
        # 保存到内存
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=120, bbox_inches='tight')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        return f"data:image/png;base64,{img_base64}"
    finally:
        plt.close(fig)


def get_microseismic_analysis(start_date, end_date, analysis_type='frequency', auxiliary_lines=None, dataset_id=None):
    """
    获取微震分析数据
    
    Args:
        start_date: 开始日期
        end_date: 结束日期
        analysis_type: 分析类型 ('frequency' 或 'energy')
        auxiliary_lines: 辅助线配置
        dataset_id: 数据集ID（可选，source_file名称或uploaded_file_id，如果指定则使用MicroseismicData，否则使用MicroseismicEvent）
        
    Returns:
        分析结果字典
    """
    # 确保日期是timezone-aware的datetime对象
    from datetime import datetime
    if isinstance(start_date, str):
        start_date = timezone.make_aware(datetime.strptime(start_date, '%Y-%m-%d'))
    if isinstance(end_date, str):
        # 结束日期设为当天的23:59:59
        end_date = timezone.make_aware(datetime.strptime(end_date, '%Y-%m-%d').replace(hour=23, minute=59, second=59))
    
    # 根据是否指定dataset_id选择不同的数据源
    if dataset_id:
        # 使用上传的数据集
        # 支持三种数据集ID: 'simulated', source_file名称, 或uploaded_file的ID
        if dataset_id == 'simulated':
            # 模拟数据集
            events = MicroseismicData.objects.filter(
                timestamp__gte=start_date,
                timestamp__lte=end_date,
                is_simulated=True
            )
            dataset_display_name = '模拟数据集'
        elif dataset_id.isdigit():
            # 上传文件ID
            events = MicroseismicData.objects.filter(
                timestamp__gte=start_date,
                timestamp__lte=end_date,
                uploaded_file_id=int(dataset_id)
            )
            dataset_display_name = f'上传文件#{dataset_id}'
        else:
            # source_file名称
            events = MicroseismicData.objects.filter(
                timestamp__gte=start_date,
                timestamp__lte=end_date,
                source_file=dataset_id
            )
            dataset_display_name = dataset_id
        
        if not events.exists():
            return {
                'success': False,
                'message': f'数据集 "{dataset_display_name}" 在指定时间范围内没有数据',
                'count': 0
            }
        
        # 转换为DataFrame
        df = pd.DataFrame(list(events.values('timestamp', 'event_x', 'event_y', 'event_z', 'energy', 'magnitude')))
        df.rename(columns={'event_x': 'x_coord', 'event_y': 'y_coord', 'event_z': 'z_coord'}, inplace=True)
        
    else:
        # 使用分析事件数据
        events = MicroseismicEvent.objects.filter(
            timestamp__gte=start_date,
            timestamp__lte=end_date,
            data_type=analysis_type
        )
        
        if not events.exists():
            return {
                'success': False,
                'message': '指定时间范围内没有数据',
                'count': 0
            }
        
        # 转换为DataFrame
        df = pd.DataFrame(list(events.values('timestamp', 'x_coord', 'y_coord', 'energy', 'magnitude')))
        dataset_display_name = None
    
    # 生成散点图
    scatter_img = generate_scatter_plot(
        df,
        auxiliary_lines=auxiliary_lines,
        title=f'Microseismic Scatter Plot ({start_date.strftime("%Y-%m-%d")} ~ {end_date.strftime("%Y-%m-%d")})'
    )
    
    # 计算密度
    if analysis_type == 'frequency':
        x_grid, y_grid, density = calculate_frequency_density(
            df['x_coord'].values,
            df['y_coord'].values
        )
        density_title = f'Microseismic Frequency Density ({start_date.strftime("%Y-%m-%d")} ~ {end_date.strftime("%Y-%m-%d")})'
    else:
        x_grid, y_grid, density = calculate_energy_density(
            df['x_coord'].values,
            df['y_coord'].values,
            df['energy'].values
        )
        density_title = f'Microseismic Energy Density ({start_date.strftime("%Y-%m-%d")} ~ {end_date.strftime("%Y-%m-%d")})'
    
    # 生成密度图
    density_img = None
    if x_grid is not None:
        density_img = generate_density_plot(
            x_grid, y_grid, density, df,
            auxiliary_lines=auxiliary_lines,
            title=density_title
        )
    
    return {
        'success': True,
        'count': len(df),
        'analysis_type': analysis_type,
        'dataset_id': dataset_id,
        'dataset_name': dataset_display_name,
        'scatter_plot': scatter_img,
        'density_plot': density_img,
        'statistics': {
            'total_events': len(df),
            'x_range': [float(df['x_coord'].min()), float(df['x_coord'].max())],
            'y_range': [float(df['y_coord'].min()), float(df['y_coord'].max())],
            'avg_energy': float(df['energy'].mean()) if 'energy' in df.columns and df['energy'].notna().any() else None,
            'max_energy': float(df['energy'].max()) if 'energy' in df.columns and df['energy'].notna().any() else None,
        }
    }
