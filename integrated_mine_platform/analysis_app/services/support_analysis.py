"""
支架阻力小波(DWT)分析服务
包括: DWT分解、事件检测、压力分布分析
"""
import numpy as np
import pandas as pd
import pywt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
from datetime import datetime
from django.utils import timezone
from ..models import SupportResistance
from data_app.models import SupportResistanceData


# 不使用中文字体，全部使用英文
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def dwt_decompose(data, wavelet='db4', level=None):
    """
    离散小波变换(DWT)分解
    
    Args:
        data: 时间序列数据
        wavelet: 小波基函数 (db4, sym4, coif4等)
        level: 分解层数
        
    Returns:
        coeffs: 小波系数列表 [cA_n, cD_n, cD_n-1, ..., cD1]
    """
    if level is None:
        level = pywt.dwt_max_level(len(data), pywt.Wavelet(wavelet).dec_len)
    
    # 执行多层DWT分解
    coeffs = pywt.wavedec(data, wavelet, level=level)
    return coeffs


def reconstruct_from_coeffs(coeffs, wavelet='db4'):
    """
    从小波系数重构信号
    
    Args:
        coeffs: 小波系数
        wavelet: 小波基函数
        
    Returns:
        reconstructed: 重构的信号
    """
    return pywt.waverec(coeffs, wavelet)


def detect_pressure_events(data, threshold_std=2.0):
    """
    检测支架阻力异常事件
    
    Args:
        data: 阻力数据
        threshold_std: 阈值倍数(标准差)
        
    Returns:
        events: 异常事件索引数组
        threshold: 阈值
    """
    # 计算统计量
    mean = np.mean(data)
    std = np.std(data)
    threshold = mean + threshold_std * std
    
    # 检测超过阈值的点
    events = np.where(data > threshold)[0]
    
    return events, threshold


def generate_dwt_analysis_plot(data, timestamps, station_id, wavelet='db4'):
    """
    生成DWT分析图表
    
    Args:
        data: 阻力数据
        timestamps: 时间戳
        station_id: 测站ID
        wavelet: 小波基函数
        
    Returns:
        base64编码的PNG图片
    """
    # DWT分解
    coeffs = dwt_decompose(data, wavelet=wavelet)
    
    # 重构去噪信号
    coeffs_denoised = coeffs.copy()
    coeffs_denoised[1:] = [pywt.threshold(c, np.std(c)/2, mode='soft') for c in coeffs[1:]]
    denoised = reconstruct_from_coeffs(coeffs_denoised, wavelet=wavelet)
    
    # 调整长度
    if len(denoised) > len(data):
        denoised = denoised[:len(data)]
    elif len(denoised) < len(data):
        denoised = np.pad(denoised, (0, len(data) - len(denoised)), mode='edge')
    
    # 检测事件
    events, threshold = detect_pressure_events(denoised)
    
    # 创建图表
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    
    try:
        # 子图1: 原始数据
        axes[0].plot(timestamps, data, 'b-', linewidth=1, label='Original Data')
        axes[0].set_ylabel('Resistance (MPa)', fontsize=12)
        axes[0].set_title(f'{station_id} - Support Resistance DWT Analysis', fontsize=16, weight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 子图2: 小波分解系数
        axes[1].plot(timestamps, denoised, 'g-', linewidth=1.5, label='Denoised Signal')
        axes[1].plot(timestamps, data, 'b-', linewidth=0.5, alpha=0.3, label='Original Data')
        axes[1].set_ylabel('Resistance (MPa)', fontsize=12)
        axes[1].set_title('Wavelet Denoising Result', fontsize=14)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # 子图3: 事件检测
        axes[2].plot(timestamps, denoised, 'g-', linewidth=1, label='Denoised Signal')
        axes[2].axhline(y=threshold, color='r', linestyle='--', linewidth=2, label=f'Threshold ({threshold:.2f} MPa)')
        if len(events) > 0:
            axes[2].scatter(
                [timestamps[i] for i in events],
                [denoised[i] for i in events],
                c='red', s=50, marker='o', label=f'Anomaly Events ({len(events)})'
            )
        axes[2].set_ylabel('Resistance (MPa)', fontsize=12)
        axes[2].set_xlabel('Time', fontsize=12)
        axes[2].set_title('Pressure Event Detection', fontsize=14)
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        # 旋转x轴标签
        for ax in axes:
            ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        # 保存到内存
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=120, bbox_inches='tight')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        return f"data:image/png;base64,{img_base64}"
    finally:
        plt.close(fig)


def generate_pressure_distribution_plot(data, station_id):
    """
    生成压力分布直方图
    
    Args:
        data: 阻力数据
        station_id: 测站ID
        
    Returns:
        base64编码的PNG图片
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    try:
        # 绘制直方图
        n, bins, patches = ax.hist(data, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
        
        # 添加统计线
        mean = np.mean(data)
        median = np.median(data)
        ax.axvline(mean, color='red', linestyle='--', linewidth=2, label=f'Mean ({mean:.2f} MPa)')
        ax.axvline(median, color='green', linestyle='--', linewidth=2, label=f'Median ({median:.2f} MPa)')
        
        ax.set_xlabel('Resistance (MPa)', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title(f'{station_id} - Support Resistance Distribution', fontsize=16, weight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 保存到内存
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=120, bbox_inches='tight')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        return f"data:image/png;base64,{img_base64}"
    finally:
        plt.close(fig)


def get_support_dwt_analysis(station_id, start_date, end_date, wavelet='db4', dataset_id=None):
    """
    获取支架阻力DWT分析结果
    
    Args:
        station_id: 测站ID
        start_date: 开始日期
        end_date: 结束日期
        wavelet: 小波基函数
        dataset_id: 数据集ID（可选，如果指定则使用SupportResistanceData）
        
    Returns:
        分析结果字典
    """
    # 确保日期是timezone-aware的datetime对象
    if isinstance(start_date, str):
        start_date = timezone.make_aware(datetime.strptime(start_date, '%Y-%m-%d'))
    if isinstance(end_date, str):
        end_date = timezone.make_aware(datetime.strptime(end_date, '%Y-%m-%d').replace(hour=23, minute=59, second=59))
    
    # 根据是否指定dataset_id选择不同的数据源
    if dataset_id:
        # 使用上传的数据集
        if dataset_id == 'simulated':
            # 模拟数据集（支架阻力没有is_simulated字段，跳过）
            return {
                'success': False,
                'message': 'Support resistance data does not have simulated datasets',
                'count': 0
            }
        elif dataset_id.isdigit():
            # 上传文件ID
            records = SupportResistanceData.objects.filter(
                station_id=station_id,
                timestamp__gte=start_date,
                timestamp__lte=end_date,
                uploaded_file_id=int(dataset_id)
            ).order_by('timestamp')
            dataset_display_name = f'Uploaded File #{dataset_id}'
        else:
            # source_file名称
            records = SupportResistanceData.objects.filter(
                station_id=station_id,
                timestamp__gte=start_date,
                timestamp__lte=end_date,
                source_file=dataset_id
            ).order_by('timestamp')
            dataset_display_name = dataset_id
    else:
        # 使用分析事件数据（旧方式，保持兼容）
        records = SupportResistance.objects.filter(
            station_id=station_id,
            timestamp__gte=start_date,
            timestamp__lte=end_date
        ).order_by('timestamp')
        dataset_display_name = None
    
    if not records.exists():
        message = f'Station {station_id} has no data in the specified time range'
        if dataset_id:
            message = f'Dataset "{dataset_display_name}" - Station {station_id} has no data in the specified time range'
        return {
            'success': False,
            'message': message,
            'count': 0
        }
    
    # 转换为数组（兼容两种模型的字段名）
    if dataset_id:
        df = pd.DataFrame(list(records.values('timestamp', 'resistance')))
        df.rename(columns={'resistance': 'resistance_value'}, inplace=True)
    else:
        df = pd.DataFrame(list(records.values('timestamp', 'resistance_value')))
    
    data = df['resistance_value'].values
    timestamps = df['timestamp'].tolist()
    
    # DWT分解
    coeffs = dwt_decompose(data, wavelet=wavelet)
    
    # 去噪
    coeffs_denoised = coeffs.copy()
    coeffs_denoised[1:] = [pywt.threshold(c, np.std(c)/2, mode='soft') for c in coeffs[1:]]
    denoised = reconstruct_from_coeffs(coeffs_denoised, wavelet=wavelet)
    
    # 调整长度
    if len(denoised) > len(data):
        denoised = denoised[:len(data)]
    elif len(denoised) < len(data):
        denoised = np.pad(denoised, (0, len(data) - len(denoised)), mode='edge')
    
    # 检测事件
    events, threshold = detect_pressure_events(denoised)
    
    # 生成图表
    dwt_plot = generate_dwt_analysis_plot(data, timestamps, station_id, wavelet)
    distribution_plot = generate_pressure_distribution_plot(data, station_id)
    
    return {
        'success': True,
        'station_id': station_id,
        'dataset_id': dataset_id,
        'dataset_name': dataset_display_name,
        'count': len(data),
        'wavelet': wavelet,
        'dwt_plot': dwt_plot,
        'distribution_plot': distribution_plot,
        'statistics': {
            'mean': float(np.mean(data)),
            'std': float(np.std(data)),
            'min': float(np.min(data)),
            'max': float(np.max(data)),
            'median': float(np.median(data)),
            'event_count': len(events),
            'threshold': float(threshold),
            'decomposition_levels': len(coeffs) - 1,
        },
        'events': [
            {
                'index': int(i),
                'timestamp': timestamps[i].isoformat(),
                'value': float(denoised[i])
            }
            for i in events[:20]  # 返回前20个事件
        ]
    }


def get_wavelet_comparison(station_id, start_date, end_date, wavelets=['db4', 'sym4', 'coif4'], dataset_id=None):
    """
    对比不同小波基函数的分析效果
    
    Args:
        station_id: 测站ID
        start_date: 开始日期
        end_date: 结束日期
        wavelets: 小波基函数列表
        dataset_id: 数据集ID（可选）
        
    Returns:
        对比结果字典
    """
    # 根据 dataset_id 选择数据源
    if dataset_id:
        from data_app.models import SupportResistanceData, UploadedFile
        
        # 判断数据集类型
        if dataset_id == 'simulated':
            # 查询模拟数据
            records = SupportResistanceData.objects.filter(
                is_simulated=True,
                timestamp__gte=start_date,
                timestamp__lte=end_date
            ).order_by('timestamp')
        elif dataset_id.isdigit():
            # 上传的数据集 (uploaded_file ID)
            records = SupportResistanceData.objects.filter(
                uploaded_file_id=int(dataset_id),
                timestamp__gte=start_date,
                timestamp__lte=end_date
            ).order_by('timestamp')
        else:
            # source_file 数据集
            records = SupportResistanceData.objects.filter(
                source_file=dataset_id,
                timestamp__gte=start_date,
                timestamp__lte=end_date
            ).order_by('timestamp')
        
        if not records.exists():
            return {
                'success': False,
                'message': f'Dataset {dataset_id} has no data in the specified time range',
            }
        
        # 转换为数组 (使用 resistance 字段)
        df = pd.DataFrame(list(records.values('timestamp', 'resistance')))
        data = df['resistance'].values
        timestamps = df['timestamp'].tolist()
    else:
        # 分析事件数据 (旧逻辑)
        records = SupportResistance.objects.filter(
            station_id=station_id,
            timestamp__gte=start_date,
            timestamp__lte=end_date
        ).order_by('timestamp')
        
        if not records.exists():
            return {
                'success': False,
                'message': f'测站 {station_id} 在指定时间范围内没有数据',
            }
        
        # 转换为数组
        df = pd.DataFrame(list(records.values('timestamp', 'resistance_value')))
        data = df['resistance_value'].values
        timestamps = df['timestamp'].tolist()
    
    # 对比不同小波
    results = []
    for wavelet in wavelets:
        coeffs = dwt_decompose(data, wavelet=wavelet)
        coeffs_denoised = coeffs.copy()
        coeffs_denoised[1:] = [pywt.threshold(c, np.std(c)/2, mode='soft') for c in coeffs[1:]]
        denoised = reconstruct_from_coeffs(coeffs_denoised, wavelet=wavelet)
        
        # 调整长度
        if len(denoised) > len(data):
            denoised = denoised[:len(data)]
        elif len(denoised) < len(data):
            denoised = np.pad(denoised, (0, len(data) - len(denoised)), mode='edge')
        
        events, threshold = detect_pressure_events(denoised)
        
        results.append({
            'wavelet': wavelet,
            'event_count': len(events),
            'threshold': float(threshold),
            'snr': float(np.mean(denoised) / np.std(data - denoised))  # 信噪比估计
        })
    
    return {
        'success': True,
        'station_id': station_id,
        'count': len(data),
        'comparison': results
    }
