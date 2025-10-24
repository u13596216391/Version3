"""
数据文件解析服务
支持CSV和ZIP文件的微震和支架阻力数据解析
"""
import pandas as pd
import zipfile
import os
import io
from datetime import datetime
from django.core.files.uploadedfile import InMemoryUploadedFile
from .models import MicroseismicData, SupportResistanceData, UploadedFile


def parse_microseismic_csv(file_content, source_filename='', uploaded_file=None):
    """
    解析微震数据CSV文件 - 支持多种格式
    
    支持的格式:
    1. Event格式: Event_ID, Event_X, Event_Y, Event_Z, Event_Energy, Locate_Mw, Event_Date, Event_Time
    2. 简单格式: timestamp, longitude, latitude, depth, energy, magnitude
    3. 混合格式: 自动检测列名
    
    Args:
        file_content: 文件内容（字节或文件对象）
        source_filename: 源文件名
        uploaded_file: UploadedFile实例
        
    Returns:
        (success, count, error_message)
    """
    try:
        # 读取CSV
        if isinstance(file_content, bytes):
            df = pd.read_csv(io.BytesIO(file_content), encoding='utf-8')
        else:
            df = pd.read_csv(file_content, encoding='utf-8')
        
        # 列名映射字典
        column_mapping = {
            'event_id': ['event_id', 'Event_ID', 'eventid', 'id', '事件ID'],
            'event_x': ['event_x', 'Event_X', 'x', 'X', 'longitude', '经度', 'x_coord'],
            'event_y': ['event_y', 'Event_Y', 'y', 'Y', 'latitude', '纬度', 'y_coord'],
            'event_z': ['event_z', 'Event_Z', 'z', 'Z', 'depth', '深度', 'z_coord'],
            'energy': ['energy', 'Energy', 'Event_Energy', '能量'],
            'magnitude': ['magnitude', 'Magnitude', 'mag', 'Mag', '震级'],
            'locate_mw': ['locate_mw', 'Locate_Mw', 'Mw', 'mw'],
            'locate_err': ['locate_err', 'Locate_Err', 'error', 'Error', '误差'],
            'velocity': ['velocity', 'Velocity', 'vel', '速度'],
            'event_date': ['event_date', 'Event_Date', 'date', 'Date', '日期'],
            'event_time': ['event_time', 'Event_Time', 'time', 'Time', '时间'],
            'timestamp': ['timestamp', 'Timestamp', 'datetime', 'Datetime', '时间戳'],
        }
        
        # 查找匹配的列
        def find_column(field_name):
            for col in column_mapping.get(field_name, []):
                if col in df.columns:
                    return col
            return None
        
        # 检查必需列（X和Y至少要有）
        x_col = find_column('event_x')
        y_col = find_column('event_y')
        
        if not x_col or not y_col:
            return False, 0, f'缺少必需的坐标列。找到的列: {list(df.columns)}'
        
        # 获取其他列
        z_col = find_column('event_z')
        event_id_col = find_column('event_id')
        energy_col = find_column('energy')
        magnitude_col = find_column('magnitude')
        locate_mw_col = find_column('locate_mw')
        locate_err_col = find_column('locate_err')
        velocity_col = find_column('velocity')
        timestamp_col = find_column('timestamp')
        event_date_col = find_column('event_date')
        event_time_col = find_column('event_time')
        
        # 准备数据
        records = []
        for idx, row in df.iterrows():
            try:
                # 解析时间戳
                timestamp = None
                
                # 方法1: 直接的timestamp列
                if timestamp_col and pd.notna(row[timestamp_col]):
                    try:
                        timestamp = pd.to_datetime(row[timestamp_col])
                    except:
                        pass
                
                # 方法2: 组合Event_Date和Event_Time
                if not timestamp and event_date_col and event_time_col:
                    if pd.notna(row[event_date_col]) and pd.notna(row[event_time_col]):
                        try:
                            date_str = str(row[event_date_col])
                            time_str = str(row[event_time_col])
                            datetime_str = f"{date_str} {time_str}"
                            timestamp = pd.to_datetime(datetime_str)
                        except:
                            pass
                
                # 方法3: 只有日期
                if not timestamp and event_date_col and pd.notna(row[event_date_col]):
                    try:
                        timestamp = pd.to_datetime(row[event_date_col])
                    except:
                        pass
                
                # 默认当前时间
                if not timestamp:
                    timestamp = datetime.now()
                
                # 创建记录
                record = MicroseismicData(
                    timestamp=timestamp,
                    event_id=str(row[event_id_col]) if event_id_col and pd.notna(row[event_id_col]) else None,
                    event_x=float(row[x_col]),
                    event_y=float(row[y_col]),
                    event_z=float(row[z_col]) if z_col and pd.notna(row[z_col]) else None,
                    energy=float(row[energy_col]) if energy_col and pd.notna(row[energy_col]) else None,
                    magnitude=float(row[magnitude_col]) if magnitude_col and pd.notna(row[magnitude_col]) else None,
                    locate_mw=float(row[locate_mw_col]) if locate_mw_col and pd.notna(row[locate_mw_col]) else None,
                    locate_err=float(row[locate_err_col]) if locate_err_col and pd.notna(row[locate_err_col]) else None,
                    velocity=float(row[velocity_col]) if velocity_col and pd.notna(row[velocity_col]) else None,
                    source_file=source_filename,
                    uploaded_file=uploaded_file,
                    is_simulated=False
                )
                records.append(record)
            except Exception as e:
                print(f"跳过行 {idx}: {e}")
                continue
        
        # 批量创建
        if records:
            MicroseismicData.objects.bulk_create(records, batch_size=500)
            return True, len(records), None
        else:
            return False, 0, '没有有效的数据行'
    
    except Exception as e:
        return False, 0, f'解析错误: {str(e)}'


def parse_support_resistance_csv(file_content, source_filename='', uploaded_file=None):
    """
    解析支架阻力数据CSV文件
    
    预期CSV格式:
    - 必需列: Station_ID, Resistance (或 station_id, resistance, 阻力值, 站号, 支架阻力)
    - 可选列: Time, Timestamp, Pressure_Level, 进尺
    
    Args:
        file_content: 文件内容
        source_filename: 源文件名
        uploaded_file: UploadedFile实例
        
    Returns:
        (success, count, error_message)
    """
    try:
        # 读取CSV，尝试不同的编码
        if isinstance(file_content, bytes):
            try:
                df = pd.read_csv(io.BytesIO(file_content), encoding='utf-8')
            except UnicodeDecodeError:
                df = pd.read_csv(io.BytesIO(file_content), encoding='gbk')
        else:
            df = pd.read_csv(file_content)
        
        # 检查必需列 - 支持更多列名变体
        station_col = next((col for col in df.columns if col.strip() in ['站号', '测站', 'Station_ID', 'station_id', 'StationID', 'stationid', 'station', 'Station']), None)
        resistance_col = next((col for col in df.columns if col.strip() in ['支架阻力', '阻力值', '阻力', 'Resistance', 'resistance', 'value', 'Value']), None)
        
        if not station_col or not resistance_col:
            return False, 0, f'缺少必需的列。找到的列: {", ".join(df.columns)}'
        
        # 时间列
        time_col = next((col for col in df.columns if col.strip() in ['时间', 'Time', 'time', 'Timestamp', 'timestamp', 'DateTime', 'datetime']), None)
        pressure_col = next((col for col in df.columns if col.strip() in ['压力等级', 'Pressure_Level', 'pressure_level', 'Level', 'level']), None)
        
        # 准备数据
        records = []
        for idx, row in df.iterrows():
            try:
                # 解析时间
                if time_col and pd.notna(row[time_col]):
                    try:
                        timestamp = pd.to_datetime(row[time_col])
                    except:
                        timestamp = datetime.now()
                else:
                    timestamp = datetime.now()
                
                record = SupportResistanceData(
                    timestamp=timestamp,
                    station_id=str(row[station_col]),
                    resistance=float(row[resistance_col]),
                    pressure_level=str(row[pressure_col]) if pressure_col and pd.notna(row[pressure_col]) else None,
                    source_file=source_filename,
                    uploaded_file=uploaded_file
                )
                records.append(record)
            except Exception as e:
                print(f"跳过行 {idx}: {e}")
                continue
        
        # 批量创建
        if records:
            SupportResistanceData.objects.bulk_create(records, batch_size=500)
            return True, len(records), None
        else:
            return False, 0, '没有有效的数据行'
    
    except Exception as e:
        return False, 0, f'解析错误: {str(e)}'


def parse_zip_file(zip_file, data_type, uploaded_file=None):
    """
    解析ZIP压缩包中的CSV文件
    
    Args:
        zip_file: ZIP文件对象或字节内容
        data_type: 数据类型 ('microseismic' 或 'support_resistance')
        uploaded_file: UploadedFile实例
        
    Returns:
        (success, total_count, file_count, error_message)
    """
    try:
        # 读取ZIP文件
        if isinstance(zip_file, bytes):
            zip_content = io.BytesIO(zip_file)
        else:
            zip_content = zip_file
        
        total_count = 0
        file_count = 0
        errors = []
        
        with zipfile.ZipFile(zip_content, 'r') as zf:
            # 遍历ZIP中的所有文件
            csv_files = [name for name in zf.namelist() if name.lower().endswith('.csv')]
            
            if not csv_files:
                return False, 0, 0, 'ZIP文件中没有找到CSV文件'
            
            for csv_filename in csv_files:
                try:
                    # 读取CSV文件内容
                    with zf.open(csv_filename) as csv_file:
                        csv_content = csv_file.read()
                        
                        # 根据数据类型解析
                        if data_type == 'microseismic':
                            success, count, error = parse_microseismic_csv(
                                csv_content,
                                source_filename=csv_filename,
                                uploaded_file=uploaded_file
                            )
                        else:  # support_resistance
                            success, count, error = parse_support_resistance_csv(
                                csv_content,
                                source_filename=csv_filename,
                                uploaded_file=uploaded_file
                            )
                        
                        if success:
                            total_count += count
                            file_count += 1
                        else:
                            errors.append(f'{csv_filename}: {error}')
                
                except Exception as e:
                    errors.append(f'{csv_filename}: {str(e)}')
                    continue
        
        if file_count > 0:
            error_msg = '\n'.join(errors) if errors else None
            return True, total_count, file_count, error_msg
        else:
            return False, 0, 0, '所有CSV文件解析失败: ' + '\n'.join(errors)
    
    except Exception as e:
        return False, 0, 0, f'ZIP文件解析错误: {str(e)}'


def process_uploaded_file(uploaded_file, data_type, file_obj):
    """
    处理上传的文件
    
    Args:
        uploaded_file: UploadedFile模型实例
        data_type: 数据类型
        file_obj: Django上传的文件对象
        
    Returns:
        (success, count, error_message)
    """
    try:
        file_content = file_obj.read()
        filename = uploaded_file.filename
        
        if filename.lower().endswith('.csv'):
            # 解析CSV
            if data_type == 'microseismic':
                success, count, error = parse_microseismic_csv(
                    file_content,
                    source_filename=filename,
                    uploaded_file=uploaded_file
                )
            else:
                success, count, error = parse_support_resistance_csv(
                    file_content,
                    source_filename=filename,
                    uploaded_file=uploaded_file
                )
            
            return success, count, error
        
        elif filename.lower().endswith('.zip'):
            # 解析ZIP
            success, total_count, file_count, error = parse_zip_file(
                file_content,
                data_type,
                uploaded_file=uploaded_file
            )
            
            return success, total_count, error
        
        else:
            return False, 0, '不支持的文件格式'
    
    except Exception as e:
        return False, 0, f'文件处理错误: {str(e)}'
