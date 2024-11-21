import os
import pandas as pd
import re
import logging

# 配置日志
logging.basicConfig(
    filename='GlucoseBITProcess.log',
    filemode='w',
    level=logging.WARNING,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# 定义标签函数
def assign_label(total_minutes):
    if 0 <= total_minutes < 10 or total_minutes >= 65:
        return 'low'
    elif 10 <= total_minutes < 25 or 45 <= total_minutes < 65:
        return 'medium'
    elif 25 <= total_minutes < 45:
        return 'high'
    else:
        return 'low'  # 默认归为低

# 定义数据目录
data_dir = '../../data/GlucoseBIT'  # 请根据实际路径调整

# 初始化一个空的列表来收集数据
data_list = []

# 定义辅助函数来识别行类型
def is_time_line(line):
    return ':' in line

def is_frequency_line(line):
    return 'hz' in line.lower()

def is_amp_phase_line(line):
    parts = line.split()
    if len(parts) != 2:
        return False
    try:
        float(parts[0])
        float(parts[1])
        return True
    except ValueError:
        return False

# 遍历每个日期文件夹
for date_folder in os.listdir(data_dir):
    date_path = os.path.join(data_dir, date_folder)
    if os.path.isdir(date_path):
        # 获取所有txt文件，并按段号排序
        try:
            txt_files = sorted(
                [f for f in os.listdir(date_path) if f.endswith('.txt')],
                key=lambda x: int(x.split('-')[0])
            )
        except ValueError as ve:
            logging.warning(f"在排序文件时出错: {ve}，文件夹: {date_path}")
            continue
        
        # 初始化累计时间在日期文件夹级别
        cumulative_time = 0  # 以分钟为单位
        
        for txt_file in txt_files:
            # 从文件名中提取段号和持续时间
            match = re.match(r'(\d+)-(\d+)\.txt', txt_file)
            if match:
                segment_num = int(match.group(1))
                duration_min = int(match.group(2))  # 持续时间，以分钟为单位
            else:
                logging.warning(f"文件名格式不正确: {txt_file}")
                continue
            
            txt_path = os.path.join(date_path, txt_file)
            
            try:
                with open(txt_path, 'r') as file:
                    lines = file.readlines()
            except Exception as e:
                logging.warning(f"无法读取文件 {txt_path}: {e}")
                continue
            
            # 预处理：移除空行和仅包含空白字符的行
            lines = [line.strip() for line in lines if line.strip()]
            
            # 初始化状态
            state = 'expect_time'
            current_data = {}
            first_error_logged = False  # 每个文件单独跟踪
            last_amp_phase = None  # 用于检测重复的幅值和相位行
            
            idx = 0
            while idx < len(lines):
                line = lines[idx]
                if state == 'expect_time':
                    if is_time_line(line):
                        # 解析时间
                        time_str = line
                        time_parts = time_str.split(':')
                        if len(time_parts) != 3:
                            error_msg = f"文件 {txt_path} 中第 {idx+1} 行时间部分分割错误: '{time_str}'，跳过该数据单元。"
                            logging.warning(error_msg)
                            if not first_error_logged:
                                print(error_msg)
                                first_error_logged = True
                            idx += 1
                            continue
                        try:
                            hours = int(time_parts[0])
                            minutes = int(time_parts[1])
                            seconds = int(time_parts[2])
                            time_sec = hours * 3600 + minutes * 60 + seconds
                            time_min = time_sec / 60  # 转换为分钟
                            # 计算总时间
                            total_time_min = cumulative_time + time_min
                            total_time_sec = total_time_min * 60
                            current_data['time_sec'] = total_time_sec
                            current_data['time_min'] = total_time_min
                            state = 'expect_freq'
                            idx += 1
                        except ValueError as ve:
                            error_msg = f"文件 {txt_path} 中第 {idx+1} 行时间解析错误: '{time_str}'，跳过该数据单元。"
                            logging.warning(error_msg)
                            if not first_error_logged:
                                print(error_msg)
                                first_error_logged = True
                            idx += 1
                            continue
                    elif is_amp_phase_line(line):
                        # 检测到意外的幅值相位行，保留第一条，跳过后续重复的行
                        if last_amp_phase == line:
                            # 重复行，跳过
                            error_msg = f"文件 {txt_path} 中第 {idx+1} 行检测到重复的幅值和相位行: '{line}'，跳过该数据单元。"
                            logging.warning(error_msg)
                            if not first_error_logged:
                                print(error_msg)
                                first_error_logged = True
                            idx += 1
                            continue
                        else:
                            # 保留第一条幅值相位行，但由于缺少时间和频率信息，无法处理，跳过
                            last_amp_phase = line
                            error_msg = f"文件 {txt_path} 中第 {idx+1} 行幅值和相位行缺少时间和频率信息，跳过该数据单元。"
                            logging.warning(error_msg)
                            if not first_error_logged:
                                print(error_msg)
                                first_error_logged = True
                            idx += 1
                            continue
                    else:
                        # 其他格式错误的行，跳过
                        error_msg = f"文件 {txt_path} 中第 {idx+1} 行时间格式不正确: '{line}'，跳过该数据单元。"
                        logging.warning(error_msg)
                        if not first_error_logged:
                            print(error_msg)
                            first_error_logged = True
                        idx += 1
                        continue
                elif state == 'expect_freq':
                    if is_frequency_line(line):
                        # 解析频率
                        freq_str = line.lower().replace('hz', '').strip()
                        try:
                            freq = float(freq_str)
                            current_data['frequency_hz'] = freq
                            state = 'expect_amp_phase'
                            idx += 1
                        except ValueError as ve:
                            error_msg = f"文件 {txt_path} 中第 {idx+1} 行频率解析错误: '{line}'，跳过该数据单元。"
                            logging.warning(error_msg)
                            if not first_error_logged:
                                print(error_msg)
                                first_error_logged = True
                            # 尝试重新同步
                            state = 'expect_time'
                            current_data = {}
                            idx += 1
                            continue
                    elif is_amp_phase_line(line):
                        # 检测到意外的幅值相位行，跳过并尝试重新同步
                        if last_amp_phase == line:
                            error_msg = f"文件 {txt_path} 中第 {idx+1} 行检测到重复的幅值和相位行: '{line}'，跳过该数据单元。"
                            logging.warning(error_msg)
                            if not first_error_logged:
                                print(error_msg)
                                first_error_logged = True
                            idx += 1
                            continue
                        else:
                            last_amp_phase = line
                            error_msg = f"文件 {txt_path} 中第 {idx+1} 行幅值和相位行缺少频率信息，跳过该数据单元。"
                            logging.warning(error_msg)
                            if not first_error_logged:
                                print(error_msg)
                                first_error_logged = True
                            idx += 1
                            continue
                    else:
                        # 其他格式错误的行，跳过并尝试重新同步
                        error_msg = f"文件 {txt_path} 中第 {idx+1} 行频率格式不正确: '{line}'，跳过该数据单元。"
                        logging.warning(error_msg)
                        if not first_error_logged:
                            print(error_msg)
                            first_error_logged = True
                        # 尝试重新同步
                        state = 'expect_time'
                        current_data = {}
                        idx += 1
                        continue
                elif state == 'expect_amp_phase':
                    if is_amp_phase_line(line):
                        # 检查是否为重复幅值相位行
                        if last_amp_phase == line:
                            # 重复行，跳过
                            error_msg = f"文件 {txt_path} 中第 {idx+1} 行检测到重复的幅值和相位行: '{line}'，跳过该数据单元。"
                            logging.warning(error_msg)
                            if not first_error_logged:
                                print(error_msg)
                                first_error_logged = True
                            idx += 1
                            continue
                        else:
                            # 解析幅值和相位
                            amp_phase = line.split()
                            try:
                                amp, phase = map(float, amp_phase)
                                current_data['amplitude'] = amp
                                current_data['phase'] = phase
                                
                                # 分配标签
                                label = assign_label(current_data['time_min'])
                                
                                # 添加到列表
                                data_list.append({
                                    'date': date_folder,
                                    'segment': segment_num,
                                    'time_sec': current_data.get('time_sec', 0),
                                    'time_min': current_data.get('time_min', 0),
                                    'frequency_hz': current_data.get('frequency_hz', 0),
                                    'amplitude': current_data.get('amplitude', 0),
                                    'phase': current_data.get('phase', 0),
                                    'label': label
                                })
                                
                                # 更新最后一个幅值相位行
                                last_amp_phase = line
                                
                                # 重置状态
                                state = 'expect_time'
                                current_data = {}
                                idx += 1
                            except ValueError as ve:
                                error_msg = f"文件 {txt_path} 中第 {idx+1} 行幅值和相位解析错误: '{line}'，跳过该数据单元。"
                                logging.warning(error_msg)
                                if not first_error_logged:
                                    print(error_msg)
                                    first_error_logged = True
                                # 尝试重新同步
                                state = 'expect_time'
                                current_data = {}
                                idx += 1
                                continue
                    elif is_time_line(line):
                        # 意外遇到时间行，可能缺少幅值相位行，跳过并重新同步
                        error_msg = f"文件 {txt_path} 中第 {idx+1} 行意外的时间行: '{line}'，跳过该数据单元。"
                        logging.warning(error_msg)
                        if not first_error_logged:
                            print(error_msg)
                            first_error_logged = True
                        # 重置状态并处理该行作为新的时间行
                        state = 'expect_time'
                        current_data = {}
                        # 不增加idx，这样可以重新处理当前行
                        continue
                    else:
                        # 其他格式错误的行，跳过并尝试重新同步
                        error_msg = f"文件 {txt_path} 中第 {idx+1} 行幅值和相位格式不正确: '{line}'，跳过该数据单元。"
                        logging.warning(error_msg)
                        if not first_error_logged:
                            print(error_msg)
                            first_error_logged = True
                        # 尝试重新同步
                        state = 'expect_time'
                        current_data = {}
                        idx += 1
                        continue
            
            # 累加持续时间
            cumulative_time += duration_min

# 将列表转换为DataFrame
all_data = pd.DataFrame(data_list)

# 检查DataFrame是否为空
if all_data.empty:
    print("数据集为空，请检查数据源。")
else:
    # 将标签转换为数字
    label_mapping = {'low': 0, 'medium': 1, 'high': 2}
    all_data['label'] = all_data['label'].map(label_mapping)
    
    # 保存为CSV
    output_csv = '../../data/GlucoseBIT/glucose_data.csv'
    all_data.to_csv(output_csv, index=False)
    print(f"数据已成功保存到 {output_csv}")
    print(f"总解析的数据单元数: {len(all_data)}")
