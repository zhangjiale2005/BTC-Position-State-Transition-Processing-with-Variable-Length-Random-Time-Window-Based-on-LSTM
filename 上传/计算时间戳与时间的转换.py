from datetime import datetime

def timestamp_to_datetime(timestamp):
    """
    将时间戳转换为可读的日期和时间格式。

    参数:
    timestamp (int 或 float): Unix 时间戳。

    返回:
    str: 格式化为 '%Y-%m-%d %H:%M:%S' 的日期和时间字符串。
    """
    dt_object = datetime.fromtimestamp(timestamp)
    return dt_object.strftime('%Y-%m-%d %H:%M:%S')

def datetime_to_timestamp(date_string, date_format='%Y-%m-%d %H:%M:%S'):
    """
    将可读的日期和时间格式转换为时间戳。

    参数:
    date_string (str): 要转换的日期和时间字符串。
    date_format (str): 日期和时间字符串的格式，默认为 '%Y-%m-%d %H:%M:%S'。

    返回:
    int 或 float: 对应的 Unix 时间戳。
    """
    dt_object = datetime.strptime(date_string, date_format)
    return int(dt_object.timestamp())  # 使用 int() 是因为 timestamp() 方法返回 float

# 示例用法
timestamp = 1741730580# 示例时间戳
date_string = "2021-10-01 12:00:00"  # 示例日期和时间字符串


readable_datetime = timestamp_to_datetime(timestamp)
print(f"时间戳 {timestamp} 转换为可读日期和时间: {readable_datetime}")
converted_timestamp = datetime_to_timestamp(date_string)
print(f"可读日期和时间 '{date_string}' 转换为时间戳: {converted_timestamp}")