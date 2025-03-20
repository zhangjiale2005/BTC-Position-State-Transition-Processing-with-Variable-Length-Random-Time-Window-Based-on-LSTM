from __future__ import print_function
import time
from datetime import datetime
import json
import ast

#输入开始时间结束时间，以二维列表输出这个时间段内所有的k线数据
timestamp = time.time()
timestamp = round(timestamp, 3)
timestamp1 = round(timestamp, 4)
timestamp2=str(timestamp1*1000)[:-2]#时间戳



import gate_api
from gate_api.exceptions import ApiException, GateApiException
# Defining the host is optional and defaults to https://api.gateio.ws/api/v4
# See configuration.py for a list of all supported configuration parameters.
configuration = gate_api.Configuration(
    host = "https://api.gateio.ws/api/v4"
)

import bisect
from sortedcontainers import SortedList



def datetime_to_timestamp(date_string, date_format='%Y-%m-%d %H:%M:%S'):
    dt_object = datetime.strptime(date_string, date_format)
    return int(dt_object.timestamp())  # 使用 int() 是因为 timestamp() 方法返回 float
data=[]


date_string = "2025-01-01 12:00:00"#input("输入起始时间，如             2021-10-01 12:00:00")  # 示例日期和时间字符串
converted_timestamp = datetime_to_timestamp(date_string)
oo=converted_timestamp
current=int(time.time())-60*60#datetime_to_timestamp(input("输入结束时间，如           2021-12-01 12:00:00"))
rt=int((current-converted_timestamp)/60)
result=[]
api_client = gate_api.ApiClient(configuration)
# Create an instance of the API class
api_instance = gate_api.FuturesApi(api_client)
settle = 'usdt'  # str | Settle currency
contract = 'BTC_USDT'  # str | Futures contract
interval = '1m'
keys_order = ['c', 'h', 'l', 'o', 'sum', 't', 'v']
nnnn=int(((current-converted_timestamp)%(2001*60))/60)
qyc=int(nnnn)
wwww=int((current-converted_timestamp)/(2001*60))
#最后一次添加到全局列表
pp=0
while converted_timestamp < current:

    _from = converted_timestamp  # int | Start time of candlesticks, formatted in Unix timestamp in seconds. Default to`to - 100 * interval` if not specified (optional)
    to = converted_timestamp+60*2000 # int | End time of candlesticks, formatted in Unix timestamp in seconds. Default to current time (optional)
    #limit = 100  # int | Maximum recent data points to return. `limit` is conflicted with `from` and `to`. If either `from` or `to` is specified, request will be rejected. (optional) (default to 100)
      # str | Interval time between data points. Note that `1w` means natual week(Mon-Sun), while `7d` means every 7d since unix 0.  Note that 30d means 1 natual month, not 30 days (optional) (default to '5m')
    api_response = api_instance.list_futures_candlesticks(settle, contract, _from=_from,to=to, interval=interval)
    #print(api_response)
    data = api_response
    # 处理数据
    # 输出结果

    if pp !=wwww:
        result.extend(data)
        converted_timestamp += 60 * 2001
        pp += 1
    else:
        index = nnnn  # 假设你想截取索引3之后的所有元素
        r = data[:index]  # 注意，切片是左闭右开的，所以要加1来包含index之后的元素
# 输出: [4, 5, 6]
        result.extend(r)
        converted_timestamp += 60 * 2001








#print(data)
def parse_to_dict(s):
    try:
        # 尝试解析字符串为字典
        return ast.literal_eval(s)
    except (ValueError, SyntaxError):
        # 解析失败则返回 None
        return None

# 处理原始数据，仅保留有效字典
after1 = list(map(str, result))

parsed_data = []
for item in after1:
    if isinstance(item, str):
        parsed = parse_to_dict(item)
        if isinstance(parsed, dict):  # 确保解析后是字典
            parsed_data.append(parsed)
    elif isinstance(item, dict):
        # 如果已经是字典，直接保留
        parsed_data.append(item)

# 示例用法



# 提取值的二维列表
result = [
    [d[key] for key in keys_order]
    for d in parsed_data
    if isinstance(d, dict) and all(key in d for key in keys_order)
]
T=[]
g=0


while g<len(result):
    finall = [[float(item) for item in row] for row in result]
    # 提取五个特征列（假设是位置1到5，对应iloc[:,1:6]）
    g += 1
#未截取

#finall为处理后的最终数据
# for row in finall:#截取操作
#     del row[6]
#    # del row[5]
print(finall)
#截取后直接拿来训练

