from __future__ import print_function
import time
from datetime import datetime
import json
import ast
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



def datetime_to_timestamp(date_string, date_format='%Y-%m-%d %H:%M:%S'):
    dt_object = datetime.strptime(date_string, date_format)
    return int(dt_object.timestamp())  # 使用 int() 是因为 timestamp() 方法返回 float

date_string = "2021-10-01 12:00:00"#input("输入起始时间，如2021-10-01 12:00:00")  # 示例日期和时间字符串
converted_timestamp = datetime_to_timestamp(date_string)
print(f"转换为时间戳: {converted_timestamp}")

api_client = gate_api.ApiClient(configuration)
# Create an instance of the API class
api_instance = gate_api.FuturesApi(api_client)
settle = 'usdt' # str | Settle currency
contract = 'BTC_USDT' # str | Futures contract
_from = converted_timestamp # int | Start time of candlesticks, formatted in Unix timestamp in seconds. Default to`to - 100 * interval` if not specified (optional)
to = converted_timestamp+60*2 # int | End time of candlesticks, formatted in Unix timestamp in seconds. Default to current time (optional)
limit = 100 # int | Maximum recent data points to return. `limit` is conflicted with `from` and `to`. If either `from` or `to` is specified, request will be rejected. (optional) (default to 100)
interval = '1m' # str | Interval time between data points. Note that `1w` means natual week(Mon-Sun), while `7d` means every 7d since unix 0.  Note that 30d means 1 natual month, not 30 days (optional) (default to '5m')


api_response = api_instance.list_futures_candlesticks(settle, contract, _from=_from,to=to, interval=interval)
print(len(api_response))
print(api_response)
data=str(api_response)
print(111111,data)

h=[]

def parse_to_dict(s):
    try:
        # 尝试解析字符串为字典
        return ast.literal_eval(s)
    except (ValueError, SyntaxError):
        # 解析失败则返回 None
        return None

# 处理原始数据，仅保留有效字典
after1 = list(map(str, api_response))

parsed_data = []
for item in after1:
    if isinstance(item, str):
        parsed = parse_to_dict(item)
        if isinstance(parsed, dict):  # 确保解析后是字典
            parsed_data.append(parsed)
    elif isinstance(item, dict):
        # 如果已经是字典，直接保留
        parsed_data.append(item)

#parsed date是一个已经处理好的元素为字典的列表












