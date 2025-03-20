import numpy as np
import torch
import torch.nn as nn
import random
import datetime
import time
from datetime import datetime
import ast
import tqdm
total=20
from tqdm import tqdm
class DynamicTimeModel(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=64):
        super().__init__()
        # 双向LSTM增强时序特征提取
        self.lstm = nn.LSTM(input_dim, hidden_dim,
                            bidirectional=True,
                            batch_first=True)

        # 自适应注意力机制
        self.attention = nn.Sequential(
            nn.Linear(2 * hidden_dim, 32),  # 双向拼接
            nn.Tanh(),
            nn.Linear(32, 1, bias=False))

        # 动态分类器
        self.classifier = nn.Linear(2 * hidden_dim, 3)

    def forward(self, x, lengths):
        # x的形状: (batch, seq_len, 23)
        # lengths: 各样本实际长度

        # 打包变长序列
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(),
            batch_first=True, enforce_sorted=False)

        # LSTM处理
        lstm_out, _ = self.lstm(packed)
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(
            lstm_out, batch_first=True)

        # 注意力权重
        attn_weights = self.attention(lstm_out)  # (batch, seq_len, 1)

        # 应用长度mask
        mask = torch.arange(x.size(1))[None, :] < lengths[:, None]
        attn_weights = attn_weights.masked_fill(~mask.unsqueeze(-1), -1e9)
        attn_weights = torch.softmax(attn_weights, dim=1)

        # 上下文向量
        context = torch.sum(attn_weights * lstm_out, dim=1)

        return self.classifier(context)




model = DynamicTimeModel()

model_save_path = r"C:\Users\HP\BTC历史数据\保存已训练最新模型的文件\model.pth"
model.load_state_dict(torch.load(model_save_path))
    # 如果需要，将模型设置为评估模式
model.eval()
#输出
def dynamic_predict(feature_sequence):
    """处理任意长度输入
    参数：
        feature_sequence: numpy数组，形状为(N,23)
    返回：
        预测类别（0/1/2）
    """
    model.eval()
    with torch.no_grad():
        # 转换为张量
        seq_tensor = torch.FloatTensor(feature_sequence).unsqueeze(0)  # (1, N, 23)
        length_tensor = torch.LongTensor([len(feature_sequence)])

        # 模型预测
        logits = model(seq_tensor, length_tensor)
        probs = torch.softmax(logits, dim=-1)

        # 决策规则
        if probs[0, 0] > 0.5:  # 无输出阈值
            return 0
        else:
            return torch.argmax(probs[0, 1:]) + 1


#获取实时数据做预测

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



data=[]
date_string = "2025-01-01 12:00:00"#input("输入监控的起始时间，如         2021-10-01 12:00:00")  # 示例日期和时间字符串
converted_timestamp = datetime_to_timestamp(date_string)

current=(time.time())#不能实时预测,因为成交量在变化，如果是历史回测请单独写脚本

api_client = gate_api.ApiClient(configuration)
# Create an instance of the API class
api_instance = gate_api.FuturesApi(api_client)
settle = 'usdt'  # str | Settle currency
contract = 'BTC_USDT'  # str | Futures contract
interval = '1m'
keys_order = ['c', 'h', 'l', 'o', 'sum', 't', 'v']
result = []
while converted_timestamp < current:

    _from = converted_timestamp  # int | Start time of candlesticks, formatted in Unix timestamp in seconds. Default to`to - 100 * interval` if not specified (optional)
    to = converted_timestamp+60*2000 # int | End time of candlesticks, formatted in Unix timestamp in seconds. Default to current time (optional)
    #limit = 100  # int | Maximum recent data points to return. `limit` is conflicted with `from` and `to`. If either `from` or `to` is specified, request will be rejected. (optional) (default to 100)
      # str | Interval time between data points. Note that `1w` means natual week(Mon-Sun), while `7d` means every 7d since unix 0.  Note that 30d means 1 natual month, not 30 days (optional) (default to '5m')
    api_response = api_instance.list_futures_candlesticks(settle, contract, _from=_from,to=to, interval=interval)
    #print(api_response)
    data=api_response
    # 处理数据

    #print((converted_timestamp-(datetime_to_timestamp(date_string)))/(current-(datetime_to_timestamp(date_string))))

    result.extend(data)
    # for item in data:
    #     if isinstance(item, dict):  # 仅处理字典
    #         # 按顺序提取值
    #         sublist = [item[key] for key in keys_order]
    #         result.append(sublist)



    converted_timestamp+=60*2001
print("实时数据从网页端读取完成")
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






keys_order = ['c', 'h', 'l', 'o', 'sum', 't', 'v']

# 提取值的二维列表
result = [
    [d[key] for key in keys_order]
    for d in parsed_data
    if isinstance(d, dict) and all(key in d for key in keys_order)
]

g=0

#print(len(result))
for row in result:
    del row[6]
    del row[5]
#print('删除多余数据完成')
#切割字符串为整数
for sublist in result:
    for i, s in enumerate(sublist):
        # 移除小数部分并转换为整数
        sublist[i] = int(s.split('.')[0])
finall=result

# while g<len(result):
#     finall = [[int(item) for item in row] for row in result]
#     # 提取五个特征列（假设是位置1到5，对应iloc[:,1:6]）
#     g += 1
#
#最近的数据集
le=len(finall)
#print("数据已处理成可预测输入")
p=[]
data = finall  # 替换为实际数据集
j=0
pbar = tqdm(total=total)
while j<total:
    start_index = random.randint(90, le - 1)
    interval = data[start_index:]
    j+=1
    Xinpu = interval
    inp=np.array(Xinpu)
    ou=dynamic_predict(inp)
    p.append(ou)
    pbar.update(1)
# 输出结果示例


from collections import Counter

# 假设 data 是给定的列表
data = p# 示例数据

# 统计元素出现次数
counter = Counter(data)

# 获取出现次数最多的元素及其次数
most_common = counter.most_common(1)

# 提取结果
most_common_element = most_common[0][0] if most_common else None


u=str(most_common_element)
p=u[7]
print(p)
