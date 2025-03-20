import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import 循环爬取交易所数据并保存成数据集
循环爬取交易所数据并保存成数据集
# 核心模型定义
import datetime
from datetime import datetime
import time
from sortedcontainers import SortedList
from sortedcontainers import SortedList
def group_extremes(data, column, window_size=21600):
    arr = [row[column] for row in data]
    n = len(arr)

    lower_group = []  # 存储前15%极端值对应的原始数据行
    upper_group = []  # 存储后15%极端值对应的原始数据行

    sorted_list = SortedList()
    current_start = 0
    current_end = -1

    for i in range(n):
        # 计算窗口边界
        new_start = max(0, i - window_size)
        new_end = min(n - 1, i + window_size)

        # 扩展右边界
        while current_end < new_end:
            current_end += 1
            sorted_list.add(arr[current_end])

        # 收缩左边界
        while current_start < new_start:
            sorted_list.discard(arr[current_start])
            current_start += 1

        # 计算分位数
        m = new_end - new_start + 1
        if m == 0:
            continue
        k_lower = int(0.15 * m)
        k_upper = int(0.85 * m)
        lower = sorted_list[k_lower]
        upper = sorted_list[k_upper]

        # 分组当前数据
        current_value = arr[i]
        if current_value <= lower:
            lower_group.append(data[i])  # 直接存储原始数据行
        elif current_value >= upper:
            upper_group.append(data[i])

    return lower_group, upper_group  # 返回两个分组
class DynamicTimeModel(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=64):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim,
                            bidirectional=True, batch_first=True)
        self.attention = nn.Sequential(
            nn.Linear(2 * hidden_dim, 32),
            nn.Tanh(),
            nn.Linear(32, 1, bias=False))
        self.classifier = nn.Linear(2 * hidden_dim, 3)

    def forward(self, x, lengths):
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        lstm_out, _ = self.lstm(packed)
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        attn_weights = self.attention(lstm_out)
        mask = torch.arange(x.size(1))[None, :] < lengths[:, None]
        attn_weights = attn_weights.masked_fill(~mask.unsqueeze(-1), -1e9)
        attn_weights = torch.softmax(attn_weights, dim=1)
        context = torch.sum(attn_weights * lstm_out, dim=1)
        return self.classifier(context)

# def map_to_value(start, timestamps, value):
#     """通用函数：将时间戳标记为指定值（0 或 2）"""
#     valid_offsets = [ts - start for ts in timestamps if ts >= start]
#     if not valid_offsets:
#         return []
#     max_offset = max(valid_offsets)
#     values = [0] * (max_offset + 1)  # 初始化为全 0
#     for offset in valid_offsets:
#         values[offset] = value       # 直接覆盖为指定值
#     return values
#
# # 具体函数（基于通用函数封装）
# map_to_zero = lambda start, ts: map_to_value(start, ts, 0)
# map_to_two = lambda start, ts: map_to_value(start, ts, 2)


def merge_operations(start, zeros, twos, existing_values):
    zero_offsets = [ts - start for ts in zeros if ts >= start]
    two_offsets = [ts - start for ts in twos if ts >= start]

    # 仅修改存在的偏移量，不调整列表长度
    for offset in zero_offsets:
        if offset < len(existing_values):
            existing_values[int(offset)] = 0
            if offset % 1000 == 0:
                print(offset / len(existing_values))

    for offset in two_offsets:
        if offset < len(existing_values):
            existing_values[int(offset)] = 2
            if offset%1000==0:
                print(offset / len(existing_values))


    return existing_values
# 数据预处理函数
def create_dynamic_dataset(features, labels, min_len=5, max_len=30):
    """请在此处接入您的时序特征数据"""
    X, y, lengths = [], [], []
    for i in range(len(features) - min_len):
        window_size = np.random.randint(min_len, min(max_len, len(features) - i))
        X.append(features[i:i + window_size])
        y.append(labels[i + window_size])
        lengths.append(window_size)
    return X, np.array(y), np.array(lengths)


# 数据加载器
def collate_fn(batch):
    batch.sort(key=lambda x: len(x[0]), reverse=True)
    sequences = [torch.FloatTensor(item[0]) for item in batch]
    labels = torch.LongTensor([item[1] for item in batch])
    lengths = torch.LongTensor([len(seq) for seq in sequences])
    padded_sequences = nn.utils.rnn.pad_sequence(sequences, batch_first=True)
    return padded_sequences, labels, lengths

e=循环爬取交易所数据并保存成数据集.finall
# 示例使用
q=[]
h=[]
if __name__ == "__main__":
    # 假设data是一个三维列表，这里创建一个示例数据
    # 示例数据：100个时间点，每个时间点是一个10x10的二维列表
    data = e
    low,high = group_extremes(data, 0)
    for i in range(len(high)):
        q.append(high[i][5])

    for t in range(len(low)):
        h.append(low[t][5])

    # 打印前10个时间点的标记结果
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y, lengths):
        self.X = [torch.FloatTensor(x) for x in X]
        self.y = torch.LongTensor(y)
        self.lengths = lengths

    def __len__(self): return len(self.X)

    def __getitem__(self, idx): return self.X[idx], self.y[idx], self.lengths[idx]


def datetime_to_timestamp(date_string, date_format='%Y-%m-%d %H:%M:%S'):
    dt_object = datetime.strptime(date_string, date_format)
    return int(dt_object.timestamp())
# 训练流程
def train_model(total_epochs=1000):
    # 初始化组件
    model = DynamicTimeModel()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    new_lst = []
    all_len = (循环爬取交易所数据并保存成数据集.rt)
    y = [1] * all_len
    afq = [num / 60 for num in h]
    afh=[num / 60 for num in q]
    print(afq,afh)
    t = merge_operations(int(循环爬取交易所数据并保存成数据集.oo)/60, afq, afh,y)
     # 输出: [0, 2, 0, 0, 2]


    for i in range(len(t)):
        y[i] =t[i]



    X=循环爬取交易所数据并保存成数据集.finall
    for row in X:  # 截取操作
        del row[5]
    print(11111111111111111111111111111111111111111, len(y), len(X),len(q),len(h) )

    X, y, lengths = create_dynamic_dataset(X, y)

    # 构建数据管道
    dataset = TimeSeriesDataset(X, y, lengths)
    dataloader = DataLoader(dataset, batch_size=32,
                            collate_fn=collate_fn, shuffle=True)

    # 训练循环
    pbar = tqdm(total=total_epochs)
    for epoch in range(total_epochs):
        for batch_x, batch_y, batch_lens in dataloader:
            optimizer.zero_grad()
            outputs = model(batch_x, batch_lens)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
        pbar.update(1)

    # 模型保存
    torch.save(model.state_dict(),r'C:\Users\HP\BTC历史数据\保存已训练最新模型的文件\model.pth')
    print('模型训练完成')
    print(X, y, len(X), len(y))

if __name__ == "__main__":
     train_model(total_epochs=500)



