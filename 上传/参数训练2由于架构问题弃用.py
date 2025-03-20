import random
from tqdm import tqdm
import numpy as np
import pandas as pd
import time
from datetime import datetime
from datetime import datetime, timedelta
import 循环爬取交易所数据并保存成数据集
total=1000#input("模型迭代次数")
start='2017-10-01 12:00:00'#input("输入训练开始的时间，如       2021-10-01 12:00:00")
end='2019-11-01 12:00:00'#input("输入训练结束的时间")



df1 = pd.read_parquet(r"C:\Users\HP\BTC历史数据\综合btc文件\综合至今的数据.parquet")#df1是完整表格
data = {'timestamp': [start,end]}#这里是转换str格式为time格式
df2 = pd.DataFrame(data)#timestamp标签下，df2[0]是开始时间，1是结束时间
df2['timestamp'] = pd.to_datetime(df2['timestamp'])
#这里定义的函数是对时间戳的转换
def datetime_to_timestamp(date_string, date_format='%Y-%m-%d %H:%M:%S'):
    dt_object = datetime.strptime(date_string, date_format)
    return int(dt_object.timestamp())

date_format = "%Y-%m-%d %H:%M:%S"
start_object = datetime.strptime(start, date_format)

after_start = datetime_to_timestamp(start)
after_end=datetime_to_timestamp(end)

print(f"起始时间和日期 '{start}' 转换为时间戳: {after_start}")
print(f"终止时间和日期'{end}'转换后为:{after_end}")

suoxu=df1.iloc[:, 0:6]
column_name = 'timestamp'

# 假设你要查找的时间戳值
start_list1 =after_start#时间戳，一串数字
end_list1=after_end#时间戳，一串数字
r=0#循环的初始化
new_timestamp_add=df2.loc[0, 'timestamp']
end_list = df1[df1[column_name] == end]
all_len=(int(after_end)-int(after_start))/60
print(all_len)
X=[]#输入的初始化
y=[]

while len(y)<all_len:
     y.append(1)
u = []
target = 1  # 买入
# 替换后的元素
replacement = 0
kais0 = ''
jies0 = ''
# 指定区间（例如，从索引2到索引6，不包括6）
while kais0 or jies0 != "结束" or '\n' or '下一步' or "开始":
    kais0 = input("输入买入的开始时间，如          2021-10-01 12:00:00")
    jies0 = input('输入买入的结束时间')
    try:
        kais0 = datetime_to_timestamp(kais0)
        jies0 = datetime_to_timestamp(jies0)
        start_index = kais0 - after_start
        end_index = jies0 - after_start
        new_lst = y[:start_index] + [replacement if x == target else x for x in y[start_index:end_index]] + y[
                                                                                                            end_index:]
    except ValueError:
        break


    # 输入开始时间和结束时间，得到all len中位于第几个位置并替换中间的量
float

target = 1  # 买入
# 替换后的元素
replacement2 = 2
kais = ''
jies = ''
# 指定区间（例如，从索引2到索引6，不包括6）
while kais or jies != "结束" or '\n' or '下一步' or "开始":
    kais = input("输入卖出的开始时间，如          2021-10-01 12:00:00")
    jies = input('输入卖出的结束时间')

    try:
        kais = datetime_to_timestamp(kais)
        jies = datetime_to_timestamp(jies)

    # 输入开始时间和结束时间，得到all len中位于第几个位置并替换中间的量
        start_index2 = kais - after_start
        end_index2 = jies - after_start

        new_lst = y[:start_index2] + [replacement if x == target else x for x in y[start_index2:end_index2]] + y[end_index2:]
    except ValueError:
        break


h=[0]
# 使用 iterrows() 方法迭代每一行
print(h)
for index, row in df1.iterrows():  # 直接遍历原始数据框df1
    current_time = row['timestamp']  # 获取当前行的时间戳
    # 检查当前时间是否在指定区间内
    if pd.Timestamp(start) <= current_time <= pd.Timestamp(end):
        # 提取五个特征列（假设是位置1到5，对应iloc[:,1:6]）
        features = row.iloc[1:6].tolist()
        features=float_list = [float(item) for item in features]
        X.append(features)
        r+=1
        k=datetime_to_timestamp(str(current_time))
        h.append(k)
        if len(h)>1 and h[-1] - h[-2]!=60:
            print("这里是"+str(len(h))+"不连续了")
        if (r + 1) % 1000 == 0:  # 注意这里用i+1来确保打印的是当前的“人类可读”迭代次数
            print(r / all_len)
    # else :
    #     print(current_time)


lengths=[]
print('数据读取完成，开始训练')







#print(X)
#print(u)
#模拟Y=[]
#y=[]
# y1=random.randint(1000000,1400000)

# NUM=100
# z=0
# #模拟数据
# X1=np.random.randint(10, 100000, size=(y1+NUM, 21))
#
# while z<NUM:
#     z+=1
#     Y.append(z+y1)
# while len(y)<NUM:
#     y.append(1)
#     lengths.append(21)
# Len= len(y)
# # 使用切片获取最后n个元素
# X = X1[-len(y):]
print(len(X))

# print('第一次时间戳请求',df2.loc[0, 'timestamp'])#第一次时间戳请求
# print('X',X[1])#21维度向量群
# print('lenX',len(X))
# print('leny',all_len)
# print('y',y)#买2，卖0，持仓1
#读取数据做训练
#第一步获取时间戳
#第二步格式化数据集
#第三步输入持仓标签
df = pd.read_parquet(r"C:\Users\HP\BTC历史数据\综合btc文件\综合至今的数据.parquet")




import torch
import torch.nn as nn

criterion = torch.nn.CrossEntropyLoss()
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


def create_dynamic_dataset(features, labels, min_len=5, max_len=30):
    X, y, lengths = [], [], []
    for i in range(len(features) - min_len):
        # 随机生成窗口长度
        window_size = np.random.randint(min_len,
                                        min(max_len, len(features) - i))

        X_window = features[i:i + window_size]
        y_label = labels[i + window_size]

        X.append(X_window)
        y.append(y_label)
        lengths.append(window_size)

    return X, np.array(y), np.array(lengths)


def collate_fn(batch):
    # 按长度降序排列
    batch.sort(key=lambda x: len(x[0]), reverse=True)

    sequences = [torch.FloatTensor(item[0]) for item in batch]
    labels = torch.LongTensor([item[1] for item in batch])
    lengths = torch.LongTensor([len(seq) for seq in sequences])

    # 动态padding
    padded_sequences = nn.utils.rnn.pad_sequence(
        sequences, batch_first=True)

    return padded_sequences, labels, lengths
#zz=create_dynamic_dataset(X,y)
#print()
#训练
# 初始化
model = DynamicTimeModel()

# 生成动态窗口数据集（需取消注释并调用）
X, y, lengths = create_dynamic_dataset(X, y)

#print(X,y,lengths)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# 数据加载
from torch.utils.data import Dataset, DataLoader


class TimeSeriesDataset(Dataset):
    def __init__(self, X, y, lengths):
        self.X = [torch.FloatTensor(x) for x in X]
        self.y = torch.LongTensor(y)
        self.lengths = lengths

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.lengths[idx]


dataset = TimeSeriesDataset(X, y, lengths)
dataloader = DataLoader(dataset, batch_size=32,
                        collate_fn=collate_fn, shuffle=True)

pbar = tqdm(total=total)
for epoch in range(total):
    for batch_x, batch_y, batch_lens in dataloader:
        optimizer.zero_grad()

        # 前向传播
        outputs = model(batch_x, batch_lens)

        # 损失计算
        loss = criterion(outputs, batch_y)

        loss.backward()
        optimizer.step()
    pbar.update(1)

print('模型训练完成')


torch.save(model.state_dict(), r'C:\Users\HP\BTC历史数据\保存已训练最新模型的文件\model.pth')




