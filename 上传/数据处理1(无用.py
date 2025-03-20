import numpy as np
import pandas as pd
from datetime import datetime
import time
import tqdm

path=r"C:\Users\HP\BTC历史数据\综合btc文件\综合至今的数据.parquet"
# 读取Parquet文件
df = pd.read_parquet(r"C:\Users\HP\BTC历史数据\综合btc文件\综合至今的数据.parquet")
# 查看数据的前几行
#print((df))

df1 = pd.DataFrame(df)
e=[]
t=[]
r=0
# 使用 iterrows() 方法迭代每一行
for index, row in df1.iterrows():
    # 将每一行的数据转换为列表
    row_list = row.tolist()
    #时间转换
    date_str =str(row_list[0])
    date_obj = datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S')
    timestamp = time.mktime(date_obj.timetuple())
    #print(timestamp)  # 秒级时间戳这个对应于y1
    #print(row_list[0])#世界时间


    del row_list[0]
    del row_list[21]
    # 输出列表

    row_list.append(timestamp)
    row_list = [float(item) for item in row_list]
    #print(row_list)
    r+=1
    if (r + 1) % 100000 == 0:  # 注意这里用i+1来确保打印的是当前的“人类可读”迭代次数
        print(r/2247160)

    e.append(row_list)


#print(e)#一组21维向量,对应于X
print(len(e))
#print(t)
with open(r"C:\Users\HP\BTC历史数据\输入输出\输入输出.txt", 'w', encoding='utf-8') as file:

    file.write(str(e))

print("数据已写入")

#y为买卖或持仓，需要人工判断后输入