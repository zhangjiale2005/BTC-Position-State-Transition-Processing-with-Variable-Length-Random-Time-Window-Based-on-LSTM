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


import 循环爬取交易所数据并保存成数据集
循环爬取交易所数据并保存成数据集
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
print(f"前15%的列表{q}")#低
print(f"后15%的列表{h}")#高


