import os

# 指定包含待重命名文件的文件夹路径
folder_path = r'C:\Users\HP\BTC历史数据\data'

# 获取文件夹内所有文件的名称列表，并依据字典序进行排序
file_names = sorted(os.listdir(folder_path))

# 遍历排序后的文件名列表
for index, file_name in enumerate(file_names, start=1):
    # 提取文件的扩展名（若存在）
    _, file_extension = os.path.splitext(file_name)

    # 构建新的文件名（仅包含数字与扩展名，若原文件有扩展名）
    new_file_name = f"{index}{file_extension}"

    # 构造旧文件与新文件的完整路径
    old_file_path = os.path.join(folder_path, file_name)
    new_file_path = os.path.join(folder_path, new_file_name)

    # 重命名文件
    os.rename(old_file_path, new_file_path)

print("文件重命名操作已完成。")