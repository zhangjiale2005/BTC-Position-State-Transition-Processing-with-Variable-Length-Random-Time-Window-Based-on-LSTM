import pandas as pd
import glob
import os

# 指定Parquet文件所在的目录
parquet_dir = r'C:\Users\HP\BTC历史数据\data'

# 使用glob模块获取所有Parquet文件的路径列表
parquet_files = glob.glob(os.path.join(parquet_dir, '*.parquet'))

# 根据文件名对文件列表进行排序
# 注意：这里假设文件名能够反映你想要的排序顺序
parquet_files.sort()

# 初始化一个空的DataFrame来存储合并后的数据
combined_df = pd.DataFrame()

# 遍历排序后的文件列表，读取每个Parquet文件并将其追加到combined_df中
for file_path in parquet_files:
    df = pd.read_parquet(file_path)
    combined_df = pd.concat([combined_df, df], ignore_index=True)

# 如果需要将合并后的数据写回到一个新的Parquet文件中
output_file = r'C:\Users\HP\BTC历史数据\最新数据\最终.parquet'
combined_df.to_parquet(output_file, engine='pyarrow')