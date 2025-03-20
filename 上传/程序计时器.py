import time
import 已训练的最新模型的推理
# 全局变量来存储开始和结束时间
start_time = None
end_time = None
def start_global_timer():
    """启动全局计时器"""
    global start_time
    start_time = time.perf_counter()  # 使用perf_counter进行高精度计时
def stop_global_timer():
    """停止全局计时器并打印运行时间"""
    global end_time, start_time
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(f"程序运行时间: {elapsed_time:.4f} 秒")
# 示例程序
def main():
    start_global_timer()  # 启动计时器

    # 模拟一些程序运行时间
      # 假设程序运行了2秒

    已训练的最新模型的推理
    # 你的程序逻辑代码应该放在这里
    # ...
    stop_global_timer()  # 停止计时器

main()