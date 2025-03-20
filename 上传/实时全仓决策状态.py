import time
from enum import Enum


class State(Enum):
    ZERO = 0
    ONE = 1
    TWO = 2


class StateMachine:
    def __init__(self):
        self._current_state = State.ONE
        self._transition_log = []
        self._listeners = []

    @property
    def current_state(self):
        return self._current_state.value

    def add_listener(self, callback):
        """添加状态变化监听器"""
        self._listeners.append(callback)

    def _notify_listeners(self, old_state, new_state):
        """通知所有监听器"""
        for callback in self._listeners:
            callback(old_state.value, new_state.value)

    def transition(self, new_output: int):
        """
        执行状态转移
        :param new_output: 新输入值 (0/1/2)
        :return: 新状态值
        """
        # 输入验证
        if new_output not in (0, 1, 2):
            raise ValueError("无效输入，只接受0/1/2")

        old_state = self._current_state
        new_state = self._determine_new_state(new_output)

        if new_state != old_state:
            self._log_transition(old_state, new_state)
            self._current_state = new_state
            self._notify_listeners(old_state, new_state)

        return self.current_state

    def _determine_new_state(self, new_output):
        """核心状态转移逻辑"""
        transition_map = {
            State.ONE: {
                0: State.ZERO,
                2: State.ONE,  # 修改点：状态1时输入2保持为1
                1: State.ONE
            },
            State.ZERO: {
                2: State.ONE,  # 状态0时输入2转移到1
                0: State.ZERO,
                1: State.ZERO
            },
            State.TWO: {
                0: State.ONE,
                2: State.TWO,
                1: State.TWO
            }
        }
        return transition_map[self._current_state][new_output]

    def _log_transition(self, old_state, new_state):
        """记录状态转移日志"""
        entry = {
            "timestamp": time.time(),
            "old_state": old_state.value,
            "new_state": new_state.value,
            "trigger": "external"
        }
        self._transition_log.append(entry)


# 使用示例
if __name__ == "__main__":
    def print_notification(old, new):
        print(f"[通知] 状态变化：{old} → {new}")


    sm = StateMachine()
    sm.add_listener(print_notification)

    test_sequence = [0, 0, 2, 1, 2, 0, 1, 0, 2]

    print("=== 开始状态监控 ===")
    for idx, data in enumerate(test_sequence):
        print(f"第{idx + 1}次轮询，输入: {data}")
        current = sm.transition(data)
        print(f"当前状态: {current}\n")
        time.sleep(0.5)