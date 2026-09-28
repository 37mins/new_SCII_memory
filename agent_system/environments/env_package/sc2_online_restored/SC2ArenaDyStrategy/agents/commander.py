import numpy as np
from agents.planner import SC2Planner
from agents.strategist import SC2Strategist
import json
import gymnasium as gym
from gymnasium import spaces
import socket
import pickle
import multiprocessing as mp
import time
import copy
import os

from collections import defaultdict
import re

def get_time(text):
    text = str(text)
    result = re.search(r'It is now (\d{1,2}):(\d{2}) in the game', text)
    if result:
        return int(result.group(1)) * 60 + int(result.group(2))
    result = re.search(r'time_seconds:\s*(\d+)', text, flags=re.IGNORECASE)
    if result:
        return int(result.group(1))
    result = re.search(r'time_formatted:\s*(\d{1,2}):(\d{2})', text, flags=re.IGNORECASE)
    if result:
        return int(result.group(1)) * 60 + int(result.group(2))
    return None

def clip_value(value):
    return max(-5, min(value, 5))

def calculate_time_diff_simple(time_str):
    # 提取时间
    matches = re.findall(r'(\d{2}):(\d{2})', time_str)

    if len(matches) < 2:
        return None

    # 计算秒数
    t1_seconds = int(matches[0][0]) * 60 + int(matches[0][1])
    t2_seconds = int(matches[1][0]) * 60 + int(matches[1][1])

    # 返回差值
    return abs(t2_seconds - t1_seconds)

def normalize_bo_name(name):
    return str(name).strip().lower()

def normalize_bo_dict(bo_dict):
    normalized = defaultdict(int)
    for name, count in bo_dict.items():
        normalized[normalize_bo_name(name)] += count
    return dict(normalized)

def read_int_env(name, default):
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default

class SC2Commander:
    def __init__(self, host='127.0.0.1', port=9999) -> None:
        self.strategist = SC2Strategist()
        self.planner = SC2Planner(host, port)

        self.waiting_bo = {}
        self.current_bo = {}

        self.bo_todo_strategy = None
        self.bo_done = None
        self.strategy_start_time = None
        self.strategy_last_time = None

        self.strategy_renew_ct = 0
        self.has_attack = False
        self.attack_ct = 0

        knowledge_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "knowledge"))
        with open(os.path.join(knowledge_dir, "data_BO2Action_1.json"), "r") as f:
            self.bo2action = json.load(f)

    def get_intent(self, b):
        if b.startswith('Attacking Instruction'):
            return 'Attacking Instruction'

        for i in self.bo2action:
            bo_name = i['bo_name']
            bo_name= bo_name.replace(' ','').lower()
            if bo_name == b.lower():
                return i['bo_intent']

        if 'techlab' in b:
            return 'Build Tech Lab'
        if 'reactor' in b:
            return 'Build Reactor'

        return 'None'

    def ct_waiting_bo(self):
        return len(self.waiting_bo.keys())

    def should_renew_strategy_by_time(self, obs):
        current_time = get_time(obs)
        if (
            current_time is None
            or self.strategy_start_time is None
            or self.strategy_last_time is None
        ):
            return False
        if self.strategy and self.strategy.get("_terminal_phase"):
            return False
        grace_seconds = read_int_env("SC2_STAGE_GRACE_SECONDS", 15)
        deadline = self.strategy_start_time + self.strategy_last_time + grace_seconds
        if current_time >= deadline:
            print(
                "Force renewing strategy by time window: "
                f"current={current_time}s deadline={deadline}s remaining_bo={self.waiting_bo}"
            )
            return True
        return False

    def renew_waiting_bo(self):
        # Selection quotas, not evidence that construction/production completed.
        self.current_bo_debits = [None] * len(self.current_bo)
        for index, bo in enumerate(self.current_bo):
            bo_key = normalize_bo_name(bo)
            if bo_key not in self.waiting_bo.keys():
                continue
            if self.waiting_bo[bo_key] > 0:
                self.current_bo_debits[index] = bo_key
            self.waiting_bo[bo_key] = max(self.waiting_bo[bo_key] - 1, 0)
            if self.waiting_bo[bo_key] == 0:
                del self.waiting_bo[bo_key]

    def release_unsubmitted_bo(self, start_index):
        """Release recorded selections only; the caller excludes submitted targets."""
        restored = {}
        for index in range(start_index, len(self.current_bo_debits)):
            name = self.current_bo_debits[index]
            if name is not None:
                self.waiting_bo[name] = self.waiting_bo.get(name, 0) + 1
                restored[name] = restored.get(name, 0) + 1
                self.current_bo_debits[index] = None
        return restored

    def ensure_opening_marine(self, obs):
        if self.strategy_renew_ct > 1:
            return

        if not self.waiting_bo.get("marine"):
            return

        normalized_current = [normalize_bo_name(bo) for bo in self.current_bo]
        if "marine" in normalized_current:
            return
        if any("attacking instruction" in bo_name for bo_name in normalized_current):
            return

        obs_text = str(obs).lower()
        has_or_building_barracks = "barracks" in normalized_current or "barracks" in obs_text
        if not has_or_building_barracks:
            return

        segment_max = max(1, read_int_env("SC2_PLAN_SEGMENT_MAX", 4))
        insert_pos = len(self.current_bo)
        for idx, bo_name in enumerate(normalized_current):
            if bo_name == "barracks":
                insert_pos = idx + 1

        preserve_model_segment = (os.getenv("SC2_ORIGINAL_BO_SEGMENT", "0") == "1"
                                  and self.planner.original_segment_active)
        while not preserve_model_segment and len(self.current_bo) >= segment_max:
            drop_idx = len(self.current_bo) - 1
            for idx in range(len(self.current_bo) - 1, -1, -1):
                candidate = normalize_bo_name(self.current_bo[idx])
                if (
                    candidate not in {"scv", "supplydepot", "barracks"}
                    and "attacking instruction" not in candidate
                ):
                    drop_idx = idx
                    break
            removed = self.current_bo.pop(drop_idx)
            print(f"Opening marine guard: dropping {removed} to fit marine")
            if drop_idx < insert_pos:
                insert_pos -= 1

        self.current_bo.insert(max(0, min(insert_pos, len(self.current_bo))), "marine")
        print(f"Opening marine guard applied: {self.current_bo}")

    def reward_planner_speed(self, obs):
        current_time = get_time(obs)
        if self.strategy_last_time is not None:
            bonus_time = self.strategy_last_time - (current_time - self.strategy_start_time)
        else:
            bonus_time = 0
        print(f"Bonus time: {bonus_time}")
        return bonus_time

    def reward_planner_bo(self):
        """
        更精细的奖励函数版本，使用Jaccard相似度和数量衰减函数

        Args:
            generated_dict: 生成的字典
            target_dict: 目标字典

        Returns:
            float: 奖励信号
        """

        if self.bo_todo_strategy == None:
            return 0

        target_dict = self.bo_todo_strategy
        generated_dict = self.bo_done
        if not target_dict:
            return 1.0 if not generated_dict else 0.0

        # 获取键集合
        target_keys = set(target_dict.keys())
        generated_keys = set(generated_dict.keys())

        # 1. Jaccard相似度计算键的匹配度
        intersection = target_keys.intersection(generated_keys)
        union = target_keys.union(generated_keys)

        if not union:  # 防止除零
            jaccard_similarity = 1.0
        else:
            jaccard_similarity = len(intersection) / len(union)

        # 2. 数量匹配度（对每个匹配的键）
        count_similarities = []

        for building in target_keys:
            target_count = target_dict[building]

            if building in generated_dict:
                generated_count = generated_dict[building]

                if target_count == 0:
                    # 目标数量为0的情况
                    similarity = 1.0 if generated_count == 0 else 0.0
                else:
                    # 使用衰减函数：similarity = e^(-α * |diff/target|)
                    # α控制衰减速度，α越大，对偏离的惩罚越严格
                    alpha = 1.5
                    diff_ratio = abs(generated_count - target_count) / target_count
                    similarity = np.exp(-alpha * diff_ratio)
            else:
                similarity = 0.0

            count_similarities.append(similarity)

        # 平均数量相似度
        avg_count_similarity = np.mean(count_similarities) if count_similarities else 0.0

        # 3. 惩罚多余的建筑
        extra_keys = generated_keys - target_keys
        extra_penalty = 0.0

        for building in extra_keys:
            generated_count = generated_dict[building]
            if generated_count > 0:
                # 惩罚与多余建筑的数量成正比
                extra_penalty += min(0.1, 0.02 * generated_count)  # 每个多余建筑最多惩罚0.1

        # 4. 综合奖励
        # 使用调和平均数或加权平均
        weights = {'key': 0.4, 'count': 0.6}

        base_reward = (weights['key'] * jaccard_similarity +
                    weights['count'] * avg_count_similarity)

        # 应用惩罚
        final_reward = max(0.0, base_reward - min(extra_penalty, 0.4))

        return min(1.0, max(0.0, final_reward))


    def renew_bo(self, obs, execution_feedback=None):
        # A failed segment gets one replan in its original phase. Normal segment
        # exhaustion still uses the existing time/quota phase transition below.
        force_renew_by_time = (execution_feedback is None
                              and self.should_renew_strategy_by_time(obs))
        if execution_feedback is None and (self.ct_waiting_bo() == 0 or force_renew_by_time):
            # one strategy phase test
            print("Current stage", self.strategy_renew_ct)
            if self.strategy_renew_ct >= 6:
                done_test = True
            else:
                done_test = False

            # reward = self.reward_planner_bo() * 0.1
            reward = 0
            speed_bonus = self.reward_planner_speed(obs)
            reward += max(0.0, min(speed_bonus * 0.05, 1.0))

            if self.has_attack:
                reward += 0.5
                self.has_attack = False
                self.attack_ct += 1

            if done_test:
                reward += 1

            self.renew_strategy(obs)

            self.strategy_renew_ct += 1

            self.bo_todo_strategy = copy.deepcopy(self.waiting_bo)
            self.bo_done = defaultdict(int)
        else:
            reward = 0
            if self.has_attack:
                reward += 0.5
                self.has_attack = False
                self.attack_ct += 1

            done_test = False

        if execution_feedback is None:
            self.current_bo = self.planner.generate_plan(obs, self.waiting_bo, self.strategy, reward, done_test, self.attack_ct)
        else:
            self.current_bo = self.planner.generate_plan(
                obs, self.waiting_bo, self.strategy, reward, done_test, self.attack_ct,
                execution_feedback=execution_feedback,
            )
        self.ensure_opening_marine(obs)
        trace = getattr(self, "command_trace", None)
        if trace is not None:
            trace.queue_ready(self.current_bo, self.strategy.get("_phase_index"))
        for bo in self.current_bo:
            self.bo_done[bo] += 1
            if 'Attacking Instruction: ' in bo:
                self.has_attack = True

        self.renew_waiting_bo()

        print("===== waiting bo =====\n", self.waiting_bo)

        bo = []
        for i in self.current_bo:
            bo.append({
                'action': i,
                'intent': self.get_intent(i)
            })
        return bo

    def renew_strategy(self, obs):
        self.strategy = self.strategist.generate_strategy(obs)
        phase_start = self.strategy.get("_phase_start_seconds")
        phase_end = self.strategy.get("_phase_end_seconds")
        if phase_start is not None and phase_end is not None:
            self.strategy_start_time = phase_start
            self.strategy_last_time = max(0, phase_end - phase_start)
        else:
            self.strategy_start_time = get_time(obs)
            self.strategy_last_time = calculate_time_diff_simple(self.strategy['timing'])

        self.waiting_bo = normalize_bo_dict(self.strategy['bo todo'])
