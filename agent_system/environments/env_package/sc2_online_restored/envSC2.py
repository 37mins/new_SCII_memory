import gymnasium as gym
from gymnasium import spaces
import socket
import pickle
import multiprocessing as mp
import time
import signal
import fcntl
import os
import select
import traceback

import sys
PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
ARENA_DIR = os.path.join(PACKAGE_DIR, "SC2ArenaDyStrategy")
if ARENA_DIR not in sys.path:
    sys.path.insert(0, ARENA_DIR)

from sc2 import maps
from sc2.player import Bot, Computer
from sc2.main import run_game
from sc2.data import Race, Difficulty, AIBuild
from argparse import ArgumentParser, Namespace
import json

from players import LLMPlayer
from tools import constants
from tools.llm import LLMClient
import random

import ray
import numpy as np
from agent_system.sc2_episode import gymnasium_transition, truncation_fields

if not os.getenv("SC2PATH"):
    raise RuntimeError("SC2PATH must point to the StarCraftII installation")

class TimeoutException(Exception):
    pass

def signal_handler(signum, frame):
    raise TimeoutException("操作超时")


def _log_sc2_runtime(*values, sep=" ", end="\n", flush=False):
    output_dir = os.path.join(
        os.getenv("SC2_RUNTIME_LOG_ROOT", "logs"), "worker-output"
    )
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"controller-{os.getpid()}.log")
    message = sep.join(str(value) for value in values)
    with open(output_path, "a", encoding="utf-8", errors="backslashreplace") as stream:
        stream.write(message + end)
        if flush:
            stream.flush()


def get_unused_port(host='127.0.0.1', port=0):
    """
    获取一个未使用的端口
    port=0 表示让系统自动分配可用端口
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, port))
        s.listen(1)
        port = s.getsockname()[1]
    return port


def parse_args():
    args = Namespace()

    args.map_name = 'Flat128'
    args.difficulty = os.getenv("SC2_DIFFICULTY", "Hard")
    # args.difficulty = "CheatVision"
    # args.difficulty = "CheatMoney"
    # args.model_name = "Qwen/Qwen3-30B-A3B-Instruct-2507"
    args.model_name = "Qwen3-30B"
    args.ai_build = os.getenv("SC2_AI_BUILD", "Air")
    args.player_name = "default_player"
    args.enable_rag = False
    args.enable_plan = True
    args.enable_plan_verifier = False
    args.enable_action_verifier = True
    args.base_url = os.getenv("SC2_EXECUTOR_LLM_BASE_URL", "http://127.0.0.1:8001/v1")
    args.api_key = os.getenv("SC2_EXECUTOR_LLM_API_KEY", "unused")
    args.own_race = "Terran"
    args.enemy_race = "Terran"
    args.enable_random_decision_interval = False  # store_true 动作的默认值
    args.bo_address = os.getenv(
        "SC2_BO_ADDRESS",
        os.path.join(
            ARENA_DIR,
            "knowledge/BO/replay_observations_Flat128_d4_TvT_Macro_g2_Result.Victory_twobase.json",
        ),
    )

    if args.enable_rag:
        raise ValueError(
            "RAG agent is not supported in this version. Please disable it with --enable_rag."
        )
    if args.enable_plan_verifier and not args.enable_plan:
        raise ValueError(
            "Plan verifier requires Plan agent to be enabled. Please enable Plan agent with --enable_plan."
        )
    if not args.base_url or not args.api_key:
        raise ValueError(
            "Base URL and API key must be provided. Please set them using --base_url and --api_key."
        )

    return args

class EnvironmentA():
    def __init__(self, host=None, port=None, seed=None) -> None:
        self.args = parse_args()
        self.seed = seed

    def run(self, host=None, port=None):
        print('Start client...')
        print(port)
        # 将生成的种子放入您的代码中
        self.args.port = port
        self.args.host = host
        map_name = self.args.map_name
        difficulty = self.args.difficulty
        model_name = self.args.model_name
        ai_build = self.args.ai_build
        player_name = self.args.player_name
        log_root = os.getenv("SC2_RUNTIME_LOG_ROOT", "logs")
        log_path = os.path.join(
            log_root,
            self.args.player_name,
            f"{self.args.map_name}_{self.args.difficulty}_{self.args.ai_build}",
        )

        # Initialize LLM service
        llm_config = {
            "model_name": model_name,
            "generation_config": {
                "model_name": model_name,
                "n": 1,
                "max_tokens": 9000,
                "temperature": 0.1,
                "top_p": 0.8,
                "top_k": 20,
                "repetition_penalty": 1.1,
                "presence_penalty": 0.0,
            },
            "llm_client": LLMClient(
                base_url=self.args.base_url,
                api_key=self.args.api_key,
            ),
        }

        # Initialize players
        join_player = Computer(
            race=getattr(Race, self.args.enemy_race),
            difficulty=getattr(Difficulty, difficulty),
            ai_build=getattr(AIBuild, ai_build),
        )
        ai_player = LLMPlayer(
            config=self.args,
            player_name=player_name,
            log_path=log_path,
            bo_address=self.args.bo_address,
            **llm_config,
        )
        host_player = Bot(getattr(Race, self.args.own_race), ai_player)

        # random_seed_value = random.randint(0, 100)

        print("Start to build game.")
        game_time_limit_env = os.getenv("SC2_GAME_TIME_LIMIT", "").strip()
        game_time_limit = float(game_time_limit_env) if game_time_limit_env else None
        if game_time_limit is not None:
            print(f"Using SC2 game_time_limit={game_time_limit}")

        res = run_game(
            maps.get(map_name),
            [host_player, join_player],
            realtime=False,
            rgb_render_config=None,
            save_replay_as=ai_player.log_path + "/replay.SC2Replay",
            random_seed=self.seed,
            game_time_limit=game_time_limit,
        )

        print(f"RES: \n {res}")
        game_result = res.name
        try:
            final_obs = ai_player.get_obs()
        except Exception as exc:
            print(f"Failed to collect final observation for reward shaping: {exc}")
            final_obs = ""
        ai_player.commander.planner.game_end(
            final_obs,
            game_result,
            attack_ct=getattr(ai_player.commander, "attack_ct", 0),
            telemetry=ai_player.get_long_loop_telemetry(),
        )

class SC2Env(gym.Env):
    def __init__(self, config):
        super(SC2Env, self).__init__()
        self.env_a_class = EnvironmentA
        self.host = '127.0.0.1'

        # 定义空间（应根据实际 A 的情况调整）
        self.observation_space = spaces.Box(low=-1, high=1, shape=(10,), dtype=float)
        self.action_space = spaces.Discrete(2)

        self.server_socket = None
        self.conn = None
        self.env_a_process = None
        self.config = config

        self.game_end_flag = False
        self.game_end_data = None
        self.reset_failed = False
        self.socket_timeout = float(os.getenv("SC2_SOCKET_TIMEOUT", "120"))
        self.reset_retries = int(os.getenv("SC2_RESET_RETRIES", "4"))
        self.reset_attempt_timeout = int(os.getenv("SC2_RESET_ATTEMPT_TIMEOUT", str(int(self.socket_timeout) + 30)))
        self.reset_concurrency = int(os.getenv("SC2_RESET_CONCURRENCY", "4"))
        self.accept_timeout = float(os.getenv("SC2_ACCEPT_TIMEOUT", "5.0"))
        self.fail_fast_on_env_error = os.getenv("SC2_FAIL_FAST_ON_ENV_ERROR", "0") == "1"
        self.pre_action_terminal_drain_timeout = float(os.getenv("SC2_PRE_ACTION_TERMINAL_DRAIN_TIMEOUT", "0.0"))

    def _safe_close(self, sock):
        """安全关闭套接字"""
        if sock:
            try:
                sock.close()
            except OSError:
                pass

    def _setup_server(self, port):
        """建立 Socket 服务端"""
        if self.server_socket is None:
            _log_sc2_runtime("Start server...")
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            # 设置 SO_REUSEADDR 允许立即重用端口，防止 reset 时端口被占用
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind((self.host, port))
            self.server_socket.listen(1)
            _log_sc2_runtime("Server listens...")

    def _cleanup_resources(self, reason='reset_or_close'):
        """清理本轮 reset/step 产生的连接、监听 socket 和子进程。"""
        if self.env_a_process:
            self._stop_owned_game(reason)
            self.env_a_process = None
        self._safe_close(self.conn)
        self.conn = None
        self._safe_close(self.server_socket)
        self.server_socket = None

        if self.env_a_process:
            try:
                self.env_a_process.terminate()
                self.env_a_process.join(timeout=0.5)
            except:
                pass
            self.env_a_process = None

    def _stop_owned_game(self, reason):
        """Stop only this EnvironmentA child's process tree."""
        import psutil
        process = self.env_a_process
        try:
            children = psutil.Process(process.pid).children(recursive=True)
        except psutil.NoSuchProcess:
            children = []
        owned = [{'pid': child.pid, 'created_at': child.create_time()} for child in children]
        for child in children:
            try:
                child.terminate()
            except psutil.NoSuchProcess:
                pass
        _, alive = psutil.wait_procs(children, timeout=2)
        if process.is_alive():
            process.terminate()
        process.join(timeout=2)
        if process.is_alive():
            process.kill()
            process.join(timeout=1)
        for child in alive:
            try:
                child.kill()
            except psutil.NoSuchProcess:
                pass
        _, alive = psutil.wait_procs(alive, timeout=1)
        alive = [child for child in alive if child.status() != psutil.STATUS_ZOMBIE]
        record = {
            'reason': reason,
            'environment_pid': process.pid,
            'children': owned,
            'alive_child_pids': [child.pid for child in alive],
            'environment_alive': process.is_alive(),
        }
        log_root = os.environ['SC2_RUNTIME_LOG_ROOT']
        os.makedirs(log_root, exist_ok=True)
        with open(os.path.join(log_root, 'cleanup.jsonl'), 'a') as stream:
            stream.write(json.dumps(record) + '\n')
        if alive or process.is_alive():
            raise RuntimeError(f'SC2 owned process cleanup incomplete: {record}')

    def truncate(self):
        self.game_end_data = {
            'obs': self.obs,
            'reward': 0.0,
            'done': True,
            'success': False,
            'attack_ct': 0,
            'last_consumed_policy_action_id': None,
            **truncation_fields(),
        }
        self.game_end_flag = True
        self._cleanup_resources(reason='interaction_limit')

    def _acquire_reset_slot(self):
        if self.reset_concurrency <= 0:
            return None
        lock_dir = os.getenv("SC2_RESET_LOCK_DIR", "/tmp/sc2_reset_locks")
        os.makedirs(lock_dir, exist_ok=True)
        deadline = time.time() + max(10, self.reset_attempt_timeout)
        while True:
            for slot in range(self.reset_concurrency):
                lock_path = os.path.join(lock_dir, f"slot_{slot}.lock")
                fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o666)
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    os.ftruncate(fd, 0)
                    os.write(fd, str(os.getpid()).encode("ascii"))
                    _log_sc2_runtime(f"Acquired SC2 reset slot {slot}/{self.reset_concurrency}.")
                    return fd
                except BlockingIOError:
                    os.close(fd)
            if time.time() > deadline:
                _log_sc2_runtime("Timed out waiting for SC2 reset slot.")
                return None
            time.sleep(0.25)

    def _release_reset_slot(self, fd):
        if fd is None:
            return
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)

    def _arm_reset_alarm(self):
        try:
            signal.signal(signal.SIGALRM, signal_handler)
            signal.alarm(max(1, self.reset_attempt_timeout))
            return True
        except ValueError:
            return False

    def _clear_reset_alarm(self, armed):
        if armed:
            signal.alarm(0)

    def reset(self, seed=None, mode=None):
        super().reset()

        self.game_end_flag = False

        self.reset_failed = False
        self.game_end_data = None
        self._next_policy_action_id = 0

        for i in range(self.reset_retries):
            slot_fd = self._acquire_reset_slot()
            alarm_armed = self._arm_reset_alarm()
            try:
                self._cleanup_resources()

                port = get_unused_port()

                try:
                    self._setup_server(port)
                except OSError as exc:
                    _log_sc2_runtime(f"port already in use or unavailable: {exc}")
                    continue

                start_event = mp.Event()
                ready_event = mp.Event()

                self.env_a_process = mp.Process(
                    target=self._start_env_a_process,
                    args=(seed, port, start_event, ready_event),
                    daemon=True
                )
                self.env_a_process.start()

                start_event.set()
                if not ready_event.wait(timeout=20):
                    _log_sc2_runtime("SC2 child process was not ready in 20s.")
                    self._cleanup_resources()
                    continue

                self.server_socket.settimeout(self.accept_timeout)
                try:
                    self.conn, addr = self.server_socket.accept()
                except socket.timeout:
                    self._cleanup_resources()
                    _log_sc2_runtime("Server timeout")
                    continue

                self._send_to_a({"msg": "RESET"})
                obs_dict = self._receive_from_a()
                if not obs_dict or 'obs' not in obs_dict:
                    _log_sc2_runtime("RESET returned no observation; cleaning up SC2 subprocess.")
                    self._cleanup_resources()
                    continue
                self.obs = obs_dict['obs']
                self._mark_terminal_if_done(obs_dict)
            except TimeoutException:
                _log_sc2_runtime(f"SC2 reset attempt timed out after {self.reset_attempt_timeout}s.")
                self._cleanup_resources()
                continue
            except OSError as exc:
                _log_sc2_runtime(f"SC2 reset socket error: {exc}")
                self._cleanup_resources()
                continue
            except Exception as exc:
                _log_sc2_runtime(f"SC2 reset attempt failed: {exc}")
                self._cleanup_resources()
                continue
            finally:
                self._clear_reset_alarm(alarm_armed)
                self._release_reset_slot(slot_fd)

            break
        else:
            msg = "SC2 reset failed after retries."
            if self.fail_fast_on_env_error:
                raise RuntimeError(msg)
            _log_sc2_runtime(f"{msg} returning a failed one-step episode.")
            self.reset_failed = True
            self.obs = getattr(self, "obs", "SC2 reset failed.")

        return self.obs

    def step(self, action):
        # 向 A 发送动作
        # print("action:\n", action)
        # Direct text callers have no validator verdict and cannot earn this bonus.
        plan_format_valid = False
        trace_origin = action.get("trace_origin") if isinstance(action, dict) else None
        if isinstance(action, dict):
            plan_format_valid = action["plan_format_valid"]
            action = action["text"]

        if self.reset_failed:
            self.reset_failed = False
            if self.fail_fast_on_env_error:
                raise RuntimeError("SC2 reset failed before step.")
            data = self._failed_step_data("SC2 reset failed before step.")
            return gymnasium_transition(data, False)

        policy_action_id = None
        if not self.game_end_flag:
            pending_data = self._receive_pending_from_a(timeout=self.pre_action_terminal_drain_timeout)
            if pending_data:
                data = pending_data
                self.obs = data['obs']
                self._mark_terminal_if_done(data)
                return self._policy_transition(data, True)

            try:
                self._next_policy_action_id += 1
                policy_action_id = self._next_policy_action_id
                message = {"msg": "ACTION", "value": action,
                           "policy_action_id": policy_action_id,
                           "plan_format_valid": plan_format_valid}
                if trace_origin is not None:
                    message["trace_origin"] = trace_origin
                self._send_to_a(message)
            except (OSError, AttributeError) as exc:
                pending_data = self._receive_pending_from_a(timeout=max(1.0, self.pre_action_terminal_drain_timeout))
                if pending_data:
                    data = pending_data
                    self.obs = data['obs']
                    self._mark_terminal_if_done(data)
                    return self._policy_transition(data, True, policy_action_id)
                if isinstance(exc, (BrokenPipeError, ConnectionResetError)):
                    _log_sc2_runtime(f"SC2 subprocess socket closed while sending ACTION: {exc}; marking this episode done.")
                    data = self.game_end_data if self.game_end_data else self._failed_step_data(self.obs)
                    self.game_end_data = data
                    self.game_end_flag = True
                    return gymnasium_transition(data, False)
                if self.fail_fast_on_env_error:
                    raise RuntimeError("Failed to send ACTION to SC2 subprocess.") from exc
                _log_sc2_runtime(f"Failed to send ACTION to SC2 subprocess: {exc}")
                data = self._failed_step_data(self.obs)
                self.game_end_data = data
                self.game_end_flag = True
                return gymnasium_transition(data, False)

            # 阻塞等待 A 的反馈
            data = self._receive_from_a()
            if not data:
                if self.fail_fast_on_env_error:
                    raise RuntimeError("SC2 subprocess returned no data.")
                _log_sc2_runtime("SC2 subprocess returned no data; marking episode failed instead of crashing.")
                data = self._failed_step_data(self.obs)

            self.obs = data['obs']
            self._mark_terminal_if_done(data)
        else:
            data = self.game_end_data

        # print(f"Step Reward: {data['reward']}")

        # 这里的 data 结构由 EnvA 的 get_action 定义
        return self._policy_transition(data, True, policy_action_id)

    def _policy_transition(self, data, action_valid, policy_action_id=None):
        # Sending to a socket is not proof that get_action() consumed the answer.
        consumed_id = data["last_consumed_policy_action_id"]
        applied = policy_action_id is not None and consumed_id == policy_action_id
        if not applied and not data["done"]:
            raise RuntimeError("SC2 nonterminal response did not acknowledge the policy action")
        return gymnasium_transition({**data, "policy_action_applied": applied,
                                     "policy_action_id": policy_action_id}, action_valid)

    def _receive_pending_from_a(self, timeout=0.0):
        if self.conn is None:
            return None
        try:
            readable, _, _ = select.select([self.conn], [], [], timeout)
        except (OSError, ValueError):
            return None
        if not readable:
            return None
        return self._receive_from_a()

    def _mark_terminal_if_done(self, data):
        if not data or not data.get('done'):
            return
        self.game_end_data = data
        self.game_end_flag = True
        self._ack_terminal_to_a()

    def _ack_terminal_to_a(self):
        try:
            self._send_to_a({"msg": "TERMINAL_ACK"})
        except (OSError, AttributeError) as exc:
            _log_sc2_runtime(f"Failed to send TERMINAL_ACK to SC2 subprocess: {exc}")

    def render(self):
        return self.obs

    def _start_env_a_process(self, seed, port, start_event, ready_event):
        """子进程运行函数"""
        output_dir = os.path.join(
            os.getenv("SC2_RUNTIME_LOG_ROOT", "logs"), "worker-output"
        )
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(
            output_dir, f"sc2-env-{port}-{os.getpid()}.log"
        )
        output_stream = open(
            output_path, "a", encoding="utf-8", errors="backslashreplace", buffering=1
        )
        os.dup2(output_stream.fileno(), 1)
        os.dup2(output_stream.fileno(), 2)
        sys.stdout = output_stream
        sys.stderr = output_stream

        # 等待父进程信号
        start_event.wait()
        time.sleep(0.1)

        try:
            if os.getenv("SC2_SEED_PYTHON_RANDOM", "0") == "1":
                random.seed(seed)
            env_a = self.env_a_class(host=self.host, port=port, seed=seed)
            # 通知父进程子进程已启动
            ready_event.set()
            env_a.run(port=port)
        except Exception as e:
            print(f"Error in env_a_process: {e}", flush=True)
            traceback.print_exc()
            raise

    def _send_to_a(self, data):
        if self.conn is None:
            raise ConnectionError("SC2 subprocess socket is not connected")
        payload = pickle.dumps(data)
        self.conn.sendall(len(payload).to_bytes(4, byteorder='big'))
        self.conn.sendall(payload)

    def _receive_from_a(self):
        if self.conn is None:
            _log_sc2_runtime("SC2 subprocess socket is not connected while reading.")
            return None
        try:
            self.conn.settimeout(self.socket_timeout)
            header = self.conn.recv(4)
        except socket.timeout:
            _log_sc2_runtime(f"Timed out waiting for SC2 subprocess after {self.socket_timeout}s.")
            return None
        except OSError as exc:
            _log_sc2_runtime(f"Socket error while reading SC2 subprocess: {exc}")
            return None
        if not header:
            return None
        length = int.from_bytes(header, byteorder='big')
        chunks = []
        bytes_recvd = 0
        while bytes_recvd < length:
            try:
                chunk = self.conn.recv(min(length - bytes_recvd, 4096))
            except socket.timeout:
                _log_sc2_runtime(f"Timed out while reading SC2 payload after {self.socket_timeout}s.")
                return None
            except OSError as exc:
                _log_sc2_runtime(f"Socket error while reading SC2 payload: {exc}")
                return None
            if not chunk:
                return None
            chunks.append(chunk)
            bytes_recvd += len(chunk)
        return pickle.loads(b''.join(chunks))

    def _failed_step_data(self, obs):
        raise RuntimeError('SC2 transport/reset failure is not a game outcome')

    def _cleanup(self):
        """清理资源"""
        if self.conn:
            self.conn.close()
            self.conn = None
        if self.env_a_process:
            self.env_a_process.terminate()
            self.env_a_process.join()
            self.env_a_process = None

    def close(self):
        """完全关闭环境"""
        self._cleanup_resources()
        self._cleanup()
        self._safe_close(self.server_socket)
        self.server_socket = None
        self._is_closed = True

class SC2Worker:
    """
    Ray remote actor that holds an independent SC2Env instance.
    """
    def __init__(self, config):
        output_dir = os.path.join(
            os.getenv("SC2_RUNTIME_LOG_ROOT", "logs"), "worker-output"
        )
        os.makedirs(output_dir, exist_ok=True)
        self._worker_output_stream = open(
            os.path.join(output_dir, f"sc2-worker-{os.getpid()}.log"),
            "a",
            encoding="utf-8",
            errors="backslashreplace",
            buffering=1,
        )
        os.dup2(self._worker_output_stream.fileno(), 1)
        os.dup2(self._worker_output_stream.fileno(), 2)
        sys.stdout = self._worker_output_stream
        sys.stderr = self._worker_output_stream
        self.env = SC2Env(config)

    def step(self, action):
        return self.env.step(action)

    def reset(self, seed=None):
        # SC2Env.reset 接受 seed 和 mode 参数，但我们这里只传递 seed
        # 注意：SC2Env.reset 可能没有实现种子设置，这里为了接口兼容性保留
        obs = self.env.reset(seed=seed)  # 如果 SC2Env.reset 不支持 seed，可改为 self.env.reset()
        # 返回 obs 和一个空字典作为 info（标准接口要求返回 (obs, info)）
        return obs, {}

    def render(self, mode_for_render=None):
        # SC2Env.render 没有参数，忽略 mode
        return self.env.render()

    def close(self):
        self.env.close()

    def truncate(self):
        self.env.truncate()

class SC2MultiProcessEnv(gym.Env):
    """
    Ray-based wrapper for the SC2 environment.
    Each Ray actor creates an independent SC2Env instance.
    The main process communicates with Ray actors to collect step/reset results.
    """
    def __init__(self,
                 config,
                 seed=0,
                 env_num=1,
                 group_n=1,
                 mode='rgb_array',
                 resources_per_worker={"num_cpus": 0.1},
                 is_train=True):
        """
        - config: configuration dictionary for SC2Env (passed to each worker)
        - env_num: number of different environments
        - group_n: number of same environments in each group (for GRPO and GiGPO)
        - seed: random seed for reproducibility
        - mode: rendering mode (ignored by SC2Env, kept for compatibility)
        - resources_per_worker: resources for each Ray actor
        - is_train: whether in training mode (affects seed range in reset)
        """
        super().__init__()

        if not ray.is_initialized():
            ray.init()

        self.is_train = is_train
        self.group_n = group_n
        self.env_num = env_num
        self.num_processes = env_num * group_n
        self.mode = mode
        self.seed = seed
        self.reset_index = 0
        self.validation_rng = np.random.RandomState(seed)
        self.training_rng = np.random.RandomState(seed)
        np.random.seed(seed)

        _log_sc2_runtime(f"Env num and group n {env_num} {group_n}")

        # Create Ray remote actors
        env_worker = ray.remote(**resources_per_worker)(SC2Worker)
        self.workers = []
        for _ in range(self.num_processes):
            worker = env_worker.remote(config)
            self.workers.append(worker)

    def step(self, actions):
        """
        Perform step in parallel.
        :param actions: list[int], length must match self.num_processes
        :return:
            obs_list, reward_list, terminated_list, truncated_list, info_list
            Each is a list of length self.num_processes
        """
        assert len(actions) == self.num_processes

        futures = [worker.step.remote(action) for worker, action in zip(self.workers, actions)]
        results = ray.get(futures)

        obs_list, reward_list, terminated_list, truncated_list, info_list = [], [], [], [], []
        for obs, reward, terminated, truncated, info in results:
            obs_list.append(obs)
            reward_list.append(reward)
            terminated_list.append(terminated)
            truncated_list.append(truncated)
            info_list.append(info)

        return obs_list, reward_list, terminated_list, truncated_list, info_list

    def truncate(self, indices):
        ray.get([self.workers[index].truncate.remote() for index in indices])

    def reset(self):
        """
        Perform reset in parallel.
        :return: obs_list and info_list, the initial observations for each environment
        """
        # Generate seeds
        if self.is_train:
            if os.getenv("SC2_ORIGINAL_GAME_SEEDS") == "1":
                # Match SC2_online.zip's per-game randint(0, 100), without
                # consuming model RNG. The recorded reset index fixes replay.
                rng = random.Random(self.seed)
                draws = (self.reset_index + 1) * self.num_processes
                seeds = [rng.randint(0, 100) for _ in range(draws)][-self.num_processes:]
            else:
                seeds = self.training_rng.randint(0, 2**16 - 1, size=self.env_num)
                seeds = np.repeat(seeds, self.group_n).tolist()
            seed_record = {
                "base_seed": self.seed,
                "reset_index": self.reset_index,
                "seeds": seeds,
            }
            _log_sc2_runtime(f"SC2 training seed record: {seed_record}", flush=True)
            seed_log_path = os.getenv("SC2_TRAIN_SEEDS_LOG_PATH")
            if seed_log_path:
                os.makedirs(os.path.dirname(seed_log_path), exist_ok=True)
                with open(seed_log_path, "a", encoding="utf-8") as seed_log:
                    seed_log.write(json.dumps(seed_record, sort_keys=True) + "\n")
            self.reset_index += 1
        else:
            seeds = self.validation_rng.randint(2**16, 2**32 - 1, size=self.num_processes).tolist()
            seed_record = {
                "base_seed": self.seed,
                "reset_index": self.reset_index,
                "seeds": seeds,
            }
            _log_sc2_runtime(f"SC2 validation seed record: {seed_record}", flush=True)
            seed_log_path = os.getenv("SC2_EVAL_SEEDS_LOG_PATH")
            if seed_log_path:
                os.makedirs(os.path.dirname(seed_log_path), exist_ok=True)
                with open(seed_log_path, "a", encoding="utf-8") as seed_log:
                    seed_log.write(json.dumps(seed_record, sort_keys=True) + "\n")
            self.reset_index += 1

        futures = [worker.reset.remote(seed) for worker, seed in zip(self.workers, seeds)]
        results = ray.get(futures)

        obs_list = [obs for obs, _ in results]
        info_list = [info for _, info in results]
        return obs_list, info_list

    def render(self, mode='rgb_array', env_idx=None):
        """
        Request rendering from Ray actor environments.
        Can specify env_idx to get render result from a specific environment,
        otherwise returns a list from all environments.
        """
        if env_idx is not None:
            future = self.workers[env_idx].render.remote(mode)
            return ray.get(future)
        else:
            futures = [worker.render.remote(mode) for worker in self.workers]
            return ray.get(futures)

    def close(self):
        """Close all Ray actors"""
        for worker in self.workers:
            ray.kill(worker)

    def __del__(self):
        self.close()

def build_sc2_envs(
        config,
        seed=0,
        env_num=1,
        group_n=1,
        mode='rgb_array',
        resources_per_worker={"num_cpus": 0.1},
        is_train=True):
    """
    Build a SC2MultiProcessEnv instance.
    """
    return SC2MultiProcessEnv(
        config=config,
        seed=seed,
        env_num=env_num,
        group_n=group_n,
        mode=mode,
        resources_per_worker=resources_per_worker,
        is_train=is_train
    )
