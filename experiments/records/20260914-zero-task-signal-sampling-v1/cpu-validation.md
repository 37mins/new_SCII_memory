# CPU 验证，2026-09-14

共35项通过：20项实际 GiGPO/driver 测试、7项收尾与 effective-config 测试、8项旧入口回归。SC2、actor RPC、checkpoint IO 在 driver 测试中使用小型替身；没有加载大模型、使用 GPU 或产生学习效果结论。

43节点使用已有 `/root/new-scii-spb-level5-96-20260908/python` 依赖，源码位于独立 CPU staging `/root/new-scii-zero-signal-cpu-20260914-r1`。这是开发候选测试目录，不是正式冻结实验部署。命令：

```bash
CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
PYTHONPATH=.:/root/new-scii-spb-level5-96-20260908/python \
/usr/bin/python3 -m unittest discover -s tests -p test_sc2_task_signal.py -v
```

20项用原计分函数和实际 `fit()` 验证：全零32局不调用 old-logprob/actor RPC、不保存伪 checkpoint；两组跳过、两组接受时只有2次 global-step更新、最终保存实际step6；对照同32局更新4次。还覆盖默认关闭、采样上限、不重试错误、败局保留、mask、输入不变和 NumPy RNG 不变。

本地以下3条命令分别通过7、3、5项：

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tests -p test_signal_sampling_record.py -v
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tests -p test_feedback_rl_launch.py -v
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tests -p test_fsdp_rl_launch.py -v
```

新 A/B dry-run 的有效配置只有 `trainer.sc2_skip_zero_signal=False/True` 不同；同一A4、4卡、每次8局、总4次和同一 Coverage seed hook。收尾测试拒绝漏计跳过的游戏、全零无更新冒充训练通过、缺失 summary、计数错误和非有限指标。`bash -n`、`git diff --check`、`shasum -a 256 -c configs/system-v1.sha256` 均通过。

最初两轮 CPU 失败保留：v1 有测试夹具问题（TensorDict 遍历，以及误把四舍五入后同 reward 当作精确零）；v2 揭示**本候选**使用 DataProto deepcopy 导致原 TensorDict 被锁，已改为直接克隆张量。v3为18项通过，v4增加默认/边界测试后20项通过。这些是候选开发返工，不能记成项目历史 bug 的修复成果。

原始日志均在43节点 `/root/new-scii-zero-signal-cpu-20260914-r1-cpu-vN.log`，SHA-256：

| 日志 | SHA-256 |
| --- | --- |
| v1 | `4f20f1a03ccf51318666dec92d041af3206ab4840ecfe7bb01a7f0c08d672ab8` |
| v2 | `8534620af6f968de9b1296e0b7396b23d7cf8cce045ab73c1d482f135d80ca3b` |
| v3 | `21aa35c6a7da463353c4f1b2a6b79bd7e9ca7c2b5f360baff5be3046889c0686` |
| v4 | `79c715f9d57c889b972092b50c1b176c83ed88b0faa7488244353a0f3b2200fe` |

实测源文件哈希：`ray_trainer.py` 为 `492dce6d222a222ab36031984f3d020a04621003515b1e9755bed03c4a2ddeb9`；`sc2_task_signal.py` 为 `f5ef9b52907bbd6dc104bb9f935c6cbef031c1b68f6cbb381418e9f1a953ec37`；20项测试文件为 `7e2b2a80aa923f7a97f072297fb48e152131fd459b9d476fdd71ea0984fa64c0`。

仍未验证：真实分布式采样/更新、模型梯度和 checkpoint、相同游戏预算下的独立游戏表现。通过CPU不代表学习问题修复。
