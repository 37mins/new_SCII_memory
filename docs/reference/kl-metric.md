# KL 指标汇总修复

`DataParallelPPOActor.update_policy()` 原先在每个 micro-batch 执行
`metrics["actor/kl_loss"] = ...`，每个 worker 只留下最后一个值。它不是整个
global step 的 KL 均值，更不是第一次参数更新前的 KL。历史日志缺少被覆盖
的值，不能从最终指标反算出正确均值。

本修复只将该行改为现有 `append_to_dict`，与旁边 `actor/ppo_kl`、policy
loss 的收集方式一致，再交给现有 `reduce_metrics`。这遵循仓库中已固定的
VERL 指标收集约定，不新增聚合框架或 KL 公式。它得到的是参与汇总的
micro-batch KL 值的算术均值，不是全局 token 加权均值，也不代表每个值在
同一组模型参数下计算；一个 global step 内可以发生多次 optimizer 更新。

改动不涉及 KL loss 张量、系数、mask、反向传播、梯度裁剪、优化器或学习率。
`actor/ppo_kl` 不变。此项修复不能保证 KL 变小，更不是策略一致性修复。

测试执行真实指标语句以及 CPU 上的真实 `update_policy()` 方法，使用小型
可微模型代替大模型；两次运行比较每次 KL、梯度、更新参数和其他指标。
未启动 SC2、大模型或分布式训练，不能冒充 FSDP 集成验证。

```bash
python3 -B -m unittest discover -s tests -p 'test_r6_kl_metrics.py' -v
```

历史 c24bec5 A/B 和当前 f6a784c 八步运行均未应用本修复。新的
`20260906-qwen-kl-metric-v1` 才单独验证新口径；历史数字不得与新口径
直接拼成连续曲线。原候选和 CPU 测试来自 163dce4/0c2e52b，此处只移植
已确认的一行与原回归测试，不改变其技术依据。

09-06 后续：从 fa76c3a 移植同一行及 c16384c 的原测试；此前 48 的 GPU
尝试被依赖缺失/显存竞争中断，不能算 GPU 验证通过。正在运行的 44a6e9d
四步快照不受开发改动影响；新结果使用新的 RUN_ID，不覆盖历史值。

Integrated Fixes v1 纳入这一行已通过CPU等价测试的统计修复，但不纳入未完成
GPU验证的额外诊断forward。整合实局是否通过，以新整合record为准；不据此
宣称PPO KL漂移根因已经解决。

## 跨卡遗漏（09-06 r7 后续）

r7 四卡保存的真实 PPO K1 均值分别为 4.866162、5.327567、7.488033、
6.291207；日志 4.866 只来自 rank0。`DataProto.concat` 原先直接使用
首项 `meta_info`，导致其他卡的 actor/critic 指标被丢弃。卡内 append 修复
并未覆盖这条跨卡路径。

现在仅对 `meta_info["metrics"]` 合并各卡标量/列表，展平后交给原有
`reduce_metrics`，不修改输入包或其他 metadata，不改变 batch、loss、
梯度或优化器。当前 KL 是全部 micro-batch 的等权平均；r7 同口径应为
5.993242，不是 token 加权平均。含 max/min 的指标仍沿用原 reducer。
没有逐卡原始记录的旧实验不反算、不覆盖历史日志。

依据是 [VERL 官方 DataProto.concat 的 metrics 特殊合并](https://github.com/volcengine/verl/blob/3af734b6145bc84903ed33f228ecab5e6c93cb00/verl/protocol.py#L829)。
仅适配本版本实际使用的字典内标量/列表；不升级依赖，也不移植其他
metadata 合并或兼容逻辑。这是统计修复，不是概率漂移修复。

回归：`python3 -B -m unittest discover -s tests -p test_cross_rank_metrics.py -v`。
测试真实 `DataProto.concat` 和 DP collector，覆盖四卡数据、不同 micro 数、
递归合并、输入不变、tensor/NumPy payload 不变及既有 mean/max 口径。
不启动 Ray 服务或 GPU；真实 r7 trace 再汇总另记在实验 record。
