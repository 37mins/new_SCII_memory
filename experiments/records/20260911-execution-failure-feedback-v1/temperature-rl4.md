# 温度探索候选：A4再训练四步并独评

2026-09-12 09:25；状态 **budget approved / not promoted**。用户已确认新增
43 GPU0–3、总2小时、约60GiB预算；只在当前温度对照支持继续后启动。
当前T0.4/0.8纯评测已完成：均0/8，高温合法率34/58，未通过有用探索晋级判断；
见[完整结果](temperature-ab.md#完整结果与晋级结论)。不修改9ae8668部署，未排后台训练。
本轮是独立新增授权，不复用先前四卡45分钟纯评测预算。

## 问题与对照

不是假定“规则写死、模型无选择空间”。Marine数量单变量已在一个场景改变胜负，
但[A4→A8日志](a8-training-signal.md)显示关键早期回答趋同，差异主要出现于后期。
先检验提高温度能否采到合法的不同早期方案和可区分回报，再做本轮学习验证。
若当前纯评测只是乱码增加、截断增加或早期计划仍相同，不自动启动短训。
不同合法计划但未观察到回报差时，证据不足；先汇报，不能凭一次波动晋级。

对照复用已经完成的A4→A8（4ad9a60，训练T0.4），不重新跑低温四步或初始评测。
候选仍恢复A-r2/global_step_4的真实模型、优化器、per-rank RNG，用已验证的
continue_a4边界进入新游戏。这是匹配历史对照，不冒充同时随机分配的A/B，也不是
旧SC2进程的逐位续接。两组基础模型、数据路径、系统文件需与冻结SHA逐项核对。

| 项目 | 本轮候选与已有对照 |
| --- | --- |
| 唯一训练变量 | rollout temperature 0.4→0.8；不是只改评测温度 |
| 更新 | 从global4开始，再做global5–8，每步8局；不是4次optimizer update |
| 实际训练seed | 45891、21243、30403、32103，每步同seed8局，保持原顺序 |
| 训练系统 | 沿用A：失败反馈关闭；当前prompt、约15条/xN、192token、过滤、规则补齐、BO、微操、reward、history不变 |
| 数值/优化 | FP32参数与归一化、BF16 autocast、cache、SWA dropout0、分层FSDP、lr1e-6及原loss不变 |
| 保存 | 仅新增step8完整checkpoint，约59.2GiB；保存后恢复进行纯评测 |
| 独立评测 | 仍T0.4、反馈开，与原始/A4/A8已完成独评一致；不让训练温度混入评测 |
| 评测seed | 42613、52620、17089、4018390961、3236598124、3297512220、773351930、1098341402，各1局 |
| 已批准预算 | 43 GPU0–3串行训练→评测，总上限2小时（8 GPU小时），不复制大权重、不自动重试/续96步 |

训练反馈关、评测反馈开是已有A与A4/A8评测口径，不是本次新增变量；结果仅说明
该共同评测接口下的表现，不直接外推无反馈部署。单次8seed不足以证明稳定提升。
09-12 08:56节点本地盘可用94GiB；启动前重新核验至少90GiB与GPU归属，资源不足
等待或另报，不删除旧产物、不改卡数、不同时再加载一个模型实例。

## 最小实现与用法

复用`run_a4_continue_node43.sh`，仅增加显式温度候选选择、独立RUN路径和单独授权
检查；不另造续训器。`run_feedback_train4_node43.sh`仅在显式
`FEEDBACK_RL_TEMPERATURE`且恢复A4时传入对应rollout.temperature，默认不变。
独评用显式标签`T08A8`定位这份新step8，仍由原`learning_eval.install`加载并核对8个seed。
现有冻结runtime与旧结果不变，源代码通过commit和私有bundle保存。

读取现有代码确认：fsdp_workers.py:700/708/742把rollout.temperature传给
actor old-logprob、batch meta与reference重算，dp_actor.py:336读取同一值做更新。
因此新batch按0.8重新生成及计算概率，不读历史0.4的old/ref数组；未改概率函数，
这项只读核对不等于cached/full数值一致性已全部验收。

候选RUN：

- `20260911-execution-failure-feedback-v1-spb2-A4-l6-temp08-rl4-4gpu-s0-r1`
- `20260911-execution-failure-feedback-v1-spb2-T08A8-l6-feedback-on-eval8-r1`

预算和纯评测结论确认后，独立冻结部署中由受控会话运行：

```bash
SOURCE_SHA=<新冻结SHA> SC2_A4_CONTINUE_AUTHORIZED=1 SC2_TEMPERATURE_RL_AUTHORIZED=1 \
FEEDBACK_RL_TEMPERATURE=0.8 A4_CONTROL_DIR=<未使用的控制目录> \
timeout --signal=INT --kill-after=15s 7185 \
  bash experiments/records/20260911-execution-failure-feedback-v1/run_a4_continue_node43.sh
```

总预算含加载、四步更新、保存和8局独评。训练失败、checkpoint不完整、空策略批次、
非有限指标或归因错误时不接独评，不制造样本、不重试。用完预算则保留当前产物并
记录未完成的阶段。启动前后均要核验manifest、实际恢复step与seed，不能只看目录。

## 验收

先逐step确认早期原始计划和实际执行队列是否产生合法差异、整局回报是否可区分、
优势来自胜负还是合法性/终局残存状态；记录全卡KL、clip、梯度与实际更新次数。
最后在共同T0.4评测下比A4、旧A8、新候选step8的逐seed胜负、累计reward、计划
退化/截断/有效率。只出现高温训练rollout涨分而共同条件独评不变，不称学习成功。

本地CPU已执行：`test_temperature_rl_preparation.py`3/3通过（1.026秒）；
`test_a4_continuation.py`5项中3通过、2因缺Torch跳过（0.995秒）；
`test_feedback*eval.py`18项中17通过、1因缺Torch跳过（3.004秒）。
配置对照确认训练仅一项温度变化；独评仅checkpoint路径与审计标签变化。
09-12服务器固定依赖CPU复核：上述3/3、5/5、18/18全部通过，分别0.617/3.438/5.131秒。
独立冻结部署`/root/new-scii-execution-feedback-20260911/code-a2894d4`，SHA
`a2894d48b09b95a0d72d9e5057d21e18bb6a3ef7`，clean，System-v1文件哈希全部通过。
私有源码备份`temperature-rl-prepared-r1.bundle`（需9ae8668前置），SHA256
`ae6e67c4ee71a0974a7d463f8121218aaf6636d605aa4d5764b3e769e94d820f`。
尚未启动模型加载或短训，未消耗新的GPU训练预算；09:25节点八卡已释放。
预算批准不替代前置验收：本次不自动越过门槛，不把2小时改作其他测试。
