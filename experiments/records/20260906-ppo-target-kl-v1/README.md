# PPO old-policy KL early-stop：备用方案，暂停实施

最新用户指令：不实施本候选、不启动其A/B。保留下面原方案和模拟作为备用，
当前转向冻结版本的真实Level6 RL，96步目标、先8步验收；KL 0.015不作为单独失败门槛。

状态：仅方案与已有日志的CPU模拟。2026-09-06用户暂停r33原样重跑，允许转为
稳定性候选；阈值、预算和验收标准确认前不实施训练器、不启动GPU。
r32仍为超时失败28/35更新，产物和部署不改；并非已完成35次或通过验收。

## 候选机制

以[SB3 2.7.0 PPO](https://stable-baselines3.readthedocs.io/en/v2.7.0/_modules/stable_baselines3/ppo/ppo.html)
的近似KL与early-stop为依据：令 `d=logπ_current-logπ_old`，使用
`expm1(d)-d`，超过 `1.5*target_kl` 则停止当前rollout batch的剩余更新。
该估计在旧策略采样条件下对应old→current KL；固定历史重放不是重新从原始
checkpoint采样，所以这里只称经验策略变化统计，不冒充精确分布KL。
不读reference概率、不调用现有封顶`kl_penalty`，也不截断d或KL。

取与训练相同的response mask，排除prompt和token padding、包括有效EOS。
当前固定文本路径使用`attention_mask`的response段；若使用多轮loss_mask，
必须沿用它而不能把非策略文本算入。r32原有2条复制样本在本次对照中不改权，
明确不构成真实RL的补齐计权验收。仅在detach后的统计中用FP64计算和累加。
全卡在当前四卡数据并行组汇总 `SUM(有效token KL)/SUM(有效token数)`，不是各卡均值再平均。
先mask再做expm1；有效token非有限时所有卡共同拒绝更新，不静默裁剪为有限数。
单卡零token仍参与聚合；全组零token沿用空策略批次拒绝更新，不造样本或无限重采样。

检查单位是一次optimizer mini-batch：四卡各2回答，合计8回答，而非整个
276回答rollout batch。复用这次forward的概率，在两次micro-batch backward
累积完、`_optimizer_step()`之前统一all-reduce并判断；触发则清空未提交梯度，
所有卡退出本batch的全部epoch，返回正常训练循环等待下一批rollout。
这与SB3在backward前判断的区别是计算了随后丢弃的梯度，参数/Adam状态均不变；
避免第二遍模型forward或保留多个大图。不是更新后回滚，也不修改游戏done/reward。
四卡使用同一聚合值及停止决定，不能rank-local break。

## 阈值提案与实际触发模拟

建议首个候选 `target_kl=0.01`，有效阈值0.015。
[Spinning Up官方PPO](https://spinningup.openai.com/en/latest/_modules/spinup/algos/pytorch/ppo/ppo.html)
默认target_kl为0.01，可作为起点；但它用k1且任务不同，不是本任务token-KL
阈值已验证的证明。SB3本身默认None，不把0.01说成其默认值。

| 检查口径 | 首次越过0.015 | 此时该口径KL | 同一参数的全批最差绝对Δlogp |
|---|---|---:|---:|
| 候选：当前optimizer mini-batch | 第8次更新前，已完成7次 | 0.080467 | 13.185547 |
| 仅作旁证：每次更新后完整276回答 | 第1次更新后 | 0.017051 | 7.982620 |

0.001/0.005/0.01/0.02/0.05五个target的mini-batch模拟均在第8次前触发；
不是GPU调参结果。第7次后的全批KL为0.040495，最差token的advantage为正。
所以标准mini-batch检查预计只能限制后续放大，未必及时；不能为让结果好看
偷偷改成全batch gate。全batch方案即使第1次后停，也不能撤销首次漂移。

## 最小改动与A/B预算提案

批准后复用当前高风险开发分支，新增原子commit/独立冻结目录和唯一RUN_ID：

- `verl/workers/actor/dp_actor.py`：detach的KL统计、跨卡停止与实际更新计数；
  已有loss与optimizer公式不改。停止时不能将计划更新数写成实际更新数。
- `verl/trainer/config/ppo_trainer.yaml`：新增actor `target_kl: null`，默认关闭。
- `tests/test_ppo_target_kl.py`：mask、方向、未封顶、不同rank token数、四进程
  同步退出、Adam不更新、默认关闭等价及下一batch能继续的最小回归。
- 实验record与现有`replay_fixed_tokens.py`：区分训练prefix与完整历史观测集，
  支持最多8次更新，但所有后验检查仍用相同276回答/3370token；输出真实计数。

A=`target_kl=null`，B=`target_kl=0.01`，除此之外相同。原始Instruct-2507、
FP32参数/温度、BF16 autocast、cache及原样本顺序不变，old/ref重新算且核验两组
相同。lr/loss截断/reward/prompt/BO/微操不改；两组具有相同无扰动观测。
建议各4卡并行，**每组最多8次optimizer更新、15分钟上限，共不超过2 GPU小时**；
启动时重新查空卡。不是8个global step，不启动SC2或96-step。
按r32实测58秒/次全批观测估计含准备约10–12分钟；超时原样记录，不自动重跑。
若上限内不足以取得完整终点分布，则证据不足，不删减观测来宣布完成。

## 验收与结论口径

机制验收：A在早停触发前复现r32更新前缀；A/B在共同更新前缀的loss、梯度和
参数一致。B四卡在同一mini-batch触发，实际7次（由运行证实），第8次的参数、
Adam状态不变；记录停止时的梯度，单独标记它未用于参数更新。默认关闭不改变
原训练；CPU多batch回归验证清空停止标志后下一batch可再更新。

稳定性验收提案：除了机制正确，停止后的完整batch同口径KL不得超过0.015，
并逐项报告p50/p90/p99/max、ratio越界率、目标clipfrac、最差正advantage样本与
全卡梯度。若首次触发已经复现r32同类严重尾部漂移，则**拒绝作为充分稳定性
修复**，即使后续最大差低于112、进程退出0或梯度全有限。已有模拟预计不能
通过这条验收；本轮A/B至多验证机制和限制进一步漂移，不预先承诺采用。
停止阈值不是参数更新幅度的数学保证；不靠继续缩阈值至几乎零更新宣布成功。

只有稳定性证据支持后才另行申请Level6短训：重新采样、跨batch持续更新，
保留训练前后checkpoint，同一明确种子组独立评测。少更新/低KL不是学习证据。
当前结论为**证据不足，不采用为正式版本**；运行后明确采用/拒绝/证据不足，
分别评价限制进一步漂移、修复最初漂移原因、改善游戏学习效果。

CPU复现：在现有Torch环境用`CUDA_VISIBLE_DEVICES='' python3 simulate.py <r32目录>`。
只读`.pt`与JSONL，不加载模型。[模拟结果](simulation-r32.json)保留完整时间序列；
原始输入/日志哈希见[已归档r32](../20260906-qwen-kl-metric-v1/full-updates-r32.json)。
