# 零 advantage：2026-09-10 只读审计

结论：所查15批没有有意义的组内回报差，GiGPO的episode和step两项均归零。
不是发现了新的梯度/分布式实现bug。13批CPU重算严格为0；两项实验的step20
存在最大5.7220458984375e-6的FP32相减残差，日志保留三位小数显示0。
因此此前“全部精确为0”的说法需要纠正。未改运行代码、未启动GPU实验。

## 对象与方法

- Level6：43八卡，SHA `6a1657f7e045dcf998495a3d0adfb331572bf60d`，
  RUN `20260910-spb-fsdp-layer-wrap-v1-spb2-l6-g2-opening-quota-rl96-8gpu-s0-r1`。
- Level7：44 GPU4–7，SHA `90b3fdc2fb7fadc1230861306b20c27540d9b736`，
  RUN `20260910-spb-fsdp-layer-wrap-v1-spb2-l7-g2-opening-quota-resume4to96-4gpu-s0-r1`。
- 共享根：`/mnt/data0601/yangjinyue/new_SCII/experiments/20260910-spb-fsdp-layer-wrap-v1/runs`。
  读取各run的`effective_config.json`、`prompt_response.jsonl`、
  `episode_outcomes.jsonl`与console；没有读取大checkpoint。

两节点实际冻结`gigpo/core_gigpo.py`的SHA256均为
`f135bd272518b49a58762da3474e2d4bc90906b053f1ae820d8fb05c2064148b`；
collector、reward manager、batch adjustment、actor loss文件也逐一核对一致。

在44已有Python依赖下，隐藏CUDA、Torch/BLAS单线程、30秒timeout执行CPU重建，
退出0且`torch.cuda.is_initialized()`为False。用AST加载冻结文件原函数，按
每个env的结局step_index分割真实回答，复用`policy_transitions`处理收尾，核对
每局决策数。history关闭时日志prompt等于anchor；以一个有效token表示每回答
的广播标量优势，复用原discounted-return、episode_norm_reward、build_step_group、
step_norm_reward函数。这是reward/分组标量重建，不是保存的原PPO全张量重放，
不声称验证了参数更新。所有所查样本均合法且条数整除卡数，无需猜测复制项。

## 实际结果

| 批次 | 每局累计reward | 每局真实回答数 | CPU重算joint最大绝对值 |
| --- | --- | --- | --- |
| L6 step15、16 | 8局均20.5 | 各5 | 0 |
| L6 step18、19 | 8局均20.5 | 各2 | 0 |
| L6 step20 | 8局均−49.45000076293945 | 各2 | 5.722e-6 |
| L7 step6、9、13、15、16、18、19、21、22 | 8局均20.5 | 各2 | 0 |
| L7 step20 | 8局均−49.45000076293945 | 各2 | 5.722e-6 |

实际配置：一批8局、同一游戏seed、同一episode group；GiGPO mean_norm、
gamma0.95、step权重1、精确anchor匹配、group filter关闭、KL不进入reward。

episode项是`(R_i−同组8局均值)/该局回答数`。所有R相同，数学上为0。
step项是`该回答的折扣回报−相同anchor组均值`。本次每个anchor组都是8条，
不是singleton误分组；组内discounted return全部相同，数学上也为0。
两回答全胜样本的reward序列为`[0,20.5]`、returns约`[19.475,20.5]`；
全败样本为`[0,−49.45]`、returns约`[−46.9775,−49.45]`。差的是不同
位置的回报，不是同一位置组内的回报，所以不能因此产生step相对优势。

L6与L7的step18、20各自8局的整段回答序列逐字相同，每局只有两条回答。
例如L6两段为：

```text
<answer>
- supplydepot x 4
- barracks x 4
- marine x 24
</answer>

<answer>
- marine x 42
- supplydepot x 8
- barracks x 3
- reactor barracks x 3
</answer>
```

L6 step18获8胜、step20全败，但算法不跨不同global step/seed比较奖励。
L7 step6有3种整局回答序列，仍得到完全相同的奖励与step回报：不仅逐字重复
会导致归零，回答有变化但奖励无法区分也会如此。
L6 step1的8局回答序列均不同，step18/20各只有1种；这是采样多样性下降的
现象证据，不单独证明是prompt、温度或某次参数更新造成。temperature0.4、
top_p0.9、192token预算未改。

## 解释边界与代码

- `gigpo/core_gigpo.py:160–170,235–241,381–386`：两级相对优势。
- `agent_system/multi_turn_rollout/rollout_loop.py:410–421`：8局一组、各局独立traj ID。
- `agent_system/reward_manager/episode.py:72–79`：完整episode reward写入回答末token。
- `verl/trainer/ppo/metric_utils.py:116–146`：统计有效回答token，不是padding均值。
- `verl/utils/logger/aggregate_logger.py:27`：三位小数显示，非全精度归零证据。
- `verl/workers/actor/dp_actor.py:439–467`：总loss还含entropy系数0.001与ref-KL系数0.01。
  即使PPO奖励项归零，仍有正则、优化器历史动量及权重衰减影响；不能说参数停止更新。

[官方GiGPO源码](https://github.com/langfengQ/verl-agent/blob/master/gigpo/core_gigpo.py)
同样用组均值构造相对优势。当前项目的episode等轨迹计权不是与upstream逐字相同，
但这些同奖励、同回答长度批次不因该差异产生有意义的优势。这里只引用中心化
机制，不声称完整算法实现审计通过。

当前最明确的问题是组内探索/回报差不足，消耗训练预算却缺少区分好坏的信号。
尚未定位多样性下降的唯一原因，也未证明这些批次的正则更新改善了游戏表现。
不据此改advantage公式、混不同seed凑奖励差、自动重采样或停训；两项训练继续冻结。
