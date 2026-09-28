# Step24 的训练信号：只读结论

2026-09-11，未改生产代码、配置或运行状态，未使用GPU。审计对象仅为本次固定输入
对照所用的 **Level7 step24**，不是Level6 step63。结论：训练目标存在少调用偏向，
后期奖励信号大量归零；训练未覆盖刚测的执行失败输入。尚未证明哪次参数更新导致
纠错顺序退化，也未定位新的优化器或执行层实现bug。

## 核实的链路

| 环节 | 已有记录与冻结代码证据 | 能说明什么 |
| --- | --- | --- |
| 模型看到了什么 | global1–24，192局、582条真实回答；输入中剩余人口最小3，没有人口余量0，也没有执行失败反馈；与纠错对照的两份完整prompt均不相同 | 上轮测的是训练中未出现的故障情形。没有证据说训练曾明确教它忽略这类反馈 |
| 计划拿到什么reward | step5同一第二轮输入，env0输出Marine42等大计划，两次调用后赢；env4输出Marine2等小计划，四次调用后赢。两局reward均20.5 | 奖励没有在这两个胜局间区分战损、资源转化或后续适应性；两计划都以Marine开头，不是纯顺序A/B |
| reward变成什么advantage | 同样上述输入，大计划本轮折扣回报20.5，小计划18.501249；对应joint advantage分别为正、负 | gamma0.95按调用次数折扣，对这组同终局收益轨迹偏向少调用，并非证明大计划战术更好 |
| 后期还有没有奖励信号 | 12/24批joint advantage为0或浮点残差；step18–24连续7批全部如此，包含全胜与全败 | 这时高胜率不能说明正向奖励学习仍在发生，全败也未必提供负向相对优势 |
| 有没有更新参数 | 按每批真实回答数、四卡凑整及mini8推算80次optimizer更新，与先前真实step24加载的全rank Adam计数80一致 | 确实更新过，不是24次optimizer更新。不能据此证明每次参数变化正确，或奖励造成了某个token变化 |

### Step5 的实际回答与优势

同一query哈希前缀`c64f73642031`；源resume日志第9、13行。

```text
env0：marine x42 → supplydepot x8 → barracks x3 → reactor barracks x3 → scv x2 → refinery x1
env4：marine x2  → supplydepot x2 → barracks x2 → reactor barracks x2 → scv x2 → refinery x1
```

此时观测为580矿、剩余人口11，**不是人口已满的纠错输入**。
两轨迹reward序列分别为`[0,20.5]`与`[0,0,0,20.5]`，episode相对项为0。
复用已完成的[冻结函数CPU审计](../20260910-spb-fsdp-layer-wrap-v1/sampling-reward-audit.md)：
18条回答复制2条凑20，实际复制索引未留存。枚举153种可能补齐，env0本轮joint
advantage为`+0.199875～+0.444166`，env4为`−1.798876～−1.554585`，符号不变。
不能把范围伪装成保存的精确原值；此次未重跑该枚举或修改gamma。

它也**不是永远奖励大计划**：step7七条两次调用的大计划轨迹都败，只有一条
继续规划的轨迹获胜；后者还包含无效回答与规则处理，不能只归因于计划大小。
正advantage是loss的输入，不等于某个token概率必然上涨或其实际参数贡献已查清。

### 零信号批次复核

本次在44隐藏CUDA、Torch/BLAS单线程、30秒timeout内，AST加载实际冻结
`gigpo/core_gigpo.py`原函数，仅用每条回答一个有效token重算广播标量；退出0，
`torch.cuda.is_initialized()`为False。不加载模型/optimizer，不计算反向传播。
12批各16条合法真实回答、各局两次调用、无复制补齐，因此不需要猜测复制索引。

- 严格为0：step6、9、13、15、16、18、19、21、22、23、24。
- step20数学上组内相同，FP32重算最大绝对值`5.7220458984375e-6`；不说精确为0。
- 12批共24次optimizer更新。reward项归零时，loss仍有entropy系数0.001和
  reference KL系数0.01；Adam历史状态及weight decay也仍生效，不能说模型停止更新。
- step18全胜、step20全败、step24全胜，但上述批次都无可用的组内好坏差。
  全部24步共127/192胜，仅为训练rollout统计，不是独立评测。

## 来源与复核入口

44两段冻结部署均clean：

- step1–4：`/root/new-scii-spb-fsdp-20260910/code-cfdd3b4`，
  SHA `cfdd3b468cf9bbe3a1f85ce8dc9a5a9ea530ffbd`。
- step5–24：`/root/new-scii-spb-fsdp-20260910/code-90b3fdc`，
  SHA `90b3fdc2fb7fadc1230861306b20c27540d9b736`。

两run根为`/mnt/data0601/yangjinyue/new_SCII/experiments/20260910-spb-fsdp-layer-wrap-v1/runs`，
RUN后缀分别是`spb2-l7-g2-opening-quota-rl4-4gpu-s0-r1`与
`spb2-l7-g2-opening-quota-resume4to96-4gpu-s0-r1`，均有experiment同名前缀。
本次只读取manifest、effective config、prompt_response、episode_outcomes、metrics、
training_seeds及冻结源码；按reset_index/逐env终局切开，仅取global1–24，排除终局
占位回答，核对每局真实决策数及整局reward与已归档指标。两run完整文件SHA、逐批
数据与真实回答摘录见[step24-training-signal.json](step24-training-signal.json)。

只读统计可用`audit_step24_training_signal.py --first-run <原四步run> --resume-run
<续训run> --output <未使用的JSON路径>`复核；本次对本地约4MB日志副本运行，退出0。
余量用数字字段解析，不能用字符串`0 remains unused`子串匹配，否则会误中10/20。

相关冻结位置：

- `gigpo/core_gigpo.py:87,160,235,381`：调用折扣、episode归一、step归一。
- `agent_system/reward_manager/episode.py:72`：完整episode reward映射到回答末token。
- `verl/trainer/ppo/ray_trainer.py:1219,1225,1311,1328`：回报、补齐、无效惩罚、优势。
- `verl/workers/actor/dp_actor.py:439–484`：PPO、entropy、ref-KL、backward及optimizer。

前两冻结版本的GiGPO文件SHA256相同：
`f135bd272518b49a58762da3474e2d4bc90906b053f1ae820d8fb05c2064148b`。
collector/reward-manager/actor文件也已对两部署逐一核验；trainer差异是已登记续训
状态边界，不用当前工作树替代这些来源。诊断开关未启用，没有保留完整原advantages、
补齐/均衡后的样本顺序、逐回答实际loss、梯度或参数差，不能从三位小数聚合日志恢复。

## 收口

证据支持优先处理**训练信号的区分度与训练场景覆盖**，不支持继续增添执行层功能。
上轮同输入5/8→0/8的局部退化事实仍成立；本审计提供可疑训练偏向和覆盖缺口，
但没有完成“某次更新导致不响应失败反馈”的数值因果定位。未宣布reward公式有bug，
不凭此改奖励、gamma、prompt、loss或启动新的GPU对照。
