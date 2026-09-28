# 配额澄清prompt：同种子整局A/B计划

状态：`completed`，2026-09-10核验。用户已批准44 GPU4–7串行，两组各4局、
共同50分钟墙钟上限，不做PPO，不自动重试。本轮不修改战术或扩大生成预算。

固定41份query的A/B已完成：有效22→28，超配额19→11，未EOS16→10；但8份原合法
回答退化，B合法队列最多展开73个动作。下一项问题是：提示的局部改善是否能传到
真实执行，还是因为执行队列变长、调用时机变化或回答重复，在新观测下失效。
这属于系统效果验证，不是学习/架构对照，也不是把合法率当成生产完成率。

## 最小对照

| 项目 | 两组共同配置 |
| --- | --- |
| 模型 | 原始Panyuqi/SpikingBrain-2.0-instruct，无项目SFT/RL；manifest `40d6f1417568d5fda1107cdafd66c7441a679b03fb148851c8f226ecc9cf884c` |
| 场景 | Level6/Harder、Flat128、Terran/Air；沿用g2资产和现有阶段待办 |
| 接管 | `SC2_FIXED_BO_PREFIX_LIMIT=-1`，Planner从00:00接管；不是完整固定g2绕过模型执行 |
| 配对 | 每组4局，实际游戏种子 `[2732,2732,43567,43567]`，按env index配对；仅2个不同seed |
| 生成 | response192、prompt4096、temperature0.4/top_p0.9/top_k0，原chat template，HF/eager/cache |
| 数值 | 参数FP32、FP32温度归一化、BF16 autocast，swa_dropout=0 |
| 其余冻结 | 约15条/x N且无展开15动作硬限、名称/前置/配额过滤、合法直通、无效补兵重排、Commander guard、Executor、微操/自动战斗、history0/off、reward |

A是B192协议的原prompt，**不是此前6/6的A**；B仅使用固定query实验中的同一段配额/
示例/结束澄清文本，不增加“同名只能一行”、硬性15条、解码约束或新停止token。
两组都保持完整输入中的阶段、观测、待办及具体one-shot，不替换成静态测试query。

复用现有`run_eval.sh B`和`handoff_eval`，两组均设`SC2_ORIGINAL_BO_SEGMENT=1`及
`SC2_EVAL_RESPONSE_LENGTH=192`。现有`run_eval.sh A`会选回6/6，不能误用。
正式启动前仅把固定输入候选的文本替换接到Planner最终query，开关默认关闭；
A关闭、B开启，CPU比较完整effective config和41组query，确认除这段文本外一致。
不通过CPU就不上卡，不顺手修改过滤/执行逻辑。候选已在本地Planner增加
`clarify_bo_quota(query)`及默认关闭的`SC2_CLARIFY_BO_QUOTA`，只在原始段协议下
调用；`scripts/train_sc2.sh`仅透传开关到Ray。没有改生产输出处理代码。

基线来源为`385829c48099c4cbcf749d0ae83c4b7761117235`，其生产代码与B192的
`71fae36ccc112d4c9cd303ab084c1e0a0c7eebec`相同。两组实际运行冻结SHA均为
`2f7c5dcbcd8d51e866d31769a7d38617d1fce510`，A关闭新开关、B开启；CPU已验证关闭
与385829c相同。复用现有实验分支/记录，部署为独立clean目录，不修改
`code-385829c`、`code-71fae36`或历史run。

唯一RUN_ID（A先运行，B串行接续）：

- `20260909-bo-segment-protocol-v1-spb2-l6-g2-opening-quota-A-eval4-s0-r1`
- `20260909-bo-segment-protocol-v1-spb2-l6-g2-opening-quota-B-eval4-s0-r1`

## 资源与停止边界

18:18共享节点42/43/44/46/47/48只读快照中没有完整空闲八卡节点；44的GPU0有其他
任务，GPU1–7空闲。建议复用44已验证环境，在同一GPU4–7上A后B串行，不改两组卡数，
不部署新Python或复制模型。预计30–45分钟，申请两组合计最多50分钟墙钟、
200 GPU分钟，含两次加载及退出宽限。不复用固定query实验已结束的30分钟预算。

两组采用共同绝对截止，A结束后确认所属进程退出/GPU释放再启动B；A异常或预算
不足以继续时停止并保留记录，不自动重试、换卡或延长。若批准前已有完整八卡节点
可用，可另报告同节点两组四卡并行的配置与预算；不自行变更本方案。

23:20:26北京时间复查：44 GPU4–7均4MiB，无计算进程；GPU0为既有其他任务PID1176690，
未干预。复用现有依赖和共享模型，不下载或改旧环境。实际入口为
`launch_quota_pair_node44.sh` → `run_quota_eval.sh A/B` → 原`run_eval.sh B`，
两组同一冻结SHA，仅prompt开关0/1不同。A后最多等30秒释放GPU；B开始前至少须余
20分钟，否则标记未启动，不自动延长。这是资源预算守卫，不改变SC2终止规则。

CPU已执行：`PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:tests /private/tmp/sc2-air-results.HBI61m/decoder/bin/python -m unittest test_quota_prompt test_fixed_prompt_ab test_original_bo_segment test_factory_prereq_candidate test_planner_air_options test_valid_bo_passthrough test_r6_planner_interface test_r6_format_reward test_g2_planner_handoff test_system_release test_opening_rl_launch test_handoff_paired_eval -q`。
107项通过，12.783秒、退出0；shell语法和diff检查通过。真实Planner/Commander的
合法、超配额、错名、半截回答用例在A/B下除query外逐字段一致；关闭开关与冻结
385829c完全一致。两组dry-run完整配置归一化输出路径后仅quota开关0/1不同。
另对41份历史query逐份核对生产变换与已完成固定输入B逐字相同、可逆且validator
判定不变，41/41通过。这些是CPU/编排验收，不替代尚未开始的整局结果。

## 如何判断价值

逐局保存初始实际seed和采样rank seed，统计所有真实消费回答的联合有效率、
超配额/重复/半截输出、Planner调用次数及游戏时间间隔、合法计划展开数量和最终
Executor队列，区分模型选择与规则动作；命令接受/单位建成仅按已有事件证据记录。
记录每局自然胜负及从开局至首次终止的累计reward，包含未调用Planner的局，后者
单独标明。原始回答、配置、outcome、退出码和已有回放索引留在各自run。

继续沿原评测统计文本达到预算的代理值；固定query实验已保存精确token/EOS，
不把它冒充未来整局的精确截断率。当前整局入口没有逐回答原始token日志，本次不为
补指标扩展模型生成模块。已有回放保存可能仅索引一份，不能声称所有逐局回放齐全。

候选若在真实query上仍净改善，且没有新的循环/队列破坏或可指明的整体退化，
才建议下一次短RL；不是直接自动晋级或开96step。若改善消失或增加重复/长队列
问题，则保留原版，先定位该差异。四局胜负只描述本次样本，不证明统计显著或学习。
游戏轨迹分岔后输入和后续采样RNG自然不同；同seed配对不意味着逐调用输入相同。

## r1启动记录

北京时间2026-09-09 23:28:54启动串行控制会话`sc2_quota_pair_r1`，共同硬截止
2026-09-10 00:18:54。23:30:12核验A manifest为formal=true、dirty=false、
mode=evaluation、world_size=4、GPU4–7；原始SpB及SC2/地图资产哈希通过。
effective config确认A quota=0、original segment=1、opening=-1、192tokens、
val_only=true、resume禁用、save=-1、实际评测4局。此时尚在初始化，未据此声称
已消费Planner回答或得到游戏结果；B尚未启动。

冻结源码：`/root/new-scii-bo-segment-20260909/code-2f7c5dc`；控制记录：
`/root/new-scii-bo-segment-20260909/quota-pair-r1`，含launcher、共同deadline、
A/B日志与退出状态。run根为
`/mnt/data0601/yangjinyue/new_SCII/experiments/20260909-bo-segment-protocol-v1/runs`。
完整历史bundle本地`/private/tmp/sc2-quota-game-r1.bundle`及44的
`/root/new-scii-bo-segment-20260909/source-2f7c5dc.bundle`已核验同一SHA256：
`c5386fabaa775943743114a48fadba334bf4ca43abb479fabdf927d7d55ad1dd`。
服务器clean/source manifest通过，并在实际已有依赖中额外跑4项候选CPU测试，
1.227秒、退出0。没有推送GitHub，不绕过此前推送权限限制。

23:32:48实际核验：A四卡模型11/11分片完成，SSESWAMoBAForCausalLM 5.17B、
swa_dropout=0；`evaluation_seeds.jsonl`实际为`[2732,2732,43567,43567]`。
已消费8份回答，四个env首轮均在00:00，query不含新澄清文本，符合A定义；
首轮回答均为SupplyDepot/Barracks/Marine各`x 2`且通过联合校验。已有`BO ori`和
`refresh build order`队列记录，确认不是仅创建会话或加载模型。
B按已批准控制入口等待A正常结束，仍未开始；尚无最终游戏结果，不声称系统改善。

## r1完整结果（2026-09-10核验）

A于9月9日23:42:12、B于23:54:01北京时间结束，两组及pair均退出0。
合计25分07秒，未触及50分钟预算；8局全部自然终止、均有真实Planner决策，
无PPO更新。9月10日00:02:48检查本实验GPU4–7已释放；未重试或续训。
两组冻结源码均2f7c5dc，模型manifest相同；actual effective config归一化run/
Ray路径后只差`SC2_CLARIFY_BO_QUOTA=0/1`。两组游戏种子均为
`[2732,2732,43567,43567]`，rank采样seed0–3及各rank初始CUDA RNG哈希一致。
A的38份输入均无澄清段，B的20份均含完整候选段，没有部署开关失效证据。

| 指标 | A 原prompt | B 配额澄清prompt |
| --- | --- | --- |
| 自然胜负 | 2胜2负 | 4胜0负 |
| 平均整局累计reward | -14.100 | 20.375 |
| 全部实际消费回答有效 | 20/38（52.6%） | 15/20（75.0%） |
| 完整条目累计超配额 | 15/38 | 3/20 |
| 解码文本回算达到192tokens | 16/38，其中15条无效 | 3/20，全部无效 |
| Planner实际调用总数 | 38 | 20 |

有效率不是末轮`val/sc2_action_valid=0.5`，也不是单位建成率；按全部
policy_action_applied=true统计。每局reward从开局累加至首次done，不漏终局。
整局未保存原生成token/EOS，达到192仍仅是文本回算代理，不能与固定query的
精确未EOS统计混同。A有一条达到192但通过validator，不称所有触顶都无效。

| env / seed | A 胜负 / 累计reward / 有效调用 | B 胜负 / 累计reward / 有效调用 |
| --- | --- | --- |
| 0 / 2732 | 负 / -48.95 / 5/9 | 胜 / 20.5 / 6/9 |
| 1 / 2732 | 负 / -48.95 / 6/11 | 胜 / 20.5 / 4/5 |
| 2 / 43567 | 胜 / 20.5 / 5/9 | 胜 / 20.0 / 2/3 |
| 3 / 43567 | 胜 / 21.0 / 4/9 | 胜 / 20.5 / 3/3 |

B的env1–3首轮真实回答为：

```text
- supplydepot x 4
- barracks x 4
- marine x 24
```

整段合法；23:44:31的`refresh build order`确实保留32个展开动作，没有截回6或15。
这是执行队列证据，不等于32个单位/建筑已经产出。A四局首轮均各x2、展开6动作。
B的env1–3第二次Planner在253游戏秒；A为107–111秒。调用减少同时伴随首段变长，
不能仅凭调用更少就说规划效率更好，也不能将胜负变化唯一归因于该首段。

B仍有5条无效。例如env1/action3反复输出
`marine x 2 / supplydepot x 2 / reactor barracks x 2`，累计超过配额；
env2/action3后半段反复交替Marine与SupplyDepot。同类重复没有消失，仍不能宣称
“prompt已修好”。本次只支持该候选在这组真实游戏中的正向局部结果；只有两个
不同seed、每seed重复2局，不作跨seed显著性或架构/学习结论，也不报告伪精确置信区间。
候选仍默认关闭，未合入正式采用配置；可申请短RL验收，但没有自动启动。

主流程自然结果和退出码完整。console在收尾出现Ray worker SIGTERM告警，不能
将其说成从未有告警，也不据此改写已完成的自然胜负。每组结果仅索引一个回放，
没有声称全部逐局回放齐全。胜负、调用与reward依据四局JSONL/outcome汇总。

[完整逐局汇总及产物哈希](quota-game-r1-results.json)由冻结`summarize_eval.py`
和既有计数核验函数只读生成，未加载模型或修改run。候选prompt先逆变换回原
示例标记再复用计数函数，避免把one-shot条目误当待办配额；58条重算判定与日志一致。
第一次CPU分析命令漏源码PYTHONPATH而退出1，补齐命令路径后成功；这是分析命令
自身的问题，不计为项目原有bug或GPU实验失败。所有历史与本轮产物保留。
