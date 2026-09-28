# 温度0.4／1.0同checkpoint评测（已完成）

用户2026-09-10批准停止Level7并释放四卡用于此前建议的温度对照。
本次只做第一阶段纯评测，不执行PPO或自动启动后续4步训练；不改变43的Level6。

## 停止记录

44的Level7续训完成到global step31，2026-09-10 20:40:34北京时间因用户请求
SIGINT退出130。续训段5–31完成27步、216局、152胜；含原1–4为31步、248局、
174胜。最后完整checkpoint为step24；25–31只有日志、没有相应checkpoint。
原result/.done保持failed/130（未完成96），另追加
`operator_stop_20260910_temperature_ab.json`注明aborted/user_requested，
不将目标step96缺失误读为step24损坏。step8/16/24、原step4及所有日志均保留。
专属timeout PID1731385/PGID1731385与四GPU worker归属核验后发SIGINT，
8个同属该run的SC2残留进程随后SIGTERM退出；没有删除文件或停止外部任务。
GPU4–7已释放，GPU0的PID1176690不是本项目，未动。

## 问题与最小对照

判断已出现回答集中的模型提高温度后，是否采样到更多合法且实际不同的计划，
这些计划是否对应不同游戏回报。模型与执行层能力不在本次因果归因范围。

A：temperature0.4；B：temperature1.0。仅该机制不同；训练默认温度与val_kwargs
均显式设置同值，真正生效的是验证分支val_kwargs。两组都do_sample=True、
top_p0.9、top_k0，192token、约15条/xN及quota说明保持不变，不改prompt。

两组使用**Level7续训step24同一权重**，在**Level6/Harder、Flat128、Terran/Air、
canonical g2、开局Planner接管**评测。这不是原始模型与训练后模型的比较，
不是Level7与Level6的难度对照，也不宣称step24就是step31。
其余FP32参数/温度归一化、BF16 autocast、HF cache、FSDP分层、SWA dropout0、
history0/off、reward、合法计划保留、对空候选和执行/战斗代码保持冻结。

游戏seed向量两组均为：
`[52620,52620,52620,52620,17089,17089,17089,17089]`。
它们来自原训练step18/20，分别出现过全胜/全败模板，属于诊断选样，
不是独立留出评测；只有两个独立场景，不能算8个独立seed。
按env_index配对，同组重复是生成采样重复。复用既有handoff_eval的worker RPC，
每rank仅在第一次生成前设置Torch seed=rank，两组相同，不逐调用重置；
后续轨迹长度不同会消耗不同随机数，不能声称所有后续token抽样逐项匹配。

## 冻结来源与入口

生产源码基准 `90b3fdc2fb7fadc1230861306b20c27540d9b736`，
当前增加的仅是本record评测入口、seed编排和测试，未改生产模块。

权重：
`/mnt/data0601/yangjinyue/new_SCII/experiments/20260910-spb-fsdp-layer-wrap-v1/runs/20260910-spb-fsdp-layer-wrap-v1-spb2-l7-g2-opening-quota-resume4to96-4gpu-s0-r1/checkpoints/global_step_24`。

模型仍是SpikingBrain-2.0-instruct经这条RL训练得到的step24，不是原始权重。
启动先校验原始资产/模型代码，再用冻结FSDP loader加载step24四rank模型、
optimizer、scheduler/RNG；纯评测不会更新optimizer。不转换权重，不复制大模型，
不恢复SC2训练环境计数。加载返回后写`checkpoint_evaluation.json`，
实际8种子由`evaluation_seeds.jsonl`核验；不能只看目录就称加载成功。

`run_temperature_pair_node44.sh`在44 GPU4–7串行A/B，
每组8局、总16局、总墙钟上限60分钟（含资产验证、两次加载及退出），无自动重试。
入口检查clean SHA、system-v1哈希、GPU归属、依赖及checkpoint小元数据/尺寸。
文件尺寸检查不是完整反序列化验收，后者由实际FSDP加载返回证明。
第一次失败保留run并退出，不自动重试或跳过空卡检查。

两RUN_ID：
- `20260910-spb-fsdp-layer-wrap-v1-spb2-l7s24-l6-g2-temp0.4-eval8-4gpu-r1`
- `20260910-spb-fsdp-layer-wrap-v1-spb2-l7s24-l6-g2-temp1.0-eval8-4gpu-r1`

输出仍在同experiment的共享runs，控制目录
`/root/new-scii-spb-fsdp-20260910/temperature-ab-r1`。
新deployment/source SHA及实际时间在启动后追加，不改旧deployment。

## 验证与判读

本地CPU命令：
`PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:tests /private/tmp/sc2-air-results.HBI61m/decoder/bin/python -m unittest test_temperature_eval test_fsdp_rl_launch -q`：
7项通过，1.094秒；两入口bash -n、git diff --check通过。
首次新增测试因把带Hydra前缀的环境变量误当裸参数而失败，修正测试读取方式后通过；
这是本次测试编排返工，不算项目旧bug修复。

完整effective-config A/B差异仅两个temperature配置键；明确断言纯评测、4卡、
8局、step24路径、192token、dropout0、开启采样及保留关键系统开关。
seed选择测试拒绝训练环境/额外重采样。CPU通过不代表GPU已启动或模型已恢复。

结果按实际seed报告自然结局、整局累计reward、Planner调用数与间隔、
格式/语义有效率、截断率、原始模型计划和最终队列差别。
无Planner游戏单列但计入整局结果。标题/措辞变化不计作不同语义计划。
不把更多无效回答、单局胜负、非零advantage本身当学习成功；
少量结果只能支持或拒绝继续短RL候选，不能证明架构优势。

## 实际启动（9月10日21:01只读快照）

冻结SHA `939c4ca956adabf1728e0cdd4a87c879593d7cbe`，部署在44
`/root/new-scii-spb-fsdp-20260910/code-939c4ca`；与90b3fdc相比生产目录无diff。
增量Git bundle在同父目录`source-939c4ca.bundle`，SHA256为
`751909b60793063975f254157c3e610b78e987c30ce9c3c95ef6ba635c4c7a99`；
依赖旧90b3fdc完整备份，已在服务器核验并完成clone/fetch，未推送GitHub。
44固定依赖、隐藏CUDA下7项CPU测试全部通过（1.407秒），旧部署保持clean。

受控会话`sc2_spb_temp_ab_0910_r1`在北京时间20:53:20启动，总上限到21:53:20。
A的manifest为20:53:57、formal evaluation、4卡，已准确加载step24而非原始权重。
21:00核验`checkpoint_evaluation.json`恢复返回、实际种子向量一致，SC2已推进
到约4分钟并有8条真实Planner回答，全部合法且已送入系统；首轮均为depot4、
barracks4、marine24。尚未得到最终胜负或A/B结论。
B在同一脚本中等待A退出0及四卡释放后运行，不是已经启动第二组。
[启动证据](temperature-ab-launch.json)保存manifest/配置/恢复/种子文件哈希。

## 完成与结果（9月10日晚核验）

两组及总控制器退出码均为0，16局全部自然终止，无PPO更新、无未调用Planner的局。
控制会话于北京时间21:22:35结束，共29分15秒；44 GPU4–7已释放，43长训未改动。
冻结SHA仍为`939c4ca956adabf1728e0cdd4a87c879593d7cbe`。
两次checkpoint恢复返回、实际种子序列、各rank初始采样RNG哈希均已逐一核对。
排除run/model/log/tmp等隔离路径后，effective config仅两个temperature键不同。
manifest是启动时快照，完成状态以result、.done及控制器记录为准。

| 指标 | A：0.4 | B：1.0 |
| --- | --- | --- |
| seed52620 | 4/4胜 | 4/4胜 |
| seed17089 | 0/4胜 | 0/4胜 |
| 全部8局平均累计reward | −14.40025 | −14.40650 |
| 真实Planner回答数 | 21 | 26 |
| 协议有效回答／真实回答 | 16/21（76.2%） | 19/26（73.1%） |
| 回答达192token且缺闭合标签 | 2/21 | 0/26 |

累计reward按每局真实终止step截断后求和，不遗漏终局反馈，不累计已结束环境的
占位日志。回答只计`policy_action_applied=true`，与终局policy_decision_count一致。
这里的有效率不是汇总日志的`val/sc2_action_valid=0.875`，也不是执行成功率。
token统计是固定tokenizer对已保存文本的CPU重编码，未保存原始生成token ID，
不能当作另一套生成长度张量的独立核验。

[逐局证据](temperature-ab-results.json)保留16局reward、调用时间、原始回答、有效性、
失败原因、输入哈希和观测摘要，以及原run小型产物SHA256。只有两个诊断种子，
不推断总体胜率或训练提升。

本地`/private/tmp/sc2-air-results.HBI61m/decoder/bin/python -B`内联CPU核对退出0：
16局逐一断言种子配对、回答数等于plans长度、有效计数、整局reward等于逐次反馈
之和及两组汇总一致；A两条未闭合回答核对一致。`git diff --check`通过。
此次只新增分析记录并更新状态，没有修改生产代码或重跑模型测试。

## 温度改变了什么，没有改变什么

开局8次相同输入，A只有一种回答；B出现两种，但新增的是超额的`marine x 48`，
被判无效。B其余7次仍为depot4、barracks4、marine24。这不是产生了第二种合法开局。

后续B确有合法的小计划，并非所有差异都无效；但也出现`attack x 1`、重复超额，
或当前待办外的Starport。冻结Planner对无效回答仍使用原6动作规则补齐；因此
这些回答的游戏结果不能全归因于模型所选计划。合法original段保留原顺序，
不会重新排序为执行器认为更合理的计划。

B env3有一条合法`Launch OFFENSE with 8 marine`，整局reward21而非其余胜局的20.5。
日志中的额外+0.5与冻结Commander的attack奖励对应，不能将它解释为战斗效果提升。
温度A/B没有PPO，不能由本次reward差直接声称优势或学习问题已解决。

## 最有信息量的失败局：小计划仍被资源顺序阻塞

seed17089第二次query在04:09，A/B八次输入哈希前缀均为`8d4d419e666c`：
11 SCV、20 Marine、90矿、0气、4兵营、4补给站。B env4合法回答为：

```text
<answer>
- marine x 1
- supplydepot x 1
- barracks x 1
- reactor barracks x 2
- scv x 1
- refinery x 1
</answer>
```

模型实际交给Executor的就是这7个展开动作，未被改成全量补兵。
唯一该队列对应SC2Worker PID1944583；console记录：

```text
(1396) refresh build order: marine, supplydepot, barracks,
       reactor barracks, reactor barracks, scv, refinery
Action blocked: Insufficient Vespene Gas (0/50).
(1661) manage_total_attack_groups: 总攻波次 1 已全灭。
(1751) bo_skipped_3: {'reason': 'timeout', 'action': 'reactor barracks'}
Action blocked: Insufficient Vespene Gas (0/50).
(1972) bo_skipped_4: {'reason': 'timeout', 'action': 'reactor barracks'}
```

两个Reactor都排在Refinery之前，没有气，逐个等到超时。这不是此前“排兵占用
兵营”的冲突复发，也不是解析器拒绝这条回答。05:52才重新规划，此时660矿、
仍0气、3 Marine；随后选择marine41/depot7/barracks2，下次反馈10:56军队已归零。
该worker后续第2–5波也全灭。日志去重的repeated次数不用于统计局数或额外事件。

因此，之前“长计划拖延反馈”的证据仍成立，但**缩短计划并不自动解决问题**：
这局已较早得到新反馈，仍遇到资源顺序阻塞和连续战损。不能由一局断言把
Refinery提前就能取胜，更不能将全部失败归因于这两个超时。

## 归因边界与唯一下一项建议

当前阶段指令明确要求Marine Reinforcement Flood、先持续Marine压力，待办提供
Marine、补给站、兵营、Reactor、SCV和Refinery，没有Starport/Viking/战斗升级。
这是冻结策略给模型的决策范围，不是把全单位表交给它自由应变。现有证据支持
优先怀疑“阶段范围＋资源顺序＋队列结束才反馈＋自动小波次”共同限制应变，
但尚未分离它们各自对胜负的因果贡献。没有新增参数更新bug的证据。

冻结代码位置（相对44 deployment，未用最新工作树代替）：

- `agent_system/environments/env_package/sc2_online_restored/SC2ArenaDyStrategy/agents/planner.py:1300`：合法段保留与无效段6动作补齐分支。
- `agent_system/environments/env_package/sc2_online_restored/SC2ArenaDyStrategy/players/llm_player.py:677`：队列耗尽才重新规划；`:763`为等待/超时分支，`:101`为20/10默认攻击规模。
- `agent_system/environments/env_package/sc2_online_restored/SC2ArenaDyStrategy/agents/strategist.py:335`：marine_pressure阶段来源，对空备选只加入末阶段。
- `agent_system/environments/env_package/sc2_online_restored/SC2ArenaDyStrategy/agents/commander.py:267`：反馈奖励与后续规划。

结论：**本次不支持将温度1.0作为已验证改进晋级短RL**；它增加了一些回答变化，
没有改变这两个场景的胜负。也不能据此认定所有提高温度的方案都无效。

建议的单一后续验证是固定失败seed17089和这条7动作计划，只把Refinery移到
两个Reactor之前，其他动作、数量、战斗及预算不变，观察50气到达、Reactor
开工/超时、下一反馈时刻和战损。它是**手工计划顺序的执行因果对照**，不是
模型学会了正确规划，也不自动将重排规则加入正式系统。具体后续Planner行为、
局数和GPU预算需先确定并获批；本轮未实施、未新增游戏。
