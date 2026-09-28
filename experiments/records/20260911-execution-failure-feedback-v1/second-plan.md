# B4第二条回答替换对照

2026-09-11。用户已批准：43四卡、同seed两组各4局，实际启动起最多45分钟。
状态：北京时间21:56:08启动、22:08:06完成，退出码0；22:21完成逐局只读核验。
实际用时11分58秒，未超45分钟预算，没有重试。

只验证第二条计划能否改变后续补兵与交战。不修生产模块、不做PPO、不续长训。
沿用[已完成的独立评测](learning-eval-analysis.md)及同一分支/record。

## 对照

两组都加载反馈短RL **B-step4** 的完整四分片checkpoint：
`20260911-execution-failure-feedback-v1-spb2-original-l6-g2-B-rl4-4gpu-s0-r2/checkpoints/global_step_4`。
该训练冻结SHA为`361cb7ae7fd86fe8811d6fcb6946d0241bf93e11`，不是A4或Level7权重。
原始资产为`Panyuqi/SpikingBrain-2.0-instruct`，manifest SHA
`40d6f1417568d5fda1107cdafd66c7441a679b03fb148851c8f226ecc9cf884c`。

一个四卡模型实例同时处理8局，env顺序为`control, replacement`重复四次。
实际SC2 seed全部`1098341402`，仅一个独立场景。GPU0–3初始化采样seed保持0/1/2/3，
不逐call重置；不是四组独立游戏seed，也不声称后续采样逐token相同。

| 调用 | 对照 | 替换组 |
|---|---|---|
| 第1答 | 同一历史B4开局：Depot4/Barracks4/Marine24 | 完全相同 |
| 第2答 | B4原答：Marine8/Depot2/Barracks3/Reactor3/SCV2/Refinery1 | A4原答：Marine42/Depot8/Barracks3/Reactor3/SCV2/Refinery1 |
| 第3答起 | 在线B4 | 在线B4 |

两份第二答按历史原文使用，保留A4标题行；不是只改变Marine一个数量。
本次唯一干预对象是**整条第二答**，不能拆称单一兵种数量因果实验。
输入/回答和来源哈希见[历史fixture](second-plan-inputs.json)，来自运行SHA
`937ae3cfcbf7340208a8f86c0abf680d733c0484`独立评测的env7、原始行8/16。
前两次query必须与对应历史SHA一致，validator必须接受，否则明确失败、不自动重试。

前两答仍正常经过冻结解析/执行器，单独记录为人工干预；游戏reward不变。
`prompt_response.jsonl`记录转发后的回答；干预时真正生成但未使用的原答另存
`interventions.jsonl`。既有collector的consumed-policy计数会包括这两答，
**不能称为自主模型决策数**；结束时`intervention_coverage.json`按每局减去两条，
另报第三次开始真实在线选择数。没有第三次调用的游戏要保留并单独标明。

## 冻结边界与入口

游戏/模型/优化配置与原B4独立评测一致：Level6 Harder、Flat128、Terran/Air、
Planner从开局、原阶段待办、约15条及数量展开、累计配额prompt、过滤/补齐、
微操/自动战斗、history0/off、step shaping off、失败反馈开启；FP32参数及温度归一化、
BF16 autocast、cache开启、SWA dropout0、eager/HF/分层FSDP、T0.4/top_p0.9/top_k0、
prompt4096/response192。原始checkpoint恢复后只验证，不做第5次训练更新。

新增`second_plan_eval.py`通过既有评测接口转发前两答；未改agent_system或verl。
现有反馈评测hook仅增加可选的回答来源说明，默认行为保持原样；现有launcher增加
显式`FEEDBACK_SECOND_PLAN=1`分支选择诊断hook，未设置时保持既有入口。

`run_second_plan_node43.sh`复用已验收launcher/环境/模型，只加载一次B4。
从独立clean SHA启动，唯一RUN：
`20260911-execution-failure-feedback-v1-spb2-B4-l6-second-plan-ab8-4gpu-s1098341402-r1`。
本地共享镜像路径、include与Python沿用43，Ray新目录`/dev/shm/sp_0911_r1`。
不复制大权重、不删除旧产物；GPU归属及空闲检查保留，不自动重试或增加局数。
控制器记录开始/结束/退出码，预算2700秒含退出；外层受控tmux同样施加45分钟边界。

## 验收

先核对真实B4恢复、0 PPO、8局实际同seed和前两query哈希，确认两组仅第二答不同。
逐组追踪完整执行队列、Marine产出/断档、首次失败、波次损失、胜负与整局累计reward；
原始生成和人工转发分开，实际产出仍用现有日志/回放证据，不把“提交动作”当建成。
若替换组改善，只支持这个场景中第二条规划段有因果作用，不等于模型学会此计划；
若无改善，也不单凭少量胜负认定模型不可训练。预算不足或边界不匹配保留失败记录。

CPU命令：
`PYTHONPATH=tests <python> -B -m unittest test_second_plan_eval test_feedback_learning_eval test_feedback_eval test_supply_order_eval`。
覆盖真实validator/解析器得到19/59个展开动作、前缀一致性、第三答恢复在线、
仅一次8局采样、有效配置对B4仅改变诊断hook、四分片加载选择、预算及Ray路径。
第一轮测试中比较基线漏设43的依赖路径，出现PYTHONPATH差异；补齐测试环境后重验。
这是诊断测试准备修正，尚未启动GPU，不算项目原有bug修复。

本地24项测试退出0：23通过、1项缺Torch跳过，1.863秒。随后补充新hook的CPU
CUDA禁用检查，随冻结部署在43验证；本地缺Torch不能宣称该项已通过。

## 冻结与部署

运行SHA `a0e5425b2b9ff09db28d0c017f7c6dc7c0d1ba58`，43独立目录
`/root/new-scii-execution-feedback-20260911/code-a0e5425`，clean detached。
基于937ae3c的私有增量bundle两端SHA256均为
`383d0b8f372d48955a0f1b13efb20a7792383c276826dc33550148e6075167dc`。
没有修改旧部署、复制大权重或推送GitHub。

新增测试后的本地25项：23通过、2项缺Torch跳过，1.837秒、退出0。
43固定依赖、隐藏CUDA：25项中24通过，包括新旧hook的CUDA禁用检查；
唯一未通过的是旧`SupplyOrderTest.test_original_model_evaluation_only_config`在
30秒超时（总38.635秒、退出1），其旧44入口仍引用共享盘依赖。本次43入口的
dry-run配置和B4检查均通过，没有重跑/修复这个无关入口，也未据此宣称25项全通过。
日志保留在`/root/new-scii-execution-feedback-20260911/second-plan-r1-cpu.log`。
启动前再次核验23个冻结生产文件哈希、两份历史query/原答哈希均一致。

受控tmux：`sc2_second_plan_r1`；控制目录
`/root/new-scii-execution-feedback-20260911/second-plan-r1`。
21:56:08前GPU0–7均0MiB/0%，实际仅用已授权0–3。旧SC2遗留进程未动。
正式入口自身和外层timeout均受45分钟限制；不自动重试。

22:00:15只读核验：`checkpoint_evaluation.json`证明分布式loader已返回global_step4，
确切路径为B-r2/step4，`val_only=True`、0 optimizer updates；8个实际游戏seed逐一
为1098341402。8份游戏boundary已生成，8条开局干预记录均匹配历史query SHA。
此时第二条回答干预尚未发生，不能宣称对照完成；后续仍需检查8局全部到达该边界。
配置/manifest/加载证据/种子/原始资产信息的哈希见[启动记录](second-plan-r1-launch.json)。
预计22:15–22:30出结果，22:41:08硬截止不变。

## 完成结果（2026-09-11 22:21核验）

`result.json`与`.done`逐对象一致：completed、exit0、8局全部自然终止，0次PPO更新。
manifest仍是clean `a0e5425b2b9ff09db28d0c017f7c6dc7c0d1ba58`；实际加载B-r2的
step4，8局实际seed均为1098341402。16条干预覆盖所有游戏的前两答；首query SHA
为`95f4e1650fe0f587aba982557916cbb0e59d9bcbf1c10092fd40664578a5d424`，
第二query SHA为`4c3972dca2ed9c0665934414648b868741a493c494c6cad57119f022f8ac3765`，
逐条一致。43节点22:19快照GPU0–7均0MiB，评测已释放GPU，未续训。

| 第二答干预 | 胜负 | 平均整局累计reward | 在线有效回答（排除人工前两答） |
|---|---|---:|---:|
| B4原答：Marine8、Depot2，其余保持原文 | 0胜4负 | -49.43775 | 7/7 |
| A4原答：Marine42、Depot8，其余保持原文 | 4胜0负 | 20.50000 | 5/5 |

两组仍是同一B4权重，第三答起恢复在线B4；不是A4权重对B4权重，也不是训练收益对照。
整局reward按`prompt_response.jsonl`每个env累加到其首次终局step（含该条），
不包含结束后批处理占位行。八局均值-14.468875，与结果记录-14.469在舍入范围一致。

| env | 组 | 结果 | 整局reward | 消费回答数 | 其中真正在线选择数 |
|---:|---|---|---:|---:|---:|
| 0 | B原答 | 负 | -49.450 | 3 | 1 |
| 1 | A替换 | 胜 | 20.500 | 3 | 1 |
| 2 | B原答 | 负 | -49.401 | 6 | 4 |
| 3 | A替换 | 胜 | 20.500 | 3 | 1 |
| 4 | B原答 | 负 | -49.450 | 3 | 1 |
| 5 | A替换 | 胜 | 20.500 | 4 | 2 |
| 6 | B原答 | 负 | -49.450 | 3 | 1 |
| 7 | A替换 | 胜 | 20.500 | 3 | 1 |

collector共计28条消费回答，其中16条是人工干预，只有12条是第三答起在线选择。
所有游戏都实际到达第三次在线调用，没有需另列的零在线选择游戏。两组在线格式有效率
均为100%，但结果相反：格式合法本身不能解释胜负，也不能作为RL学好的证明。

例如B原答组env0第三答已经在线选出`marine x 34 / supplydepot x 6 /
reactor barracks x 3`，仍然失败。A替换组env1第三答只选出各1个Marine、Depot、
Barracks、Reactor、SCV，仍然胜利。仅凭这些后续回答不能判定执行时机或生产因果链。

本次支持在这个固定场景中继续追踪**第二段计划内容及其执行时序**，反驳“下游系统
无论给什么合法计划都必败”的强判断。不能归因为Marine数量单变量：Depot数量和原文
标题也随整答替换；也不能称模型已学会此选择，或失败反馈训练已经有效。
一个游戏seed每组重复4次，不能外推多seed胜率或架构优势。新run回放的第二答前完整
tracker前缀、Marine真实产出/损失和断档尚未核对，下一项应只读对齐这些既有产物，
不凭0/4→4/4顺手改prompt或开启长训。

证据位于本节对应RUN目录，主要SHA256：

- `result.json` / `.done`：`8aa479c545a3cbf69e1284eb2874289c85fb75d83bc950c13e7c0e05e99347e3`
- `manifest.json`：`b17b7e98343ceab48427e73062cfff9753d41755e168a000694dbf2ca6b5d37a`
- `effective_config.json`：`b4ac4d68b8475fff2bef6d009ede656507da51ee29cc23870228b423b3fc586b`
- `prompt_response.jsonl`：`786c70c09585cd9be2956905317f7fe8cf53ece929d35e6073ae2d34a5c6c472`
- `interventions.jsonl`：`6d7f64480438332b597536ccd03b58db2a8858c845352eda01dfc8f55430174a`
- `intervention_coverage.json`：`db86a79f34680f7da4aa714b0a3303baeb297b64168c9311cb588c7a46d9828b`

退出前console含一条Ray worker收到SIGTERM的清理期告警；结果、8份自然终局回放、
覆盖记录与控制器均已完整落盘且退出0。本次不把该告警遗漏，也不把它改写为游戏失败。

## 回放与执行链核对

仅下载本RUN八份回放、trace、日志和小型provenance；未上GPU、重跑游戏或修改生产代码。
运行`audit_second_plan_intervention.py /private/tmp/sc2-second-plan-replays.7gpJJh`，
使用已有`/private/tmp/sc2-air-results.HBI61m/decoder/bin/python -B`，退出0。
[机器可读审计](second-plan-replay-audit.json)含全部源文件哈希、8局生产/死亡账本与
代表局env0/1的队列、BO映射、动作意图及波次日志行号。复用s2protocol的75689解码器，
时间以game loop / 22.4换算；建成看tracker，不把dispatch前trace当作执行成功。
初次工具输出超长截断，之后缩减重复报告再完成核验；这是报告输出处理，不是运行失败。

### 开局确实一致，但不声称恢复了所有隐藏状态

第二答在iteration1462（261.071秒）交给执行器。此前8局的398条tracker事件完全相同；
342条game事件中唯一差异是`SSetSyncPlayingTimeEvent.m_syncTime`的现实时间相差1秒，
所有选择、指令和目标更新记录相同。两组首波均在4:30.536创建、4:49.107出发。
这比仅比较query更强，但不等于持有全部引擎/随机数内部状态。

最终执行队列与人工转发回答一致：B为19项，A为59项，均按原顺序展开；A的标题没有
变成动作。不是解析器丢掉了后面的Marine，也不是补兵规则把两组重新改成同一计划。

### 103秒生产断档发生在哪里

所有B原答局的时间链一致：

| 游戏时间 | 已有证据 |
|---|---|
| 4:21.071–5:01.607 | 依次提交第二段8个Marine；trace有8项对应意图 |
| 5:01.786–6:05.536 | 执行头部转到2个Depot、3个Barracks；回放确认5个建筑真正开工 |
| 6:05.536–6:44.821 | Reactor占据当前BO，0/50气；39.286秒后超时，目标未提交 |
| 6:45.000 | 才再次调用Planner并恢复Marine命令 |
| 5:19.464–7:02.857 | 回放实际没有新Marine出生，间隔103.393秒 |

建筑开始时间为5:04.420、5:11.562、5:33.438、5:47.321、6:09.866。
后三个兵营确实开工，不能再写成“生产建筑命令全部失效”。头部命令推进与建成不是同一时刻。
Refinery在Reactor后方，第一次超时前没有执行到它。反馈在6:44.821才产生，而首两波
已经分别于5:23.750、6:03.214全灭：**首次反馈发生得比这一轮战斗崩溃晚**。

代码依据均为冻结`a0e5425`的`players/llm_player.py`：720–735只有队列结束/失败反馈
才重新规划；737–745取单一current_bo；822–842按22个iteration间隔检查、超过10次
后超时；902–968遇资源不足等待返回，不推进后续BO；1023–1027提交后才推进索引。
维护/微操仍在654–659先执行，不是等待BO时整套战斗停止。这里证实的是串行队列和
计划资源分配的作用，**不是发现代码偏离了其既定实现，也没有直接修改它**。

### 产出与战损

下表中5–7分钟指标每组4局都相同，整局数字因第三答后的在线选择不同而略有区别。

| 指标 | B原答 | A替换 |
|---|---:|---:|
| 5分钟现存Marine | 27 | 27 |
| 5–7分钟Marine新产出 / 损失 | 3 / 30 | 18 / 10 |
| 6分钟现存Marine | 4 | 26 |
| 7分钟现存Marine | 0 | 35 |
| 整局Marine产出 | 41–65 | 48–49 |
| 整局Marine损失 | 40–65 | 14 |
| 整局SCV损失 | 12 | 0 |

B组5–6分钟26个Marine死亡，其中17个由敌方VikingAssault击杀；不是一概被空中单位
击杀。env0首名工人5:25.045死于敌方基地附近(29,20)，后11名在9:01–9:21于自家
矿区被LiberatorAG击杀。不可把这两次不同位置/时间的损失都写成生产工人被抽走。

A第二段实际只推进23个Marine命令，就因人口阻塞于7:02.321反馈重规划，剩余19个
Marine和建筑尚未执行。仍有6:40.714–7:24.286的43.571秒出生间隔。
所以**胜利不证明A的42Marine→8Depot顺序已正确解决供给与反馈问题**。

### 因果结论的边界与下一步

已证实：有效计划能够原样改变执行队列，关键时间段的资源投入/补兵和最终reward有
明显差别。因此系统不是完全没有合理选择空间；格式100%有效也不保证游戏决策好。
但尚不能把4/4胜全部归因于多产Marine：第三答前第一项额外Marine在5:19.643才出生，
双方老兵战损从5:05.670/5:05.714已经分岔。在5:19.640前两组产兵均为3个，
损失已经是17比9。第二波集结点也从(73,26)变为(74,25)。这些是同一整段干预的
后续游戏响应，未单独控制；其最早的微操/交战差异尚未定位到具体指令，不能猜测为
某个随机数、模型权重或微操bug。

当前最小的下一项候选是只把B原答`marine x 8`改成`marine x 42`，Depot仍为2，
标题、其他BO、B4权重和后续在线行为不动，从而把现有整答干预缩为单数量干预。
这是待批准的纯评测方案，不是当前已启动实验，也不直接做RL或修改默认系统。
这轮没有检查PPO梯度/advantage，不能用游戏因果证据宣称训练已经学好。

## Marine数量单变量对照（本轮已授权，启动前）

用户确认本轮独立预算：43 GPU0–3，一个B4实例同时处理A/B各4局，Level6，
实际seed均1098341402；从实际启动起最多45分钟，不做PPO、不新增checkpoint、
不复制大权重、不自动重试。22:40北京时间只读检查八卡均0MiB、无GPU进程。
启动前仍重查，不终止任何其他任务。本节沿用同一实验record，不覆盖上面的整答对照。

可证伪问题：在相同B原答中，仅增加Marine配额是否足以复现此前胜负/补兵分岔？
两组首答仍是共同历史开局。第二答如下，除了8/42没有任何文本差异：

```text
<answer>
- marine x 8       ← 对照8，候选42；实际回答不含本注释
- supplydepot x 2
- barracks x 3
- reactor barracks x 3
- scv x 2
- refinery x 1
</answer>
```

候选取B历史原文仅替换`- marine x 8`为`- marine x 42`，不取A原文、不加标题。
实际解析队列分别19/53项。env0/2/4/6为`control_B_answer`，1/3/5/7为
`marine_42_only`；第三答起两组恢复在线B4。人工前两答、丢弃的原始生成和后续
自主选择分别记录。场景、模型、失败反馈、prompt、解析/过滤、微操、history、
reward、cache、精度与采样预算全部沿用上一节；两组共享一个实例而非两个并发模型。

复用`second_plan_eval.py`，仅增加显式`install_marine_only`入口及候选答案构造；
默认`install`仍为历史整答对照。薄入口`run_marine_count_node43.sh`选择此hook，
其余启动和45分钟控制复用原入口。生产manifest和agent_system/verl不修改。
新RUN：`20260911-execution-failure-feedback-v1-spb2-B4-l6-marine-count-ab8-4gpu-s1098341402-r1`，
Ray目录`/dev/shm/mc_0911_r1`，控制目录`marine-count-r1`，均须未使用。

验收先核对B-r2/step4实际恢复、0次PPO、8个游戏seed、两次query哈希及19/53项队列；
再对齐Marine真实产出/断档、首次失败及重规划时间、战损、自然胜负、整局reward。
不只看胜负。与上轮一样，战斗差异可能早于增产，不能默认收益全部由出生数量中介。
两组依旧只有一个独立游戏场景，不报告跨种子的胜率置信结论、不据此称RL已有效。

本地CPU：`PYTHONPATH=tests <decoder-python> -B -m unittest test_second_plan_eval
test_feedback_learning_eval test_feedback_eval`，21项，19通过、2项缺Torch跳过，
1.689秒、退出0；包含19/59/53项真实解析队列、第二答仅一处文本差、第三答恢复在线、
effective config只改变hook的验证。随后补充新hook同样不得提前初始化CUDA的检查，
需在43固定依赖且CUDA隐藏的CPU环境完成；不重跑上次无关旧44入口的NFS超时测试。

### 本轮已启动

运行SHA `e25ce325bf61e86f5934bbea7969da7b8bea639a`，43独立clean目录
`/root/new-scii-execution-feedback-20260911/code-e25ce32`；私有bundle两端SHA256
`a1ba8de0aea73b944e4e383912e84b89cb08fae2bd6bb0c061ed01e54dd135dc`。
23项生产文件hash全通过。补充新hook后的本地21项19通过/2跳过，1.654秒；
43固定Python、CUDA隐藏的21项全部通过，11.658秒、退出0。
CPU日志：`/root/new-scii-execution-feedback-20260911/marine-count-r1-cpu.log`。

北京时间22:46:14在tmux `sc2_marine_count_r1`启动，控制目录
`/root/new-scii-execution-feedback-20260911/marine-count-r1`，硬截止23:31:14。
外层timeout2685秒并留15秒退出，内层同样有45分钟总截止，不自动重试。
启动前八卡均空闲、主机available455GiB；仅使用本轮授权GPU0–3。

22:48:20实际effective config归一化新旧源码/输出/Ray目录后，仅诊断hook不同；
g2、train/test parquet分别对照内容SHA一致。22:50:52确认分布式checkpoint loader
已返回B-r2/step4，`val_only=True`、0 optimizer updates；8局实际seed均1098341402，
8个独立游戏边界、8条共同首答干预已经出现。此时尚未到第二答，不宣称候选完成。
详见[实际启动证据](marine-count-r1-launch.json)，含模型加载、种子及配置/日志哈希。
预计22:58–23:08出结果，预算不随状态检查延长。未推送GitHub、未动旧部署/权重。

### Marine单变量结果与改进决策（22:59以后核验）

本轮实际22:55:40结束，用时9分26秒，控制器/结果退出码均0，8局自然终止、
0次PPO。43八卡于22:59核验无GPU进程；没有自动接续其他实验。
runtime仍为e25ce32，B-r2/step4实际恢复、种子、16次干预、19/53项队列均通过。
结果和.done内容一致，SHA256：
`baa636b63a2c5a06859e92bc0fefac94395dbdf77d436c01e882c08b68d0f255`。

| 同一B4、同一游戏seed | 第二答Marine8 | 仅改为Marine42 |
| --- | ---: | ---: |
| 自然胜负 | 0/4胜 | 4/4胜 |
| 平均整局累计reward | −49.45 | 20.5 |
| 后续真实在线回答合格数 | 4/4 | 5/5 |
| 5–6分钟实际Marine出生／死亡（每局） | 3／26 | 9／10 |
| 6–7分钟实际Marine出生／死亡（每局） | 0／4 | 9／0 |
| 整局Marine出生／死亡 | 65／65 | 47–49／14 |
| SCV死亡（每局） | 12 | 0 |

25条实际消费回答中16条是人工前缀/干预，只有9条在线模型选择；不能报成模型
25/25自主正确。各局累计reward包含自然终止，无漏局。八局只有一个独立seed，
不把同seed的4/4视作总体胜率估计或生成跨seed置信区间。

干预前398个tracker事件逐条一致；342个game events只有env6的loading wall-clock
同步时间差1秒，选择/指令均相同。首次生产/交战分岔与整答对照复现：对照在
05:01完成8个Marine的提交尝试后转建Depot/Barracks；06:05开始等缺气Reactor，
06:45才反馈重规划。Marine实际出生05:19.46–07:02.86断档103.39秒。
这次对照第三答已改成Marine34，仍4/4败，说明失败反馈后的补救并不能逆转该局
先前损失。不是单纯“最后生成的兵少”：对照整局造得更多，却全部损失。

候选的42个Marine并未全部完成：第二段提交到第23个后仍发生人口阻塞，07:02.32
反馈。后续4局真实B4第三答均将Depot放在Marine前，3局之后有新Marine出生，
另1局直到获胜未再出生。不能把它写成“42全执行且无阻塞”；审计同时保存末次
出生到终局的无出生尾段，避免只看相邻出生间隔而漏掉未恢复生产的局。
两组战损差异仍早于第一个额外Marine出生：05:00–05:19.64均出生3个，死亡已为
17对9。因此支持**该数量选择在此场景改变系统轨迹与结果**，不声称全部收益都
由后来多出生的单位解释，也没有定位最早交战分岔的全部中介。

**决策：** 结束本seed的数量消融，不再重复Marine42或把42硬编码成策略。当前
系统并非没有有用选择；格式相同合格也不代表计划等价。独立评测原始/A4/B4为
3/8、5/8、2/8，故保留A4作为后续学习候选、B4作为失败参照，暂不晋级反馈训练。
不是宣称A4已显著优于原始模型；见[下一轮学习验收建议](rl4.md#下一轮学习验收建议)。
没有更改prompt、reward、Executor或模型算子，没有启动新训练。

本地8份回放/完整trace快照：`/private/tmp/sc2-marine-count-results.nHoULN`；
[逐局审计](marine-count-replay-audit.json)含实际队列、原始在线回答、命令尝试、
真实出生/死亡、失败原因和源文件哈希。实际只读命令：

```bash
/private/tmp/sc2-air-results.HBI61m/decoder/bin/python -B \
  experiments/records/20260911-execution-failure-feedback-v1/audit_second_plan_intervention.py \
  /private/tmp/sc2-marine-count-results.nHoULN \
  --expected-sha e25ce325bf61e86f5934bbea7969da7b8bea639a
```

退出0，审计SHA256 `190bdff92c8b032bff4098533e8274728871873f2b4254dbb9fea413c393abdd`。
脚本仅增加显式冻结SHA和末尾未恢复生产统计；初次复用因只允许playing而非loading
同步时钟差异退出1，随后核对确切字段，仅允许这两类时钟值，保留差异、不忽略指令。
这是分析脚本兼容返工，不算项目bug修复。旧整答8局重算后所有既有逐局字段不变。
21项相关CPU回归19通过、2缺Torch跳过，1.487秒；23项生产源码与冻结版本逐字一致。
