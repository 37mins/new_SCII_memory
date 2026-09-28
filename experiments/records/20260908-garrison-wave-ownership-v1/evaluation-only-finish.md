# 纯评测无决策终局收尾

2026-09-08，用户批准只修纯评测收尾。基于本分支clean提交
`2bd252e11c9948827b10af52072ed21b471c3619`；其相关源码与上一局冻结
`bba87e3a2e6d136d0c9facd031f827f06867faac`逐字相同。
沿用现有隔离工作树和分支，不改原deployment、run或r6，不新建实验分支。
以下先记录CPU阶段；随后获批的单局在线验收已完成，见末节。无PPO或长训。

## 原因与最小修改

上一局09:17自然Victory、累计reward +20、真实Planner决策0条，但纯评测
沿用`gather_rollout_data`，在没有policy transition时抛异常，主进程exit1。
证据为[原run结果](r1-result.json)，而不是根据当前工作树猜测。

只改两个已有方法：

- `TrajectoryCollector.multi_turn_loop`：仅在SC2、`trainer.val_only=True`且
  `is_train=False`时，将每局既有累计reward、success、工具调用数、真实决策数
  及已消费回答作为DataProto的meta_info返回。batch保持None，非张量样本表为空，
  不创建训练样本。仍调用原`policy_transitions`检查归因、排除终局未消费回答。
- `RayPPOTrainer._validate`：读取这份评测元数据，按游戏汇总指标；实际模型回答
  仍进入原可选generation logger，无决策局没有假回答占位。其他路径沿用原逻辑。

没有新增开关、抽象框架或异常重试。原`gather_rollout_data`及其空批次报错逐字
不变，训练rollout、采样、PPO、loss、参数更新、奖励公式、prompt、BO、微操、
cache与精度全部冻结。训练过程中`val_only=False`的validation也不改变，不能
把这次独立评测修复称为训练空批次问题已解决。
`configs/system-v1.sha256`只登记两个新源码哈希，不修改配置值。

## 指标与使用

沿用`EVAL_ONLY=1`的原纯评测入口，不增加配置。新正式游戏必须另用新SHA、
新RUN_ID部署；不要直接复用上一局入口默认RUN_ID覆盖旧结果。

`val/<data_source>/test_score`是该数据源每局累计环境reward的等权平均；
`val/success_rate`是所有游戏的等权平均，含无决策局。工具次数同样每局一次。
新增`val/sc2/episode_count`、`no_policy_episode_count`、`policy_decision_count`，
显式区分游戏数和模型真实决策数。有效动作率无样本时保持N/A，不伪造0%或100%。

这会纠正纯评测原先按回答数重复表示整局reward、以及跨不同大小batch平均胜率
的计权口径；不是改变游戏reward或训练loss。旧数值若回答数不等，不能与新
test_score直接当同口径比较，应从原始每局reward重算。

结果写入器`scripts/run_record.py`没有修改。只有实际退出码0、有限评测指标、
足额自然终局且无训练更新才完成；本次没有把上一局failed改成completed。
空游戏集合、不合法的未消费非终局回答继续失败，截断游戏不冒充自然终局。

## CPU证据与限制

Python3.9.6、NumPy2.0.2。测试从源码提取真实方法执行；模型、Ray/SC2交互和
DataProto传输是轻量测试替身，不加载Torch/模型。额外调用原DataProto一致性
检查确认仅元数据对象长度0、不含策略张量。没有声称完整分布式栈验收通过。

~~~bash
python -B -m unittest discover -s tests -p test_sc2_evaluation_without_policy.py -v
~~~

12项回归通过：无决策胜/负、有无决策混合、正常决策、多源、多batch不等数量、
非法归因、空游戏、训练空批次拒绝、训练混合批次只保留真实credit、边界隔离，
以及原记录终局事实到结果写入器的CPU闭环。
以旧冻结源码执行同一无决策胜/负用例，两者都在原空批次检查失败；
以候选方法执行，通过并得到test_score +20/-50、success 1/0、policy_decisions0。
混合样例为一局无决策+20、一局两次决策累计-47，结果应为-13.5、胜率0.5，
而不是漏掉第一局或将第二局按回答数重复计权。

CPU闭环读取真实r1终局事实，在临时测试目录生成结果：候选指标+模拟exit0得
completed、0更新、0有效决策；exit1或截断仍failed。该临时结果是CPU测试产物，
不是重新跑了一局，也不是改变原游戏的退出码。
最初测试替身缺少类型注解依赖及collation适配导致的测试装配失败不计项目bug。

原冻结回归输出见[before](cpu-eval-frozen-before.txt)，候选见[after](cpu-eval-after.txt)。
相关旧测试、源码冻结和入口回归输出见[cpu-eval-related](cpu-eval-related.txt)。
另58项通过：终局归因10、自然终止8、驻防8、发布/入口26、格式奖励6；
加新增12项共70项。源码manifest及git diff --check通过。
AST复核：collector只改变multi_turn_loop，trainer只改变_validate；gather、
vanilla/dynamic采样、训练fit、优化器与其余方法均与bba87e3一致。

CPU阶段结论是纯评测收尾通过CPU验收、尚未完成新GPU/SC2验收。下一步如获授权，
用独立新RUN_ID做单局完整入口验证；不重跑训练、不增加战术修改。
本次提交及本地bundle保留回滚点；未重试此前被拒绝的GitHub推送。

## 已批准的单局在线验收 r2

2026-09-08，用户批准下一项单局验收，启动时状态planned，现已completed。runtime固定为
`e1e25ce1d14362b209e34ec73871ebaeb63538a6`；另行提交的编排只指定新源码、
RUN_ID和临时目录，不向该冻结源码追加改动。比较对象为本记录的r1：自然胜局、
但纯评测空策略异常exit1；不重跑对照、不覆盖其失败状态。

RUN_ID：`20260908-garrison-wave-ownership-v1-spb2-l6-g2-s2621640384-evalfinish-r2`。
13:48北京时间只读核验44 GPU1空闲、未发现同名评测或SC2进程，预定run及
code-e1e25ce尚不存在。复用同节点GPU1及原依赖，不复制/下载模型或改共享环境。
启动前再查GPU及重复任务。仍为原始SpB、Level6/Harder、Flat128、Terran Air、
canonical固定g2、实际seed2621640384，6BO/96token、history0/off、dropout0、
HF/eager/FP32参数及温度归一化/BF16 autocast/cache、shaping off，全部同r1。
只有此前已批准的纯评测收尾机制不同，训练采样和游戏行为不变。

入口 `run_eval_finish_node44.sh` 复用runtime内原train_sc2入口和全部override，
钉住源码SHA，保留GPU空闲检查、dirty拒绝及完整资产校验。最多1800秒，TERM
宽限30秒，一局一GPU，无PPO、无自动重试。输出仍在共享本experiment/runs；
新Ray目录`/dev/shm/sc2_garrison_evalfinish_20260908_r2`，
tmux `sc2_spb_evalfinish_r2`。原r1 Ray路径和deployment均不复用或清理。

验收看自然结果、exit0、有限指标、完整replay/config/.done；若无决策，
episode_count1/no_policy_episode_count1/policy_decision_count0，reward与终局
日志一致，没有假策略样本。胜负不设硬性门槛，但需回放/命令比较是否出现游戏
行为变化。有真实决策时仍核对完整统计，不要求为了覆盖空批次而强行控制游戏。
验收失败保留产物，不越权修改或重跑；通过也不追加seed或启动长训。

启动记录：编排提交`d13728fec1f3d1cfa98b421d074d97fadd378fbb`，私有bundle
SHA256 `dcdb1a81ad2ff723c2464c100418aa323425b8b2a1f477eab5feab89ffb41676`。
SCP曾在SSH握手阶段失败，未创建run或启动模型；随后SSH/rsync成功部署，
不是GPU实验重试。13:58:54再次核验GPU1空闲后创建受控tmux，13:59:17写入
正式manifest，14:01:13确认模型已加载、SC2推进且实际seed为2621640384。
14:02:20复核实际148项启动参数，规范化源码、run和Ray路径后与r1完全一致；
模型manifest、BO、已登记游戏源码、依赖版本、world_size和GPU选择均相同。
CPU预检另以`--cfg job`解析成功：val_only=True、resume_mode=disable、单卡、
env.rollout.n=1、greedy；预检没有加载模型或启动游戏。冻结源码哈希全部通过。

## r2在线结果与验收结论

14:05:09北京时间结束，正式运行约352秒；result、.done及编排退出码均为0，
status=completed。14:07:09确认tmux退出、GPU1无计算进程、冻结源码clean；
旧r1 result哈希仍为`242e93e3b774494307a7e11c5844939f73dbc718692511d28f6c14081bc509b6`。
没有自动重试、追加游戏或启动训练。完整指标、配置核验和产物哈希见[r2结果](r2-eval-result.json)。

| 同seed单局 | 原r1 | 收尾修复r2 |
|---|---|---|
| 自然结果／游戏时长 | Victory／09:17.14 | Victory／09:17.14 |
| 整局累计reward | +20 | +20 |
| 真实Planner决策／PPO更新 | 0／0 | 0／0 |
| 评测退出码／状态 | 1／failed | 0／completed |
| 入编队后驻防覆盖 | 0 | 0 |

新指标为`val/text/test_score=20`、`val/success_rate=1`、
`val/sc2/episode_count=1`、`no_policy_episode_count=1`、`policy_decision_count=0`。
终局有1条生成但未消费的回答，`policy_action_applied=false`、
`semantic_action_evaluable=false`，与r1文本完全一致，开头为：

~~~text
Based on the provided game state, here's a breakdown and analysis to help you make strategic decisions:
~~~

它没有影响本局，也没有当成策略样本。可评判模型动作0条，动作有效率为N/A。
旧环境仍返回`val/sc2_action_valid=1`等默认标志，且`semantic_action_evaluable=0`；
本次没有改这些标志，不能把它们当成100%模型动作有效率。在线只覆盖无决策胜局；
无决策负局、有无决策混合及训练拒绝空批次仍是CPU证据，不冒充各跑过一局。

复用`audit_r4_reactor_pair.py`及`audit_wave_orders.py`只读解码新旧回放，均exit0。
两局491条我方命令完全一致，tracker事件流逐字节一致（SHA256
`985e26f907d1ce5d57838d3c36c17d8581589b21af3f98cbeefb9a4540cabb99`）。
逐事件比较仅`SSetSyncLoadingTimeEvent`和`SSetSyncPlayingTimeEvent`时间戳不同，
不声称整个replay文件字节相同。两局均97 Marine出生/43死亡、3 Tank出生/0死亡，
46 SCV（含初始12）/1死亡；9波次中8波进入进攻，第9波创建后即临近终局。
没有Viking产出，不能据此判断其存活或对空应对已修好。

结论：**接受纯评测无策略样本收尾修复**，本seed下游戏行为保持相同。
这是评测边界修复，不是RL、概率更新、跨seed胜率或模型能力验收。下一项需回到
真实Planner参与的系统验证，预算另行确认；当前不追加机制或长训。
本分支保留原子提交和私有bundle备份，未修改原deployment、r6或main，未推GitHub。

## r3：真实Planner参与验证（completed，接口有证据，单局败）

用户随后要求开始验证有Planner参与的完整游戏。本次只安排SpB一GPU一局，
1800秒墙钟上限、TERM宽限30秒，无PPO、无自动重试。继续使用44 GPU1及
只读`code-e1e25ce`；14:52只读核验该卡无进程，同名新run和后台会话须启动前再查。
不新建branch、复制模型或改依赖；runtime仍为`e1e25ce1d14362b209e34ec73871ebaeb63538a6`。

RUN_ID：`20260908-garrison-wave-ownership-v1-spb2-l6-g2-s2807211443-planner-r3`。
唯一评测变量是实际seed从2621640384改为2807211443；后者是历史r4初评第一局，
初评/step8各消费8/4条真实回答，见[r4参与记录](../20260906-integrated-fixes-v1/r4-planner-participation.json)。
历史模型、游戏源码不同，不能据此保证当前也进入Planner，也不作为本次因果对照。
选择第一项已记录的非空样例，不按新胜负挑seed，不增加g1–g5转换逻辑。

新薄入口`run_planner_eval_node44.sh`仅移除旧的第4项种子选择hook，采用冻结
环境原生`RandomState(1000)`首项，CPU已算得2807211443。仍保留同seed的游戏
Python随机数设置。除hook和run/Ray路径外，其余参数必须逐项与r2相同；最终
`evaluation_seeds.jsonl`必须为base_seed1000/reset_index0/seeds[2807211443]。
源码、模型、prompt、6BO/96tokens、解析/补齐规则、自动战斗、reward、cache、
dropout、精度和history全部不变，不截短g2、不提前强制Planner接管。
新Ray目录`/dev/shm/sc2_garrison_planner_20260908_r3`，tmux`sc2_spb_planner_r3`。
输出仍为本experiment共享runs中的唯一目录，r1/r2及其失败状态不覆盖。

验收须至少有一条非终局的实际消费回答；保存完整query、原始回答、解析与最终
BO队列，再与Executor命令及回放对齐，区分模型选择、无效回答规则补齐、命令
提交与单位实际产出。仅`policy_action_applied=true`或退出码0不算有效计划执行。
并核对自然终局、整局累计reward、有效动作分母、per-game评测指标与PPO更新0。
若无有效模型动作或没有消费回答，则该部分未覆盖，不靠假决策、追加seed或
改协议宣布通过。单局胜败和回答有效率不作为学习提升、总体胜率或RL验收。

编排SHA `426d12415e8b1be4db5b40dcdd4045bd6532f054`，私有bundle已校验，
SHA256 `629ffb87b9f4cfaa4f0cae54b1628e4676f9af9322967c99fb25c343bfc634a4`。
CPU语法与差分检查通过；44上dry-run及Hydra `--cfg job`解析exit0，147项参数
对r2除种子hook、run/Ray路径外无差异。14:59:07再次查空卡/重名后启动tmux，
14:59:31正式manifest写入；15:02:21确认模型已加载、游戏推进且实际seed正确。
15:03:52复核实际config、模型/BO/源码哈希和版本，同样只有预定种子变化。

15:08:19结束，正式运行约528秒，exit0、自然Defeat、游戏时长14:13.39、
累计reward -49.5。7条回答全部实际消费，6条完整有效（85.7%），PPO更新0。
评测正确记录episode_count1/no_policy_episode_count0/policy_decision_count7。
旧`val/sc2_action_valid=1`不是7条回答的合格率，不以它报告100%。
15:09:45核验本次tmux与GPU进程已退出，GPU1已被其他任务使用；没有清理他人进程。
冻结runtime仍clean，旧run没有覆盖。完整query、回答、命令及哈希见[r3结果](r3-planner-result.json)。

### 本局验证了什么

6条合法回答的生产动作与Planner最终计划完全一致，Commander最终队列也一致；
未触发开局Marine guard。合计36项中，25项生产命令可从trace匹配到同帧SC2
回放指令：17 Marine、6 SiegeTank、2 Marauder。这里是命令提交，不是25个
单位均已产出或存活，更不能把整局固定BO产兵全部记到模型名下。

例如10:20原始回答为：

~~~text
Here are the Build Orders with attack instructions:
- marine
- mule
- siegetank
- marauder
- marine
- marine
~~~

最终队列顺序不变；3 Marine、1 SiegeTank、1 Marauder分别在回放loop
13996、14092、14272、14364、14456收到指令，生产建筑tag和raw ability一并保存。
MULE则被旧Executor明确跳过。其余合法计划中也有同样现象，共6项MULE跳过；
ScriptPlayer原有自动MULE仍独立运行，不能因此说游戏里没有MULE，也不能说
Planner这6项控制了它的召唤。最后一份Marine计划有1项超时、1项已开始但未提交、
2项尚未开始即自然终局。未宣称所有合法BO都完整落地。

首轮10:04原始回答包含`- siege tank`，被现有validator拒绝，规则结果为
`supplydepot, marine, marauder, marine, starport, marauder`。这6项单独归为
无效回答的规则处理，其中Starport不能归因于模型主动选择。

### 明确的阻断与证据边界

10:45模型合法选择`vikingfighter`，解析和队列都保留。11:09开始执行后，
日志10次明确报告`Facility under construction.`，11:49按原等待阈值跳过。
回放中Starport在loop14000（10:25）开工，直到loop17267（12:50）被摧毁，
没有SUnitDoneEvent；Viking训练命令0、产出0。直接阻断是生产前置未完工，
不是本轮解析器把Viking删掉。施工为何一直未完成尚未定位，不猜测是工人死亡
或建造覆盖，也不由一次等待超时直接定性Executor有bug。

Planner首次接管时军队已只有2 Marine、1 Marauder；48 SCV随后降为10:45的
19个、11:49的3个。整局共117 Marine、7 Tank、6 Marauder均死亡，SCV56个
（含初始12）均死亡。驻防覆盖进攻编队仍为0，9波中7波进入进攻；本局败说明
消除该冲突不等于整体战斗可靠，但不同seed胜负不能当成该修复的因果A/B。

只读复用`attribute_plans.py`、`audit_wave_orders.py`和`replay_summary`，并按
refresh iteration分段对齐trace及回放。可选sc2reader顶层载入因回放缺少
cache_handles失败，未改依赖；已有s2protocol解码成功。该分析工具限制不是
项目运行bug，不能计作修复成果。产出未逐一匹配到具体模型命令，保留这一边界。

结论：**有真实Planner决策的评测收尾、合法计划保留及实际SC2命令提交在本局
得到验证；完整计划落地和Level6作战效果尚未通过，RL未测试。** 最小下一项是
只读核对该Starport的建造工人／建造命令与中断过程，确定未完工原因后再讨论
单点修复；本次不顺手改MULE、prompt、超时、战术或训练，也不追加GPU游戏。

后续只读[星港施工审计](r3-starport-audit.md)已定位：开工后0.982游戏秒，施工SCV
被Banshee击杀；无改派命令、无接续施工命令，最终星港未完成即被摧毁。
这是施工中断后的接班缺口，不是Viking被解析删除；尚未修改或做修复后游戏验收。
