# 早期生产计划：训练实际在比较什么

2026-09-13，只读已有A/B四步产物和冻结源码；无GPU、SC2、新PPO或生产修改。
结论：**采到了不同数量，不等于采到了能区分胜负的训练对照。当前证据更支持
检查训练场景/策略覆盖，不支持继续加Depot奖励或把问题定性为新的优化器bug。**

## 范围

两组都是原始SpB、Level6/g2，A关闭执行失败反馈、B开启。runtime
`361cb7ae7fd86fe8811d6fcb6946d0241bf93e11`；各4个global step、每步8局，
实际游戏seed为2732、43567、42613、52416。不是4次optimizer update。
沿用[完整记录](rl4.md#r2完整结果)及[原始日志审计](rl4-reward-audit-final.json)，
12份原始日志/配置文件哈希、终局归因、四步指标复核通过；两组result/.done相等、
退出0、完整step4。未加载权重、优化器或新增采样。

本轮按prompt的`Current Stage: Marine Reinforcement Flood`选每局第一次进入
增援阶段的回答，**不是一律取第二次模型调用**：前两批部分游戏需3–9次调用才进入
该阶段。相同game time也不等于相同输入；排名按完整query哈希分组，history关闭。
[逐条结果](early-training-signal.json)保存64个阶段入口、原答/行号/输入哈希与标量界。

## 训练组缺什么

| 批次 | 第一条增援计划选择的Marine数量（只统计合法回答） | 结果 | 能提供的比较 |
| --- | --- | --- | --- |
| A step1/2 | 2/4，以及2/4/8/42 | 两批均8胜 | 有计划差异，没采到赢/输对照；输入不全相同 |
| A step3 | 8条全为42 | 8负，advantage日志0 | 同输入、同计划、同分，没有相对排名 |
| A step4 | 6条42、2条8 | 8胜，全为20.5，advantage日志0 | 同输入确有数量差异，但此seed结果不区分 |
| B step1/2 | 2/4/42，以及4/8 | 两批均8胜 | 非零信号不能直接当作更强生产计划的证据 |
| B step3 | 7条4，另1条无效 | 8负 | 有后续分岔，没有该批获胜路线 |
| B step4 | 6条4、2条8 | 7胜1负 | 确有结果排名，但根本没采到42 |

八个训练批中七个全胜或全败；同分/异分与胜负仍分开，不能说七批全部零优势。
已知Marine8→42改变胜负的seed1098341402不在这四个训练seed中，其04:21输入
也没有出现在这64个阶段入口。此前[单变量干预](second-plan.md#marine单变量结果与改进决策2259以后核验)
证明的是该场景有用选择，不意味着本轮训练已比较过它，更不意味着应固定输出42。

## 具体回答得到了什么信号

冻结代码为`GiGPO mean_norm`，gamma=.95按Planner决策数折扣，step权重1；
episode项按轨迹决策数均分，无效回答另扣.1。先计算折扣回报，再复制补齐及平衡，
最后求相对优势。原始复制ID/平衡排列/逐token advantage张量不在这份快照中。
以下只重建整条回答的标量，**不是Marine数量token的直接梯度或真实参数变化**。

| 真实回答 | 结果 | 本轮标量证据 |
| --- | --- | --- |
| B step1/env5，04:13、第70行，Marine42→Depot8→… | 胜，20.5 | 同输入6答，step项−0.026511；考虑可能的episode代表行后，联合优势在−0.028594至−0.017136 |
| A step2/env0，04:01、第121行，Marine42→Depot8→… | 胜，20.5 | 该anchor只有自身；step项0，episode项保守界−0.0425至−0.01875 |
| A step4，第153–160行，42与8两类 | 都胜，20.5 | 两答/局、全合法、无需补齐，数学上的相对优势为0；日志吻合 |
| B step4/env0，第265行，Marine4→Depot1→… | 负，−49.431 | 穷举31种单条复制，第二答联合优势始终负：−71.776至−61.482 |
| B step4，其余7条，Marine4或8 | 全胜，20.5 | 同样穷举全部复制位置，第二答联合优势始终正；各行数值见JSON |

B第70行的42并非“错误答案被惩罚”的证据：它的折扣回报为17.5761875，同输入
其他回答为16.290125、18.05、18.50125、16.697378、18.50125。大家都获胜，但剩余
规划次数和少量附加分不同；相对均值使它略负。不能据此宣称更新一定降低了42的
token概率，也不能把其他场景的手工干预结果偷换成这条回答的真实质量标签。

B第四批31条真实决策补至32，且全合法。对31种可能复制位置完整枚举：只有复制
第265行的标量结果同时符合日志两端−64.868/+20.238；此时第二答env0约−61.482，
两个Marine8胜局各+20.238，五个Marine4胜局约+12.479至+17.756。
**这是日志约束下的推断，不是找回了复制ID或PPO张量。** 重要结论不依赖选中该位置：
31种情况下，上述失败回答都负、其余胜利回答都正。因此没有证据说这一批把赢输
颠倒，也不应把已知复制补齐直接定为本次模型行为差异的根因。

计算依据为冻结`gigpo/core_gigpo.py:87,174,338`、
`verl/trainer/ppo/ray_trainer.py:201,1219,1310`及
`agent_system/reward_manager/episode.py`；Git blob哈希保存在JSON。
[GiGPO v3，§4.1–4.2](https://arxiv.org/html/2505.10978v3#S4)
给出轨迹与相同状态内的相对回报比较。本项目的轨迹均分、复制补齐是本地适配，
不能把这些全部说成论文原样实现。本轮不新增公式，也不重做已无胜率收益的gamma对照。

## 收敛判断与下一项

尚未发现新的实现偏差。更直接的证据是：训练没有稳定地在关键早期状态采到
有游戏结果差异的选择；已有信号会强化当前场景里能赢的短计划，却不能保证迁移
到需要另一种投入节奏的场景。后半句是覆盖不足的解释候选，不是已证实的参数根因。
现有产物不足以追踪某次更新怎样改变42/8的条件概率，不扩展Adam/精度诊断。

下一项建议进入**训练场景覆盖短验证**，不再改奖励：以现有A4作为起点，固定
Level6/g2，单独考察已知存在计划差异的seed1098341402，从开局自然采样，不注入
历史首答或Marine42。先看真实轨迹是否重访关键输入并产生有区分度的回报，最多
新增1个global step并保存checkpoint；同分或没有采到目标状态时明确记录“覆盖
未建立”，不能靠一次更新或KL有限称为学会。prompt/采样/反馈关闭/BO/reward/
优化器全部沿用A4续训配置；它是学习可行性诊断，不是完整单变量A/B或泛化评测。
本轮只提出该方案，未新增入口或启动任务。建议资源为43确认空闲后的GPU0–3，
最多45分钟、8条rollout、A4续至global5，新增完整checkpoint约60GiB，不自动重试；
资源预算须用户另批，不能复用前次纯评测授权。训练后独评另报预算。
不能直接把已用于诊断/训练的seed再称为未见留出集，也不自动续96步。

CPU实际执行：

```bash
PYTHONPATH=experiments/records/20260911-execution-failure-feedback-v1 python -B \
  -m unittest test_early_training_signal test_b4_training_signal test_a4_training_signal -v
python -B experiments/records/20260911-execution-failure-feedback-v1/audit_early_training_signal.py \
  --output <新的JSON路径>
```

12项测试通过，完整审计退出0。第一次逐字比较旧审计时，B第三批mean_reward的
末位相差约7e−15；确认全部原始哈希/逐局值一致后，只对派生均值允许1e−12算术差，
其他字段仍精确比较。此为诊断脚本处理浮点序列化，不计作项目bug修复。

## 已批准的场景覆盖单步（2026-09-13）

状态：completed（9月13日17:40北京核验），用户已单独批准43 GPU0–3、8局、45分钟、约60GiB新checkpoint。
RUN_ID：`20260911-execution-failure-feedback-v1-spb2-A4-l6-scene1098341402-rl1-4gpu-r1`。
父checkpoint是A-r2的`global_step_4`，不是B4、原始模型或A8。恢复模型、优化器、
调度器及原loader保存的rank RNG；沿用已验收的耗尽dataloader尾部处理。
父快照没有SC2状态，本次明确新开局、reset_index=0，实际8局seed均1098341402。
这是显式指定诊断场景，不冒充原16-bit训练随机数序列的下一次抽样。

入口`run_a4_scene1_node43.sh`复用A4续训配置。仅将四个新epoch改成一个、保存频率
改成1，并选用`continue_a4.install_scene`指定上述种子。调度器总预算仍96，
实际只运行global5，保存后退出，无初评/后评、无global6、无自动重试。
命令来源记录关闭；当前源码中的只读记录hook不启用。模型/prompt/192-token预算/
采样/反馈关闭/history关闭/g2/微操/自动战斗/reward/loss/精度/cache/优化器不变。
独立`code-<SHA>`、控制目录`a4-scene1-r1`、Ray路径`/dev/shm/a4s_0913_r1`；
checkpoint继续保存在43已授权本地镜像run根，不写故障共享盘或旧run。

首先验收确实恢复global4、八局实际种子、自然模型回答和恰好一个global更新、
完整global5 checkpoint。再按第一条增援query哈希检查是否重访04:21关键输入
`4c3972dca2ed9c0665934414648b868741a493c494c6cad57119f022f8ac3765`，记录计划数量、
整局reward、胜负及advantage。若全同分/没有关键状态/没有不同计划，则明确
“本次未建立有区分度的学习对照”，不注入答案、不追加奖励、不续训补样本。
一次更新不能证明学习改善，独立评测须另批。严重运行错误、空策略批、非有限
指标或归因异常停止并保留失败；资源不足不擅自换卡数或清理旧checkpoint。

启动前本地CPU回归：`PYTHONPATH=.:tests python -B -m unittest
test_a4_continuation test_sc2_command_trace test_feedback_rl_launch -v`，31项中29项
通过、2项需要Torch的恢复测试跳过，待43固定环境CUDA隐藏补测。dry-run逐项比较
新入口与原A4续训，effective overrides仅epoch数、保存频率和种子选择hook不同，
确认target_steps=5/start_step=4，旧续训仍target8；disabled记录hook行为回归通过。

16:12:21北京，冻结`1689ba46f6307b8a4d0a7825d78b3fda40e587ea`的独立后台控制器
`sc2_a4_scene1_r1`已启动，硬截止16:57:21。43的CUDA隐藏回归31项全部通过；
另与历史`4ad9a60`运行的真实effective_config比较，归一化路径后只存在上述三项
差异，dry-run保留在`/dev/shm/a4_scene1_config_b8qte_jx/dry`。
私有备份`/root/new-scii-execution-feedback-20260911/sc2-a4-scene1-1689ba4-20260913.bundle`
SHA256=`02c3a6204c0c9910e259b4eccbcea29d208dfd5646715af31e4e9e399580f98b`，
依赖已有完整e23eea3源码；远端bundle verify通过。启动前GPU0–7均空，按授权只用0–3；
本地盘约94GiB空闲，A4十二个分片和data.pt及原退出0记录完整。此处仅记录控制器
启动；模型实际恢复、八局seed及最终PPO/保存状态继续以run产物为准。

16:15–16:16复核：`a4_continuation.json`确认原loader已完成A4模型/optimizer恢复，
loaded_global_step=4、只允许运行global5、无SC2状态恢复。八个本RUN专属SC2进程
已创建，并进入开局Planner调用；当前是global5的rollout，尚无本次PPO更新完成结论。
实际manifest的start_step=4/target_steps=5、四卡、training均正确。不是另跑评测，
也没有续到96；配置中的96仅为冻结调度器设置。预估16:35–16:50完成，实际取决于
游戏/回答长度和checkpoint写入，16:57:21为授权硬截止而非保证成功的完成时间。

### 单步完成结果

16:25:56控制器退出0，总耗时815秒；训练result/.done一致、global4→5、恰好新增
1个global step，全部指标有限，无空策略批。step5的4组model/optimizer/extra_state
及data.pt已保存，文件大小复核通过；模型哈希由结束归档记录，本轮不再加载大权重。
17:38核验43八卡均空闲，无本实验tmux。未启动后评或global6。

| 指标 | 结果 |
| --- | --- |
| 八局自然终局 | 6胜2负，无截断 |
| 整局累计reward均值 | 2.875（六局20.5、两局−50） |
| 真实决策有效率 | 16/18，88.9%；训练含补齐指标为85%，口径不同 |
| 已知04:21输入重访 | 8/8，完整query SHA均匹配预登记值 |
| Advantage日志范围 | −70.514至+24.962，不再全0 |
| PPO clipfrac / reference KL | 0.004 / 0.012；PPO K1日志四舍五入为−0.000 |
| 裁剪前梯度范数 | 88.241 |

八局从开局自然生成同一首答Depot4/Barracks4/Marine24，没有历史回答注入。
04:21第二答中，env0/1/3/4/6/7自然选择Marine42＋Depot8，六局均胜；
env2/5选择Marine8＋Depot2，两局随后进入第三答并败。第三答分别含不可选的
`techlab starport`、`techlab`，前者还有重复列表且达到192-token预算。
**本次证明目标状态覆盖、计划差异和回报差异能同时出现在真实训练批中**；
不是Marine单变量因果对照，因为Depot数量也不同且后续轨迹不同。

24行原始log中6行来自已结束游戏的未消费输出；按各局实际终止边界排除，
保留18条真实决策，不能把那6条重复reward或自由文本当成有效模型决策。
复用冻结源码`credited_episode`/`policy_transitions`复核奖励归因，累计均值与
日志一致。源文件哈希、逐局结果/回答行号和指标见
[完成审计](a4-scene1-r1-result.json)；全文回答和分片清单保留在原run，不复制入Git。
结束阶段日志仍有Ray worker SIGTERM提示；
控制器和训练均退出0、全部终局/PPO/保存完成，不把提示单独计作训练失败。

该6/8是**更新前A4采集的训练rollout**，不是step5训练后胜率，不能宣称已学会。
这次的进展是打通了有区分度的场景训练批，并实际更新保存。下一项应比较A4与
新step5的同配置独立评测，检查选择是否改变、是否减少后续无效回答以及游戏表现；
评测预算另批，不根据一次更新自动续长训，也不据此增加奖励/战术/精度机制。

### 已批准的A4/step5配对独评

9月13日用户明确批准：43 GPU0–3串行A4、A5scene各8局，同一场景seed1098341402，
合计最多45分钟；无PPO、新checkpoint、权重复制或自动重试。17:48健康快照为八卡
空闲、主机available约451GiB、本地盘剩余35GiB；启动前再查占用。本轮小产物写本地，
不删除现有checkpoint。四卡匹配两份原生四分片，串行避免两个checkpoint并发加载。

复用`run_feedback_eval_node47.sh`及`feedback_eval`的已验证纯评测/隔离回放入口；
新增显式场景选择`FEEDBACK_SCENE_EVAL=1`，仅接受A4与本次A5scene。旧learning评测
默认feedback-on，本轮两组都保持训练时的feedback-off，不能悄悄恢复为on。
模型之外的prompt/阶段/BO/192 tokens/T0.4/top_p0.9/history/off/奖励/微操/数值路径/cache
完全相同；无回答回放、无固定Marine42。模型/optimizer仍由原加载器恢复，但验证入口
val_only=True、save_freq=-1、0次optimizer更新；不把加载optimizer称为继续训练。

两组都新开游戏，实际seed逐局核对为1098341402，生成前使用既有hook设置各rank的
Torch seed=rank（0/1/2/3），比较保存的CPU/CUDA RNG哈希，不使用训练6/8作为独评
对照。由于模型输出会改变后续调用长度，不能宣称每个后续token共享同一个随机抽样。
主指标为04:21输入重访率、原始选择的Marine/Depot联合分布、真实有效率、整局累计
reward和胜负；无Planner游戏照计整局结果，已结束游戏未消费输出不算决策。
若两组未重访相同输入，先报告轨迹分岔，不强行归因成同输入决策改善。

训练场景内独立采样可以判断该更新是否改变已见场景表现，不能证明新seed泛化。
不规定本次必须涨胜率；结果相同是“未观察到改善”，而不是据此立即改奖励或长训。
计划RUN_ID为`20260911-execution-failure-feedback-v1-spb2-<A4或A5scene>-l6-scene1098341402-eval8-4gpu-r1`；
控制入口`run_a4_scene_eval_node43.sh`，独立`code-<SHA>`、控制目录`a4-scene-eval-r1`，
Ray分别`/dev/shm/se_A4_0913_r1`和`/dev/shm/se_A5scene_0913_r1`。
原1689ba4部署、A4/A5 checkpoint和训练结果不改。当前为准备状态，未把CPU通过当作上卡。

本地CPU：`PYTHONPATH=.:tests python -B -m unittest test_feedback_learning_eval
test_feedback_eval test_a4_continuation -v`，23项中20项通过、3项Torch测试跳过。
配置对照确认两评测组仅checkpoint路径与记录标签不同，且都保持feedback-off；
原feedback-on评测、原A4续训入口默认行为不变，CUDA隐藏服务器回归随后补测。

17:56:26北京，独立tmux `sc2_a4_scene_eval_r1`从冻结
`6e7ba0ae5a011f0882a23a5072a25211355eb216`启动，硬截止18:41:26。
43 CUDA隐藏CPU回归23项全部通过，原模型/SC2资产及system-v1源码hash检查通过。
与1689ba4的生产代码、配置和train_sc2入口无diff；只增加本轮评测编排和验收。
私有Git bundle SHA256为
`4ae9c48e91ccf2082953b615a20ddb47c73c685e9906985ebf6ef185eeaffde8`，
远端verify通过，依赖已保留的1689ba4完整源码。启动前八卡全空，0–3按授权使用。
两父run的result/.done、source SHA、模型manifest和分片stat均通过。
控制器先A4后A5scene；当前A4加载中，尚不据此声称SC2已开始或新权重已评测完成。

18:01复核：A4的`checkpoint_evaluation.json`已确认加载返回、global_step=4、
val_only=true、optimizer_updates=0、feedback=0。`evaluation_seeds.jsonl`记录本次
reset实际八局均为1098341402，四份sampling-rank记录确认初始seed分别0/1/2/3。
17:58:35开始创建八局Flat128、Level6（Harder）Terran/Air游戏，17:58:40进入
首次Planner调用；18:00四卡占用约16.6–16.8GiB且正在推理。因此A4已实际评测，
不是只完成准备；A5scene此时仍待串行启动，尚无本轮胜负或学习效果结论。

### 配对独评完成结果

两组均完成：A4于18:04:02、step5于18:10:53北京结束，控制器及两run退出0；
总867秒，小于45分钟授权。18:13快照43八卡已空，无自动重试或续训。
冻结源码仍为clean `6e7ba0a`，实际分布式恢复分别返回global4/5；
两组val_only、0次optimizer更新，各8局自然终止、无截断、无无模型决策游戏。

| 指标 | A4 | 新step5 |
| --- | --- | --- |
| 胜率 | 7/8（87.5%） | 8/8（100%） |
| 平均整局累计reward | 11.75625 | 20.5 |
| 真实有效回答 | 16/16 | 16/16 |
| 每局Planner调用 | 2 | 2 |
| 第二答Marine42 / Marine8 | 7 / 1 | 8 / 0 |

逐局实际seed都为1098341402，四rank初始CPU/CUDA RNG哈希逐一相同；
effective-config差异仅checkpoint、模型标签、隔离输出/model工作目录和Ray路径。
16对prompt逐字相同，所有第二答重访04:21输入hash `4c3972dc…`；
16对回答只有env1第二答（原log第10行）不同：Marine8→42，其余字节相同，
Depot仍为8。该局A4负、reward -49.45；step5胜、reward20.5；其他7局均胜且
原回答相同。这个变化是在线模型自然输出，不是人工42替换。
所有回答均通过既有合法性判定，未见本轮输出退化；没有第三答可用于评价后续纠错。

这是一次训练后在**已见场景**中的正向配对观察，比只看训练rollout更有信息；
但只有1对胜负改变，不能据此断言稳定提升、跨seed泛化或项目已修好。
同一游戏seed的8次采样不能当作8个独立场景，故不以跨场景置信区间包装结果。
下一项建议仅用既有、预先固定的多seed评测组比较A4与step5，保持反馈关闭和
其他条件不变，检查是否牺牲其他场景；新增GPU预算另批，不先续96步或新增机制。

本轮只读核验包括：result与.done相等、runtime SHA/clean、加载记录、实际游戏种子、
四rank采样RNG、逐局终止边界、32条真实回答与累计reward、配置差异白名单。
均通过；各run结束时仍有Ray清理SIGTERM提示，但在自然终局和结果完成后，
不能据此把退出0的整轮评测记成失败。摘要/源文件hash见
[配对结果](a4-scene-eval-r1-result.json)，完整回答和回放仍留原run，未改旧产物。

### 下一项：八种子配对评测

问题：step5在训练场景的正向选择，是否以其他场景退化为代价？复用9月11日就已
声明的`learning_eval.GAME_SEEDS`，顺序不变：42613、52620、17089、4018390961、
3236598124、3297512220、773351930、1098341402。每组每seed一局，共16局。
最后一个是本次step5新增训练用场景，单列它与其他七个场景；不能将历史已用于
诊断的种子组称为盲测或全新保留集，也未证明前七seed在更早训练中从未出现。

仍是A4对其scene-step5，Level6/g2，反馈off，无回放/人工回答，T0.4、top_p0.9、
192-token预算、原段协议、quota澄清、history/off及数值路径全部保持上轮状态。
旧多seed A4评测feedback-on，不可直接与本轮step5对比，所以两组需共同重新评测。
只复用原串行控制器增加`FEEDBACK_SCENE_SET=historical8`选择和
`learning_eval.install_multiseed`薄入口；未改agent_system、verl、configs或train_sc2。
默认single行为保持原样，所有旧部署和结果不变。

候选CPU命令：`PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:tests python -B -m unittest
test_feedback_learning_eval test_feedback_eval test_a4_continuation -v`。
本地25项中22项通过、3项因无Torch跳过；确认新旧评测配置仅seed hook不同，
同组A4/step5仅checkpoint及模型标签不同，旧feedback-on入口未改变。
服务器CUDA隐藏回归和正式加载仍待实际执行，不把本地通过当成评测启动。

18:23健康快照：43八卡空闲、主机available约451GiB、系统盘剩余34GiB；本地
checkpoint和依赖沿用上轮，未复制/删除资产。提出本轮独立预算：GPU0–3串行
各8局、合计45分钟（3 GPU小时上限），无PPO、新checkpoint、大权重复制或自动重试。
预算尚未获确认，不复用上一轮单场景45分钟的授权。

确认后仍调用`run_a4_scene_eval_node43.sh`，另需
`SC2_A4_MULTI_EVAL_AUTHORIZED=1`及`FEEDBACK_SCENE_SET=historical8`；
从新clean SHA独立部署，控制目录`a4-multiseed-eval-r1`，Ray前缀`me_`，
RUN_ID为`20260911-execution-failure-feedback-v1-spb2-<A4或A5scene>-l6-multiseed-eval8-4gpu-r1`。
启动前重新查卡与目录，若已有产物不覆盖；不因上轮已通过而省略checkpoint恢复核验。

交付逐seed胜负、整局reward、真实有效率/调用次数、原始选择变化；无Planner局也
纳入整体均值。区分同输入选择变化与不同状态轨迹，汇报配对胜负改变数，不以总胜率
一项宣布学习成功。优先排除回答或其他场景明显退化；单轮8seed各1局仍不足以证明
稳定收益，不自动续训或再加修复机制。GPU状态和时间以实际启动后的记录为准。

用户随后明确确认本轮四卡45分钟预算。冻结runtime为
`e8d8de164ed5312c33f9905613176503a9d1832c`，43独立`code-e8d8de1`部署，
与上一轮6e7ba0a相比agent_system、verl、gigpo、configs和train_sc2无diff。
私有增量Git bundle已在43 verify通过，依赖已保留的6e7ba0a；SHA256：
`569b98535d4e67d339f1096aa9ad444c622d450d1356de15a5aeb3141dc1a56c`。
服务器CUDA隐藏CPU回归25项全部通过（8.796秒），system-v1源码hash检查通过。

18:40健康检查及启动前复查八卡全空；18:43:45北京在独立tmux
`sc2_a4_multi_eval_r1`启动控制器，硬截止19:28:45。双方父run的.done/result、
来源SHA、model manifest和checkpoint分片stat均通过；未覆盖旧文件。
18:44状态是A4加载准备中，当前尚不以目录存在或preflight通过声称实际SC2已运行。
控制器按A4→A5scene串行，不做PPO、不会自动续第6步或96步。

18:46复核A4加载和开局已成功：`checkpoint_evaluation.json`确认返回global4、
val_only=true、optimizer_updates=0、feedback=0；实际`evaluation_seeds.jsonl`
与声明的八seed逐项一致。18:45:49开始八局Harder/Terran/Air，随后进入首次
Planner调用并记录初始化采样RNG。故A4已实际评测，step5仍待串行启动。
预计两组从启动起约20–35分钟，硬截止不变；尚无本轮结果或泛化结论。

### 八种子配对结果：未退化，也未观察到游戏收益

9月13日19:25核验：A4于18:54:53、step5于19:06:08北京完成，两run及控制器退出0，
总1343秒，43八卡已空。两组分别真实恢复global4/5、val_only、0次PPO，各8局自然
终止，无无模型决策游戏。实际seed、四rank初始CPU/CUDA RNG逐项匹配；源码仍clean
`e8d8de1`。配置差异仅checkpoint/标签/独立路径，没有悄悄开启反馈或改游戏机制。

| seed | A4胜负 / reward | step5胜负 / reward | 每组Planner调用 / 有效回答 |
| --- | --- | --- | --- |
| 42613 | 负 / -49.382 | 负 / -49.382 | 2 / 2 |
| 52620 | 胜 / 20.5 | 胜 / 20.5 | 2 / 2 |
| 17089 | 负 / -49.450 | 负 / -49.450 | 2 / 2 |
| 4018390961 | 胜 / 20.5 | 胜 / 20.5 | 2 / 2 |
| 3236598124 | 负 / -49.401 | 负 / -49.401 | 4 / 3 |
| 3297512220 | 胜 / 20.5 | 胜 / 20.5 | 2 / 2 |
| 773351930 | 胜 / 20.5 | 胜 / 20.5 | 2 / 2 |
| 1098341402（step5训练场景） | 胜 / 20.5 | 胜 / 20.5 | 2 / 2 |

两组均5/8胜、平均整局reward **-5.716625**，逐seed胜负变化0/8；其他七seed均4/7胜。
两组各32行原log，按实际终止边界排除14行已结束游戏输出后，均18次真实决策、
17次有效（94.4%）。汇总`val/sc2_action_valid=1.0`不是这18次的总体有效率，
本报告采用逐决策分子/分母，不用100%覆盖无效回答；此次未修改该统计模块。

18对模型输入逐字相同，实际回答只有两处变化：

- seed52620第二答：Marine8→42，Depot仍8，其余文字不变；两局本来都胜。
- seed3236598124第三答：step5少了末尾4条，但两组都无效；第四答相同，最终都负。

故不能称为“模型完全没变”，也不能称为“学习效果已改善”。本轮未观察到新增无效
回答或胜率退化，但小样本不构成无退化保证。上一轮同训练seed的7/8→8/8仍是有效
局部观察，并未被这轮推翻；它尚未转化为当前多seed样本的胜负改善。
尤其三个失败seed的第二答都已选择Marine42/Depot8：不能继续把失败统一解释为
“Marine数量选得不够”，也不能仅凭模型报出42断言已造出42或执行/战斗没有问题。
后续若继续分析，先复用这三局日志/既有消融确认剩余失败的具体边界，不新增奖励、
prompt或战术机制，不因无退化直接续96步。新GPU实验预算仍须另批。

完成核验覆盖.done/result一致、源SHA/clean、恢复、seed/RNG、终止边界、逐局reward、
有效回答、原回答/投影差异及配置差异；全部使用小型既有产物，无新增GPU任务。
摘要和源文件hash见[八种子结果](a4-multiseed-eval-r1-result.json)，全文回答/回放留原run。

随后完成[三个败局与同计划胜局的只读回放审计](multiseed-failure-boundary.md)：
Marine实际出生66/66/68、死亡65/66/67；同计划胜局出生66、死亡36。
已排除本轮“Marine42没有产出”的解释，确认第二阶段不开放后期对空选项，且长队列
执行中不因阶段时间到期自动重新规划。尚未证明改变这些边界即可提高胜率，未新增GPU。

## 回到学习主线：A4→step5闭环审计（2026-09-13）

本节按用户最新要求收敛范围：只核对既有回答、奖励、训练信号及配对独评；
不继续过程奖励候选的命令归因、Factory、坐标或SCV诊断，不启动GPU或修改运行机制。
这些候选仍未采用，其记录器修补不计作原项目学习问题的修复。

训练权威是`1689ba46f6307b8a4d0a7825d78b3fda40e587ea`及
[原始完成摘要](a4-scene1-r1-result.json)，不是当前分支HEAD。
43上的原始manifest、effective config、console及回答/终局日志哈希与摘要一致；
源码对照确认当前`agent_system/`、`gigpo/`、`verl/`、`configs/`相对该SHA无差异。
同场景独评SHA为`6e7ba0ae5a011f0882a23a5072a25211355eb216`；
八种子独评SHA为`e8d8de164ed5312c33f9905613176503a9d1832c`。

| 环节 | 实际证据 | 能说明什么／不能说明什么 |
| --- | --- | --- |
| 回答→整局回报 | 八局第二次输入逐字相同；Marine42/Depot8六局均胜、各20.5；Marine8/Depot2两局均负、各−50 | 本批存在计划和结果差异，不是全部同分；两种数量同时变化，不是Marine单变量因果结论 |
| 回报→训练信号 | GiGPO mean_norm、gamma .95、step权重1、无效惩罚.1；global5 advantage范围−70.514～24.962、pg_loss −.054、裁剪前grad_norm 88.241 | 信号并非全零；有限梯度不能证明逐参数更新正确，也不保证某个数量token的概率变化方向 |
| 真实更新与保存 | A4完整续训global4→5；ppo_epochs=1，lr=1e−6，4卡、每卡micro-batch=1；退出0、完整step5 | 新增1个global step，不是仅1次optimizer update；日志中lr显示0.000只是三位小数格式化 |
| 同训练场景独评 | A4→step5：7/8→8/8胜，平均整局reward 11.75625→20.5；16对真实输入相同，只有一条第二答Marine8→42，Depot仍8，对应负→胜 | 有局部正向观察；只有一个改变胜负的配对，不能宣称稳定提升 |
| 八种子独评 | 两组均5/8胜、reward均−5.716625；18对真实输入相同，2条回答改变，逐seed胜负全相同，真实有效率均17/18 | 未观察到跨场景游戏收益或本面板退化；不是完全没变，也不是学习验收通过 |

样本口径：训练24条原始输出中，6条是终止后未消费输出；18条真实决策中16条有效。
四卡补齐后指标对应20行、有效率.85，不能与16/18混用。代码先按轨迹计算折扣回报，
再复制补齐、平衡、施加无效惩罚并求相对优势；episode回报由
`agent_system/reward_manager/episode.py`提供，核心顺序在
`verl/trainer/ppo/ray_trainer.py:1219,1225,1310`，优势公式在
`gigpo/core_gigpo.py:174,338`。本轮没有把复制补齐直接认定为漂移或不学习的根因。
原始复制ID、平衡排列及逐token advantage/更新前后概率张量未保存于这些小型产物；
因此本节不冒充逐回答精确优势或参数更新审计，也不为补齐它另起重放。

判断：尚未发现新的实现偏差。此前八个A/B训练批有七个全胜或全败，确实缺少稳定的
胜负对照；但不能说七批的所有优势都为零。选中有结果分岔的场景后，现有奖励已能
产生非零信号，并观察到一次局部正向行为变化。故“模型完全没有学习空间”及
“必须先加过程奖励才能学”都没有得到支持。当前最直接的训练瓶颈候选是有效比较的
场景/策略覆盖不足；它尚不是三个剩余败局的已证实根因。

三个失败seed（42613、17089、3236598124）的第二答都已为Marine42/Depot8；
继续统一提高Marine数量没有依据。同样，八种子面板包含已用于诊断的场景及本次
训练场景，不能称作未见测试集；单次采样差异不能替代条件概率或显著提升证据。

唯一下一项建议是**现有奖励下的训练场景覆盖短RL对照**，不是继续修记录器：
两组从同一个A4完整状态开始、各4个global step/每步8局；A沿用原采样规则，
B只替换预先冻结的训练种子表，包含已确认有胜负分岔的1098341402和现有三个失败场景。
同一训练组内仍是同seed的8条自然采样，不硬塞Marine42，不改reward、prompt、温度、
BO、反馈、执行器或优化器。随后用相同预先固定的评测条件比较两组及A4，分别报告
训练覆盖、非零信号来源、回答变化与游戏收益。若再次只得到同分或局部变化，就明确
不支持晋级，不能靠累计步数称为学会。该方案尚未实现、未占卡，训练及独评的资源预算
需另行确认；不复用既往授权，不自动续96步。

### 场景覆盖A/B：已批准实施

用户批准43 GPU0–3串行，两组各新增4个global step、每步8局，合计3小时；最终各保存
一个完整step8，约120GiB新产物写共享盘，原始模型/依赖及A4父checkpoint读取本地。
不删除旧产物、不自动重试或续96步；共同条件独评另报预算。
22:54北京时间只读检查：43八卡无GPU任务，本地盘剩约34GiB，共享盘约2.54TiB，
故本轮不把新checkpoint继续写满本地盘。历史SC2残留进程未清理，未停止任何任务。

| 组别 | global5–8的实际游戏种子（每步同seed八局） | 起点 |
| --- | --- | --- |
| A | 45891、21243、30403、32103（原RandomState(0)跳过四批） | 原A-r2完整global_step_4 |
| B | 1098341402、42613、17089、3236598124（预先声明的诊断场景表） | 同一份global_step_4 |

B包含已知分岔及失败场景，是有意改变训练分布，不能冒充盲采样或未见测试集。
两组都是SpB、Level6/Harder、g2、Air对手；从开局自然调用Planner，反馈off、history
off、192-token、T0.4、GiGPO mean_norm及原奖励/优化器均不变。不是从step5继续。
完整模型/optimizer/每rank RNG由原loader恢复；driver seed0、SC2全新游戏且reset_index0，
不声称恢复旧SC2进程。`trainer.total_training_steps=96`只保留旧调度器配置；本轮
`total_epochs=4`、起点4、目标8，控制器拒绝额外批次，不会自动运行到96。

唯一行为差异在实验hook `continue_a4.py`：B用有界四种子序列替换训练游戏RNG；
A仍走原分支。`run_feedback_train4_node43.sh`只增加显式选择该hook的开关，默认关闭。
两组都调用反馈关闭的A入口；控制器串行、检查空卡、共同截止、检查每组32局/4步/
完整保存。核心模型、loss、reward、Planner及Executor源码未修改。
旧A8虽可作历史参照，但其SHA为4ad9a60，后续加入过可选记录代码；本轮两组同SHA
执行，不用未经逐路径证明的跨SHA结果冒充唯一变量对照。

入口：`SOURCE_SHA=<clean pushed SHA> SC2_A4_COVERAGE_AUTHORIZED=1
A4_CONTROL_DIR=<unused directory> bash experiments/records/20260911-execution-failure-feedback-v1/run_a4_coverage_node43.sh`。
运行ID：`20260911-execution-failure-feedback-v1-spb2-A4-l6-coverage-{A,B}-rl4-4gpu-r1`；
输出根为`/mnt/data0601/yangjinyue/new_SCII/experiments/20260911-execution-failure-feedback-v1/runs`。
用新冻结部署，不修改1689ba4、旧A4/A8和既有结果。

观察每批真实回答/有效投影/组内回报差/非零优势来源，以及完整更新与保存；训练胜率
来自不同场景，不能直接用B训练均值减A均值宣布学习改善。正式收益必须等共同条件
独评，现有八seed面板也只算诊断回归，不算未见泛化。NaN、空策略批、归因错误、存储
失败或超时明确停止，保留失败，不补假样本、不自动重采样。

CPU：新增种子表边界、第五批拒绝、旧单场景入口兼容及A/B effective config对照；
两组配置差异仅setup_hook（独立输出路径等身份字段除外）。本地首次测试因新增测试
缺少gamma_ab导入路径失败，调整互斥参数校验的执行顺序后通过；这是本轮测试/入口
返工，不计作原项目bug。实际CPU命令及服务器执行状态在启动记录中补齐。
