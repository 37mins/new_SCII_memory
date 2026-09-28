# 执行失败后重规划候选

9月13日最新范围：[回到学习主线的闭环审计](early-training-signal.md#回到学习主线a4step5闭环审计2026-09-13)。
A4→step5已有非零训练信号和单场景7/8→8/8的局部观察，但多seed仍5/8→5/8；
尚未证实新实现bug或跨场景学习收益。暂停过程奖励候选及其Factory/归因诊断，
用户随后批准[场景覆盖短RL对照](early-training-signal.md#场景覆盖ab已批准实施)：
43四卡串行A/B各4步，共同3小时，约120GiB新checkpoint写共享盘；入口和CPU对照
已准备，实际启动/完成状态以新增launch记录为准。独评另批，不自动续96步。

9月13日新增[第三答起在线验证](supply-reward-candidate.md#第三答起在线获批实局验证9月13日)
已完成：只保留前两答历史前缀，43八卡8局、547秒、退出0、无PPO。4局在线Depot
真实建成并恢复产兵，另外4局未新增Marine；8局均胜，奖励未区分上述实质进展。
模型/游戏/奖励冻结；这不是训练学习验收，也不自动启用纠错bonus。
随后[自然失败场景筛选](supply-reward-candidate.md#9月13日失败场景筛选不继续在必胜场景推导奖励修复)
完成：24局中14败，没有记录到人口阻塞反馈；不能据此断言从未阻塞，也不足以支持
Depot奖励修复。已有第二答Marine8→42的失败场景有因果证据，下一项转向核对
早期生产选择的训练信号；本轮只读/CPU，无新增GPU任务或生产修改。
该[早期训练信号核对](early-training-signal.md)随后完成：A第四批8/42全胜同分；
B第四批7胜1负但只有4/8，获胜回答的相对优势确为正。更早两条42胜局的回答标量
却略负，不能等同于数量token概率下降。用户已另批场景覆盖单步：A4恢复global4、
seed1098341402八局，只到global5保存；43 GPU0–3，45分钟，无评测/重试/长训。
冻结`1689ba4`，16:25北京完成、退出0、完整step5，17:38核验八卡已空。
八局都重访04:21关键输入：自然42/8组合六胜、8/2组合两负，advantage不再全0；
这次建立了有回报差异的训练批，但6/8仍是更新前A4的rollout，不是训练后评测。
见[完成结果](early-training-signal.md#单步完成结果)；未开后评、未续长训。
用户随后已批准[A4/step5配对独评](early-training-signal.md#已批准的a4step5配对独评)：
43 GPU0–3串行各8局、同一场景、feedback-off，合计45分钟，无PPO/新checkpoint。
冻结`6e7ba0a`，17:56–18:10完成、两组退出0、无PPO，43卡已释放。
A4/step5为7/8对8/8胜、reward11.75625对20.5、均16/16有效回答；相同04:21
输入下一对回答Marine8→42、对应由负转胜，其余回答不变。这是已见单场景的
正向观察，只有一对改变，未证明稳定提升或泛化；未续RL，见[配对结果](early-training-signal.md#配对独评完成结果)。
随后原8seed配对评测已于19:06完成：冻结`e8d8de1`、43四卡、两组退出0、无PPO。
A4/step5均5/8胜、reward -5.716625、真实有效17/18，逐seed胜负相同；两条回答改变
但没有游戏收益。未见退化不等于学习验收通过，未续96步，见[多seed结果](early-training-signal.md#八种子配对结果未退化也未观察到游戏收益)。

最新结果（9月11日22:59核验）：独立评测原始/A4/B4为3/8、5/8、2/8。
随后同一B4、同seed，仅第二答Marine8→42的纯评测为0/4对4/4，0次PPO、退出0。
[结果及真实产兵证据](second-plan.md#marine单变量结果与改进决策2259以后核验)
支持保留A4为下一轮学习候选，不支持把手写42规则或B4晋级为健康版。
9月12日08:14起核验：A4→A8新增四步完成、完整保存，自动独评00:15退出0。
A4/A8同八seed均5/8、逐局胜负不变；未续96。[新增学习信号分析](a8-training-signal.md)
确认早期计划高度趋同，第7/8步的非零优势主要区分末轮合法性和终局附加分，
不能把连续运行完成称为游戏学习已改善。见[实际续训记录](rl4.md#a4追加四步a8独评已批准启动准备)。
下一项[温度单变量对照](temperature-ab.md#完整结果与晋级结论)已于09-12 09:21完成；
A4/seed42613，T0.4/T0.8均0/8，真实有效数20/20对34/58；第二轮合法队列2种对8种。
升温增加多样性但未改善游戏回报，不晋级短训；09:25确认43八卡释放。
后续[温度短RL＋共同条件独评](temperature-rl4.md)入口与CPU配置对照已准备，
复用已完成T0.4对照，只补T0.8从A4再训4步；四卡2小时预算已确认、服务器26项CPU通过，
前置晋级证据不足，尚未上卡/后台排队，新增预算未使用。
以下为各轮历史过程，旧的“运行中”描述不代表当前状态。

09-12后续：[首攻20/40对照](first-attack-threshold.md)已单独获批，43四卡串行
各4局、共同45分钟、不做PPO。runtime `e3c0695`，服务器13项CPU全通过；
10:30完成、两组退出0，均0/4胜、平均reward -49.382/-49.553。40组仅2局触发首攻，
实际推迟到8:25/8:32，首波仍在约38–40秒后全灭；另2局无首攻记录，未查明原因。
40组每局进入2–5次在线回答，仍未获胜。不晋级修复/长训，下一步只读核对已有回放。
后续[八局回放归因](first-attack-replay-audit.md)已完成：两波40 Marine均真实战死，
没有驻防MOVE拉走；无首攻两局峰值都是36，分别反复卡人口、补给后又卡资源/Tech Lab。
优先验证失败后能否纠正前置顺序，不继续搜索首攻阈值。本次没有修改运行模块或新增GPU实验。

2026-09-11；状态：r3纯评测A/B已完成，各2/4胜，反馈接口接通但未纠正人口阻塞。
固定输入原始SpB有5/8纠正顺序，Level7 step24为0/8，不能外推为Level6训练结果。
用户已另批[原始SpB的Level6短RL A/B](rl4.md)：43八卡分两组、各4步、最多3小时，
两组各完整记录3步，16:43因主机RAM触发Ray保护而失败，没有完整step4 checkpoint；
见[启动记录](rl4-launch.json)。r1资源竞争/r2存储阻塞的失败记录保留。
17:43已按授权[清理本次失败分片与临时目录](cleanup.md)，日志和完整权重保留，未重试。
随后另批r2在43 GPU0–3串行重跑两组各4步，总上限3小时；20:02控制器退出0，
A/B均完成4步并完整保存，分别24/32胜、23/32胜。runtime为361cb7a，不改训练机制、
不沿用不完整checkpoint。实际状态见[短RL记录](rl4.md)。
用户另批[原始/A4/B4独立配对评测](learning-eval.md)，四卡串行、各8局、最多2小时，
20:07接续的原始组因Ray临时目录路径过长失败，0局/0更新，未自动重试；
用户确认后，短目录r2于20:17在43 GPU0–3启动，原始组已进SC2；A4/B4顺序接续，
共同硬截止22:16。不改奖励或续长训。
[离线奖励审计](reward-discrimination.md)
确认部分执行改善得到相同reward，尚不能把它当作唯一根因。
系统线另获批准：44八卡进行[SupplyDepot顺序纯评测](supply-order.md)，
复现既有人口阻塞并只移动一个列表项，检查合理选择能否实际执行；
人工干预不算模型选择。已于16:14完成：B每局确实完成8个Depot并新增2个Marine，
A先重复人口阻塞；但两组均4/4胜、每局reward20.5，不能宣布学习提升。
用户已允许撤销“实际预扣、但目标生产命令明确尚未提交”的选择额度，作为重新
可选项，不是必须补做。候选默认关闭，不合入正式基线，不影响 gamma A/B 或其他任务。

基于 `ce6a64be74a2a65dc426c65aec31ae3122e0a4b0`，独立分支
`codex/fix-execution-failure-feedback`，工作树
`/private/tmp/sc2-execution-feedback-20260911`。父工作树、历史 runtime 与 run 不改。
相关 Planner／Executor 源码与 Level6 父实验 `6a1657f7e045dcf998495a3d0adfb331572bf60d`
一致；[63步审计](../20260910-spb-fsdp-layer-wrap-v1/learning-bottleneck-63step-audit.md)
保存回放与日志依据。目标是验证失败后是否能及时选择不同的执行计划，不预称已经
修复 RL 不学习的根因。

## 变更与边界

`SC2_EXECUTION_FAILURE_FEEDBACK=1` 启用；默认 `0` 仍按原逻辑跳过失败 BO。
启动入口 `scripts/train_sc2.sh` 将它显式传入 Ray，并保存至 effective config。
它是一个执行反馈机制，联动三个生产文件及入口，不称为单参数语义修复。

| 位置 | 候选行为 |
| --- | --- |
| `agents/commander.py` | 每个展开项记录真实选择预扣；只释放失败项及未访问后缀实际扣过的额度，每项一次。未扣过的规则补项不增加配额。反馈重规划保持原阶段；普通队列耗尽仍用既有阶段切换。 |
| `players/llm_player.py` | 仅既有映射失败／重试阈值耗尽触发。下一游戏帧重新规划，不继续旧后缀，不对一次事件同帧循环采样。固定 BO 不受影响。 |
| `agents/planner.py` | 真实 observation 的前置条件过滤和原 query 组装完成后，追加独立执行反馈。名称过滤、约15条／`x N`、原解析与补齐规则不变。 |

反馈包含旧回答 policy ID、原阶段、失败动作和最近阻塞原因、未提交后缀、实际恢复
额度，以及预走位 MOVE 的提交尝试。队列明确标作“解析回答＋既有规则”，不能把
整个队列算为模型独立选择。额外反馈不构造 policy ID、reward 或 PPO 样本；下一条
真实回答继续通过现有 socket 与归因处理。原非切阶段的待结算攻击奖励仍只结算一次，
未改奖励公式；规划次数改变可能改变实际累计奖励，不能声称数值必然相同。

已处理前缀不恢复，也不统称“已完成”：现有 `run_actions` 不提供足以证明成功的
结构化返回。本次不追查提交后的引擎拒绝或最终产出。当前超时项未调用其目标生产
命令，但可能尝试过 MOVE；清除本地预走位缓存不撤销已发命令，MOVE 尝试不冒充
引擎接受。正常提交、MULE 跳过、攻击、微操和自动战斗保持原样。

不增加计时重规划、强制重试或新战术；不恢复6条截断，不改模型、数值路径、采样、
history、reward 或 loss。当前拟验证配置仍为 history off、step shaping off。
已有明确失败后的重规划机会不保证模型会作出更好选择，也不保证组内产生奖励差。
若失败项是从未预扣的规则补项，且释放后待办仍为空，则记录
`skip_reason=no_selectable_quota` 后沿用原跳过，不虚增额度或强行发空待办 query。

## 依据

冻结 `llm_player.py` 的映射失败、timeout 位于目标生产命令提交之前；两个正式
提交路径返回后立即推进索引。`Commander.renew_waiting_bo` 则在生成计划后就预扣。
所以不能只清空队列，也不能按后缀总数直接增加额度。9月11日 CPU 边界探针复现
超时索引1→2、Planner调用0、选择后的待办已空；是合成替身，不是新实局。

[Voyager（2023）](https://voyager.minedojo.org/) 将环境反馈和执行错误用于迭代修订；
[公开 action.py](https://github.com/MineDojo/Voyager/blob/main/voyager/agents/action.py)
的 `render_human_message` 分离环境反馈、执行错误与观测。这里只参考反馈接口思路，
不复制其框架，不声称该工作使用 SC2 配额账本或证明本候选能提升 RL。
额度回收是用户批准、由本项目实际预扣记录限定的适配规则。

## CPU 验证与后继

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:tests python -m unittest \
  test_execution_failure_feedback test_empty_bo_frame test_g2_planner_handoff \
  test_original_bo_segment test_valid_bo_passthrough test_system_release -q
```

本地实际使用 `/private/tmp/sc2-air-results.HBI61m/decoder/bin/python` 运行上述六个
模块及 `test_r6_natural_termination test_r6_terminal_policy_credit test_r6_format_reward
test_quota_prompt test_planner_air_options test_fsdp_rl_launch test_gamma_ab`：
共133项，130通过、3项因本地没有 pinned Torch 跳过，8.106秒，退出0。
其中新增机制19项全部通过；跳过的是未修改的 gamma/Torch 测试，不伪称已经执行。
`bash -n scripts/train_sc2.sh`、`git diff --check` 及全部23个 source manifest 哈希
检查通过。这里没有 CUDA、真实模型或游戏，CPU 正确不等于实局／学习验收通过。

[CPU完整输入示例](cpu-example.json)保存合成观测、实际额度回收与生成的完整query。
例中前缀Marine不恢复；失败Reactor与后缀Marine各恢复1；规则Refinery从未预扣，
没有通过回收虚增。它仍可由**已有**前置条件规则列入候选。
两条手写回答 `- refinery`／`- marine` 均通过原validator，分别解析成不同队列，
没有强制重做Reactor。这是接口证据，不是模型已经学会纠错的结果。

## 已批准的实局 A/B

9月11日用户明确批准：47 八卡分两组四卡并行，A/B 各4局，从控制器启动起合计
最多1小时（最高8 GPU小时），不做 PPO、不自动重试；不停止44 gamma A/B或43训练。
A 用 GPU0–3、开关0，B 用 GPU4–7、开关1，两组来自同一冻结 SHA。
基础候选 `811986a7cf09040dfd7c31176ca4dfd3169e198c` 已保存完整私有 Git bundle；
薄评测入口单独提交后再部署。实际 SHA 由 `SOURCE_SHA` 强校验并写入两个 manifest。

两组均加载既有 Level7 SpB 的 `global_step_24`：
`/mnt/data0601/yangjinyue/new_SCII/experiments/20260910-spb-fsdp-layer-wrap-v1/runs/20260910-spb-fsdp-layer-wrap-v1-spb2-l7-g2-opening-quota-resume4to96-4gpu-s0-r1/checkpoints/global_step_24`。
父 run SHA 为 `90b3fdc2fb7fadc1230861306b20c27540d9b736`；原模型为
`Panyuqi/SpikingBrain-2.0-instruct`，本次不是原始权重实验。四rank模型、optimizer、
extra_state和data由既有loader恢复，但val_only禁止更新；SC2状态不续接，评测重新开局。
先核验小文件/分片长度，再以四rank真实load返回及step24记录确认加载，目录存在不算通过。

每组实际种子为 `[52620,52620,17089,17089]`，按env index配对，只有两个独立场景。
每rank首次生成的Torch seed同为rank，不在每次Planner调用前重置。两组新观测可能
因反馈分岔；不宣称后续采样token逐字一致。无手写回答注入或历史回答替换。
Level6=`Harder`，Flat128，Terran对Terran/Air，g2资产，Planner从开局接管；
T0.4/top_p0.9/top_k0，192token，history0/off，shaping off，约15条/`x N`及
累计配额说明、名称/前置过滤、合法计划保留、对空备选、既有规则补齐均冻结。
FP32参数/温度归一化、BF16 autocast、HF生成cache/eager、分层FSDP与SWA dropout0冻结。
核验资产仍用 `configs/system-v1.sha256`、`configs/spikingbrain2.sha256` 和既有
SC2 build75689/Flat128哈希；Python2组共用47已验证依赖，不下载或复制权重。

入口 `run_feedback_pair_node47.sh` 并行调用 `run_feedback_eval_node47.sh A/B`；
`feedback_eval.py` 只选择四局配对种子、配对初始采样RNG和记录游戏边界。
复用 `refinery_order_eval.install_game_observers` 的只读采集与每局日志隔离，
**不安装**其首两条回答干预。旧命名 `order_diagnostic_game_pid` 仅作为游戏关联键。
原始query/回答保存在prompt_response日志，实际队列/命令/失败反馈保存在SC2 trace。
每组独立Ray/cache/run路径；cache目录隔离不改变模型生成cache开关。

RUN_ID 为 `20260911-execution-failure-feedback-v1-spb2-l7s24-l6-g2-{A|B}-eval4-s0-r1`，
均保存到共享新项目下 `experiments/20260911-execution-failure-feedback-v1/runs/`。
若目录已使用则停止，不覆盖；启动和结束由控制目录记录，结果与退出码保存到各run。

验收首先判断：真实未提交失败后能否收到反馈；原阶段实际预扣额度是否只释放一次；
下一次真实回答是否改变计划并影响实际命令，能否避免重复阻塞。记录失败到下次
Planner的游戏时间/调用次数、原始回答/最终队列、有效率、截断、累计reward和胜负。
提交过的前缀不得恢复，规则补项不得制造配额，不得伪造模型决策。
若B未遇到可触发失败或只是重复同一失败计划，不能宣布纠错能力通过。
若只有系统指标改善也不能宣布RL已经学会；少量胜负不作为学习增益。
机制验证通过才另行申请短 RL 与训练前后配对评测，不自动续96步。

资源快照（09:55 北京时间）：42/43/44/46/47/48/197 全部 GPU 有任务，198认证失败。
43/47/48 的共享目录 stat 5秒超时；这不是 SSH 或 GPU 故障。未停止、部署或启动
任何任务，不能将该快照当作之后永久无卡。

## r1 启动与中止记录

冻结 SHA `9e86f0cb9778cca8af965163514d848cf414742e`；部署
`sc2-a100-47:/root/new-scii-execution-feedback-20260911/code-9e86f0c`。
完整私有 bundle 已在同根备份，SHA256
`735e617900397783e640cd0d9bae178169c8023c308d177bf4145e12715a49b8`；未推送GitHub。
机制与入口合并CPU回归94项：本地93通过、1项缺Torch跳过；47固定依赖、CUDA隐藏
补跑入口全部6项通过（9.145秒），包含setup hook不提前探测CUDA；23项源码哈希通过。
step24小文件/分片长度预检通过，不等于实际分布式加载通过。

11:24:36 GPU检查为空，11:26:11（北京时间）启动受控会话
`sc2_feedback_ab_r1`，外层真实使用3585秒timeout＋15秒强制结束上限。
但另一组 `inner_mem` Python进程2121391–2121398在11:26:04–05创建，随后占用八卡。
先前检查不构成GPU预约，出现空卡检查与正式初始化之间的资源竞争。
两组已到TaskRunner/worker注册准备，但没有checkpoint加载完成证明或游戏记录。
11:28:02仅向本次两个timeout控制进程2121628/2121629发SIGINT；不停止外部任务。

| 组 | 退出时间（北京时间） | 退出码 | 自然游戏 / 回答 / PPO更新 |
| --- | --- | --- | --- |
| A | 11:28:08 | 130 | 0 / 0 / 0 |
| B | 11:28:06 | 130 | 0 / 0 / 0 |

控制目录 `.../pair-r1` 保存started/ended、A/B.done、pair.done与日志；两run保留
`result.json`原始状态failed及所有配置。实验层原因记为资源竞争后的主动中止，
不是模型失败或候选效果失败；`metrics_finite=false`在此表示没有指标，不是NaN证据。
A console SHA256 `94c771a90fac2a7088bb87e06433b6a4ec6f92c3732d46c3a4ac6d10f7f73dab`，
B console SHA256 `a17ddedd6d797a3ea1a9878c94d12f6221c4d38df6abd090541b60df70f44a51`。
尚不能评价模型加载、纠错或胜率，更不能宣称RL已学会。没有新PPO或游戏结果。

11:29只读检查43八卡为0MiB，无Ray/训练进程或tmux，仍有旧run的SC2 CPU残留；
现有Python与headers目录存在，未清理、部署或启动。更换节点/重试须重新确认。

## 已批准的43节点 r2

用户随后批准改到43八卡，A/B各4局，新RUN，最多57分钟，不做PPO、不自动重试。
使用 `run_feedback_pair_node43.sh` 薄包装复用原入口；只选43已有的
`/root/new-scii-spb-level5-96-20260908/{python,include}`、r2唯一目录与更短时限。
outer timeout3405秒＋15秒强制结束，总计3420秒。两组机制和模型配置仍与r1一致，
没有修改 `feedback_eval.py`、Planner、Executor、reward、采样或训练代码。
两组RUN后缀由r1改为r2，Ray/cache为 `/dev/shm/sc2_feedback_{A,B}_0911_r2`。
47旧deployment与r1失败产物不变。43新deployment仍用独立
`/root/new-scii-execution-feedback-20260911/code-<SHA>`，控制目录 `pair-r2`。
11:33:55只读GPU快照八卡0MiB；启动前会再次核对，不将检查视为节点预约。

### r2 实际启动状态（11:43:51 北京时间）

源码 `a496d0678ca5a29ccfe356a7d927b2ce01688ef0`，clean部署
`sc2-a100-43:/root/new-scii-execution-feedback-20260911/code-a496d06`。
完整私有bundle在同根，SHA256
`7311b44fdfd274952064b2c6aad15be6ae74b50ab1614d9cc7e02f00a88e1980`。
43的Torch2.7.1+cu128/Transformers4.55.2/Ray2.48.0/Gymnasium1.2.3与47一致；
CUDA隐藏的入口测试7/7通过（10.321秒），23项source哈希通过；未修改核心模块。

11:38:51启动 `sc2_feedback_ab_r2`；控制目录 `pair-r2`，57分钟硬上限到12:35:51。
A/B RUN分别为：
`20260911-execution-failure-feedback-v1-spb2-l7s24-l6-g2-A-eval4-s0-r2`、
`20260911-execution-failure-feedback-v1-spb2-l7s24-l6-g2-B-eval4-s0-r2`。
两组asset校验与模型代码准备完成，step24分片长度/小文件预检通过；
**实际分布式checkpoint加载、配对种子落盘和SC2游戏尚未确认，不能称评测已跑上。**

11:41–11:43读共享运行目录与依赖目录stat超时；读取effective_config的小型CPU
诊断也在15秒上限返回124。该124是只读诊断的退出码，不是实验退出码。
主进程B=3610926、A=3610928均存活但处于Dl，wchan分别为
`nfs_wait_bit_killable`、`rpc_wait_bit_killable`；没有Ray初始化后日志或真实游戏证据。
GPU0–7各约4MiB、利用率0%，不是已在进行模型推理。此时阻塞属于共享存储等待，
不归因给模型、反馈候选或GPU容量。原进程仍受既有57分钟控制器限制，等待恢复；
未中断/重试、未增加预算、未进行PPO，47 r1全部产物继续保留。

### r2 存储阻塞定位（12:04 北京时间）

11:56核对两个主进程的当前syscall及内核栈：均在`mkdir`各自共享run的
`hydra`目录，等待`nfs4_wait_clnt_recover → nfs4_proc_getattr → nfs_permission`。
这发生在模型加载之前，不是checkpoint加载慢。43内核在启动前已记录NFS锁丢失，
但尚未确定是客户端状态、网络还是服务端对该客户端的处理异常。

11:59对同一项目目录、原模型config和step24 rank0分片进行有超时的只读stat：
43、47均超时124；44三项均成功。12:04，43仍卡在原位置，44仍能读取资产。
因此不能称整个共享存储或资产普遍不可用。44八卡有其他实验，不迁移GPU任务到44。

44上有界`du -sb`核算：原模型11,126,702,315字节，SC2 4,938,576,448字节，
FLA/MoBA合计10,544,502字节，完整step24 checkpoint 63,568,352,897字节；
总计约74.2GiB。43本地系统盘347GiB可用，既有Python依赖可复用。
最小绕行方案是复制这些必要资产及父run小型provenance到43独立本地目录，
并令新run日志使用本地目录；仅改日志路径不足以消除后续共享盘依赖。
该方案尚未复制、部署或重启：大体积复制与新RUN预算须另行批准。
不修改全局NFS挂载、不停止其他任务、不跳过optimizer状态恢复来缩小复制量。

## 已批准的43本地存储 r3

r2在12:35:11北京时间按原时限结束，A/B均退出137、pair退出1；原主进程已退出，
没有checkpoint加载或游戏证据，失败产物保留。本次不重复停止已经退出的进程。
用户批准：仅复制必要资产，最多30分钟；校验通过后，以新RUN启动相同八卡A/B，
各4局、最多57分钟，无PPO或自动重试。不是新的模型/协议候选。

新增可选`FEEDBACK_STORAGE_ROOT`，默认仍为`/mnt/data0601`。它统一替换本record的
资产、FLA/MoBA、父run/checkpoint、小文件预检和输出路径；Hydra日志随输出目录迁移。
43调用示例：设置`FEEDBACK_STORAGE_ROOT=/root/new-scii-execution-feedback-local-20260911/shared`
及`FEEDBACK_REPLICATE=r3`，再调用原`run_feedback_pair_node43.sh`；源码SHA、授权标记
和未使用的`PAIR_DIR`仍必须显式提供。原runtime、模型及配置文件未改变。

复制从44读同一份原模型、SC2、FLA/MoBA与完整step24，包含模型/optimizer/extra/data
及父run manifest/effective config；目标仅为上述43独立目录，不修改共享源资产。
12:40:44开始计时，硬截止13:10:44。初始SSH中转仅约0.7MB/s，12:48在同一时限内
切换为受控tmux的节点直连rsync，复用已传片段，不重新计时。仅复制本机已信任的
43公开主机密钥到44隔离控制目录；使用44已有认证，未复制私钥、关闭主机校验或
修改全局SSH配置。直接传输限速180000KiB/s，启用checksum/append-verify，无重试循环。
中转退出2记录保留，最终以独立direct退出状态及校验为准，不把部分文件当完整资产。

本地入口回归8项：7通过、1项缺Torch跳过；反馈机制19项通过。
新增测试确认路径替换后无残留`/mnt/data0601`配置引用，规范化路径后配置相同。
macOS系统Python第一次缺numpy且测试遇到路径别名断言；改用已有依赖Python并修正
测试路径规范化后通过。这是本次测试准备问题，不算项目运行bug修复。
正式启动前还须在43固定依赖做完整CPU回归、验证资产、再次确认八卡空闲。
预定RUN为原前缀`20260911-execution-failure-feedback-v1-spb2-l7s24-l6-g2-`
加`A-eval4-s0-r3`或`B-eval4-s0-r3`；目前仍是准备，不是模型加载/评测成功。

43首次补测中，新增本地存储测试及CUDA隔离测试通过；两个旧dry-run测试仍把
FLA/MoBA默认路径指向故障共享盘，各30秒超时。随后仅为这两个CPU测试提供
临时本地存储根，避免诊断测试访问外部资产；没有改生产代码或放宽断言。

### r3 已实际开局（13:08 北京时间）

冻结SHA `d155c96eaccf9aa809544945b387f0e17e1ca5bd`，43独立`code-d155c96`部署。
复制13:00:35退出0；模型/SC2/source哈希、step24元数据和14个checkpoint文件哈希
13:03:04验收通过。43固定依赖CPU入口8/8通过（4.398秒），完整私有bundle已备份。
13:04:17启动`sc2_feedback_ab_r3`，硬截止14:01:17；A用GPU0–3，B用4–7。
两组均已实际恢复step24并各启动四局SC2，首次观测/Planner请求有逐游戏记录；
实际种子均为`[52620,52620,17089,17089]`。没有PPO更新或已完成的自然终局。
从44读取r2实际配置，与r3落盘配置规范化路径后无差异；r3两组仅反馈开关不同。
这验证了本次本地存储绕行与模型恢复，不代表反馈纠错效果通过或项目RL已修好。
精确路径、预算、校验与启动证据见[r3-launch.json](r3-launch.json)；
完整权重/优化器状态指纹见[r3-checkpoint.sha256](r3-checkpoint.sha256)。

### 首批效果证据（13:17北京时间，B未结束）

A已于13:12:57完成、退出0：4局自然终止，2胜2负；平均整局累计reward=-14.475，
8条可评估回答全部语义有效，无PPO。B此时仅完成3局（2胜1负），最后一局继续，
不得拿其部分胜率与A完整胜率比较。

B的env0/1均真实收到人口阻塞反馈，来源policy2、已处理前缀13条，失败目标为
`marine`，原文含`Supply blocked (Available: 0 < Required: 1.0)`及
`Construct a SupplyDepot immediately`。随后真实回答分别以
`- marine x 29`、`- marine x 1`开头，SupplyDepot仍在后面；两条回答均被应用，
又产生前缀0条的人口阻塞。下一轮回答仍以Marine开头。
这些证据来自`prompt_response.jsonl`中的完整反馈/回答，且
`semantic_action_evaluable=true`、`policy_action_applied=true`；不计已结束游戏的
占位回答。接口已接通，但这两局没有纠正关键生产顺序，不能称纠错能力验收通过。
未修改候选、未停止剩余游戏、未追加试验；A组GPU0–3完成后释放，不重复占用。

### 最终结果（13:18:54北京时间）

A/B与pair均退出0，8局均自然终止，无PPO；13:19:24确认43八卡全部释放。
两组逐种子的胜负和整局累计reward完全相同：52620两次均胜、20.5；17089两次
均负、-49.45。两组均2/4胜，平均reward=-14.475。

| 口径 | A无反馈 | B反馈 |
| --- | ---: | ---: |
| 真实模型决策 | 8 | 16 |
| 有效／可评估真实回答 | 8/8 | 11/16 |
| 真正含失败反馈的回答 | 0 | 4 |

B的24条原始回答日志包含8条结束游戏占位记录，上表及累计reward已排除它们。
5条不合法真实回答全部出自没有收到执行反馈的env2，不能说是反馈直接造成。
env0/1共4条真实反馈回答均保留Marine在先，未解决已明确告知的人口阻塞。
结论：本地存储启动与反馈传递验证完成；纠错能力未通过，未见本轮胜率/reward收益，
不据此晋级RL或称项目修好。仅两个独立种子，保留候选为默认关闭，不追加试验。
完整逐局结果、统计口径及artifact哈希见[r3-results.json](r3-results.json)。

## 固定失败输入：原始 SpB 对 step24（完成）

用户批准四卡串行、从启动起最多20分钟，原始SpB与Level7 step24各8条回答，
不启动游戏、不做PPO、不自动重试。只比较权重；r3 prompt、tokenizer、T0.4/
top_p0.9/top_k0、192token、FP32参数/温度归一化、BF16 autocast、cache、
分层FSDP及SWA dropout0保持不变。复用43已校验本地资产，不再复制或下载模型。

选择r3 B日志第9、13行，即env0在06:13和06:53的两次真实阻塞输入。env1首次
输入与env0逐字相同，不能算独立场景；本测试明确是同一seed52620内的两个时点。
每个输入使用采样seed `11000 + 4*输入索引 + rank`，两个权重前分别重置，共16答。
两时点都有人口余量0、12工人、660/1060矿，待办有SupplyDepot额度8且映射存在。
这证明候选/基本资源可用，不证明未知建造位置一定成功；本测试不作施工验收。

主观察为原回答经现有validator合法，且SupplyDepot排在Marine/SCV等本场景消耗
人口的动作前；另报首动作、完整回答、解析顺序、长度与EOS。合法名称/数量本身
不算纠错。无效回答的解析只作诊断，不声称是经过规则补齐的最终Executor队列。
保留原始输入及token哈希，检查无输入截断；不修改示例或强制模型先造SupplyDepot。
原始优于step24只提示这两个输入上的行为退化，两边都失败只提示当前模型/输入
尚未呈现纠错能力，均不直接证明训练实现bug或应修改prompt。

入口`run_fixed_feedback_node43.sh`调用`fixed_feedback_compare.py`，复用已有
`fixed_prompt_ab.load_config`、`TrajectoryCollector`、FSDP/HF生成和完整checkpoint
loader。先原始权重生成，再四rank加载step24模型/optimizer/extra；计数前后不变，
不会更新optimizer或加载/运行SC2状态。两权重共用同一个四卡worker，配置不变。
新RUN：`20260911-execution-failure-feedback-v1-spb2-original-vs-l7s24-fixed2x4-4gpu-r1`。
预定43 GPU0–3；代码由新SHA冻结在原反馈部署根的新目录，结果用43本地runs根。

CPU机制与评分回归20/20通过（0.439秒），shell语法与diff检查通过。新增评分最初
没有向原解析器传choice_map，CPU测试即发现；只修诊断脚本，不算项目bug成果。
真实原输入按日志SHA及两个prompt SHA锁定；上卡前补查固定依赖与token化。

13:50:25北京时间，43 GPU0–3空闲检查通过后，以`sc2_feedback_fixed_r1`启动，
硬截止14:10:25；冻结SHA `491746227e12646d3e6c1434e92b02c85a8cb39e`，独立
`/root/new-scii-execution-feedback-20260911/code-4917462`。完整私有bundle已在43
备份，SHA256 `2887605f38b1589159ef8f64c36fe95dcd08e19262b3f3a1b3515a8d09f25753`。
43既有依赖CPU回归20/20通过（0.968秒），完整输入1450/1452 tokens与实际
preprocess输出逐token一致，未截断。核心agent_system/verl/训练入口相对r3无差异。
13:51仅确认原始权重读取完成、FSDP初始化，尚无生成或step24加载完成证据。
这里是启动快照，不冒称模型纠错成功；最终结果以后续result与.done为准。

13:54:49完成全部16答，13:55:01退出0，共4分36秒；13:55:27核验八卡均0MiB。
四rank均实际恢复step24，Adam计数80且推理前后不变，没有PPO或新游戏。

| 同输入、同采样种子 | 原始SpB | Level7 step24 |
| --- | ---: | ---: |
| 现有validator合格 | 8/8 | 8/8 |
| SupplyDepot先于耗人口动作，且为首项 | 5/8 | 0/8 |
| 自然EOS，无192token截断 | 8/8 | 8/8 |

同一首次失败输入、seed11000，原始输出`SupplyDepot → Barracks → Marine`，
step24变为`Marine → SupplyDepot → Barracks`，数量都为1。5个原先正确顺序的
配对回答全部转为Marine在先；另外3对两组都未纠正。完整回答和token保留在
四个rank JSONL，逐对结果/哈希见[fixed-input-r1-results.json](fixed-input-r1-results.json)。

因此本次支持“这两个固定输入上的训练后纠错顺序退化”，不支持“模型根本没有
生成纠正计划的能力”。也不能由此认定优化器bug、所有场景退化、或当前Level6
step63同样退化；这里测试的是Level7 step24。原始组也不是8/8正确。没有额外
无反馈对照，不能断言原始的5次重排完全来自反馈而非观测本身。
这些是同一局两个相关时点的条件采样计数，不报作独立胜率或总体置信区间。
未改prompt或晋级新RL；下一步优先只读核对原训练中这种顺序选择收到的奖励及
更新信号，尚未定位具体实现偏差，不先增加新的修复机制。

[Step24训练信号只读审计](step24-training-signal.md)已核对global1–24：582条
真实回答不含人口余量0或失败反馈输入；12/24批奖励优势为0或微小浮点残差，
包括最后连续7批。另复核同胜利reward下的少调用偏向。未新增GPU实验或修改
生产模块，不把这些证据冒称为已定位某次参数更新的根因。

2026-09-12：[首攻回放审计](first-attack-replay-audit.md)后的下一项是
[同一失败输入的示例顺序对照](example-order.md)。仅交换示例中Marine/Depot两行，
复用A4权重和原生成路径，各8答、不启动游戏或PPO。用户批准43四卡20分钟；
18:25–18:30完成、退出0并释放GPU。合法Depot优先5/8→7/8，但完整且合法纠错
5/8→5/8；B有2条192token截断、另1条Barracks超配额。证据不足，暂不采用。
原始回答、配对随机数、四rank A4加载及Adam计数15不变均已核验，未启动游戏或PPO。

下一项[原始SpB对A4同输入](original-a4-fixed-input.md)复用上述A4原示例8答，
只补原始SpB同输入同seed8答。用户批准43 GPU0–3、最多15分钟；不采用B示例，
不重跑A4、SC2或PPO。18:44–18:47完成、退出0并释放GPU；原始完整合法纠错3/8、
A4为5/8，3对改善、1对退化。本输入不支持A4整体退化，也不足以证明一般RL收益。
原始仍有1条重复超额/截断；两者纠错都不稳定，下一项建议核对已有A4训练信号。

[A4训练信号审计](a4-training-signal.md)已完成：108条真实回答没有人口余量0或失败
反馈输入；同一完整输入下未采样出合法的相反Depot/生产顺序。step3/4的回报相等、
advantage/PG loss为0；step4 Marine8与42也同为20.5。尚未定位实现bug，不把零
PG信号写成参数未更新。下一项先核对既有B4反馈训练的覆盖及回报区分，不新增GPU。

[B4训练信号审计](b4-training-signal.md)完成：74条失败反馈中只有4条人口阻塞，
1条Depot-only优先、3条仍Marine在先；71条反馈anchor为单样本，step相对项没有排名。
正确Depot回答的episode项保守界0～0.005，不能把全批非零advantage当作纠错学会。
未修改运行模块或新增GPU；下一项仅拟定真实解除阻塞的过程奖励候选并离线防刷审查，
不立即改reward或重复既有gamma/顺序实验。

[解除人口阻塞奖励候选](supply-reward-candidate.md)已完成设计/离线审查，未接入RL。
历史事件复核了实际产出差异，但人工干预不能计为模型纠错；B4现有日志也不足以
追认完成事件。12项CPU契约测试通过不代表SC2归因通过。拟议每局最多一次奖励仍有
“先堵再解”的首次激励风险，暂不采用；下一步只补唯一命令来源证据，审批后才接入候选。

用户批准后，已在独立诊断分支实现[命令来源记录](command-provenance.md)，默认关闭。
95项CPU回归通过（新增16项），测试中的prompt、reward、BO和实际请求保持不变；
Depot完成可保留原命令来源，Marine出生仍明确未知。没有启动GPU/SC2/PPO或启用奖励，
下一项需要仅记录的实局验收；CPU通过不代表归因链或奖励候选已经可用。
用户现已批准43八卡、最多45分钟，复用Level6/seed52620顺序A/B各4局，两组都加记录；
不做PPO或新奖励。21:00–21:08完成、退出0；八局均胜、每局20.5，382条目标命令ACK
和人工/模型标签已核验。66次Depot开工位置相对命令位置均偏移(+0.5,+0.5)，严格坐标
关联0/66，尚未通过来源验收。先核验该记录器假设，不将其算作项目原有bug修复；
实局产物已保留、八卡释放，不自动重跑或接RL。

随后CPU坐标审计复现原0/66：66条实际命令只有12个目标位置，全部为半整数；
ceil与floor(x+0.5)都可得到66个唯一候选，但现有数据无法区分引擎规则。
28项测试通过不代表来源修好，未采用取整、未改reward或新增GPU实验。
下一项仅补开工时原下令SCV的订单目标证据，范围与缺口见同一记录。
该快照现已实现并通过34项CPU回归：按原下令tag保存订单坐标/tag/缺失状态，
不改变原归因判定或游戏命令。随后获批的r2于22:05–22:13在43八卡完成，退出0：
66次开工均观察到唯一SCV订单目标等于建筑中心；其中16次仍有多条该SCV历史命令。
记录验收通过，原归因仍为0/66，未启用奖励或PPO。八局均胜、每局20.5，但仅一个
游戏seed，前三答均回放/人工干预，不能算模型学习效果。详见同一记录及
[r2结果](commandtrace-r2-results.json)；下一项先用现有事件核对16次历史命令歧义。
随后[生命周期CPU复核](commandtrace-r2-lifecycle.json)查明16次均混入已完成旧命令，
按既有开工/完成事件排除后，66次在记录范围内可唯一解释；没有按FIFO或坐标取整。
实际运行归因仍0/66，未改运行模块。64个Depot完成都来自历史/人工回答，在线模型
的2个仅开工，不能作为模型纠错奖励正样本。下一项仅拟议修可选记录器的关联，
先做同输入CPU A/B，不修改reward或新增GPU实验。
9月13日用户确认后，仅记录器关联候选已实现：同66次真实输入的实际方法CPU A/B
开工关联0→66、完成0→64，非归因字段不变，反例保留未知。没有新实局/PPO，
不能把该结果写回r2或称为学习修复；实现与复现命令见[同一记录](command-provenance.md#记录器关联候选2026-09-13cpu通过)。
随后另批43八卡15分钟r3于9月13日11:20–11:28完成，退出0，无PPO；66次开工、
64次完成实时关联与独立审计一致，旧r2仍保留零关联。64次完成全部源自历史/人工
回答，在线2个只开工，不能用于证明纠错学习或启用奖励。见[实局结果](command-provenance.md#r3结果记录关联通过学习未验证)。
