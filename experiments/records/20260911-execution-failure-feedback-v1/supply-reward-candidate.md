# 真实解除人口阻塞：奖励候选的离线审查

最新状态（2026-09-13）：第三答起在线的实局评测已完成；4局在线补Depot真正建成，
随后恢复产兵，其余4局未产出新Marine。8局均胜，但回报几乎未区分纠错。
前序失败仍来自历史回放；奖励候选不自动采用，详见文末实局结果。

2026-09-12。**只完成设计和 CPU 审查，未接入训练、未新增 GPU 实验。**
开发基线 `89f9aee538687b13c658291e90b761b4cf2ae28b`；本次仅增加本 record 下的
分析、测试和文档。r6、B4、deployment、prompt、reward、GiGPO 和已有产物不变。

## 只提出这一个候选

先限定 **Marine 人口阻塞 → 实际补给建成 → 实际恢复产兵**，不同时加入缺气、
TechLab、进攻或存活奖励。选择人口问题，是因为已有 [Depot 顺序实局对照](supply-order.md)
证明了执行差异却同 reward；不是认为它占多数失败，也不是认为必须 Depot-first 才正确。
B4 只有4条此类反馈，63条是 Reactor 缺气，本候选不能覆盖所有纠错问题。

拟议事件链：

```text
真实模型生产动作因人口不足失败，未提交生产命令
→ 模型合法选择的 Depot 被提交、在失败后开工
→ 同一 Depot tag 确认完成，人口上限确实增加
→ 完成后提交的模型 Marine 生产命令，确实产出一个新 Marine
→ 本局首次满足，结算一个事件；再造建筑或再次阻塞不重复结算
```

按RUN/reset_index/env区分游戏，不能用seed作为唯一键。这里限制环境事件结算次数，
不改变现有PPO复制补齐的计权。时序与来源完整也只是进展证据，不是反事实因果证明。

Depot-only 回答允许参与：恢复生产可以来自后继模型回答，不要求同一答必须含 Marine，
更不按固定行顺序判优。反之，规则补出的 Depot/Marine、固定 BO、人工替换回答
不能作为这项“模型纠错事件”的证据；系统产出仍单独统计。这个保守范围可能漏掉
“模型补给、自动规则产兵”的有效系统改善，不将其误报为系统没有改善。

完成/产出必须有唯一的命令来源链。不能使用“关键词出现”“validator 接受”
“BO 删除”“命令函数返回”“supply_left 上升”代替：最后一项也可能只是死兵腾人口。
既有施工中 Depot、既有排队 Marine 不计成本次纠错；未完成、被取消或建前被摧毁不计。
无法确认来源时记 `insufficient_evidence`，不冒充“已经执行失败”或模型的负样本标签。

建议供后续确认的系数为 **α=0.5、本局最多一次**，`r_new=r_old+α·I(event)`。
选择0.5只是保守起点：不超过现有末次格式分，且为胜利基础分20的2.5%；
**并非论文规定、并非已证明合适、目前未启用**。不按建成数量、调用次数或解除次数加分。
独立评测同时报原任务 reward 和候选附加分，不把附加分上升称为游戏表现改善。

## 公开依据及适用边界

- [MiniHack，NeurIPS Datasets and Benchmarks 2021](https://datasets-benchmarks-proceedings.neurips.cc/paper_files/paper/2021/hash/fa7cdfad1a5aaf8370ebeda47a1ff1c3-Abstract-round1.html)
  提供可组合的环境任务；其[公开 RewardManager 源码](https://minihack.readthedocs.io/en/stable/_modules/minihack/reward_manager.html)
  用事件检查环境转移，`repeatable=False` 时跳过已完成事件，`collect_reward`取出后清零，
  `reset`重置事件。本候选只借用“真实事件、首次结算、按局重置”的机制，不移植整个框架。
  引用的是核验于2026-09-12的 stable 文档源码，不声称逐字对应2021初版。
- [Ng、Harada、Russell，ICML 1999](https://people.eecs.berkeley.edu/~russell/papers/icml99-shaping.pdf)
  讨论奖励塑形的策略不变性；标准势函数形式是 `γΦ(s')−Φ(s)`，有相应条件。
  **这里的一次性正事件奖励不因此拥有策略不变保证。** 我们不改 gamma、不宣称潜势塑形，
  也不声称该论文支持0.5这个数值或此处的 SC2 事件定义。

SC2 的动作归属、Depots/Marine 事件链和系数，是本项目针对已观察问题提出的适配假设，
没有公开证据证明这套具体组合必然改善当前 SpB/GiGPO。它是待验证的目标设计候选，
不是已定位实现 bug 的修复。

## 用历史产物审查的结果

只读核对评测 runtime `4d39d06658364689518d5d6f44a2af4bb5f73b49`：
`…-supply-order-ab8-8gpu-s52620-r1`，8局、实际seed52620、退出0、0次PPO。
小型事件/干预/manifest/result的哈希和逐事件出处见 [审查JSON](supply-reward-candidate-audit.json)。
B4来源、四条回答的行号沿用 [B4已核验审计](b4-training-signal.md)，未加载checkpoint。

| 证据 | 离线判断 |
| --- | --- |
| 顺序B四局，06:13后各建成8Depot、产出2Marine；均20.5 | 存在真实执行改善；前三答均为人工转发且没有第四答，不能据此给模型纠错分 |
| 顺序A四局，06:13后建成0/0/0/1Depot、均没有新增Marine；均20.5 | 不能仅因一个Depot完成就宣布恢复产兵 |
| B4四条人口失败反馈，有一答 `Depot1 → Barracks1` | 模型确有改选；现有prompt日志不足以确认唯一来源的建成/恢复产兵，四条均不追认新奖励 |
| 事件中的 `policy_action_id` | 取的是回调发生时的 `last_consumed_policy_action_id`，不是建造命令创建者；不能靠这个字段反推归属 |

最后一项依据为 `supply_order_eval.py:46–64`；其观察器仅用于该纯评测。
本次按建筑tag联结开工/完工事件，**没有发现此样本的ID跨调用变化**，不捏造反例。
但字段语义本身不保证来源；`commander.py:338`输出的队列只有action/intent，
`llm_player.py:449`的失败反馈也明确写 `parsed_answer_plus_existing_rules`。
这是新候选所缺的归因证据，不是说旧诊断日志或这8局PPO已经误奖励——该评测没有PPO。

12个CPU测试验证拟议判定契约：首次事件、按局限次、人工/规则排除、未知归因、未完工、
旧施工、死兵腾人口、未恢复生产，以及理想化相对优势计算。**这些是合成契约测试，
不证明真实 SC2 已能生成所需来源链，不称为训练奖励修复验收通过。**

## 两个明确的限制，不能用“测试通过”掩盖

**限次只能防无限重复，不能消除第一次故意制造阻塞的收益。** 顺畅提前补给的策略
可能拿不到这0.5，先堵再解反而拿到。这是奖励目标改变的真实风险；本轮不再加第二种
惩罚、潜势函数或新启发式来覆盖它。用户接受这个有界辅助目标之前，不建议接入RL。

**增加环境分数不自动产生有效相对优势。** 冻结 GiGPO 的71/74反馈是单样本anchor，
加奖励后该step中心化项仍为0。若只一局解除而其余七局没有，episode项可区分；
若八局都得到同样附加分，这一部分仍中心化为0。
理想化例子：八局原reward相同、全合法、无padding、每局五答，只有一局加0.5，
episode项增量为该局每答+0.0875、其余每答−0.0125。此例不是B4原张量或实际更新预测，
也不代表改善只强化了纠错那一答；episode项会分给全轨迹。

## 最小接入路径与晋级条件（待批准，未实施）

先只补**模型/规则/人工来源→命令→建筑tag/生产结果**的记录，不改执行时机或队列。
复用原始回答、现有BO索引、施工回调；只匹配能唯一确认的对象，不按“最近一次回答”
猜归属，不建设通用事件框架。候选需要的观测从失败时至实际完成，不能只看相邻query。
CPU首先验证模型Depot-only后续恢复、重复回调、完成跨Planner调用、规则补项及未知归属。
若真实命令接口无法唯一匹配，明确停在证据不足，不静默使用时间最近的指令。

实现涉及 `planner.py`返回的动作来源、`commander.py`队列元数据、`llm_player.py`
命令/回调记录；只增加元数据。确认记录可信后，奖励候选才在现有反馈发送/自然终局
路径按事件发生时间一次性结算，不倒填旧transition、不生成虚假决策、不改GiGPO。
事件产生者ID与奖励接收transition分别记录；延迟结算仍由现有算法分配信用，不能
把两个ID不同直接叫归因bug，也不能承诺只更新原纠错回答。

记录路径验收后再另行批准小型RL A/B：两组从同一原始SpB开始，反馈均开，冻结
B4的模型/prompt/g2/Level6/采样192tokens/history/优化器/gamma；唯一训练差别是
该事件bonus开关，记录器两组相同。预登记同一四批seed2732/43567/42613/52416，
各4个global step、每步8局、step4完整checkpoint。建议43 GPU0–3串行，合计最多
3小时、约120GiB、不自动重试；这是**尚未授权的预算建议**，启动前重新检查资源。

先看归因、限次、真实恢复率/延迟、奖励/advantage是否区分纠错；再用独立配对评测
判断权重是否更会纠错，评测预算另批。不以4步胜率波动宣布学习改善，不直接续96步。
如果没有自然出现足够的可归因事件，结论是覆盖不足，不调seed或人工造正样本冒充RL。
实际接入属于奖励语义变化，届时使用独立候选branch/worktree和新实验ID；两组从各自
已提交的冻结SHA启动，本设计记录不代替未来实验的授权、manifest或effective config。

复现离线审查：

```bash
python -m unittest discover -s experiments/records/20260911-execution-failure-feedback-v1 \
  -p 'test_supply_reward_candidate.py' -v
python experiments/records/20260911-execution-failure-feedback-v1/audit_supply_reward_candidate.py \
  --snapshot /private/tmp/sc2-supply-reward-review.fw0MkW --output <未使用JSON路径>
```

本地用 `/private/tmp/sc2-air-results.HBI61m/decoder/bin/python` 执行，12项通过、审查退出0，
重复生成JSON逐字节一致；另跑原A4/B4的8项回归通过。上述四个运行源码与B4 SHA无diff。
旧小型事件原件备份名 `supply-reward-evidence-20260912.tgz`，SHA256：
`66a8fd388a764e5b8b8159d656f363b2179127de8ac38d1737417dcc1a298db5`；
计划与本次增量Git bundle共同备份至43的 `/root/new-scii-execution-feedback-20260911/`。
**当前结论：候选有公开机制依据，但归因接入未验收、首次造堵激励未消除，尚不采用。**

## 9月13日：记录器验收后，复核真实在线纠错

本次只读分析；未新增GPU实验、未修改运行代码或奖励。r3冻结SHA为
`95f6e2185acb37d792970010429e8828e0c40164`，原始SpB、Level6/g2、实际游戏
seed52620。先核对[已登记r3证据](commandtrace-r3-results.json)及逐文件哈希，
再核对实际消费回答、执行队列、命令ACK、建筑tag和自然终局；不拿当前源码替代历史。

### 模型选择影响了执行，但本批奖励没有区分

四条实际在线回答来自第四轮同一06:53人口阻塞输入：47/47人口、1060矿、0气、
35 Marine、12 SCV。prompt SHA256为
`45f6664d8ca8e4079b360c5ac69fcba97ddf4745b19d7a30e1067ca00e3f849f`。
模型有效回答按原顺序进入Executor，没有把人工Depot计入在线结果。

| 实际在线回答 | 执行证据 | 自然终局 | 整局reward |
| --- | --- | --- | --- |
| env0、6：Depot1 → Barracks1 → Marine1；env6另有Reactor1 | 06:53.04下令且ACK成功，07:02.68同tag Depot开工 | 07:20胜利，Depot进度82.5%，尚未完工，未新增Marine | 各20.5 |
| env2、4：Marine2 → Depot2 → Barracks2；env4另有Reactor2、SCV2 | 此后没有记录到新Depot/Marine命令或开工 | 07:17胜利，执行索引仍为0，未新增Marine | 各20.5 |

第一组不是“命令没执行”：已提交并开工，但观察在自然胜利时结束。不能将未完工
当作执行失败，也不能断言继续游戏一定会完工。第二组只能确认上述两类命令未发出，
不能从记录器的有限覆盖推广为所有命令都没有发生。两组回答数量等也不同，不冒充
只改顺序的严格因果A/B；只有一个游戏场景，不能外推胜率或学习能力。

原始回答在`prompt_response.jsonl`第25/27/29/31行（env0/2/4/6）。例如env0：

```text
<answer>
- supplydepot x 1
- barracks x 1
- marine x 1
</answer>
```

对应env0、6 trace的181号回答、183号队列、184号命令、186号ACK、187号开工
均有原始证据；同一建筑tag为4369940484（tag仅在各自游戏内联结）。最终边界
保存`progress=0.824999988079071`。详细原文、事件与输入哈希见[只读审计JSON](online-correction-audit.json)。

冻结`agents/planner.py:1077–1099`的胜利终局是outcome奖励加末次格式分；
本run的逐调用state shaping关闭（同文件1139–1145）。实际四局累计均20.5，
没有Depot完成/恢复产兵的专门奖励。其他阶段/攻击分不能被概括为所有胜局恒为20.5；
本轮也没有PPO，不能把同reward直接写成实测advantage或证明训练一定学不到。

### 哪些证据仍不能给候选奖励验收

- r3第三轮06:13的8条原始回答均过冻结validator，其中2条Depot优先、6条Marine优先。
  它们被历史/人工回答替换，**没有执行**，不能追认纠错成功。位置为
  `interventions.jsonl:17–24`；成对采样造成重复，2/8不是八个独立场景的成功率。
- 上述第四轮失败的来源都是第三轮历史回放动作，不是模型自身决策；即使以后补齐
  完工/产兵证据，也不满足本候选最初限定的`failure_origin=model`。不能静默放宽。
- B4旧训练四条人口反馈逐条复核原始日志及七项文件哈希：第78/86/94行仍Marine优先，
  第127行Depot优先。与[B4既有审计](b4-training-signal.md)一致，缺少当时命令归属的
  完工证据；不能用r3新记录器倒填历史证据，也不能凭下一query仍没人口判定建造失败。
- r3的Depot记录关联已经验收，不再为覆盖率扩展；Marine出生到唯一生产命令的归属
  仍未知。候选“先造堵再解堵”的首次激励风险也仍在。

**结论：未确认新的执行实现bug；在线选择→队列→下令→开工已出现真实差异，
但自然终局截断了完工观察，现有奖励未区分这四个样本。纠错bonus仍不采用。**
上文的奖励RL预算只是旧设计建议，不是当前下一项，也没有获得新增启动授权。

唯一建议的下一项是独立纯评测：保留前两答历史前缀，第三答起不再替换在线回答，
在06:13同一已记录失败输入检查模型选出的计划能否走到建成/恢复产兵；现有记录器、
模型、prompt、规则和reward全部冻结。相比第四答多约40游戏秒观察机会，不保证完成。
这是评测hook的干预边界变化，不是更改正式Planner接管策略，更不能算自然RL的
`failure_origin=model`正样本。拟议8局、八卡最多15分钟、0PPO、无新checkpoint、
不自动重试；必须另行确认预算和资源，**本轮未实施或启动**。

### 复核入口与产物

新增分析器仅读取小型JSONL，校验输入及冻结validator哈希；不会加载模型/训练环境。
未完成事件归类是审计标签，不被运行时导入或作为奖励。

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:tests python -m unittest \
  test_sc2_command_trace test_depot_recorder_replay test_supply_order_eval \
  test_online_correction_audit -q
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:tests python \
  experiments/records/20260911-execution-failure-feedback-v1/audit_online_correction.py \
  --snapshot /private/tmp/sc2-commandtrace-r3.kKRGqU/verified \
  --association-audit /private/tmp/sc2-commandtrace-r3-association-audit-20260913.json \
  --b4-snapshot /private/tmp/sc2-feedback-r2-final.hrjkuX/B \
  --output <未使用JSON路径>
```

实际使用`/private/tmp/sc2-depot-cpu-20260913.Gq5Waa/bin/python`，39项CPU回归通过，
分析退出0。结果`/private/tmp/sc2-online-correction-review-20260913.json`，SHA256：
`d962ad7a95813c811c23c43aa22d325b8a15c20a0e1ced4f24224c8c245edff5`。
输入原件位置和哈希已在结果中保存；结果及Git增量作为小型证据另行备份，不复制权重。

## 第三答起在线：获批实局验证（9月13日）

用户已批准上述八卡8局、15分钟纯评测。状态：completed，12:41:24退出0，
8局自然胜利、0PPO。沿用现有诊断分支，不改正式训练接管或新建奖励实验。
RUN_ID：`20260911-execution-failure-feedback-v1-spb2-original-l6-g2-online-from3-8gpu-s52620-r1`。

唯一实验变化是评测干预边界：原r3回放前三答（第三答A/B不同），本轮只回放前两答，
第三答校验历史完整prompt SHA后原样转发生成结果；之后继续在线。第三答若无效，
交给原validator/过滤规则处理，不因诊断入口强制报错、补Depot或替换为合法答案。
两条前缀仍标历史，第三条起标在线；这不是新增两个实验组，不沿用A/B胜率称谓。
八个实例共用seed52620；成对采样seed仍为`0,0,1,1,2,2,3,3`，不能算八个独立场景。

`online_correction_eval.install`调用原`supply_order_eval.install(prefix_answers=2)`；
默认3保留旧诊断。只联动回答转发边界、来源标签与评测说明，不改模型、观察器、
reward、执行队列或任何训练算法。`run_online_correction_node43.sh`复用旧启动器，
固定原始SpB、Level6/Harder、g2开局Planner、Terran/Air、Flat128、feedback开、
约15条/`x N`和配额说明、192token、history0/off、state shaping off、cache开、
FP32参数/温度归一化、BF16 autocast、HF/eager/分层FSDP、SWA dropout0。
全部模型、游戏、依赖哈希沿用r3，载入原始模型而非任何RL checkpoint。

预定43 GPU0–7，一模型实例八局。12:25北京时间核验八卡0MiB、无compute/tmux任务，
主机可用448GiB。使用43现有本地模型/SC2/Python，不访问共享盘大资产、不复制权重。
新部署为`/root/new-scii-execution-feedback-20260911/code-<新SHA前缀>`，控制目录
`online-from3-r1`，Ray临时目录`/dev/shm/oc_0913_r1`，run仍在同实验本地runs根。
源码先提交、私有bundle备份后再独立冻结；旧95f6e21 deployment、r3和失败记录不动。

预算从后台实际启动起最多900秒（885秒加15秒退出余量），不做PPO、不新存checkpoint、
不重试、不续训。停止条件为超时、输入哈希偏离、运行错误或资源检查不满足。
不因合法但Marine优先或输了游戏而改prompt/采样；自然结束未完工保留为观察不足。

验收逐局记录第三轮输入/原始回答/转发回答、实际消费与来源、队列、下令/ACK、
Depot开工/完工、Marine出生及各自时间，并记录胜负与首次done前累计reward。
Marine唯一生产命令归属仍未知，只报时间上的恢复产出，不宣称完整模型产兵归因。
记录是否到达失败输入、额外重规划次数、回答有效性。旧r3仅作工程参照，不以少量
同seed胜负宣称学习改善或奖励通过；人工制造的历史失败仍不满足原候选的模型失败来源。

本地CPU回归41项通过，含无效在线回答原样保留、第三轮输入偏离拒绝、来源标签及
五套入口effective config对照；除hook、运行/临时目录身份外配置一致。启动命令：

```bash
SOURCE_SHA=<冻结SHA> CONTROL_DIR=<新control目录> \
SC2_ONLINE_CORRECTION_AUTHORIZED=1 \
bash experiments/records/20260911-execution-failure-feedback-v1/run_online_correction_node43.sh
```

训练源码、规则与记录器相对95f6e21无diff；CPU通过不代表本轮已经进入游戏。

### 实际启动

12:32:17北京时间在43 GPU0–7启动受控会话`sc2_online_from3_r1`，硬截止12:47:17。
冻结SHA `e23eea3219b9bbf00345e7d78f9cd9fd85fa4452`；新部署`code-e23eea3`，
旧部署只读。启动前41项服务器CPU通过（3.852秒）、24个登记源码哈希通过。
私有增量bundle本地及43均核验SHA256：
`f4735f2b5704dbe57a59d29e08d47f227e41efcb50d096b06994e39565ad4e7c`。

12:34:09已进入SC2并请求首轮Planner；11个模型分片加载完成，八卡实际参与生成。
`manifest.json`确认clean/evaluation/world_size8；`supply_evaluation.json`确认
原始权重step0、0 optimizer update、resume disabled、前两答回放且第三答起在线。
实际种子文件为reset_index0、八次52620，逐项一致。effective config经仅替换新旧
代码/RUN/临时目录后，唯一差异为`worker_process_setup_hook`，没有重做初评或PPO。

12:38:54八局均已通过第三轮输入哈希校验；8份生成文本等于转发文本，其中4条
Depot在先、4条Marine在先。这是运行快照，不作为建成、纠错成功或游戏结果。

### 完整结果：在线纠错能走到建成与恢复生产

12:41:24控制器、`.done`和`result.json`均退出0，用时547秒；八局均自然胜利，
没有新checkpoint或PPO。12:43确认八卡均0MiB、会话退出。收尾Ray仍有SIGTERM提示，
但八局自然结果及完整保存均成立，不把正常结束后的提示当作本次模型/训练失败。

| 第三答真实选择 | 在线Depot完成 | 之后Marine出生 | 整局累计reward |
| --- | --- | --- | --- |
| env4/5：Depot1 → Barracks1 → Marine2 | 各1 | 各8 | 各20.5 |
| env6/7：Depot1 → Barracks1 → Marine1 | 各1 | 各2 | 各20.5 |
| env0/2/3：Marine在先 | 0 | 0 | 各20.5 |
| env1：Marine在先，第四答无效 | 0 | 0 | 20.0 |

四个Depot均来自第三答的`validated_answer`，不是人工或规则补项：06:13.57提交，
06:16.07开工，06:37.50同tag完成，人口上限47→55。独立订单/完成历史审计与实时
归属一致。env4/5完成同帧开始提交模型Marine命令，随后连同第四答各共8条；env6/7
第三答各提交1条模型Marine命令。第一批Marine于06:55出生，env4/5的第二批07:15出生。
这比r3“只建到82.5%就自然结束”多了真实完工和后续生产证据。

不能把所有出生兵都计为模型独立选择：env6/7第四答无效，旧规则各补了1条Marine
命令，来源为`postprocess_or_salvage`；因此每局两兵并非“两条模型命令”。旧规则
还各补建成1个额外Depot，不计入表中的在线Depot完成。逐兵出生
与唯一生产命令的匹配仍未知，本轮仅验收建成、提交与恢复产出的时间链，没有扩展记录器。
env1第四答也产生过规则Depot命令，07:02.68开工，终局进度75.8%；未完工，
且不是合法在线选择，不能计为表中的在线Depot成功。

34次真实消费为16次历史＋18次在线；在线有效15/18，其中第三答8/8有效。
40条回答日志含6条已结束游戏的占位，不能重复累计其terminal reward。
三个无效回答都是第四答：env1重复列表、超出Barracks余量；env6/7重复Marine且结尾
不完整，触发原格式校验。例如env6：

```text
<answer>
- supplydepot x 1
- mule x 1
...
- marine x 1
- marine x 1
...（后续仍重复）
-
```

以上用省略号展示重复片段，完整原文在第31/32行；env1为第26行。
未由此改prompt或token预算。终局格式分仍沿用旧逻辑：env1末答无效所以少0.5，
env6/7第五答恢复合法所以仍20.5。平均累计reward为20.4375，日志20.438是舍入。
这一次观测到的0.5差来自末答格式，不是补给完成/纠错奖励。

**验收结论：当前场景下，原始SpB有可执行的在线纠错能力；模型选择没有被系统
完全抹去。先补Depot的四局均解除人口阻塞并恢复生产，但回报未相应区分。**
不证明失败反馈本身的因果收益（没有无反馈对照），也不证明RL学习改善或普遍4/8
成功率。游戏seed只有一个、采样成对，且新旧raw回答并非逐字相同，不能把与r3的
差异解释成严格逐token复现或单seed胜率提升。八局前两答均为历史回放，原奖励
候选的“失败来自模型”条件仍不满足；先堵再解激励风险未解决，因此不自动启用bonus。

本轮只保留评测诊断，不修改默认训练接管；未新增执行bug修复、未续训。
下一步应优先解决这类真实执行进展如何形成合理训练信号，不继续改已验证的
Depot记录器，也不把“有执行空间”当作“模型已经学好了”。新机制另行确认。

源配置及44个小型JSON/JSONL文件与服务器原件SHA256逐个一致。
[逐局结果与哈希](online-from3-r1-results.json)；完整审计
`/private/tmp/sc2-online-from3-r1-audit-20260913.json`，SHA256
`4139831f689a91c3fb065d8fbed4b146364ad71ff252ffe0de795567f81516d4`。
证据包`sc2-online-from3-r1-evidence-20260913.tgz`，SHA256
`543f42098206889006ae7ba905b8fc1427906f8bcef1d2db22afdf1f2f5bc7fe`，原run/replay保留。
复现仅需小型产物，不能把诊断脚本的语法/字段核对算作项目旧bug修复：

```bash
python experiments/records/20260911-execution-failure-feedback-v1/audit_online_from3.py \
  /private/tmp/sc2-online-from3-r1.428Lqs/completed --output <未使用JSON路径>
```

实际CPU审计退出0、重复结果逐字节一致；41项相关回归再次通过。没有新的GPU测试。

### 9月13日：失败场景筛选，不继续在必胜场景推导奖励修复

本轮只读筛查[自然配对评测](learning-eval-analysis.md)的原始/A4/B4共24局，
不是重跑在线第三答实验。冻结runtime为`937ae3cfcbf7340208a8f86c0abf680d733c0484`；
66个已登记JSON/JSONL哈希逐项一致，三组`.done`、加载权重step、0 PPO及实际种子
核验通过。按首次真实终局截断，每局消费回答数、reward与原审计一致。
[逐反馈原文、来源行及第二答对照](failure-scene-screen.json)由标准库脚本生成。

| 筛选对象 | 已有证据 | 本轮决策 |
| --- | --- | --- |
| 24局中的14败 | 记录到34次缺气、12次其他失败反馈；0次人口阻塞反馈 | 没有找到符合当前Depot候选的自然失败现场；次数不是独立样本数 |
| 唯一人口阻塞反馈 | A4/seed1098341402，07:02、第24行；仍答Marine19→Depot8，最终胜 | 不能用它证明不纠错会输 |
| 05:49人口阻塞败局 | 来自A4、seed42613、首攻阈值40的另一次实验 | 不能悄悄移到当前首攻20、原始SpB并称为只改纠错 |
| 已有Refinery提前 | Level7-step24权重在Level6/seed17089，A/B各0/2胜 | 局部资源链改善未转成胜利，不重做并冒充新修复 |

“0次人口阻塞反馈”只描述已记录的、被消费的query；超时前阻塞可能未生成反馈，
不等于这14局从未卡人口。也没有证明无反馈、无失败或任意reward都正确。
Refinery完成结果见[旧审计的资源顺序节](../20260910-spb-fsdp-layer-wrap-v1/learning-bottleneck-63step-audit.md#已完成消融的判读)，
旧`refinery-order.md`的“运行中”标题不作为当前状态。05:49证据见
[首攻回放审计](first-attack-replay-audit.md)；两者都不是本轮新实验。

更有信息量的已知失败现场是seed1098341402的**04:21第二段计划**，不是06:45
首次缺气反馈：当时首两波已经全灭。既有[Marine数量单变量对照](second-plan.md#marine单变量结果与改进决策2259以后核验)
用同一B4，只把第二答Marine8改为42，Depot仍2，其余不变，0/4胜→4/4胜；
说明这一处早期选择确有改变结果的空间，不代表42是通用最优或模型已经学会。
首次交战差异早于额外Marine出生，不能把全部收益归结为新增单位数量。

本轮进一步核对：A4/B4八个第二query的全哈希逐对一致，16条回答全部合法。
A4的Marine数量为`42,8,42,42,42,42,42,42`，B4为`8,4,4,4,4,4,4,8`。
A4胜而B4败的seed为4018390961、773351930、1098341402；但A4选42仍在
42613、17089、3236598124失败。不同checkpoint、每个输入仅一条样本，且其余动作
也不同，不能据此把整个差距归因于Marine数量或新增反馈机制，更不能手写42规则。
原始模型在其中六局的第二query早至01:47，与A4/B4不是同一状态，不混做同输入三模型比较。

**决策：Depot过程奖励仍不实施，当前自然失败样本准入未通过。** 下一项先复用
A/B四步训练产物，核对早期第二阶段是否采到具有结果差异的计划、这些回答实际
得到什么回报/相对优势；只区分已有标量日志与缺失的逐样本PPO证据。不再测已接通
的Depot记录器，不同时改prompt、首攻或reward。已有A8“早期趋同、晚期主要区分
合法性”的[分析](a8-training-signal.md)直接复用，不当作新的A/B四步因果证据。
若缺必要样本或更新张量，明确缺失再申请最小验证，不重复原来的整局消融。

生产源码从937ae3c到当前e23eea3的差异已按Git blob核对：是命令来源/建成记录及
可选`trace_origin`透传；训练、策略、reward没有在本轮改变。诊断前缀、卡数及运行
版本不同的结果仍分开报告，不把旧自然评测声称为当前部署逐字复现。

复现：`python -B experiments/records/20260911-execution-failure-feedback-v1/audit_failure_scenes.py
--output <新JSON路径>`；实际运行退出0，另两项CPU边界回归通过，未占GPU或启动游戏。
这只是诊断结论和场景选择，不计作新增项目bug修复。
