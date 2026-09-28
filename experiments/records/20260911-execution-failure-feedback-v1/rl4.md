# Level6失败反馈：原始SpB短RL A/B

状态：r1 failed，2026-09-11 16:43北京时间两组均因Ray主机内存保护终止、退出1。
两组各有3个完整global step指标，无完整step4 checkpoint，四步验收未通过；
用户另批r2四卡串行重启，17:56启动、17:59确认A首批实局和Planner，B等待A完整结束。
沿用本record和分支；不改已有候选实现或旧deployment。

## 问题与单一变化

[Level6审计](../20260910-spb-fsdp-layer-wrap-v1/l6-training-signal-followup.md)
确认原63步中41批奖励优势接近零，1767次真实模型输入仅一次明确人口余量0，
且没有执行失败反馈。纯评测验证了反馈接口，但尚未证明纠错能被RL学到。

两组均从原始`Panyuqi/SpikingBrain-2.0-instruct`开始，不加载Level7 step24、
Level6 checkpoint或项目SFT。A `SC2_EXECUTION_FAILURE_FEEDBACK=0`，
B `=1`；唯一方法差异为本record已定义的反馈/实际额度释放/失败后重规划机制。
这是一个联动机制，不冒称只增加一句prompt。B附加真实错误文本，基础prompt不变。
用户此前只批准的纯评测预算不复用，本次为另行明确批准的短RL。

## 冻结配置

SpB原始manifest SHA256：
`40d6f1417568d5fda1107cdafd66c7441a679b03fb148851c8f226ecc9cf884c`。
源checkpoint内容标识：
`de52e8ef993afc763f9d4fb498e8d8302ef0bc13bd7dfd53c6ae5f77f43698b5`。
仅使用既有SpB训练态kernel补丁和`swa_dropout=0`，不另改模型算子。

Level6=Harder、Flat128、Terran/Air、canonical g2资产，Planner开局接管；
当前marine_pressure阶段来源、名称/前置条件过滤、合法BO保留、规则补齐、
对空备选、旧微操、自动战斗均不改。约15条/数量展开及累计配额说明启用，
旧6/6环境变量存在但原段协议不采用其截断，不重新解释为15个展开动作。
history0/off，step shaping off，原reward/终止归因保持。

FP32参数和温度归一化、BF16 autocast、HF/eager/cache、原分层FSDP；
T0.4/top_p0.9/top_k0、prompt4096/response192。GiGPO mean_norm、gamma0.95、
step权重1，原无效动作惩罚、entropy/ref-KL、lr1e-6和loss裁剪均冻结。
每组world4、全局mini8、每卡micro1/累积2；两组相同，但不称与原八卡父run
逐位一致。采样仍为训练模式，不加载任何评测setup hook或手写回答干预。

env seed0启动既有训练RNG，四批预期实际游戏seed为2732、43567、42613、52416，
各重复8次。启动后逐条核对training_seeds，不称32个独立场景，不强制后续
采样轨迹逐字相同。观察中的失败若不出现，不额外注入或修改种子来制造结果。

## 资源、入口与产物

用户批准43八卡并行：A GPU0–3、B GPU4–7，各4个global step，每步8局；
两组共64个训练游戏。`total_training_steps=96`仅保持旧学习率调度预算，
`total_epochs=4`限制实际步数，没有第5步/96步续训命令。
各只在step4保存完整model/optimizer/extra/data，约新增120GiB；保留旧产物。
最多3小时含启动/保存/退出，不自动重试。训练后独立评测另报预算。

使用已校验43本地资产和固定Torch2.7.1+cu128/Transformers4.55.2环境，
不依赖故障共享盘、不下载或再次复制权重。现有python目录是依赖overlay，
解释器为系统Python3.12。启动要求本地至少180GiB可用并通过write/fsync探针。

运行前缀：`20260911-execution-failure-feedback-v1-spb2-original-l6-g2-`，
后缀分别`A-rl4-4gpu-s0-r1`、`B-rl4-4gpu-s0-r1`。
运行根：`/root/new-scii-execution-feedback-local-20260911/shared/yangjinyue/new_SCII/experiments/20260911-execution-failure-feedback-v1/runs`。
两个Ray/tmp/cache目录分别为`/dev/shm/sc2_feedback_rl4_A_0911_r1`、B同式；
目录隔离不关闭生成cache。原始权重只读，各run的模型权重仍用既有只读软链接。

从新clean SHA和唯一控制目录启动：

```bash
SOURCE_SHA=<冻结SHA> SC2_FEEDBACK_RL_AUTHORIZED=1 PAIR_DIR=<新控制目录> \
timeout --signal=INT --kill-after=15s 10785 \
bash experiments/records/20260911-execution-failure-feedback-v1/run_feedback_rl_pair_node43.sh
```

部署时使用独立`/root/new-scii-execution-feedback-20260911/code-<SHA>`，
受控tmux保存起止时间和两组退出码。入口拒绝占卡、dirty tree、已用run；
不终止他人进程。模型初始化及真实SC2开局确认前只报准备/启动中。

## 验收和已做验证

看完整query/原始回答/失败原因/恢复额度和真实决策ID，区分B真正收到反馈的
回答与无反馈回答；关注同失败是否重复、SupplyDepot是否前置以及实际执行。
实际命令和最终建成不混称。失败额度不得重复恢复，规则补项不得冒充模型选择。
每步核对自然终止、advantage分布、梯度、KL/clip、整局reward/胜负和完整保存。
空策略批次、归因错误或非有限数按原异常边界处理，不制造伪决策或无限重采样。
0胜不独立判bug，有限梯度/高有效率也不等于学会。

本轮只能验证反馈进入真实RL、连续更新和训练行为；若B没有实际失败反馈，
或仍重复错误/优势为零，结论为未验证纠错学习，不晋级长训。
不能由前后几个训练batch、调用次数增加或A/B少量胜负宣布学习提升；保留step4，
后续需同条件原始/训练后配对评测，预算另批。

本地CPU实际运行`test_feedback_rl_launch test_execution_failure_feedback
test_empty_bo_frame test_g2_planner_handoff test_original_bo_segment
test_valid_bo_passthrough test_system_release`：89项全部通过，12.661秒。
完整effective config比较：相对已验收四卡入口只变步数/保存频率与节点路径；
A/B仅反馈开关不同。确认无旧checkpoint恢复、无评测hook、非val_only。
两个入口shell语法与git diff检查通过。真实43依赖回归和GPU启动另记录。

43首次CPU组合（部署581ba1e）在45秒上限退出124；其中旧run_train1对照仍使用
共享依赖路径。拆分后的广泛组合也超时，未取得逐项输出，不能宣称所有服务器
回归通过或已定位每个超时位置。没有GPU或训练run，保留两次CPU失败记录。
随后仅执行新增A/B入口和反馈机制20项，43全部通过，1.141秒、退出0。
本地拆分后的两项配置测试也通过，1.251秒。没有修改launchers/生产模块或重算
GPU预算；本轮不扩展对无关旧测试的诊断，不把诊断准备问题算项目bug修复成果。

## 实际启动

运行SHA `6effc640767891da50c4c70e111c9d510f47b587`，43部署
`/root/new-scii-execution-feedback-20260911/code-6effc64`保持clean。
私有完整Git bundle的本地/43 SHA256一致：
`5ff4bafedcdc7f4d8acff8f05f1a039e6c4f1a954fcef04122ecbf2a8ed8fe2e`。
本轮相对341defb未改agent_system、verl、gigpo、scripts或configs，仅增加薄入口、
测试和记录；旧候选行为没有顺手调整。23项系统源码哈希全部通过。

15:35:08北京时间启动`sc2_feedback_rl4_r1`，控制目录
`/root/new-scii-execution-feedback-20260911/rl4-pair-r1`；硬截止18:35:08。
启动前八卡无compute任务，本地空闲291,938,430,976字节，write/fsync通过。
两组落盘manifest为formal/clean、world4、training/target4、resume disabled；
规范化目录后的effective config仅反馈开关0/1不同，不含共享盘路径。
原模型/SC2资产校验与模型kernel准备完成，未读取旧RL checkpoint。

B于15:36:52、A于15:37:00创建Flat128/Harder/Terran/Air游戏，均已打印开局
`Generate new plan... [0]`；实际首批各8个2732，与预期一致。
15:38:20八卡约11GiB/卡并执行生成；此时没有完整模型回答账本或完成的更新指标，
不冒称已完成step1。`0/96`进度条来自保留的调度预算，不是启动了96步。
完整manifest及实际配置差异见[启动记录](rl4-launch.json)。

## r1结果与阻塞（17:00后只读核验）

控制目录A/B.done及pair.done均退出1，pair于16:43:50北京时间结束。
A result结束16:43:35、B16:43:47；两组均completed_steps=3、metrics_finite=true、
checkpoint_complete=false。所有失败记录保留，没有自动重试或更改配置。

| global step／实际seed | A胜负／mean reward | B胜负／mean reward | A/B有效动作率（训练指标） |
| --- | --- | --- | --- |
| 1／2732 | 7/8；11.571 | 7/8；11.625 | 66.1%／73.5% |
| 2／43567 | 8/8；20.625 | 8/8；20.500 | 78.6%／94.4% |
| 3／42613 | 0/8；−49.434 | 0/8；−49.416 | 68.8%／94.4% |

前三步A的PPO K1=0.015/−0.007/0.001，B=0.005/0.007/0.002；
clipfrac A=0.011/0.010/0.011，B=0.014/0.002/0.013；
裁剪前grad_norm A=171.377/2.634/25.757，B=73.935/5.179/8.942。
以上取console舍入值、K1是已有采样统计，不是reference KL或精确分布KL。
两组前三步advantage均非全零；有限数值和较高有效率不等于学习改善。

第四批seed52416：A八局已结束4/8胜，进入`ray_trainer.py:1387`的保存阶段，
尚未写完整step4指标；B仍在`_collect_training_batch → generate_sequences`，
只记录3局自然结束且3胜，不得写成3/8或与A的完整八局直接比较。
A只留下4个model分片，大小不齐且无optimizer/extra/data完整集合；B无step4。
不读取大分片来宣称可恢复，也不能从这个不完整状态做训练后配对评测。
17:43经用户授权删除上述4个不完整分片；删除前大小/哈希与失败记录一致。
原日志和失败状态保留，详情见[清理记录](cleanup.md)。

直接失败证据：Ray报节点内存`957.34GB / 1007.51GB = 0.95021`，超过0.95保护阈值，
分别杀掉A WorkerDict与B WorkerDict。A栈为`_save_checkpoint`，B栈为rollout生成；
报错列出的A保存worker每个约49.4GB，B生成worker约44.4–44.5GB。
这是主机RAM压力，不是CUDA显存OOM、NaN或反馈机制已被证明错误。
当前只确认同节点并发在A保存/B生成重叠时触发保护，尚未将全部约957GB拆分归因
到进程RSS、共享内存和缓存，因此不声称只靠某个内存参数即可解决。

本轮实际消费回答A153、B151；A反馈0，B反馈44，其中2条明确人口阻塞反馈的
下一回答都合法且SupplyDepot在前。示例：

```text
<answer>
- supplydepot x 1
- barracks x 1
- reactor barracks x 1
- scv x 1
</answer>
```

这证明真实反馈进入本次RL并出现纠错选择，不证明这些行为是训练后新学到；
此前原模型固定输入已有纠错能力。两组前三批总胜负相同，且第四步/独立评测缺失，
不能宣布反馈带来学习提升。下一项先核清RAM峰值，安排隔离或串行资源的最小
保存/短训验收；需新RUN与预算确认，不直接重启长训、不关闭Ray内存保护。

## r2串行重启

用户确认43四卡串行A/B，总上限3小时、预计新增约120GiB、不自动重试。
两组均使用GPU0–3；A通过退出0、四步指标与完整checkpoint检查且显存释放后才启动B。
若A失败或资源未释放，B不启动，不杀他人任务。共享盘仍超时，所有资产和输出
沿用已验证的43本地路径，不复制大权重。r1及其失败日志不覆盖。

沿用同一入口的`FEEDBACK_RL_SERIAL=1`模式，RUN后缀改为
`A-rl4-4gpu-s0-r2`、`B-rl4-4gpu-s0-r2`；Ray临时目录改为
`/dev/shm/sc2_feedback_rl4_{A,B}_0911_r2`，控制目录`rl4-serial-r2`。
其余启动方式和总deadline与前文相同。一个外部只读进程每20秒记录`/proc/meminfo`
和前20个RSS进程，按current_arm及日志时间对应保存阶段，不修改Ray保护或训练内存参数。

模型、四卡mini/micro/累积、种子、四步/每步八局、step4保存、精度/cache、prompt、
reward、BO和微操均不变；A/B唯一方法差异仍为执行失败反馈机制。改变的是两组调度
由并行转串行及B的物理GPU号，不把r1/r2运行轨迹差异当作反馈效果。
验收仍要求完整step4 checkpoint及后续另批的独立评测，不以有限数值或短期胜率宣布学习。

本地实际CPU回归：串行/并行入口effective config全量相同、A/B仅反馈不同，
并执行反馈机制回归，共21项通过、1.134秒；未加载模型或启动游戏。
两入口`bash -n`、`git diff --check`通过；agent_system/verl/gigpo/scripts/configs无改动。

r2已启动：runtime SHA `361cb7ae7fd86fe8811d6fcb6946d0241bf93e11`，43独立部署
`/root/new-scii-execution-feedback-20260911/code-361cb7a`。完整私有bundle本地/43哈希
一致；服务器21项CPU回归通过（1.719秒）、23个系统源码哈希通过，核心源码与r1无diff。
17:56:54启动受控`sc2_feedback_rl4_serial_r2`，硬截止20:56:54。A实际配置与r1
归一化RUN/deployment路径后无差异；manifest为formal/clean、原始SpB、四卡、target4、
无checkpoint恢复。17:58:53创建Harder/Terran/Air游戏，17:59确认开局Planner调用；
首批实际seed为8个2732。此时尚无完成的global step；B未启动，不能说两组都已在训练。
详见[串行启动记录](rl4-serial-launch.json)。

## r2完整结果

A于18:42:40、B于20:01:58（北京时间，run result）结束；控制器20:02:20退出0。
两组均完成4/4 global step、各32局自然终止、指标有限、无空策略batch，
各有完整step4：4model+4optimizer+4extra_state+data，仍须真实加载后才能称恢复成功。
每组约59.2GiB；未删除既有checkpoint。runtime仍为361cb7a，未改训练机制。

| step／seed | A胜负／mean reward | B胜负／mean reward | A/B记录advantage范围 |
| --- | --- | --- | --- |
| 1／2732 | 8/8；20.375 | 8/8；20.438 | −3.520..2.737／−3.109..2.252 |
| 2／43567 | 8/8；20.625 | 8/8；20.500 | −1.616..1.124／−2.320..1.865 |
| 3／42613 | 0/8；−49.382 | 0/8；−49.499 | 0..0／−8.903..6.842 |
| 4／52416 | 8/8；20.500 | 7/8；11.759 | 0..0／−64.868..20.238 |

A共24/32胜，B23/32。A消费108条回答、94有效；B198条、178有效，其中74条收到
执行反馈。B产生更多可见计划变化和非零advantage，但没有训练胜率改善证据。
B第四步grad_norm135.407、PPO K1 .001、ref KL .044，均为console舍入统计；
该批7胜1败，不把大advantage直接认定数值bug，也不认为有限梯度说明学会。

完成日志console SHA256：A `021dbee4cf23ddcf0e560d44365c833ba4f055866abc48c632df65e816814efa`；
B `0a1612e90be4ca75c471583760af79d760066e9e71101287cb995958b26071ba`。
[完整小日志审计](rl4-reward-audit-final.json)保存逐局/逐回答/反馈/奖励及哈希；
[奖励可区分性](reward-discrimination.md)和[独立评测](learning-eval.md)是下一项，
不以此次短RL成绩自动晋级96步。

## 下一轮学习验收建议

9月11日晚：[独立评测](learning-eval-analysis.md)原始/A4/B4分别3/8、5/8、2/8；
[第二答Marine单变量](second-plan.md#marine单变量结果与改进决策2259以后核验)
同一B4从0/4变4/4，证明该场景有可影响胜负的规划选择。不再重复该seed的手写
答案消融，也不把这一个胜负对直接做SFT/DPO数据来宣称泛化或RL学习。

下一项建议从A-r2的完整step4模型及optimizer出发，再做4个global step，到step8
后用既有同一组8个评测seed比较A4/A8。训练保持A的反馈关闭，评测仍统一反馈
开启，与已完成A4评测相同；不改基础prompt、reward、gamma、采样、BO或微操。
不使用本轮手写回答hook，不续B4，不自动96步。A4只是当前有正向观察的候选，
不是已验收的长期健康版，也不据此认定失败反馈普遍有害。

资源待用户独立批准：43 GPU0–3串行训练/评测，总上限2小时、预计新增约60GiB，
仅step8保存完整checkpoint，不自动重试。沿用四卡匹配A4的四分片和mini/micro；
不把换八卡、改变累积次数同时带入本轮。若主机空间或卡不足，不擅自清理或扩预算。

A-r2未保存sc2_state.pt；不能宣称无缝续接游戏RNG。启动前必须核对小型data/extra
状态并沿用已有“模型/optimizer恢复＋新游戏边界”入口做CPU回归，明确新训练
种子及耗尽dataloader的处理，不伪造原SC2状态；实现与冻结SHA尚未准备完毕，
本节只是待批准方案，不称已经排队/启动。原始/A4评测不重复，step8独评唯一变量
是新增训练后的权重。模型能否保持合理生产选择、避免重复错误及改善留出游戏，
必须分别报告；有效率或少量胜率波动不能代替学习验收。

### A4追加四步＋A8独评：已批准，启动准备

用户已批准上一节43四卡、训练/评测合计2小时、约60GiB的安排，并要求睡前接续。
不复用Marine干预预算。23:13核验八卡0MiB、无GPU任务，本地153GiB可用、
主机MemAvailable约455GiB；历史SC2遗留进程未清理，不作为新增任务终止授权。

使用A-r2（361cb7a）的完整step4；四个model分片各5,555,629,511字节、optimizer
各10,332,463,827字节，4个extra及data.pt完整。23:16在固定依赖CPU读取1947字节
data.pt确认保存点是最后一次yield之后、StopIteration之前，8-worker真实状态重建
后关闭已耗尽尾部，再迭代四轮得到[1,1,1,1]；没有丢弃真实batch。
四rank小型extra均记录scheduler last_epoch4、lr1e-6；未在CPU加载大model/optimizer。

续训hook `continue_a4.install`复用已验证的`gamma_ab.pair_driver`关闭耗尽迭代器，
明确driver seed0，不重置已由原loader恢复的GPU rank RNG；仅将独立game RNG设置
为RandomState(0)跳过已记录前四个seed的位置。新run reset_index仍从0开始，
global step为5/6/7/8；实际游戏seed分别45891、21243、30403、32103，每步重复8局。
这是声明的新游戏边界，不是恢复缺失的sc2_state或原driver RNG。
原data/model/optimizer/per-rank RNG均由已有loader读取，实际返回后写
`a4_continuation.json`。未增加概率诊断、early-stop或模型内部修改。

评测复用`learning_eval`，仅增加A8标签并要求global_step8实际恢复。种子仍是
42613、52620、17089、4018390961、3236598124、3297512220、773351930、1098341402；
每seed一局，不重跑原始/A4评测。训练反馈关闭，独评反馈开启，分别与先前A训练
和A4独评一致；原始回答全为在线模型，不安装任何历史回答替换hook。

控制器 `run_a4_continue_node43.sh` 在未使用目录记录状态，两阶段共享7200秒，
外层timeout7185秒＋15秒退出；从控制器实际启动计时，不能重置预算。训练四步、
32局自然终止、无空batch、指标有限、step8完整保存且GPU释放后才启动独评。
独评使用相同GPU0–3，不与保存并发；任一失败或超时停止，不自动重试或续96。
若训练耗尽总预算，记录评测未运行，不延长。保留全部原始和失败产物。

新增RUN：

- `20260911-execution-failure-feedback-v1-spb2-A4-l6-s4plus4-4gpu-s0-r1`
- `20260911-execution-failure-feedback-v1-spb2-A8-l6-feedback-on-eval8-r1`

根目录沿用43本地runs；Ray目录分别`/dev/shm/a4t_0911_r1`和`/dev/shm/a8e_0911_r1`。
唯一控制目录由启动记录登记，模型和依赖不复制，只有新源码快照。
本地21项18通过/3缺Torch跳过，2.280秒；完整effective config与原A训练仅差恢复
参数、hook及其PYTHONPATH，A4/A8评测仅差权重路径/标签。服务器固定依赖CPU
回归和真实模型加载尚待完成，不能据此称实验已跑上。
本轮准备先遇到macOS旧Bash空数组nounset、配置测试未列hook导入路径两处返工，
均在本地CPU阶段纠正；不是项目原有bug修复，没有GPU试错或预算重开。

#### 实际启动与收尾安排

运行SHA `4ad9a605cc6a91419fd2a2ef5d6e8a2af72aac32`，43独立clean部署
`/root/new-scii-execution-feedback-20260911/code-4ad9a60`。私有完整bundle两端SHA256
`237a8b2a96c76349db014e9ddf4879462199cb2cc8a7a11105127218911097d7`；未推送GitHub。
49项本地46通过/3缺Torch跳过（4.344秒），服务器CUDA隐藏18项全部通过
（11.089秒），23项生产源码hash全部一致。没有修改agent_system/verl/gigpo/scripts/configs。
与实际A-r2训练和A4独评的[全量配置差异](a4-continue-r1-effective-diff.json)已核对；
先前只漏归一化三个Ray临时路径字段的审计断言退出1，补齐路径归一化后通过，
没有改变实际模型、训练或评测配置。

控制器 `sc2_a4_continue_r1` 于北京时间23:28:27启动，控制目录
`/root/new-scii-execution-feedback-20260911/a4-continue-r1`；两阶段硬截止9月12日
01:28:27。23:31确认原loader从A-r2/step4返回；8局实际seed均45891，Harder实局
已建立且开局调用Planner，进入global step5的rollout阶段，不称已完成一次PPO。
23:33快照尚无完整prompt_response批记录或完成的新global step；
[实际启动证明](a4-continue-r1-launch.json)保留这些未完成边界，不把显存占用当作学习完成。
23:34补核：继承的共享游戏日志已记录23:32:19的在线首答执行队列
（4Depot、4Barracks、24Marine），随后到iteration2104并持续战斗。没有安装人工
干预hook；这证明已实际消费Planner计划，不只完成模型加载。共享日志不能据此
拆出每局独立回答计数，完整统计等本轮原始账本和结果后再做。

启动后发现本轮新编排的证明路径错误：hook按既有default_local_dir写到
`checkpoints/a4_continuation.json`，4ad9a60控制器误从run根读取。未修改冻结部署、
未重启训练。23:32:37核验真实恢复证明后，仅在run根**新增逐字节副本**供原控制器
读取，两份SHA256均为`db26ef3e6effd24f312a5074b15c289dde4437d3014f428d31a55362673af430`，
来源/时间及原始路径登记在控制目录`proof-alias-receipt.json`并收入启动证明。
不是生成新的“加载通过”声明，也不是变更模型/optimizer。开发入口已改正读取路径，
新增回归并通过（5项3通过/2缺Torch跳过，0.492秒），本次runtime仍保持4ad9a60。
这是新实验编排返工，不算项目原有bug修复；原记录和原文件没有覆盖。

step8完整保存且全部原验收通过后，原控制器串行启动上述A8独评RUN；失败保留产物，
不接评测、不自动重试。没有重复原始/A4评测，也没有增加其他四卡实验或96步任务。

#### 9月12日完成核验

训练00:01:06完成新增global5–8，胜率8/8、8/8、0/8、0/8，全部自然终止；
`.done/result.json`一致，退出0、无空batch、step8的13个文件大小与记录一致。
后续独评实际加载step8、无PPO，00:15:51完成，控制器completed/0，总47分24秒。
同八个既有seed，A4/A8均5/8胜、逐局胜负相同；reward −5.715750/−5.776625，
消费回答合法22/23对21/23。没有重复原始/A4独评，没有自动长训；08:14核验43八卡空闲。
32局训练和8局独评的原始结果仍在各run。新增[信号审计](a8-training-signal.md)
使用已有日志和CPU，不改冻结配置；运行完成与学习效果验收必须分开。
