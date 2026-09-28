# 已验证修复整合验收

09-07 22:47北京：[Marine/Reactor占用候选](r4-reactor-conflict.md)完成同seed
Level6/g2单局验收：3个Reactor实际完成，BO超时2→0，Marine满槽堆单93/103→0/103；
3个Factory Tech Lab保留、至少4辆坦克出场。c15d02b、exit0；首次Planner截点，
不是胜率或RL验收，原版和其他模块未改。

09-07 21:48北京：[待开工建筑挂件保护候选](r4-pending-addon-fix.md)完成单局
同 seed Level6/g2 验收：3 个 Factory Tech Lab 完成，至少3辆坦克实际出场；
10:00走完237项固定BO并到首次Planner，exit0。首次截点、无模型回答/RL，
无自然胜负或完整回放；依据为SC2原始观测，不当成胜率提升。原版保留。

09-07：[三次Factory Tech Lab未开工原因](r4-factory-techlab-cause.md)：
补给站与挂件占地重叠；选址只保护已出现场上的工厂，遗漏SCV待开工订单。
三次仍选同一工厂、本地标完成。只读回放和CPU结构核验，不是已实施修复。

09-07：[两份固定 BO 回放损失定位](r4-preplanner-replay-losses.md)完成。
74 Marine及41 SCV全死；第4波未集结完成即全灭；三次Factory Tech Lab本地
标完成却未开工，Tank连续阻塞。仅离线证据，不是新修复或模型退化结论。

09-07 20:27北京：[完整固定 BO 扩展](r4-python-rng-prefix.md#完整固定-bo-扩展09-07追加授权completed未到达planner)
两局结束、均exit0，但都11:26自然败、未到首次Planner；BO停在#172/237。
数值快照/动作/进攻时序相同，原始trace仅敌方建筑显示ID不同。接管验收未通过，
没有新增PPO候选或续训；结果不能归因于模型，详见同一record。

09-07 20:03北京：[固定 BO 子进程 RNG 短验证](r4-python-rng-prefix.md)完成。
独立候选 `7aad777`，两次同seed两分钟前缀、无模型/GPU；87个trace记录帧完全
一致。只支持短前缀复现，不宣称整局败因/KL已修复；原r4未改、未续训。

最新：用户授权停止新增机制，转入[真实Level6 RL：96步目标、先step-8验收](real-rl-stage8.md)。
以下四步结果仍保留历史身份，不重标为新配置的结果。

09-07追加授权：[仅修空BO同帧循环，先CPU回归](empty-bo-frame.md)。运行代码
只有`while → if`，prompt/过滤/reward/微操/训练配置不变。候选`42329b1`在44
固定依赖下CPU 105/105通过，无跳过；没有启动GPU重跑。

随后用户批准[新版本r4真实八步验收](real-rl-r4.md)，09:23北京从`ca7885d`
在44 GPU0–7启动。09:29核验正在初始评测，八卡worker、实际SC2游戏及种子
均确认；train8尚未开始，不把初评称为PPO更新。初评完成后自动接train8和
step8配对评测，失败即停止后续段，第9步未排。旧r3仍failed且产物保留。
见[实际启动记录](real-rl-r4-launch.json)。

09-06 23:51北京：[r6只读历史审计](r6-history-audit.md)复核625/768胜及冻结源码。
新八卡r2初始评测在训练器配置检查失败，0局、0更新，train8未启动；
不是下述旧四步结果。失败证据见[启动记录](real-rl-r2-launch.json)。
09-07 用户批准仅补评测入口`env.rollout.n=8`；r3已从`aa1d20d`在44八卡启动。
00:44:55北京初始评测完成：1/8胜，八局自然结束，真实动作284/302合法。
00:45:21自动接上train8；完成5/8步、145次optimizer更新，每步25项checkpoint保留。
08:13:09因第6批env4的空BO同帧循环中断，原结果`failed / exit 1`；1,891次相同
空回答未推进游戏，不能当成正常长局。前五批2/40胜；第6批7局自然负、1局未结束。
step-8评测与后续96步未启动；08:18核验本任务退出、GPU compute进程为空。
见[故障原因与完整现场](real-rl-stage8.md#r3第6批失败空bo同帧循环)、
[失败归档](real-rl-r3-livelock-stop.json)。reward按原始八局等权另报。
见[首步证据](real-rl-r3-step1.json)、[连续训练证据](real-rl-r3-training-progress.json)。
未修改训练或游戏模块；新修复须先批准，不能将五步结果称为验收通过。
见[初始基准与回放限制](real-rl-r3-initial-eval.json)、
[实际启动记录](real-rl-r3-launch.json)。

状态（09-06 16:13北京）：16:08完成4/4 PPO更新，退出0，各步13项必需
checkpoint均非空；32局自然结束，1胜31负。运行与归因核对通过，但学习
效果和策略一致性未通过，不启动96步。见[完成审计](result-node44-r1.json)。
目标是让已验证改动真正进入主线默认配置，不再重复
单项奖励A/B。整合可运行不等于胜率或KL问题全部解决。

来源为已验证修复链 `8cb4f18`；它包含最新私有main `d42096e`，没有主线分叉。
完整备份 `/private/tmp/new-scii-before-verified-release-20260906.bundle`，SHA-256
`0c19be5b38aa52bf69efc8c0846320a389cff30962cc368f4b654a5b374f591a`。
临时发布分支仅用于这次跨多个既有修复的集成；回归后向main非强制快进，
源代码SHA先保存远端再运行。原main、r6/r7、44四步与47失败产物均保留。

本次仅调整默认配置，复用原实现：关闭per-call状态奖励；SpB默认SWA dropout=0；
通用入口默认Instruct-2507。自然终止、终局归因、格式奖励、位置和编号修复继承，
KL只继承汇总一行，不带入额外forward诊断。完整范围见
[整合说明](../../../docs/integrated-fixes-v1.md)。

## 配置与运行

RUN_ID `20260906-integrated-fixes-v1-qwen-instruct2507-l6-4gpu-rl4-s0-r1`，
入口 `run_node44.sh`。原始Qwen3-4B-Instruct-2507 revision
`cdbee75f17c01a7cc42f958dc650907174af0554`，模型manifest SHA-256
`c82106ca091f4797af1217cca7578618bb2a5cd7c4136ca0d2d06c0a77ed00a7`，无SFT/RL续训。
复用44已验证依赖和共享原始权重，不下载、不安装新环境。

Level6 Harder/Air、Flat128/Base75689、canonical g2、seed0，4个global step、
每步8局，共32局预算。每步保存四卡完整checkpoint（13个文件），全部保留。
HF/eager/cache开启，6 BO / 96 tokens，GiGPO mean_norm/gamma .95、lr1e-6、
grad_clip1、KL系数.01、entropy .001、mini8/micro1/epoch1均不变。
与44旧四步比：卡数8→4、状态奖励legacy→off、KL改为完整micro-batch汇总；
这不是严格单变量收益A/B，也不与旧KL数值拼成同一口径曲线。

源码部署 `/root/new-scii-integrated-fixes-20260906/code-<SHA>`，独立Ray目录
`/dev/shm/sc2_int_l6_4_r1`，tmux `sc2_int_l6_4_r1`。输出在共享
`/mnt/data0601/yangjinyue/new_SCII/experiments/20260906-integrated-fixes-v1/runs/<RUN_ID>`。
预计2–4小时含保存；6小时仅运维上限，不把超时当游戏败局。

## 验收与结论边界

启动前完整CPU回归、源码/模型/地图哈希及有效配置对照，确认原任务done且
GPU4–7无进程。旧部署不修改，运行目录不复用。已有启动器继续拒绝dirty源码、
缺资产和重复run；不新增调度框架或无界重试。

主要验收是32局自然结局、4次有限更新、真实动作归因与奖励守恒、逐步checkpoint
完整，日志无legacy per-call奖金。分别报全局胜率、有模型决策的局数、完整
动作合格率、reward、截断、KL和梯度。若整批无模型动作则保留失败并停止，
不复制终局伪决策凑训练样本，不擅自加采样预算。

KL汇总CPU已证明不改变更新，实局只验证可运行；本次不证明更新前log-prob
完全一致，也不自动放行长训。原始日志、manifest、effective config、done和
checkpoint是最终结果依据；CPU通过不能提前写成实局成功。

## 整合预检（09-06）

冻结源码 `1f2d8b5ac25d13cb4dc2b735725806fa713bb092` 已保存私有origin。
44固定依赖下执行 `CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
PYTHONPATH=<source>:/root/new-scii-spb2-level5-control/python /usr/bin/python3 -B
-m unittest discover -s tests -v`：74/74通过，无跳过，7.597秒。
本地22项启动器测试通过；40项r6测试中38通过、2项因缺Torch跳过。第一次
本地启动器测试发现配置哈希待同步，更新已明确修改的system-v1.env哈希后全过；
未关闭或放宽源码冻结测试。

`DRY_RUN=1 VERIFY_ASSETS=1` 执行完整新入口，源码、数据、模型、SC2与地图
哈希全部通过；128项override规范化后与旧四步仅差卡数和shaping开关。
没有新KL诊断forward；相对旧四步actor只差KL汇总一行，相对8cb4f18核心
Python逐字不变。详情及日志哈希见 [预检证据](preflight-r1.json)。

旧44八卡四步于12:59北京正常完成，32局2胜，step1–4各25个非空checkpoint
文件，退出0；其最后一步reward=-33.164，不支持此前三步reward持续上升的
外推。GPU4–7已核验无进程；0–3仍有其他任务，不占用它们。

## 主线发布与启动

私有GitHub `main` 已从 `d42096e` 非强制快进到 `271875f`，旧主线保留为
annotated tag `archive/main-before-fixes-20260906`。实际冻结运行源码是
`1f2d8b5`，与发布SHA之间仅差实验记录，不差任何代码或配置。

13:09:58北京创建独立tmux，13:10:09记录正式manifest。13:11核验
`formal=true / dirty=false / world_size=4 / GPU=4,5,6,7`，实跑128项override
规范化run路径后与预检完全相同；四个Ray WorkerDict已各占一张批准的GPU，
SC2已开始执行。未占用其他任务的GPU0–3。

实跑证据见 [launch-r1.json](launch-r1.json)。当前没有完成结果、更新指标或
checkpoint，不能据此称整合验收通过；预计总耗时2–4小时，完整结果以`.done`
和逐步checkpoint核验为准。运行中不再修改这份冻结源码。

## 运行快照（09-06 15:57北京）

[只读快照](progress-20260906-1557.json)：前三步reward依次为
-35.902、-48.023、-47.774；胜率1/8、0/8、0/8，有效率95.3%、98.8%、99.2%。
32局已全部自然结束，最终一批0胜；尚未核验第4次更新与checkpoint，故仍为
running。前述reward是现有训练统计口径，不是全部自然对局的独立评测均值。

并行KL r7完成后确认actor指标跨卡汇总只保留rank0，详见
[KL完成审计](../20260906-qwen-kl-metric-v1/README.md#r7-完成审计09-06-1557北京)。
本实验相同收集路径，不能将console的PPO/reference KL称为全卡均值。
不原位修改运行或据此宣布整合已通过；维持长训held。

## 完成结果（09-06 16:13北京）

16:08:27北京正常结束，`.done`与`result.json`一致，4/4更新、32/32自然
对局、无截断。step1–4各13项必需文件均非空，共52项；原始文件保留共享run，
结果JSON记录各步大小、最终模型分片哈希、配置与日志哈希，不覆盖旧结果。

| step | rollout胜率 | 训练episode reward | 动作有效率 |
| --- | --- | --- | --- |
| 1 | 1/8 | -35.902 | 95.3% |
| 2 | 0/8 | -48.023 | 98.8% |
| 3 | 0/8 | -47.774 | 99.2% |
| 4 | 0/8 | -47.188 | 99.5% |

总胜率1/32=3.125%，没有上升趋势。表中reward沿用trainer统计口径，不是
所有自然对局的等权平均；7局没有模型决策，不能混用分母。单seed、变化中的
训练策略不构成独立评测；描述性二项Wilson95%区间约0.55%–15.74%，不能
据此声称泛化胜率已经精确确定。

奖励/归因核对：按每局首次终止过滤日志中已结束环境的重复响应，625次真实
交互中618次有模型动作，7次无模型动作的终局被排除；25局有模型决策。
609/618回答有效，所有应用动作的policy ID与Executor消费ID一致。使用本次
冻结`policy_transitions`离线重构，25局的动作奖励总和均与环境奖励守恒；
与原训练credit日志最大差1.87e-6。整合run未额外保存PPO输入张量，因此
不将该重构称为逐token PPO张量验证，也不将动作接受称为单位建成。

配置为step shaping off，console有419条off标记、0条legacy step奖金标记；
Ray会去重日志，标记数不是调用总数。日志指标有限、response截断率四步均0；
actor指标仍受rank0-only汇总限制。并行KL专项的EOS漂移和生成尾部差异未解决，
本次只通过连续执行与已记录归因检查，不是“项目彻底修好”或长训准入。

16:10:50只读快照44八卡无compute进程；没有借此启动新任务。两项实验均
已结束并归档后关闭`sc2-r2`监控；r6/r8未启动，长训仍held。
