# SpB Level5 八卡96步

2026-09-08，用户明确改为：从43/44选一个节点，用当前最健康版本跑SpB Level5
96个global step。agent曾误解为替代Level6并暂停后续训练；不是用户明确取消Level6。
状态failed：16:27:55首批全无模型决策，退出码1，完成0/96步，无checkpoint。

## 版本选择及边界

采用目前系统修复完成相应验收、且已覆盖真实Planner消费的冻结runtime
`e1e25ce1d14362b209e34ec73871ebaeb63538a6`。对应最近的SpB r2/r3：同seed纯评测
收尾通过、驻防不抢进攻队，另一seed消费7条回答、6条合法，合法计划保留且生产
命令已对齐回放。Tech Lab选址和Marine/Reactor空位修复均包含在这版。
这些是系统接口/局部机制证据，不是该版96步RL或数值稳定性已通过的证明。

不是退回带swa_dropout0.5的历史r7，也不是使用SFT或历史RL权重。未加入星港
接班、early-stop、RMSNorm候选、改变cache或新的战术。最近星港施工审计只作为
已知限制，不阻止此次用户明确授权的长训，也不夹带修复。

原始`Panyuqi/SpikingBrain-2.0-instruct`／SSESWAMoBAForCausalLM（5.17B），资产
manifest SHA256 `40d6f1417568d5fda1107cdafd66c7441a679b03fb148851c8f226ecc9cf884c`，
revision `sha256:de52e8ef993afc763f9d4fb498e8d8302ef0bc13bd7dfd53c6ae5f77f43698b5`。
采用原prepare_spb_model的已登记训练kernel补丁；权重文件只软链接共享原资产。
SWA dropout0、HF/eager、FP32 FSDP参数与温度归一化、BF16 autocast和生成cache
保留；不下载权重或修改原模型目录。

## 配置与预算

43 GPU0–7，Level5=`Hard`、Flat128/Base75689、Terran Air、canonical g2、env seed0。
每global step8条rollout，总96步/768条rollout；seed实际写入training_seeds.jsonl。
固定BO先运行、Planner接管之后才产生真实训练决策；无Planner局不能伪造样本。
本轮是同一系统的新难度/步数运行，不宣称与历史r7或Level6的单变量因果对照。

GiGPO mean_norm、gamma .95、lr1e-6、mini8/micro1、PPO epoch1、grad_clip1、
entropy .001、reference KL系数 .01及原loss裁剪/复制补齐口径冻结；八卡每卡
累积1个micro后更新。自然终止、step shaping off、6BO/6bullet、4096/96tokens、
history0/off、temperature .4/top_p .9，以及prompt/解析/Executor/微操/自动战斗
均同r3。合法BO不额外补兵重排，对空备选和无效回答的旧规则保持不变。

直接训练：val_before_train=False、val_only=False、resume_mode=disable，不运行
Level5初评、不把44那批Level6初评当Level5基准。每步记录完整指标、回答和结果；
按既有96步启动习惯每8步保存完整model/optimizer/extra/data与模型配置，全部保留，
不启用checkpoint清理。共12个checkpoint，预计约0.8TB；不每步保存以免占满共享盘。
第一份完整checkpoint在step8，提前失败仍保留日志，不冒称有早期权重备份。
72小时运维上限沿用既有SpB96步入口；完成时间在首步实测后估计，不承诺八卡线性加速。

若出现NaN/Inf、空策略batch、归因错误、运行异常或checkpoint失败，走冻结代码
原失败路径并保留产物，不自动重试、不制造决策或改变规则续跑。单个KL高、某步
0胜或未持续涨分不单独判bug。训练rollout胜率不是独立评测或架构优越性证据。

## 部署与运行入口

16:04北京时间只读核验43八卡0MiB、内存约933GB可用、系统盘387GB可用，44
仍有外部任务和本项目两卡初评，故选择43。用户此前已明确授权停止43旧Qwen30B
服务；15:59已确认释放八卡，日志保留。本轮不再停止任何他人任务。

新目录 `/root/new-scii-spb-level5-96-20260908`，runtime子目录`code-e1e25ce`只读，
编排放`orchestration-<SHA>`。43系统Python3.12.3同44；只将44已验证的
`/root/new-scii-spb2-level5-control/python`（8.1GB）与Python3.12/平台编译头复制到
该独立目录，经新项目暂存`staging/spb-l5-96-node43-20260908`中转校验，不改共享
旧环境。Torch2.7.1+cu128、Transformers4.55.2、Ray2.48.0、Gymnasium1.2.3必须
实际导入确认；FLA/MoBA和SC2继续读取已有共享资产。

部署CPU预检发现43未安装系统Python头，独立复制的`pyconfig.h`仍引用平台子目录，
因此CPATH必须同时包含`include/python3.12`与`include`。只修编排的头文件搜索路径，
不改Python包、模型kernel或冻结runtime；此前两次预检失败保留为部署记录，不算项目bug修复。

入口 `bash experiments/records/20260908-garrison-wave-ownership-v1/run_spb_level5_96_node43.sh`。
RUN_ID `20260908-garrison-wave-ownership-v1-spb2-l5-g2-rl96-8gpu-s0-r5`，输出至
`/mnt/data0601/yangjinyue/new_SCII/experiments/20260908-garrison-wave-ownership-v1/runs/<RUN_ID>`。
独立Ray `/dev/shm/sc2_spb_l5_96_0908_r5`，tmux `sc2_spb_l5_96_r5`。
启动前再次核验八卡空闲、RUN未使用、clean SHA、模型/BO/SC2哈希和有效配置。
保留原bundle/失败记录，不改旧部署、不新建分支、不重试此前被拒的GitHub推送。

## 2026-09-08 启动核验

编排SHA `a30fd566397eb011c226467bbe969d616fb3ddf2`，runtime保持`e1e25ce`。
私有备份`/root/new-scii-spb-level5-96-20260908/source-a30fd56.bundle`，SHA256
`17452ab4169ec289bf8165584cf5cb5a5297cb3fad76507b2e01c4f80d505cbf`。
依赖tar SHA256 `9bf6f92c7ff129c33d0e3c509b7202af4c5148a6fb4d4a35dc5be12307acf6b6`；
头文件tar `6fbdb8347ffac295052f2fea96fcd143f219fd34a22b37a946cdef80dbc718c3`。

CPU dry-run `/dev/shm/spb_l5_96_dry_6959833`未加载模型、未启动游戏。
实际依赖导入通过；该effective command加`--cfg job`后由
`RayPPOTrainer._validate_config`验证通过，并显式断言八卡/8rollout、96步、save8、
非评测、禁续训、SpB dropout0、FP32参数/归一化、history off及Hard难度。
两次头文件预检失败均发生于模型加载前，最终CPATH修正后退出码0。

16:20:10八卡再次确认无compute进程，磁盘可用379GB、共享盘7.5TB；
16:20:11 tmux `sc2_spb_l5_96_r5`创建成功，pane PID3322556。
控制台`/root/new-scii-spb-level5-96-20260908/rl96-a30fd56-console.log`。
截至16:21仍在资产核验，不能据此声称模型已加载或发生PPO更新。

16:21:31.693586实际manifest生成：formal=true、dirty=false、mode=training、
world_size=8、GPU0–7、目标96步，模型/BO哈希与登记一致。随后11个权重分片
加载完成，日志确认为SSESWAMoBAForCausalLM 5.17B、swa_dropout0.0、use_cache=true。
16:23八个SC2Worker已开始第一批游戏，八卡均有本任务worker。首批实际游戏seed
为`[2732,2732,2732,2732,2732,2732,2732,2732]`（base_seed0/reset0），是现有
GiGPO同组rollout口径，不冒称八个独立种子。本时点尚无完成PPO更新或step checkpoint。

本地验证：`bash -n run_spb_level5_96_node43.sh`、`git diff --check`通过；
`python3 -m unittest discover -s tests -p test_system_release.py`实际26项通过。
环境导入、Hydra CPU配置验证均退出码0。新任务仍需观察真实更新、checkpoint和
游戏结果；部署/启动成功不能替代学习效果验收。

16:27:55结果：首批seed2732八局均自然获胜、policy_decision_count均0，collector
报`No consumed policy actions in this batch; refusing a PPO update`后退出码1。
completed_steps0、无checkpoint；八胜来自固定BO/系统行为，不能作为SpB学习结果。
原run、console、.done和result保留，不重标成功、不重试。console SHA256
`0895daab4f960310b2c87488ec405a63fa98ac52c7a1a6be63261f9339e2d356`。
随后用户单独授权42八卡Level6 96步，记录沿用[Level6实验](spb-rl4.md)。
