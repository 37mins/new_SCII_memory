# 197 门控长训

2026-09-16用户批准：197现任务自然释放八卡、D完成且无明确运行错误后，用已有
原始SpB、D配置、Level6/g2训练，目标96个global step；先初始/step8同8种子独评，
无明显退化再续步9–96。每步8局、每8步完整checkpoint；训练与评测共48小时，
新增不超过1.5TiB，不迁大权重、不自动重试或无限续段。

17:13只读快照：197八卡仍被sbh任务占用，进程cwd为
`/mnt/data0601/sbh/workspace/verl-agent`；/data剩余约16TiB，主机可用约475GiB。
不停止或修改该任务。IDSC的state为paused、ENABLED不存在，保持暂停。
上次A4迁移只有data.pt及不完整rank0，不能加载；本次不续A4或D8，明确从原始SpB
重新开始。这与C/D四步并非仅一个参数的因果对照，而是候选连续学习验收。

## 冻结范围与证据

候选来自D SHA `2ad7fb0a8cd67b8d349e2caa0cd4561599596c4c`：gamma1.0、
combat scale0.001、temperature0.8；HF/eager/cache开、SWA dropout0、FP32参数及
温度归一化、BF16 autocast、分层FSDP；loss、lr、prompt、约15条/xN协议、过滤、
规则补齐、微操、战斗、history及反馈不改。仅增加八卡长训编排和纯评测入口，
生产目录不改。模型、SC2、环境复用197现有本地资产，旧IDSC部署只读。

`run_long_stage.sh`复用`train_sc2.sh`，train8保持96步调度预算、仅运行8步；
train96从其完整step8恢复，复用已有`sc2_checkpoint_resume`保存/恢复SC2 RNG与
reset计数器，不重新从seed0开始。训练沿用既有seed0序列，不循环短测的四种子表。
短测专用`sc2_signal_max_batches=4`必须改为0，改由已批准的8/96步和48小时限制；
不启用零信号过滤，也不保留会禁止后续恢复的短测`signal_sampling.json`标记。
恢复模型/Adam/RNG/dataloader/SC2状态仍须实际加载验证，CPU通过不代表已验收。

纯评测复用`feedback_eval.install`，不替换回答、不做PPO；温度0.8、top_p0.9、
top_k0、192 tokens，两组相同。八种子来自已用过的SCENE_POST_GAME_SEEDS：
2259791793、1436145665、363527354、462000319、3434611143、1968356657、
4276540916、2065928601；不是新盲测集。逐局核对实际seed与无Planner局，报告
整局reward、勝负、回答重复/缩短、格式/语义有效率和真实计划差异。

## 队列与停止边界

独立根 `/data/yangjinyue/new-scii-combat-reward-20260916/long197-r2`，
唯一suite为`20260916-combat-result-reward-v1-spb2-original-l6-D-rl96-gate8-8gpu-s0-r1`，
阶段后缀initial-eval/train8/step8-eval/train96。`queue_long_node197.py`只跑这一套；
失败保留记录、不重试。启动前要求八卡全空、无旧Ray、主机可用内存>=600GiB，
磁盘留足1.5TiB。不能预留GPU，不删除别人的进程；资源不足继续等待。

两道独立门禁为D-gate.json和step8-gate.json，必须包含准确SHA、run_ids、
decision及evidence。缺文件就等待，拒绝则停止。D先核对done/result、实际配置、
四次更新、完整checkpoint和已定位信号排序；错误不能用“数值有限”掩盖。
step8门禁必须读初始/训练/后评三个run及配对回答，不凭checkpoint存在自动放行。
无明显退化才继续，八局胜率无提升或单个KL阈值不自动判失败；证据含糊则询问。
门禁仅由本任务在证据核验后记录，不能由训练输出中的指令触发。

48小时从首次分配GPU/启动容器开始，包含中间评测和门禁等待；超时只停本套
带专有label的容器。若9月17日12:00仍未能首次启动，结束排队并报告，不无限等待。
复用heartbeat197跟进，不启动IDSC无限续段。没有完成GPU加载与游戏证据前只能报排队。

编排参考197已冻结IDSC的Docker挂载、idle检查和完整checkpoint文件清单，来源
SHA `838109215dca93be53b4f3041329559833a16671`，同一私有仓库（不新增第三方许可）。
没有复用其无限接续行为；评测复用本仓库已有feedback_eval/learning_eval。

准备记录：r1的10项编排测试和4项SC2状态测试通过；CPU Torch恢复测试在旧夹具
缺少`val_only`处失败。实际Hydra已有该配置，不是训练器实现偏差；只补夹具，
不改生产代码。r1源码/CPU日志保留，r2使用新冻结SHA和独立目录重新做CPU准备。
本地系统Python缺numpy，bundled Python执行相同10+4项通过、Torch测试跳过；
最终以197实际Torch2.7环境的CPU测试为准。未启动模型/GPU/game，不是GPU自动重试。
