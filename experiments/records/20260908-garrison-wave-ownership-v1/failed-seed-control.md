# 原始 SpB：失败 seed43567 的历史对照

2026-09-08，用户批准立即补同seed评测。状态completed：20:55后台启动，
21:30:45退出0，八局自然Defeat、0次PPO更新；同seed原始模型也是0/8胜。
逐局累计日志reward均值−49.119375，真实消费184条回答、122条严格有效（66.30%）。
同seed既有训练批次、模型接管和两份回放分析见[结果与归因](failed-seed-analysis.md)。
不重训、不恢复96步。下面保留启动过程，不用历史running快照覆盖最终状态。
原Level6 run `20260908-garrison-wave-ownership-v1-spb2-l6-g2-rl96-8gpu-s0-r6`
完成4步后环境超时，19:23:07退出1；save8尚未触发，checkpoints为空。
因此只补原始模型，与该run第二批既有结果作历史对照，不冒称两份checkpoint的
独立配对评测，也不假装重建了历史模型采样RNG状态。

历史第二批是完成第1个global step后、尚未进行第2步更新的模型：seed43567重复
八次，0/8胜、平均episode reward -48.868，真实Planner消费146条、日志有效率75%。
首次消费前的兵力/经济与回答、实际命令应一起解释；无Planner局不从游戏分母剔除。
公开报告中逐局累计环境reward与训练invalid-action penalty后的指标分别列出，不混用。

## 冻结与最小入口

runtime仍为`e1e25ce1d14362b209e34ec73871ebaeb63538a6`，44已有
`/root/new-scii-garrison-wave-20260908/code-e1e25ce`只读。复用已验证Python覆盖目录
`/root/new-scii-spb2-level5-control/python`及系统Python3.12头，不复制依赖或下载权重。
原始Panyuqi/SpikingBrain-2.0-instruct，manifest SHA256
`40d6f1417568d5fda1107cdafd66c7441a679b03fb148851c8f226ecc9cf884c`；无项目SFT/RL。
Level6/Harder、Flat128/Base75689、Terran Air、canonical g2、6BO/96token、history0/off，
dropout0、FP32参数/温度归一化、BF16 autocast、HF/eager/cache，prompt/reward/执行器
和微操全部同历史run。保留采样temperature .4/top_p .9/top_k0，不使用评测默认greedy。

新编排`run_spb_failed_seed_node44.sh`调用原train_sc2.sh纯评测路径，显式四卡八局、
val_only=True、resume_mode=disable、save_freq=-1；GPU1–4必须空闲，最多3600秒加60秒
终止宽限，不自动重试或额外采样。预计30–60分钟。44于20:34八卡0MiB且无compute进程；
42仅后四卡空闲、43/46/47/48有其他任务，均不终止。

评测原生RNG范围不包含43567，不能只设置env.seed=43567。复用已验收
`r4_pair_seed.install`的Ray worker setup hook，只给验证环境装入一次性八个43567
种子源；保留原reset的worker调用、游戏逻辑和种子日志。独立文件
`spb_failed_seed_eval.py`不改任何冻结源文件；拒绝第二批、训练模式或checkpoint替换。
`paired_seed_selection.json`明确标注历史训练seed来源，evaluation_seeds逐个验收。

RUN_ID `20260908-garrison-wave-ownership-v1-spb2-original-l6-g2-s43567-eval8-r1`，
结果写原experiment共享runs的新目录。编排在44独立
`/root/new-scii-spb-failed-seed-20260908/orchestration-<SHA>`，以私有Git bundle备份。
CPU验证需覆盖真实reset派发八个43567、只一批、训练及恢复路径拒绝；随后Hydra
dry-run核对实际卡数、八局、采样参数、零更新及源码/资产哈希，再启动受控tmux。

## 判读

若原始模型在同seed也持续失败，支持该局面本身/冻结系统存在困难，不证明RL完全
无效；若原始模型明显更好，增加训练退化的怀疑，但历史对照不能单独定位根因。
比较胜负、逐局累计reward、Planner消费量、回答有效率和代表性原始回答。
游戏种子对齐不意味着GPU采样随机状态恢复；只一个种子的八次采样不能推广到全难度。

CPU结果（20:39）：新seed回归3项、原纯评测12项、发布/入口26项，共41项通过；
使用`/private/tmp/sc2-air-results.HBI61m/decoder/bin/python -B -m unittest discover`
逐个执行对应测试文件。`bash -n`及`git diff --check`通过，核心源码/config无diff。
纯评测仍使用已有入口的1GiB Ray object store预算；这是已存在的评测资源配置，
与历史训练的自动object store大小不同，不能声称所有运行参数逐字相同。

44实际CPU种子回归3项及Torch2.7.1/Transformers4.55.2/Ray2.48/Gym1.2.3核验通过。
首份编排afacfec的Hydra检查拒绝SC2_ROLLOUT_N重复追加：纯评测原入口先设1，
本编排需显式覆盖为8。因此只将该override的`+`改成`++`；旧CPU目录
`/dev/shm/sc2_failed_seed_cpu_afacfec`保留，未加载模型或启动游戏。此为新编排
预检返工，不计作项目原有bug修复。正式RUN_ID尚未使用，修正提交后重新冻结编排。

编排81699b3已通过44的Hydra和实际RayPPOTrainer._validate_config CPU验收。
20:47:58最终启动门禁退出1，20:49只读确认GPU0被其他人的PID634572占用；
没有创建tmux、正式run或模型进程。20:51用户明确允许GPU1–4四卡跑同样八局，
仅修改GPU选择、对应空闲检查和world_size，重新提交编排。此处新CPU配置需验收
四卡八局；上文八卡验收是此前状态。历史训练为八卡，本次为四卡，HF每卡回答
分组可能不同，不能声称硬件布局和模型采样RNG完全配对。旧源码、旧预检均保留。

## 四卡正式启动

编排SHA `721ff1e773ce0ada3fae10d3d52ab245272179b2`。44实际Hydra解析和
RayPPOTrainer._validate_config通过：world4、env rollout8、val rollout8、
val_only/val_before_train=True、resume disable、temperature .4/top_p .9、save -1。
20:55:50再次通过GPU1–4空闲、新RUN及tmux不存在检查，启动
`sc2_spb_failed_seed_r1`，pane PID643923。本地console
`/root/new-scii-spb-failed-seed-20260908/eval-721ff1e-console.log`。

20:56:14.874239 manifest确认formal/clean、runtime e1e25ce、mode evaluation、
world4/GPU1,2,3,4、原SpB模型manifest和无resume。paired_seed_selection已写出
八个43567；20:56:43 evaluation_seeds尚未产生，不能用选择记录代替游戏已启动。
TaskRunner PID646952已运行，尚未有PPO指标或.done。

私有备份均在44独立根目录：source-afacfec.bundle是完整历史，SHA256
`5facb71c90d5ad86bbe7dd60b987f756e6fbec24221d232b8463ea3da39cfb65`；
source-81699b3.bundle增量SHA256
`f8155976021d25443dd636818151d32cb9c2233e7b28ff85e7ebd362316ebb3b`；
source-721ff1e.bundle增量SHA256
`460e60477d6ebb720b32d997e1e19be7a1e93f9834f23cd3e54f96e301143a87`。
未重试被拒的GitHub推送；模型和原runtime未复制或修改。间歇SSH握手关闭
发生在部署/只读查询阶段，不能当作后台评测失败或启动第二个GPU任务的理由。

20:59:13本地日志确认SC2推进至约04:52，执行冻结g2；21:00:41只读验收确认
evaluation_seeds精确为base1000/reset0/[43567]*8，八个SC2Worker均实际创建游戏。
模型日志为SSESWAMoBAForCausalLM 5.17B、swa_dropout0.0、use_cachetrue。
新旧manifest的source_commit/source_files_sha256/model_manifest_sha256/bo_sha256
逐项相同；实际actor/ref及algorithm配置、BO预算、history、环境参数、采样
temperature/top_p已对齐，显式保留四卡/八卡及纯评测资源布局的差异。
无.done、无actor更新指标、未见Traceback/RuntimeError；不将其称为胜率或学习验收。
预计21:25–21:55出结果，入口最多3600秒加60秒终止宽限，不自行延长或重试。
