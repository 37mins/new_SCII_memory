# 开局 Planner 的 SpB Level6 RL：96步目标、step8验收

2026-09-09用户批准。状态：aborted，首批被空候选循环阻塞，14:28:54北京时间
按用户授权中止；原启动器result/.done为failed、退出码1，未改写。最终为7局自然胜、
1局未完成、PPO更新0、无checkpoint。见[空候选审计](opening-empty-candidates-audit.md)、
[停止记录](opening-rl8-stop.json)与[仅CPU验证的Factory候选修复](factory-prerequisite-candidate.md)。
旧runtime和结果保留，新GPU实验尚未启动。
复用本experiment与候选分支，不新建训练机制。
冻结runtime `e3ffac9e2da577679b9b980d66678e4cc72b06aa`，即刚完成的B组。
源码不改；只新增独立编排。原始模型、A/B、旧r6及全部失败产物保留。

## 验证与预算

检验开局就调用Planner后，SpB能否连续完成真实GiGPO/PPO更新，回答与游戏表现
是否明显退化。总目标96个global step；本次8epoch、每epoch一个数据batch，
因此只执行global step1–8，共64条rollout，不是8次optimizer更新。
`trainer.total_training_steps=96`保留原总预算/调度器；save_freq=1，保留每步
model/optim/extra_state八卡分片、data.pt、配置与tokenizer，禁止自动清理。
12小时训练墙钟上限含正常保存、不自动重试；预计约3–6小时，首步后校准。
首段约0.5–0.7TB checkpoint；是否继续到96待step8验收，届时重新核对剩余容量。
不在此入口自动评测、重新初评或进入step9。

自然终止、真实模型决策、奖励归因、补齐计权、KL/clip/梯度、原始回答与checkpoint
共同验收。NaN/Inf、明确归因/执行错误、空策略batch或checkpoint不完整需要停下
处理；不添加early-stop、不改变既有loss/复制补齐、不以低胜率或单个KL阈值判bug。
默认空batch仍报错，不启用现有可选skip，不额外重采样。运行未报错不等于学习有效。

## 冻结配置

模型为原始 `Panyuqi/SpikingBrain-2.0-instruct`，无项目SFT/RL续训；manifest SHA256
`40d6f1417568d5fda1107cdafd66c7441a679b03fb148851c8f226ecc9cf884c`。
模型直接读共享盘，保留已登记训练kernel补丁；swa_dropout0、HF/eager/cache、
FSDP参数FP32、FP32温度归一化、BF16 autocast不变。Level6/Harder、Terran Air、
Flat128/Base75689、seed0，训练实际seed逐批记录，不宣称八条rollout是八个独立seed。

`SC2_FIXED_BO_PREFIX_LIMIT=-1`只取消初始固定237条，不转换成阶段待办；当前
marine_pressure阶段来源、BO扣减、prompt、解析器、有效BO直通、无效回答补齐、
对空备选、微操、自动波次、驻防归属均同B。6BO/96tokens、prompt4096、history0/off、
无call20人为终止、step shaping off与reward均冻结。规则动作仍不能算模型选择。

训练算法沿冻结入口：GiGPO mean_norm、gamma0.95、PPO epoch1、mini8/micro1，
八卡每卡累积1；AdamW lr1e-6、weight_decay0.01、grad_clip1、entropy0.001、
reference KL系数0.01、temperature0.4/top_p0.9/top_k0，原loss截断不变。
PPO_DIAGNOSTICS=0，使用已有指标，不另开概率诊断。评测专用的四局seed/生成RNG
hook不带入训练；这属于已批准评测→训练切换，不称为A/B只改权重。

## 部署与启动

13:08核验44八卡0MiB，无GPU进程；其余共享节点无完整空闲八卡。
44旧父编排444866为SIGSTOP，初评子进程已僵尸，不恢复该套件；没有终止外部任务。
44原依赖 `/root/new-scii-spb2-level5-control/python` 为Torch2.7.1+cu128、
Transformers4.55.2、Ray2.48.0、Gymnasium1.2.3，现有Python3.12头文件可用。
共享盘7.2TB可用、/dev/shm491GB可用、主机available约521GB，启动前复查。
只复制已提交源码到独立 `/root/new-scii-g2-planner-handoff-20260909`，
`code-e3ffac9`固定runtime，`orchestration-<SHA>`保存本次入口；不改旧deployment。

入口 `run_opening_train8_node44.sh`；RUN_ID
`20260908-g2-planner-handoff-v1-spb2-l6-opening-rl96-gate8-8gpu-s0-r1-train8`，
输出为本experiment共享runs下同名目录，Ray目录 `/dev/shm/sc2_opening_rl8_0909_r1`。
tmux `sc2_spb_opening_rl8_r1`受控运行，开始前核验空卡、clean SHA、资产哈希和
未使用RUN_ID。保存完整CLI/effective config、manifest、编排SHA与退出状态。
GitHub推送此前被拒，本轮不绕过；以本地commit及用户节点完整Git bundle备份。

## step8后验收

复用B原始评测，不重复初评。按env_index使用同四局种子
`[2732,2732,43567,43567]`、同采样参数与数值路径，比较step8的自然胜负、整局
累计reward、有效回答、重复/缩短/格式变化，计入无Planner局。初始为3/4胜、
reward3.012、39/45有效；只有两个独立游戏seed，不足以宣称小幅提升或架构优势。
初评为四卡而checkpoint为八卡，现有加载器按world_size查分片，不能直接四卡
resume。末评前须核对已有权重导出/加载方式及评测布局，不静默改分片或伪称配对；
此处暂不实施新转换器或启动额外评测。真实训练不中断于该后续编排准备。
通过后从step8模型/optimizer/scheduler/RNG继续，不重新初始化Adam；续训种子
序列也须显式核对，不能把重置游戏序列称为完整状态延续。

## 启动前验证

本地执行 `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:tests /private/tmp/sc2-air-results.HBI61m/decoder/bin/python -m unittest test_opening_rl_launch test_handoff_paired_eval test_system_release.LauncherTest -q`：
25项通过、退出0、6.965秒，无模型/GPU/SC2。入口shell语法、diff检查通过；
与e3ffac9相比agent_system/verl/scripts/configs无任何差异。新增测试实际dry-run
确认八卡/八步/总目标96/save1、非val、无初评和续权重、开局prefix-1、dropout0、
FP32路径和所有被冻结的训练参数。远程依赖/Hydra及实际启动证据启动后追加。

## 实际启动（2026-09-09）

编排SHA `34c90a49ca8fdd641f75cdbacabb30df06c41de3`，runtime仍为clean e3ffac9。
完整bundle已在本地和44核验SHA256
`f4a7f9076d4dfce339be03db3002f088ebe2b7b0552e82cd8c186e77be60b1bb`，
44路径 `/root/new-scii-g2-planner-handoff-20260909/orchestration-34c90a4.bundle`。
本次无GitHub推送。远程实际dry-run、Hydra compose、冻结训练器_validate_config
通过；训练数据恰一行、8epoch/96总目标/save1、完整model/optimizer/extra保存。
与B effective config差异仅为评测→训练控制、8卡8局、记录路径、移除评测seed hook、
恢复原训练Ray object-store默认；无新增游戏或优化机制，初始权重完全相同。

13:18:42再次确认八卡空闲、RUN与Ray目录未使用后，启动tmux
`sc2_spb_opening_rl8_r1`，pane PID998163。控制日志
`/root/new-scii-g2-planner-handoff-20260909/opening-rl8-r1-console.log`。
13:19:08正式manifest为training/clean/formal、world8/GPU0–7；目标本段8、总96。
13:20模型11/11 shard加载完成，actor/ref日志swa_dropout0、use_cachetrue；
首批seed为八个2732（同组八条rollout，不是八个独立seed）。

13:21:43首轮回答 `- supplydepot\n- barracks\n- marine` 已被接受，
policy_action_applied=true、policy_action_id=1；13:22:44确认八个SC2实际进程
1003781–1003788，comm为Main_Thread，所以单用pgrep默认名称SC2_x64会漏报。
游戏推进至02:23，日志出现BARRACKSTRAIN_MARINE执行命令和后续Planner请求。
八卡各约40.5GB显存，GPU有计算。此时无Traceback/OOM、无.done，也尚无已完成
PPO更新；不能称8步连续性或学习效果已通过。[启动证据](opening-rl8-startup.json)。
