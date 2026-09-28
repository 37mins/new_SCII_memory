# SpB Level5：有界跳过无决策批次

2026-09-08，状态running/infrastructure_wait：18:41后台启动，18:44进入游戏，
18:49–18:50确认NFS阻塞；首批未结束、0次PPO，尚未通过在线跳过验收。
用户批准处理无决策批次后恢复43八卡Level5 96步；
42的Level6运行不变。旧run、deployment和失败结果保持只读。

## 已确认原因

对照是`20260908-garrison-wave-ownership-v1-spb2-l5-g2-rl96-8gpu-s0-r5`，
runtime `e1e25ce1d14362b209e34ec73871ebaeb63538a6`，编排`a30fd56`。
首批八局同组seed2732全部自然胜利、消费Planner回答数0；16:27:55退出1，0次PPO。
冻结llm_player.py:64加载237条g2，:660只在队列耗尽后renew_bo。代表性日志在
08:21、BO #191附近已经获胜。终局后生成的八条回答均policy_action_applied=False。
不是OOM、KL或合法回答被误删；固定BO控制全局时确实没有模型训练样本。
旧console SHA256 `0895daab4f960310b2c87488ec405a63fa98ac52c7a1a6be63261f9339e2d356`。

## 唯一机制

不改变模型接管条件和policy_transitions。默认仍拒绝空策略训练；仅本候选显式
启用有界跳过。collector核对每局都是自然终局、没有任何消费决策后，返回游戏
元数据而不是PPO张量。截断、无终局证据、未消费的非终局回答仍报错。

driver在生成边界保存empty_policy_batches.jsonl，继续同一global step的采样；
不调用advantage、old/ref、backward、optimizer、scheduler或checkpoint，不推进
global step或训练数据迭代器。下次reset沿原training_rng推进，不重设/筛选seed。
每次用原gen_batch深副本，防止环境kwargs被pop后丢失；模型权重保持原样。
这不是新增学习算法，也不是胜率筛选；只按“是否存在可归因的模型决策”处理。

边界是连续第3个空批次、或全程第4个空批次停止，触发批也完整记录。成功完成
96次global step最多包含99批/792局，较768局最多多24局；失败也不超过100批/
800局，且仍受原72小时上限限制。限额只是保守资源边界，不是性能阈值。
只允许新run、resume_mode=disable；本候选未实现跨run恢复跳过预算，不能将其
宣传为已验收的断点续训。原model/optimizer/extra/data checkpoint格式与save8不变。

实际改动是rollout_loop.py的自然终局标记和空批次元数据、ray_trainer.py的有界
生成循环、run_record.py的结束审计。结束审计用reset_index/env_index对齐原
episode_outcomes，额外局必须自然结束且policy_decision_count=0；不伪造训练步。
全部游戏胜率与含PPO更新批次胜率分开解释，前者包含被跳过的固定BO胜局。

## 冻结配置与启动

原始Panyuqi/SpikingBrain-2.0-instruct、SSESWAMoBAForCausalLM 5.17B，无项目SFT/RL。
模型manifest SHA256 `40d6f1417568d5fda1107cdafd66c7441a679b03fb148851c8f226ecc9cf884c`。
Level5 Hard、Flat128/Base75689、Terran Air、canonical g2、base seed0、每批8局。
GiGPO mean_norm、lr1e-6、gamma.95、mini8/micro1、PPO epoch1、grad_clip1、
reference KL .01及原loss裁剪和复制补齐均不变。FP32参数/温度归一化、BF16 autocast、
HF/eager、cache true、swa_dropout0、history0/off、6BO/6bullet、4096/96tokens、
temperature.4/top_p.9、reward和全部游戏代码冻结。

复用43已有Python3.12.3依赖目录和编译头，不部署新依赖、不下载模型、不停其他任务。
新只读源码目录`/root/new-scii-empty-policy-20260908/code-<SHA>`；沿用当前隔离
branch/worktree，提交和私有Git bundle备份，不改r6或旧deployment，不推GitHub。

~~~bash
FROZEN_SHA=<记录的完整SHA> bash experiments/records/20260908-sc2-empty-policy-batch-v1/run_spb_level5_node43.sh
~~~

RUN_ID `20260908-sc2-empty-policy-batch-v1-spb2-l5-g2-rl96-8gpu-s0-r1`，根目录
`/mnt/data0601/yangjinyue/new_SCII/experiments/20260908-sc2-empty-policy-batch-v1/runs`。
43 GPU0–7、tmux sc2_spb_l5_empty_r1、Ray目录/dev/shm/sc2_spb_l5_empty_0908_r1。
运行时以manifest/effective_config/source_commit为准。未确认加载和游戏推进前不称已跑上。

## 验收

CPU首先验证：空批→真批不推进global step；第三连续/第四累计空批停止；异常
不重试；关闭候选保持旧行为；真实样本和reward不变；终局审计计入全部额外游戏。
GPU先核对首批seed2732复现并被记录为零更新，随后原种子序列自然推进；首个
真实批次产生有限PPO更新后保持授权96步运行。若因连续空批停止，不自动增加
限额/换seed/改BO。NaN、归因错误、checkpoint错误继续原失败处理。
通过这一边界不等于学习或游戏表现改善；不因跳过胜局后训练曲线变低判模型退化。

## CPU结果（18:36，尚未上卡）

使用已有`/private/tmp/sc2-air-results.HBI61m/decoder/bin/python -B -m unittest discover
-s tests -p <文件名>`，分别实际运行：test_sc2_empty_policy_retry.py 10项、
test_sc2_evaluation_without_policy.py 12项、test_r6_terminal_policy_credit.py 10项、
test_r6_natural_termination.py 8项、test_system_release.py 26项，共66项通过。
新测试调用实际collector、driver采样方法、原SC2MultiProcessEnv.reset及结束审计；
模型、Ray/SC2 transport和checkpoint内容为轻量替身，不代表GPU/游戏验收。
原reset三次确认为2732→43567→42613，每组重复8次，未额外设seed。

`shasum -a 256 -c configs/system-v1.sha256`、`bash -n`入口、`git diff --check`
通过。对e1e25ce的源码差异核对：agent_system/environments、gigpo、verl/workers
及原env配置均无变化；没有修改模型、prompt、BO、reward、微操、loss或cache。

## 部署与启动证据

冻结runtime/入口SHA `7c34e82abe655dbef0bc6b3e85a43cf435fe3722`，服务器目录
`/root/new-scii-empty-policy-20260908/code-7c34e82`，detached clean。
私有备份`/root/new-scii-empty-policy-20260908/source-7c34e82.bundle`，SHA256
`b08329e06891e4f1b8ada6228eb7e36c1cfac804eceb6ae416516ad01e757682`，与本地一致。
两次SSH在连接阶段关闭，未启动任务；只读确认跳板恢复后部署成功。未修改SSH配置。

43新目录实际10项CPU候选回归通过。dry-run输出在
`/dev/shm/sc2_empty_policy_preflight_7c34e82`；Hydra `--cfg job`及实际
RayPPOTrainer._validate_config的CPU调用通过。与旧r5的effective_config比较，
归一化源码/run/Ray路径后，除实验名只多两个限额和一个日志路径。最初分析脚本
遗漏Ray路径归一化导致断言失败；纠正后通过，不是运行代码修复或GPU失败。

18:41:35再次确认43八卡0MiB、无计算进程、clean SHA、目标run未存在。随后
tmux sc2_spb_l5_empty_r1启动，pane PID3338692；console位于
`/root/new-scii-empty-policy-20260908/rl96-7c34e82-console.log`。
截至18:42:08资产校验已经通过模型分片/SC2/map，尚不能据此宣称模型完成加载或PPO更新。

18:42:14.957187 manifest记录formal=true、dirty=false、mode=training、world_size8、
GPU0–7、target96，模型和BO哈希与r5一致；两个限额在实际Hydra配置中为3/4。
八个actor/ref worker初始化完成，实际SSESWAMoBAForCausalLM 5.17B、swa_dropout0、
use_cache=true。18:44:42已见八个SC2Worker推进到01:40，首批实际seed2732重复8次。
此时GPU每卡约1.3–1.5GiB，模型参数按原配置offload；不是八卡上已经发生PPO更新。

## 18:50存储阻塞：与空批次逻辑分开记录

18:47读取共享目录开始阻塞；43于18:49执行带5秒超时的stat返回124。
八个本任务游戏子进程3344426–3344433处于DN状态。18:50:32只读内核栈确认：
3344426等待nfs_start_io_write/nfs_file_write；3344432等待
nfs4_wait_clnt_recover/nfs4_client_recover_expired_lease。不是依据GPU利用率猜测。
本地console最后更新时间18:48:26.759，543027字节；最近BO约05:43/#99–100。

43的共享产物只读快照：training_seeds一批，empty_policy_batches和episode_outcomes
尚未产生，.done不存在，PPO metric行0；未见Traceback/RuntimeError。不能把仍存活
的tmux或游戏启动称为训练更新通过。内核等待本身尚不能确定NFS服务器根因。

42原Level6仅做一次只读交叉检查：18:50:33同共享目录stat也返回124，本地console
最后更新时间18:48:26.122。没有停止或修改42任务。两台任务均保留，不重启、不
重新挂载、不扩大资源或改日志路径。下一在线验收仍是存储恢复后观察首批自然结束，
确认零更新跳过后沿既定种子流进入下一批；目前不能判断候选在线生效或学习表现。

18:52本地日志恢复少量增长，但18:53:52同目录stat仍返回124；游戏仅推进到
约06:14/BO #115。因此是持续/间歇NFS阻塞，不能因部分日志更新就称存储恢复。
首批尚未自然结束，未产生empty_policy_batches记录或PPO指标。任务保持运行，
不擅自延长既有环境超时；若触发原超时则保留失败，不自动重试。
