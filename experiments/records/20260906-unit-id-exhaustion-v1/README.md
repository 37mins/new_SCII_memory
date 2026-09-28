# 单位编号耗尽修复

最终追加：八卡Qwen四步于09-06 12:59北京完成，退出0，32局2胜、零截断，
step1–4各25个checkpoint文件非空。reward依次-38.566/-24.295/-18.709/-33.164；
PPO KL依次.271/1.986/.028/.000，均为日志精度，末micro reference KL仍是旧口径。
源码44a6e9d与legacy奖励未改，不能据此称KL已健康或长训有收益。原始结果
SHA-256为`6c918007c5beb3831ad6e8b03e710a07a56a4ffb40ffdfba3301465e175a347a`。
下一次是[已验证修复整合验收](../20260906-integrated-fixes-v1/README.md)，不续96步。

当前状态：四卡单步于 09-06 09:15 北京时间完成，0/8 胜。用户随后批准
先跑八卡 4 步、逐步保存 checkpoint，判断是否值得接续；96 步仍不直接启动。
4 步已于 10:09 北京时间在 44 八卡启动，clean/pushed `44a6e9d`；
68 项 Linux CPU 测试通过，除卡数和步数外所有 override 与单步一致。
另按用户要求准备八卡 SpB Level 5 / 96 步；46被新任务占用后，用户批准
改到47，用内存盘依赖/缓存和共享盘输出；已于10:39正式启动。
后续状态：47 的 SpB 于11:09自行失败，完成1/96更新、无checkpoint；第二批
8局均在Planner动作被消费前获胜，空PPO批次触发既有保护。用户已要求暂停
长训，核验时进程已退出、八卡空闲，无需发送终止信号。全部产物保留，
不重启；当前开展独立 KL 验证。44四步冻结实验不原位修改。heartbeat仍PAUSED。

## 依据与修改

从实际长训基线 `f6a784cc1e0b0b83e2a2f6e280c7d38847b100ff` 隔离，旧目录
`/root/new-scii-qwen-generation-position-20260905/code-f6a784c` 不修改。
[两次现场栈](https://github.com/CeraveNew/new_SCII/blob/6b2a0119a19bd8affbf5b02ebd043511c077b037/experiments/records/20260905-qwen-generation-position-v1/l6-step3-stall-review.json)
停在 `BasePlayer.tag_to_id`；
[原包对照与 Mac/Linux 复现](https://github.com/CeraveNew/new_SCII/blob/6b2a0119a19bd8affbf5b02ebd043511c077b037/experiments/records/20260905-qwen-generation-position-v1/tag-id-origin-and-repro.json)
确认原包也在 0–999 满表后死循环。这是每局累计映射的对象数，不是存活兵数。

只改 `players/base_player.py::tag_to_id`：前 1,000 个不同 tag 保持原取模和
冲突查找；满后以映射长度分配新整数。不删除旧映射、不重编号，重复 tag
返回原号。依据是现有映射只追加、不回收，以及下游按整数反查真实 SC2 tag。
前 1,000 个编号恰好占满 0–999，此后每次追加 length，故编号连续且唯一。
没有引入新的规划算法或执行完成语义。现有选择正则 `\[(\d+)` 支持多位整数。
四位以上编号是必要的表示范围扩展，不声称整个 Executor 已验证正常。

分支 `codex/fix-unit-id-exhaustion` 隔离首次解冻的观测编号边界，同时避开
另一开发分支未验证的 KL 统计候选；相对 f6a784c，生产目录只差这一个函数。
新文件哈希加入 `configs/system-v1.sha256`，原哈希不变；旧结果全部保留。

## 实验配置与验收

使用原始 Qwen/Qwen3-4B-Instruct-2507，revision
`cdbee75f17c01a7cc42f958dc650907174af0554`，无 SC2-SFT/RL；模型 manifest
SHA-256 为 `c82106ca091f4797af1217cca7578618bb2a5cd7c4136ca0d2d06c0a77ed00a7`。
仍为 Flat128 / Base75689、Harder / Air、canonical g2、seed 0、GiGPO
mean_norm、lr 1e-6、grad_clip 1、HF/eager、cache 开启、no-thinking。
prompt、解析器、6 BO / 96 response tokens、legacy reward、微操、自动战斗、
优化器和已有 batch-copy 全部冻结；不续训练失败权重。

```bash
bash experiments/records/20260906-unit-id-exhaustion-v1/run_node44.sh
```

入口复用旧 stage 启动器，仅新增 Level 6 一步预算。预计 30–60 分钟，含保存；
两小时运维超时和 900 秒通信超时保留。RUN_ID 为
`20260906-unit-id-exhaustion-v1-qwen-instruct2507-l6-4gpu-rl1-s0-r1`。
完整配置、资产哈希、日志、结果与 checkpoint 由既有记录器写入
`/mnt/data0601/yangjinyue/new_SCII/experiments/20260906-unit-id-exhaustion-v1/runs/<RUN_ID>`。
启动前对照旧 Level 6 的 128 项 override，排除路径后只允许
`trainer.total_epochs`、`trainer.save_freq` 从 96 变 1。

主验收：满表分配及时返回且唯一/反查/旧号稳定；8 局自然终止、1 次有限
PPO 更新、退出 0、完整 checkpoint。另报 Planner 参与局数、严格完整合格率、
胜率、KL、梯度和截断率，不拿小样本胜率当修复门槛。若实局未使用扩展编号，
一步通过仅说明未发现接口回归，不代替满表测试，也不保证 96 步无其他故障。

## 验证

本地 `python3 -B -m unittest discover -s tests -p test_unit_id_allocation.py -v`
5/5 通过：前 1,000 次等同旧逻辑；10,000 次碰撞输入唯一且反查正确；边界
重复 tag 不占新号；新局重置；扩展编号经实际选择/动作构造/反查可用。
测试提取真实生产函数，五秒测试超时防止旧 bug 挂住。本结果不是 SC2/PPO
验证。

44 的固定依赖下全套 CPU 测试 66/66 通过，无跳过；资产哈希通过，128 项
override 除路径及已登记的两项步数外全同。AST 对照确认 base_player 其余
方法完全不变，详见 [预检记录](preflight-r1.json)。运行源码
`f083ec58eeeb271c8abafaf98f917762898ce152` 已推送私有 origin，旧 f6a784c
保持 clean；新目录 `code-f083ec5` 冻结后未修改。

08:34 已核验实际 manifest 为 formal / clean、配置与预检相同，四个模型
worker 存活、SC2 游戏推进，见 [启动记录](launch-r1.json)。目前没有最终
PPO/胜率结论；当时尚未授权 96 步接续。

## 追加八卡长训授权（09-06）

**后续状态覆盖本段原接续安排：用户要求重新评估 0 胜率下的长训依据，
当前 held。下面仅保留已准备配置的来源，不再构成自动启动许可。**

用户随后要求使用八卡并排上长训。复用本 record/分支，不再开新分支；
短测完全结束且通过消费 ID、奖励、自然终局、有限更新及 checkpoint 审计后，
核验 44 全部八卡空闲，再启动 `run_long_node44.sh`。短测不被中途扩卡或重启。
只准备好 launcher 不算已经开训；接续状态单独记录。

长训 RUN_ID 为 `20260906-unit-id-exhaustion-v1-qwen-instruct2507-l6-8gpu-rl96-s0-r1`。
从原始 Instruct-2507 重新开始，96 次 GiGPO/PPO 更新，每步仍 8 条 rollout，
共 768 局预算；不是从单步 checkpoint 续训。唯一配置差异是 world size 4→8、
更新数 1→96，以及每 8 步保存一次 checkpoint；保存不改变 loss 或优化器，
用于避免故障后丢失全部进度，保留全部 12 个保存点，约需 0.6 TB 共享盘空间。
48 小时为运维上限，不修改游戏终止规则。无八卡测量前不声称速度翻倍。

全局 PPO mini-batch 仍为 8、每卡 micro-batch 为 1；现有 FSDP 代码会把
每卡 mini-batch 从 2 分到 1，不修改其累积或损失实现。不同 world size 的
轨迹和数值不保证逐位相同，不能把长训与四卡短测当作严格配对性能对照。
全部生产目录和核心哈希仍与 f083ec5 相同，不夹带 KL、prompt 或战斗修改。
若短测失败、有新的阻塞异常或八卡被其他任务占用，则不启动、不杀他人进程。

## 短测最终结果与暂缓原因

[完整结果](result-r1.json)：1/1 更新、退出 0、13 个 checkpoint 文件完整；
8 局均自然败局、均调用 Planner。实际消费 263 条回答，其中 249 条完整合格
（94.68%），14 条因 `marine x N` 被现有协议拒绝；预算截断率 0。
消费 ID、奖励守恒、现有 validator 逐条复核通过。grad_norm 1.172、
PPO KL 0.323、clipfrac 0.004，指标有限；这不是长期数值健康或学习收益证明。

全部局终局项都是 -50，但 episode reward 均值为 -32.299，范围约
-48.706 到 -24.952。例如 4 次决策的败局为 -48.706，47 次决策的败局为
-24.952；这些差异是否正确反映接近获胜，尚未证明，不能把可计算的优势
直接视作有用的胜率学习信号。下一步先核对奖励来源和败局生产/战斗链路。

执行日志所见最大编号为 973，没有看到扩展编号被选为 executor；这不是
整个映射的最大容量。实局通过不能替代已通过的满表离线回归，也不能称
本次已经在真实游戏跨过 1,000 的边界。八卡 launcher 9614753 已准备，
只跑过 CPU/dry-run，未创建正式长训目录、未启动 tmux 接续。

## 四步学习探针（09-06）

用户批准先做 4 次更新，不直接跑 96 步。入口 `run_four_node44.sh`，RUN_ID
`20260906-unit-id-exhaustion-v1-qwen-instruct2507-l6-8gpu-rl4-s0-r1`。
44 在 10:02 北京时间八卡均空闲；启动前再次核验。沿用原始 Instruct-2507，
从头开始而非加载四卡单步权重；与 f083ec5 生产代码、算法和游戏配置完全相同。
相对单步仅 world size 4→8、更新数 1→4；每步仍 8 局，全局 PPO batch 8，
共 32 局。保存频率仍为 1，保留 step 1–4 全部 checkpoint，不轮转删除。

问题是首次更新后的策略是否出现早期改善，而非再证明能退出 0。逐步报告
自然胜负、episode reward、Planner 完整有效率、response clip、PPO KL、
梯度以及消费 ID/终局归因。第 1 批来自初始模型，第 2–4 批才来自更新后模型。
全部 32 局应自然结束、4 次更新有限、无挂起；如果仍全败或 reward 的改善
只来自调用次数增加，不自动延长。少量胜局也不足以证明稳定提升，需要连同
奖励来源、动作执行和数值检查判断。禁止仅凭退出 0 或有效率高就放行长训。

预计约 2–3 小时（单步四卡历史约 43 分钟，八卡实际速度未知），六小时为
运维上限，不改变游戏终止。每个八卡 checkpoint 必须有 24 个 rank 分片
（model/optim/extra_state 各 8）及 `data.pt`，四份总预算约 0.2 TB，共享盘
当时可用约 13 TB。`extra_state` 保存 scheduler 和各 rank RNG；现有学习率为
constant、warmup=0，4 步预算不会把学习率提前衰减到零。

续训应使用原生 `trainer.resume_mode=resume_path` 指向本次
`checkpoints/global_step_4`，恢复 optimizer/scheduler/RNG/dataloader 和 global step，
保持八卡、原始 reference 权重，并将新日志写入唯一后续 RUN_ID，不覆盖本次。
这次不改 trainer/resume 实现；当前 launcher 仍是 from-scratch，接续前需为
已有加载接口补薄入口及区间审计，不能把 actor 权重路径替换当作完整续训。
SC2 重置序列与控制进程状态不在该 checkpoint 中，接续前还须核对 seed 日志；
目前仅核对了存取代码，未声称断点恢复已经实跑或与不中断训练逐位等价。

## SpB Level 5 长训（09-06）

用户追加授权在另一空闲节点跑 SpB 96 步。46 在 10:09 北京时间整节点空闲；
47 虽八卡空闲但系统盘满，不选用。入口 `run_spb_node46.sh`，RUN_ID
`20260906-unit-id-exhaustion-v1-spb2-instruct-l5-8gpu-rl96-drop0-s0-r1`。
使用 `configs/spikingbrain2.sha256` 固定的原始
`Panyuqi/SpikingBrain-2.0-instruct`（SSESWAMoBAForCausalLM），无项目 SFT/RL。
共享盘权重只读，现有 r7 训练内核补丁由 `prepare_spb_model.py` 在新 run
目录应用，不修改源权重、cache 或注意力路径。

沿用此前用户明确要求的健康 `swa_dropout=0`：薄 launcher 设置
`SPB_DISABLE_SWA_DROPOUT=1`，通过已有 HF config override 传入
`+actor_rollout_ref.model.override_config.swa_dropout=0.0`。FSDP 的 actor/ref
均读取同一 override，HF rollout 使用 actor 模型；原始配置 attention_dropout
已经为 0，use_cache 为 true。未设置这个可选开关的历史入口保持原样。
此开关不修改 trainer loss、prompt、奖励、BO、微操或自动战斗。

Level 5=Hard/Air，canonical g2、seed0、每步8局，共768局；96次更新，
每8步保存完整 checkpoint，保留12份，约0.6TB。相对当前 Qwen 的 Level 5
配置，只选择 SpB 模型及其必要依赖、关闭已有 SWA dropout，再扩到96步。
不是拿 Level 5 SpB 与 Level 6 Qwen 胜率直接比较，也不能单独证明架构优势。
预计约1–2天，首步实测后修正；72小时仅是运维上限。

46 当前缺少已验证 Python，旧共享环境已漂移，不能直接使用。申请复制44的
固定 Python 到46独立 `/root/new-scii-spb-unitid-20260906/python`；不安装或
改动旧环境。FLA/MoBA 复用历史 stageguard 源码，启动前记录来源、哈希、
实际导入位置并通过 CPU/配置预检。此部署与44正在运行的源码隔离。
长训只在依赖和资产核验通过后启动；首批单独检查 actor/ref dropout 实际值、
有限 PPO 更新和消费归因。若出现实现异常，不把它算作模型败局。

10:21 再查46，GPU0–5已被非本实验 Gemma vLLM 服务占用，未启动SpB，
不终止这些服务。用户随后明确批准改到47：入口 `run_spb_node47.sh` 只将
Python、临时文件及 HF/Triton/Torch 磁盘缓存放到独立
`/dev/shm/spb-unitid-20260906`；KV cache仍开启。源码部署在共享盘
`/mnt/data0601/yangjinyue/new_SCII/deployments/spb-unitid-20260906/code-<sha>`，
run ID与训练配置不变。47没有tmux，最终使用系统现有screen持久会话，
`SCREENDIR=/dev/shm/spb-unitid-20260906/screen`，避免满系统盘。
依赖来源为已批准的44归档；46副本保留，不清理任何旧资产。

47首轮69项CPU测试通过；随后FLA导入预检因系统没有 `Python.h` 而失败，
尚未建正式run，也未更新模型。将44同版本Python3.12头文件复制到独立
`/dev/shm/spb-unitid-20260906/include`，通过 `CPATH` 指向该目录；不安装
系统包，不改变FLA/Triton源码或训练配置。旧d540ce6部署和失败预检日志保留，
用新clean SHA重新预检。

新部署 `d3e707356cf177c71bab02ff1c7a90728f251759` 已推送私有origin；
[47预检](preflight-spb-node47-r1.json)通过：69项CPU测试、SpB/SC2资产哈希、
FLA/MoBA导入及配置差异检查。与Qwen四步相比，只有模型及SWA dropout、
Level 6→5、步数和保存频率不同，其他非路径override一致。
10:39 北京时间已启动screen `spb_l5_96_drop0_r1`，随后核对实际manifest；
尚不能把启动成功或预检通过写成首次PPO更新通过。

[正式启动核验](launch-spb-node47-r1.json)：10:43时实际manifest为formal/clean，
96步、world_size=8、配置与预检一致。actor及reference的两份模型配置均打印
`swa_dropout: 0.0`、`attention_dropout: 0`、`use_cache: true`；HF rollout复用
actor模型。真实模型为5.17B参数的SSESWAMoBAForCausalLM，不是Qwen或SFT。
8个SC2 worker已开始Hard/Air游戏，尚无完成的PPO step，无启动异常。
