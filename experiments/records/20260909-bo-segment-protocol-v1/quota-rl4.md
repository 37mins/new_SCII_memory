# 配额澄清候选：4步真实RL验收

状态：`failed`。2026-09-10 00:56:27北京时间退出1；首批8局完成，actor反向传播
CUDA OOM，0/4个global step完成、无checkpoint。用户批准的44 GPU4–7、4步/
每步8条rollout、最多4小时预算未超限；没有重试或启动96步。

候选来自[整局A/B](quota-game-ab.md)：原始SpB同种子2/4→4/4胜，回答有效
20/38→15/20，但只有两个不同游戏seed、仍有5条无效回答。这只支持进入短训，
不证明能学、不证明架构优势。此次不是prompt A/B，而是已评测候选的连续RL验收。

## 冻结配置与入口

使用`run_quota_train4_node44.sh`，传入准确`SOURCE_SHA`，正式运行还须显式设置
`SC2_SHORT_RL_AUTHORIZED=1`。从本记录所在的独立clean部署启动，不修改旧deployment。
生产模块保持整局A/B的`2f7c5dcbcd8d51e866d31769a7d38617d1fce510`，仅增加薄编排、
CPU测试和本记录；最终启动SHA另行登记。仍用同一分支，不新增平行协议/实验目录。

模型为原始`Panyuqi/SpikingBrain-2.0-instruct`，不加载项目SFT/RL checkpoint；
manifest SHA256 `40d6f1417568d5fda1107cdafd66c7441a679b03fb148851c8f226ecc9cf884c`。
权重直接读取共享资产，复用44已验证Python依赖，不下载、不改模型源码。

场景保持Level6/Harder、Terran/Air、Flat128/g2资产及现有阶段策略；Planner从
00:00接管，`SC2_FIXED_BO_PREFIX_LIMIT=-1`。保留约15条/x N、无展开15动作上限，
`SC2_ORIGINAL_BO_SEGMENT=1`、`SC2_CLARIFY_BO_QUOTA=1`。192/4096生成预算、
temperature0.4/top_p0.9/top_k0，HF/eager/cache、参数FP32、FP32温度归一化、
BF16 autocast、swa_dropout=0不变。保留合法直通、无效补兵、对空备选、Executor、
微操/自动战斗、history0/off及现有reward；step shaping仍off，无call20终止。

训练使用既有GiGPO mean_norm/PPO，学习率1e-6、mini-batch8、每卡micro-batch1、
PPO epoch1、grad_clip1、entropy0.001、reference KL系数0.01，loss截断不变。
四卡意味着既有框架每卡梯度累积2次，区别于旧八卡训练的1次；不能声称卡数无影响。
不改补齐样本计权、不加early-stop或诊断hook，不把已知风险说成已经解决。

每global step重新采样8条rollout，共4个global step，最多32条rollout；不是4次
optimizer update。同一训练集1行，4个epoch限制实际训练为4步。沿用旧训练入口
`trainer.total_training_steps=96`仅固定优化器调度配置，不授权运行96步；无自动续训、
无resume、无训练前/中额外评测。训练seed0，保存实际`training_seeds.jsonl`；
训练自身采样流不同于4局评测，不冒称与A/B逐局配对。

每步保存完整checkpoint（模型、优化器、extra_state及数据迭代状态），保留原模型
manifest与全部历史。记录实际RL reward/success、KL/clip/梯度、所有原始输入回答、
自然终止/真实决策与有效率。不同KL口径分开，不拿动作有效率或数值有限证明学习。
训练后独立评测另行确认预算，本次不重新跑初始评测。

RUN_ID：`20260909-bo-segment-protocol-v1-spb2-l6-g2-opening-quota-rl4-4gpu-s0-r1`。
输出根：`/mnt/data0601/yangjinyue/new_SCII/experiments/20260909-bo-segment-protocol-v1/runs`。
Ray临时目录：`/dev/shm/sc2_quota_rl4_0910_r1`；独立后台会话，不连接他人Ray。

## 资源与验收

00:16北京时间：44 GPU4–7均4MiB且无compute进程；GPU0为其他任务，不停止。
42也仅四卡可用，43/47/48八卡均忙，46仅一卡空闲。44可用内存482GiB、
共享盘6.7TiB、/dev/shm491GiB；checkpoint写共享盘，不写剩21GiB的系统盘。
启动前再次检查任务归属和唯一目录；资源被占用就不启动，不换卡/自动重试。

已批准4卡最多4小时（16 GPU小时，含加载和退出），预计时长须由第一步实测修正。
若完成4步则立即退出；触及预算保留日志及已完成checkpoint，不自动延长。
明确运行错误（NaN/Inf、空策略批次、归因错误、存储故障）须停止并定位，
不造伪决策、不无限重采样。0胜或KL单个阈值本身不单独判bug。

验收首先看四步是否真实完成更新、完整checkpoint能核验、回答是否明显重复/
缩短/失效以及游戏指标是否退化。4步趋势不作显著性结论；即使训练运行通过，
仍须独立配对评测才能判断效果，不自动发布正式版或晋级长训。

CPU验证：`PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:tests /private/tmp/sc2-air-results.HBI61m/decoder/bin/python -m unittest test_quota_rl_launch test_quota_prompt test_fixed_prompt_ab test_original_bo_segment test_opening_rl_launch test_system_release -q`，
55项通过、10.043秒、退出0；bash语法及diff检查通过。真实启动器dry-run生成的
manifest/effective config确认training、4卡、4步、8条rollout、save_freq=1、
quota=1、original segment=1、192tokens、禁用resume/eval/额外诊断；CPU通过不等于
模型加载、游戏或PPO已启动。源码循环为4个epoch、每epoch1个数据batch，故不会因
优化器调度保留96而偷偷续训；checkpoint默认保存model/optimizer/extra及data.pt。

00:30–00:32复核42/43/44/46/47/48，无完整空闲八卡节点。43为sbh的RSA任务、
47为oylt的inner_mem任务、48为byin等任务；44 GPU0进程工作目录为cjy/horizon。
没有确认到本SC2项目占用八卡的运行，未终止任何进程。44 GPU4–7仍无compute进程；
旧`sc2_spb_l6_rl4_r4`会话只剩bash，不把会话名当成正在训练。新run/Ray/control
目录均未使用，现有依赖与资产可复用。模型、游戏、更新次数不因资源检查变化。

## r1启动核验

北京时间2026-09-10 00:34:58启动`sc2_quota_rl4_r1`后台会话，四小时上限至
04:34:58，完成4步即退出。冻结SHA：`6e40a5cc47dd5004a3dbb9a433c520cc6f497b54`；
源码`/root/new-scii-bo-segment-20260909/code-6e40a5c`，控制记录
`/root/new-scii-bo-segment-20260909/quota-rl4-r1`。生产源码与2f7c5dc无差异，
dce6155新增的入口未再修改；6e40a5c仅记录资源批准。

本地完整bundle `/private/tmp/sc2-quota-rl4-r1.bundle`及44
`/root/new-scii-bo-segment-20260909/source-6e40a5c.bundle`哈希一致：
`df8e5ef61d4ef4f8128d36765bcc02f0355adc6b3e29abf838af6b661bcaed2f`。
两端完整历史核验通过，未推送GitHub，未修改旧部署。首次服务器bundle核验因命令
不在Git目录而退出1，改在既有repo执行核验后成功；当时未上GPU，不计为模型/
训练失败。部署后实际44 Python下5项CPU入口/prompt测试通过，1.351秒、退出0。

00:36实际manifest确认formal=true、dirty=false、mode=training、world_size=4、
target_steps=4、resume禁用；model_manifest与整局A/B相同，源码/模型/SC2哈希
全部通过。effective config确认quota=1、original segment=1、opening=-1、192、
8rollout、save_freq=1、history0/off、dropout0、FP32参数/温度归一化、无额外hook。
00:36:40模型分片11/11加载完成、SSESWAMoBAForCausalLM 5.17B，四卡worker存活；
首批实际训练seed为8个2732，不声称八个不同seed。00:36:42已创建Flat128、
Harder/Terran/Air游戏，日志确认固定BO保留0条。此时尚无更新或胜负结果，
不把初始化完成说成RL/学习验收通过。日志MFU不支持SpB而为0只是该统计不适用，
不作为GPU没有工作或训练错误证据。

00:38:14复查四卡worker PID为1474572/1474794/1474795/1474796，每卡约9.5GiB。
00:37:24八局均在游戏00:00收到Planner回答并刷新队列；已有实际执行记录。
首轮可见两类合法计划：SupplyDepot/Barracks/Marine各x2（6个展开动作），以及
SupplyDepot x4、Barracks x4、Marine x24（32个展开动作）。后者真实进入Executor
队列，没有截回6或15；这是接入证据，不是产出完成率或学习效果。此时首批仍在
采样，尚未完成global step1/PPO更新，不能报告胜率、KL或梯度验收结果。

## 后续96步授权与续接边界（2026-09-10）

用户随后要求4步结束后自动安排96步长训，充分利用GPU。这是新增长训目标，
不改变当前r1的4步/4小时限制；不原位延长当前deployment。已设置本任务内
每5分钟跟进 `sc2-96`：核对短训结束和重复任务，完成/异常/需要决策时报告，
无实质变化保持安静。此时只是自动跟进已建立，**96步进程未启动，也未提交
不带条件的服务器队列**。该跟进依赖本机与应用保持运行。

只读核验发现恢复边界：`ray_trainer.py:1005`起能设置step号并恢复actor模型/
Adam/学习率调度器/worker RNG及data.pt；但`envSC2.py:676`起把独立
`training_rng`及reset_index重新初始化，现有checkpoint没有记录这一对象。
直接resume会重用首批游戏种子，不是不中断的原种子流。另`scripts/run_record.py:127`
要求日志step恰为1..target、局数为target×8，从step5续到96的正确日志也会
被它错误归为不完整。这是续训入口/归档的覆盖缺口，不是已证实本轮学习异常。

已询问用户选择：最小补齐上述续训边界后从step4接到总计96步，或保留代码
另起原始模型96步。在选择确认前不修改冻结模块、不启动长训。若采用续训，
需先CPU核验实际游戏seed序列、step编号和归档边界，再从独立SHA启动；不能
以加载初始Adam、偷偷重置环境采样或复制旧metric行伪造连续训练。
长训仍限本轮同模型/场景/prompt/优化配置，任何明显错误或回答退化先报告，
不拿单个KL阈值或少量胜负作硬结论。启动前按实际checkpoint体积核对存储和
保存频率；短训所有权重与失败记录保留，不为长训删除历史产物。

## r1失败结果（01:02只读核验）

首批8局全部自然终止，6胜2负；43条真实消费的模型回答中29条有效（67.44%），
43/43最终输入包含新配额澄清。完整日志有104行，包含结束后的非决策占位记录，
不能把104当模型动作数。逐局累计reward为
`[20.5,-49.5,20.5,-49.913,20.5,20.0,20.5,20.5]`，均值2.886；按env顺序累加到
首次done，不重复累加后续占位。八局均为seed2732，这是首批rollout结果，不是
训练后评测或学习收益；也不能与此前两个不同seed的4/4直接作学习对照。

游戏结束后调用`fsdp_workers.py:620`的`actor.update_policy`，在
`dp_actor.py:476 loss.backward()`报错：

```text
torch.OutOfMemoryError: CUDA out of memory.
Tried to allocate 19.25 GiB.
GPU 0 ... 17.30 GiB is free.
this process has 61.89 GiB memory in use.
57.13 GiB is allocated by PyTorch, 1.27 GiB reserved but unallocated.
```

报错worker PID1474572对应已核验的物理GPU4；异常中的GPU0是进程逻辑编号，
不是使用了他人的物理GPU0。只确认反向传播显存峰值不足，现有栈未给出19.25GiB
对应的内部张量，不能直接归因为prompt、学习率、cache或断言八卡必定修好。
没有逐optimizer-update日志，因此0个完整global step不等于有证据证明已执行
0次optimizer.step；本次未留下可继续或独立评测的更新后权重。

result.json与.done逐字段一致，process_exit_code=1、completed_steps=0、
checkpoint_complete=false，checkpoints目录为空。`metrics_finite=true`来自空的
已完成step指标集合，不代表已验收梯度/参数更新。控制.done亦退出1，结束于
00:56:27，墙钟21分29秒；00:58 GPU4–7已自行释放，无主动kill、无自动重试。
自动跟进`sc2-96`已按失败边界暂停；先处理OOM再谈续训边界，不改运行源码。

保留原run全部产物。小型证据SHA256：

| 产物 | SHA256 |
| --- | --- |
| result.json | cb231f931930725773ea92e87a5affa1669a37b65a38d0a1a75be73deef8c3c4 |
| manifest.json | 9e262f35395cc0f5447af8ceb3949a2ad308277bcaf75796ee97ec5086ceaaef |
| effective_config.json | 90d30af10bede7573aa8c6bde7f995bde18eba5418af58691bda01a9eb698295 |
| prompt_response.jsonl | ddcdfc16960c2002b466b5c49f562f76be44c1fff978ea322d15bad6f3e8c090 |
| episode_outcomes.jsonl | 2923680b91c625609da056f0cdf75ecdab47daef5af3650fec7f223d393a1e77 |

此次只接受“真实游戏及Planner接入完成”的结论，连续RL/完整更新验收失败。
下一项先只读核对FSDP包裹与反向显存峰值来源，再提出一项有证据支持的最小
重试配置和预算供用户批准；不在此heartbeat内改精度、cache、prompt或训练机制。
