# Reactor 空槽修复：完整游戏配对验收

2026-09-07，用户批准。状态：completed（r2两局自然败局，均exit0，无PPO）。此前首次 Planner 前的单局只证明
Reactor/Tech Lab 开工及产出，不证明自然胜负或模型接入后的表现。

本次两组各一局 Level 6（Harder）、Flat128、Terran vs Terran Air、canonical g2。
允许原 Planner 在开局 BO 耗尽后继续规划，运行到自然终止。无 PPO、不续训。
唯一组间游戏代码差异是本地 Marine 生产者空槽过滤。

| 组 | 冻结游戏 SHA | 44 GPU | RUN_ID 后缀 |
| --- | --- | --- | --- |
| control | `836460a8533b6f0429f9e9d401ac9715adc55f5b` | 0 | `control-qwen2507-l6-g2-s2621640384-r2` |
| slots | `c15d02b08085ddfcf20cff5517d10ceae9fa2af0` | 1 | `slots-qwen2507-l6-g2-s2621640384-r2` |

RUN_ID 共同前缀 `20260906-integrated-fixes-v1-r4-reactor-pair-`。
根目录 `/mnt/data0601/yangjinyue/new_SCII/experiments/20260906-integrated-fixes-v1/runs`。
游戏 deployment 为 `/root/new-scii-r4-python-rng-20260907/code-<短SHA>`，只读。
独立编排 deployment 以本记录所在提交冻结；两组共享 launcher 和种子选择钩子。

模型固定原始 `Qwen/Qwen3-4B-Instruct-2507`，revision
`cdbee75f17c01a7cc42f958dc650907174af0554`，资产及依赖沿用 44 原部署。
不使用 SFT 或 RL checkpoint。FP32 FSDP 参数、FP32 温度归一化、BF16 autocast、
eager、HF 生成 cache 保持；评测沿用冻结入口 greedy（do_sample=False）。
prompt/legacy、6条展开动作、96 response tokens、4096 prompt tokens、history=0/off、
reward shaping=off、自动战斗和旧微操不变。两组均不开 Planner 人工终止。

实际游戏 seed 必须为 `2621640384`，Python 子进程种子开关两组均为 1。
该 seed 是原 `RandomState(1000)` 八局序列第4项。共用的 `r4_pair_seed.install`
通过 Ray 2.48.0 原生 worker_process_setup_hook，在原 make_envs 返回后只跳过
validation_rng 前3次抽取；不启动额外游戏、不改训练 RNG、核心源码或 SC2 reset。
`paired_seed_selection.json` 与原 `evaluation_seeds.jsonl` 必须吻合。

使用 `run_r4_reactor_pair_node44.sh control|slots`；它调用各自冻结
`scripts/train_sc2.sh` 的 EVAL_ONLY=1 单卡单局路径。
每组上限30分钟墙钟，最多两张卡同时；每组1次 reset、不自动重跑。
资源检查通过才启动；若超时/模型加载失败/环境异常，保留失败产物，不能当败局。
与前次 prefix 测试相比取消“首次 Planner 即止”，这是本次两组共同的评测范围，
不是又一项组间算法变化。单卡只用于本次推理，不改变/重跑历史八卡 RL。

交付自然胜负、整局累计 reward、Planner 调用与原始回答、有效动作率、BO 超时、
挂件完成和 Tank/Marine/SCV 产出及损失、资源和回放。没有模型调用的游戏单列，
不得算成模型规划成功；单 seed 两局不能推断总体胜率或架构优势。
验收重点是修复是否延续到自然完整游戏，以及是否暴露新的具体阻塞。
若游戏胜负不同仍需结合时序，不能把单局涨分当作统计显著提升。

CPU：`/private/tmp/sc2-planner-executor-venv/bin/python -B -m unittest discover
-s tests -p test_r4_pair_seed.py -v` 两项通过（仅验证种子选择和非法入口拒绝，
不冒充模型/游戏验证）；`bash -n` 通过。已核对两冻结 SHA 的游戏代码差异仅为
Marine 空槽条件及对应哈希，其他 agent_system、verl、train_sc2.sh、system-v1.env 相同。

## 保留的启动失败

编排 `5e83850` 的两组 r1 于北京时间23:19:36启动，23:19:52前退出（各 exit 1）。
完整游戏数0、模型加载0、PPO更新0；原 run、console 和 `.done` 保留。
原因：新增 `+env.prompt_response_log_max_prompt_chars=0` 与入口已有同名 `+` 参数
冲突，Hydra ConfigCompositionException。dry-run只记录CLI，没有实际执行Hydra解析，
不能证明入口可启动。本次只将编排中该参数改为 `++`，用新 r2 和新冻结编排提交；
先用 `--cfg job` 实际解析完整配置。游戏SHA、模型和所有游戏/生成设置不变。
这是诊断编排自身返工，不计作项目原有bug修复。没有重复消耗游戏或GPU实验预算。

## r2 实际启动

编排 SHA `8968d1dafdfee04b024f354451db81d86a9ea637`，44 独立目录
`/root/new-scii-r4-python-rng-20260907/pair-8968d1d`。
两组原始模型 manifest、游戏SHA、mode=evaluation 和单卡归属已实读核验；
`evaluation_seeds.jsonl` 都只有 `{"base_seed":1000,"reset_index":0,"seeds":[2621640384]}`。
两份 test.parquet 实际均1行。未进行 PPO、未加载 RL checkpoint。
现有依赖实录 Torch2.7.1+cu128、Transformers4.55.2、Ray2.48.0、burnysc2 7.1.3。

控制组 tmux `sc2_reactor_pair_control_r2`；候选 `sc2_reactor_pair_slots_r2`。
北京时间23:26已确认模型加载和真实SC2开局（各约31GB显存）；23:33两组都已进入
Planner，超过20次调用而未人为终止。首条实际回答两组相同且被消费：

```text
- marine
- supplydepot
- mule
- siegetank
- marauder
- Launch OFFENSE with 8 marine
```

独立只读结果分析入口 `audit_r4_reactor_pair.py RUN_A RUN_B`；用现有
s2protocol5.0.16.97563.0 / mpyq0.2.5 解码75689 replay。
复用旧已校验回放 `/private/tmp/r4-preplanner-replay.Z1yZ2Q/repeat1/replay.SC2Replay`
的回归已实际执行：全部出生、死亡、建造开工计数与旧归档完全一致。
新游戏结果必须等终局后再读，不能用这项回归替代本次游戏验收。

北京时间23:36只读快照（非终局）：两组 trace 都保存到 frame5500。
control 两个 Reactor 超时仍精确发生在2332/#110、2556/#114；观察到1个完成Reactor。
slots 无 BO 跳过记录，观察到3个完成Reactor；两组3个Factory Tech Lab均已完成。
control 已消费42条模型回答，其中40条有效；slots 40条中36条有效。
这是不同观测下的中途快照，不能直接当作最终有效率或模型能力对照。

## 最终结果

[原始结果、manifest和artifact哈希](r4-reactor-paired-results.json)。同实际seed
`2621640384`，两组自然终止、未截断、各1局、各exit0、PPO更新0。
控制组北京时间23:41:16结束，修复组23:37:54结束，GPU0/1已核验释放。
两组总墙钟分别约17分钟、14分钟，未超过各30分钟上限；未启动额外游戏。

| 指标 | control：仅挂件选址修复 | slots：再加Marine空槽过滤 |
| --- | --- | --- |
| 自然胜负 | 0/1胜 | 0/1胜 |
| 游戏时长 | 24:10.5 | 19:02.5 |
| 整局累计环境reward | -45.45 | -45.00 |
| 被消费的模型回答 | 70 | 54 |
| 完整有效回答 | 67/70（95.7%） | 49/54（90.7%） |
| 完成Barracks Reactor | 1 | 3 |
| 完成Factory Tech Lab | 3 | 3 |
| Reactor BO超时 | 2 | 0 |
| 全部BO超时 | 8 | 0 |
| Marine实际出生／死亡 | 306／306 | 249／249 |
| Tank实际出生／死亡 | 41／41 | 31／31 |
| Marauder实际出生／死亡 | 21／21 | 16／16 |
| SCV累计出生／死亡（含开局12个） | 56／56 | 56／56 |
| 终局矿／气存量 | 75／260 | 2870／715 |

出生、建造完成、死亡由replay tracker事件计数，不是Executor命令数或trace采样估计。
回答有效率使用实际被消费、可语义评估回答上的现有validator布尔值，不代表战术正确。
累计reward对所有环境交互求和，已与现有rollout累计值核对；不能因候选少负0.45就宣布
游戏收益。两组都调用了真实模型，不存在零Planner游戏被漏掉。
不同游戏时长使累计产量不能直接当作生产效率比较。

**机制验收通过，游戏效果改善未获支持。** 候选确实消除了指定两个Reactor超时，
但本seed比对照早失败约5分钟，不能称为整体加强，更不能宣布Level6已修好。
这次不是架构比较，也没有训练，不能据此评价RL或学习能力。

目前最明确的剩余现象是兵确实生产了，但没形成可持续的有效战斗力：两组都未生产
Viking或Medivac，replay仅有SprayTerran装饰升级，无Stim、CombatShield、攻防升级。
control的41辆Tank中18辆被Battlecruiser击杀，slots的31辆中15辆；Marine分别
117/306、87/249被Battlecruiser击杀。不能把所有损失都归为战列巡航舰，也不能只凭
这些统计断言具体某个战斗函数有bug。下一项应只读对齐敌空军出现时的阶段待办、
Planner原始回答与攻击命令，查清为何仍形成这一配兵和消耗模式，再提出单点候选。
本轮不改策略、微操或数值路径，不追加RL和GPU实验。

本地证据根 `/private/tmp/sc2-reactor-pair.cGSFdy`，两个完整RUN_ID对应子目录保存回放、
trace和结果；控制组只取必需文件，服务器原全量日志保留。分析脚本自身已通过旧回放
出生/死亡/建造开工/Marine与SCV击杀归因回归，以及本次两组artifact哈希和reward核对。
正常评测不生成prefix测试专用cleanup.jsonl；首次打包因误列该文件返回2，但gzip校验
及其余必需文件完整，已核验后取回。这是取证编排问题，不是实验失败或新系统bug。

## 补充SpB模型对照（用户追加批准）

状态completed（2026-09-08北京时间00:03启动、00:23:24自然败局、exit0、无PPO）。
只补修复后 `c15d02b` 的SpB一局，对照已完成的Qwen slots-r2，
不是再跑一轮旧/新Executor A/B。无项目SFT/RL的原始
`Panyuqi/SpikingBrain-2.0-instruct`，内容标识
`sha256:de52e8ef993afc763f9d4fb498e8d8302ef0bc13bd7dfd53c6ae5f77f43698b5`，
manifest `configs/spikingbrain2.sha256`；权重读共享原资产，不下载、不改原模型。

RUN_ID `20260906-integrated-fixes-v1-r4-reactor-pair-slots-spb2-l6-g2-s2621640384-r1`。
44 GPU0空闲且Qwen已结束后启动，单局、单卡、最多30分钟，不做PPO/不自动重试。
复用本记录launcher的 `spb` 选项、相同游戏SHA、种子钩子、Level6/地图/对手/g2、
prompt/解析器/微操/reward、greedy、4096/96预算、history=0/off及FP32/BF16路径。
模型自带tokenizer/chat template/特定内核属于模型差异，不宣称序列化token完全相同。

使用现有SpB入口默认 `SPB_DISABLE_SWA_DROPOUT=1`，实际override `swa_dropout=0`；
原snapshot `attention_dropout=0`。`model_type=sse_swa_moba` 时冻结HF rollout保持
`use_cache=True`，不走旧SpB1 `gla_swa` 的禁cache分支。沿用已有r7训练态内核兼容
准备脚本，权重仅只读软链、模型代码复制到新run，未加入新kernel候选。
需要现有FLA/MoBA只读源码路径及 `/usr/include/python3.12` 编译头，
Torch/Transformers/Ray与Qwen完全相同；不安装或替换共享依赖。

冻结编排 `67b121fb6a3eeebc6e896d9d498dd18a3af7d50a`，部署
`/root/new-scii-r4-python-rng-20260907/pair-67b121f`，tmux
`sc2_reactor_pair_spb_r1`；日志在该目录同级 `pair-spb-67b121f.log`，正式输出
仍在共同experiment runs根下。现有依赖导入及完整Hydra配置解析通过，
有效配置相比Qwen只差模型、SpB特定dropout/依赖和记录路径。
CPU隐藏CUDA的导入检查触发Triton driver错误，改为只让已授权空卡可见后原依赖直接
导入成功（模型加载0、张量显存分配0）；未改库、模型或kernel，不算系统修复。

运行中的模型日志两次打印 `swa_dropout: 0.0`、`use_cache: true`，
`SSESWAMoBAForCausalLM contains 5.17B parameters`；Qwen实际为4.02B。
因此本次是相同外部系统的模型对照，不是等参数架构因果验证。
SC2引擎实录Map Flat128、Bot Terran vs Computer Harder(Terran, Air)，
原始 `evaluation_seeds.jsonl` 的seed为2621640384。

### SpB与Qwen修复组最终对照

[SpB完整结果和哈希](r4-reactor-spb-result.json)。相同c15d02b游戏代码、相同seed，
首轮纯文本Planner query逐字一致，SHA256
`b7dfd3a02652d430310a0396ec1f2160663dc049fa14b835b3d7bb1fafe463a0`。
后续观测随不同回答演化，不能当作全程同一批输入。原始记录未覆盖，Qwen没有重跑。

| 指标 | Qwen Instruct-2507 slots-r2 | 原始SpB2 slots-r1 |
| --- | --- | --- |
| 自然胜负 | 0/1胜 | 0/1胜 |
| 游戏时长 | 19:02.5 | 22:19.3 |
| 整局累计环境reward | -45.00 | -49.50 |
| 被消费的模型回答 | 54 | 56 |
| 整体有效回答 | 49/54（90.7%） | 43/56（76.8%） |
| Reactor完成／指定Reactor BO超时 | 3／0 | 3／0 |
| Factory Tech Lab完成 | 3 | 3 |
| Marine实际出生／死亡 | 249／249 | 268／268 |
| Tank实际出生／死亡 | 31／31 | 38／38 |
| Marauder实际出生／死亡 | 16／16 | 20／20 |
| SCV累计出生／死亡（含开局12个） | 56／56 | 56／55 |
| 终局矿／气 | 2870／715 | 2095／232 |

SpB最后仍有1个SCV，但SC2已报告自然Defeat；不能用“单位没有全部死光”否定引擎
终局。两组都未产生Viking/Medivac及实质性战斗升级。SpB的38辆Tank中19辆被
Battlecruiser击杀，另有268名Marine全部损失；与Qwen同样存在严重战斗损耗。
SpB后期有两个siegetank BO超时（frame6644、6958），不是Reactor修复回退；
本次只记录，未定位这两个后期超时的具体资源/建筑条件，未作新修改。

首轮实际SpB回答如下；失败原因明确是非canonical名称 `siege tank`，不是标题行。
下一轮改用 `siegetank` 即被判有效。`policy_action_applied=true` 表示回答已送入
游戏，不等于整份回答合格或每个动作最终完成。

```text
Here are the Build Orders with attack instructions:
- supplydepot
- mule
- marauder
- siege tank
- marauder
- marauder
```

**结论：模型对照执行完成，未获得Level6胜率提升证据。** SpB这局坚持更久、
累计产量更高，但依然败局且回答有效率更低；不能用时长/累计产量证明更强，
也不能用两次单seed纯评测判断RL学习或架构优势。两个生产可靠性修复在SpB上
同样工作；“完全没有产出”不符合这次回放。下一项仍应从已保存的交战时序、
阶段待办与模型选择核对对空配兵及战斗损耗，不自动开长训或改策略。

SpB资产取证只打包model_preparation元数据、日志和回放，没有复制run/model中的
权重软链。00:26:30核验GPU0已释放、专属SC2/tmux已退出、两个冻结deployment仍clean。
已复核原模型manifest、相同BO哈希、两组实际effective config、首轮query、
累计reward/真实决策计数及回放/console哈希；首次CPU导入检查失败记录上文保留。
日志目录的 `Qwen3-30B` 是旧静态标签，实际身份以manifest及SSESWAMoBA加载记录为准。
