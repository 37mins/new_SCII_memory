# r4 子进程 RNG：两分钟固定 BO 复现验证

2026-09-07，状态 completed；用户批准的两局 CPU-only 短测已结束，无模型调用、GPU 占用。
依据见 [固定 BO 分岔审计](r4-fixed-bo-divergence.md)。不是 KL 诊断或胜率实验。

## 唯一候选与冻结边界

在 r4 `ca7885d` 的 `envSC2.py::_start_env_a_process` 创建 EnvironmentA 前，
若 `SC2_SEED_PYTHON_RANDOM=1`，执行 `random.seed(seed)`。默认 0 完全保留原行为。
这是让侦察/选址现有 Python random 使用已有 episode seed，不改变选址算法。
候选复用 `codex/fix-unit-id-exhaustion` 的独立工作树；此前提交与 r4 的
agent_system/configs 无运行代码差异。原 r4 deployment、run 和结果不修改。
固定 BO、Planner、解析器、微操、reward 和训练数值路径均不改；无模型初始化。

## 配置与验收

44 节点，Flat128、Terran vs Terran、Harder/Level6、Air、canonical g2（237 项），
两局均 seed `2621640384`，开关为 1。这是一项候选的同配置重复，**不是 on/off A/B**。
预算仅两局，各游戏时间 120 秒、墙钟硬限 300 秒，串行；失败即停止、不自动重试。
复用原 Python3.12.3 / burnysc2 7.1.3 和 SC2 资产，不部署新依赖。

薄入口只调用真实 `SC2Env.reset(seed)`，接收截止时 terminal 并 ACK，永不发送
ACTION；提前收到 Planner 请求直接报错。用已有 game_time_limit 截止，不改终止实现。
其原始 Tie/reward 只存档，不解释为自然败局或用于训练。两个独立日志根避免串局。

比较相同 iteration 的资源快照、真实建筑坐标和 BO 开始/命令帧号；不以占位 [1,1]
作为落点，不要求跨局 SC2 tag 相同。全部符合才支持“这个 seed 的两分钟前缀可复现”；
不同则保留首处分岔证据。两局不能证明完整游戏确定性、RL 改善或历史败局根因。

RUN_ID 为 `20260906-integrated-fixes-v1-r4-python-rng-prefix120-s2621640384-repeat{1,2}-r1`，
分别写入现有 integrated-fixes-v1/runs。冻结 SHA、effective config、起止时间、
退出码、driver、trace、replay 和哈希跟随 run 保存。

```bash
bash experiments/records/20260906-integrated-fixes-v1/run_r4_rng_prefix_node44.sh
python3 -m unittest discover -s tests -p 'test_sc2_child_seed.py'
```

CPU 验证覆盖开关关闭时随机流不变、开启后在不同父进程随机状态下同 seed 相同、
不同 seed 不同，实际调用子进程入口方法而非另一套实现。该 unittest 2/2 通过；
另跑 test_r6_natural_termination.py 8/8、test_empty_bo_frame.py 6/6 通过。
launcher 的 bash -n、源码 manifest 与 git diff --check 通过；44 上 seed 测试 2/2。

## 实际结果

冻结源码 **`7aad7773e993417cde5ead3484aa6433e7788a83`**，已在 private origin 保存。
44 部署 `/root/new-scii-r4-python-rng-20260907/code-7aad777`，tmux `sc2_r4_rng_prefix`。
repeat1 北京20:02:04–20:02:45，repeat2 20:02:45–20:03:20，均 `EXIT_CODE=0`。
20:06核验两局进程和本任务tmux均已退出，候选和原r4均clean，未启动更多游戏/RL。

| 核验项 | 两次结果 |
| --- | --- |
| 完整 trace | 87 个记录帧逐字段相同；两个文件 SHA-256 都为 `b28d175c1a639c19f52372260c3e58be6e7ecb46c66b0e06fdac84838e1b2df1` |
| 资源快照/完整观测 | 68 组 / 12 组，全部相同 |
| BO 开始记录 / 提交命令 | 13 项 / 12 次，时刻、内容相同；最后 orbitalcommand 仍等待中 |
| 首座供给站 | 命令iteration112；135帧观测真实落点 `(29,23)`、施工3%，相同 |
| 旧分岔点 BO#9 barracks | 两次均iteration537提交，无原先515/537差异 |
| 终点 | 最后回调iteration671；上次时钟显示01:59，下一游戏步触发120秒上限 |
| 模型参与 | 两次0次generate_plan、0个回答；terminal的消费动作ID均为null |

effective config 仅两个输出路径不同。日志中的 `Qwen3-30B` 是原 EnvironmentA
固定的日志名称，**没有加载该模型，也没有加载 Qwen/SpB 权重**。
结束后只读核验地图、SC2 Base75689、g2 BO 哈希，与冻结的
`scripts/verify_assets.py` 登记逐字一致，分别为：
`3916dbd2bb662a6e9c9bd27089906630c8b0a42fdbab51fc4401cf1df7b9b212`、
`115f65a49191c504f6336f7d5b7818b23dbba3d7a45cacf9142c86c959abaa41`、
`304b1f4ff12fe30c5e6d5f7b8eaef8ed6823ca65afed3a3706cae10e3639c494`。

结论：**该候选在这个seed的两个短前缀中通过复现检查**。旧日志显示同seed早期已
分岔，补齐Python RNG后本次未再出现；这支持它是需要修正的复现性缺口。
但没有做同期on/off A/B，也没有观测首次Planner前完整十分钟或战斗，不能宣称
整局败因已定位、RL退化被解释或KL问题修好。开关仍默认关闭，未并入当前RL运行。

审计脚本 [audit_r4_rng_prefix.py](audit_r4_rng_prefix.py) 只读原产物；
[JSON结果](r4-python-rng-prefix.json) 保存逐项BO、资源、真实建筑观测和产物哈希。
提取命令为 `ssh sc2-a100-44 'CUDA_VISIBLE_DEVICES="" timeout 30s /usr/bin/python3 -B -'
< experiments/records/20260906-integrated-fixes-v1/audit_r4_rng_prefix.py`。
首次返回长度不足后完整重新提取，未重复游戏。若继续，下一项应单独批准延长同一
复现验证到首次Planner前；不由此改PPO、战术或直接续96步。

## 完整固定 BO 扩展（09-07追加授权，completed；未到达Planner）

用户已批准验证首次Planner前的完整阶段，复用同一record/branch。游戏源码及
configs仍逐字等于`7aad777`，只扩展测试入口 `first-planner` 模式，命令为：

```bash
bash experiments/records/20260906-integrated-fixes-v1/run_r4_rng_prefix_node44.sh first-planner
```

仍在44 CPU串行两局、同seed/同g2/Level6/同RNG开关，不加载模型。RUN_ID中的
`prefix120`改为`preplanner`，不复用旧目录。每局墙钟上限600秒、游戏上限1200秒；
首次Planner请求即截停，或提前游戏结束则单独记录未到达边界。不自动重试。

测试专用Player仅在构造完成后包装Planner的发送函数：发送首个非terminal包前
保存完整内存trace、BO索引、游戏帧/时间、原始query，再原样发送；不改协议载荷，
不消费RNG、不推进游戏。父进程收到query后使用原有owned-process清理停止该局，
不发送ACTION，不伪造terminal/reward。此处是诊断截点，不是自然游戏结局；中途
关闭不保证产生可播放replay，以boundary_trace和原始日志为完整前缀权威证据。

验收要点是两局均遍历237项并到达首次请求；逐帧比较资源、观测/真实建筑位置、
命令与进攻事件，比较首次query和接管状态。完成/跳过BO不等于建筑/单位已建成。
若分岔，明确首次差异；若完全一致，只支持这个seed的接管前复现，不判断胜率、
学习或KL。原r4、两分钟冻结部署和旧产物保持不动。

实际从clean/pushed SHA `5b24db566007b514686f61242ad999b5d296725b` 启动；44的
独立目录为 `/root/new-scii-r4-python-rng-20260907/code-5b24db5`。北京20:18:49
tmux `sc2_r4_rng_preplanner` 启动，20:20:59已确认SC2实际运行到06:11/BO#98，
无模型调用。入口截点回归2/2、RNG回归2/2通过，远端截点回归2/2通过，
`git diff 7aad777 -- agent_system configs`为空；不是新的游戏行为候选。

两局分别于北京20:18:49–20:23:09、20:23:09–20:27:35运行，退出码均0，
状态均为 `ended_before_first_planner`。**都在11:26自然Defeat，尚未走完固定BO；
不能把程序正常退出说成“完整固定BO→Planner接管验收通过”。** 无额外游戏或RL。

| 指标 | 两局共同结果 |
| --- | --- |
| BO进度 | 启动173项；0–171共172项标记完成/跳过；#172 supplydepot未完成；总共237项 |
| 提交命令 | 149次，内容与帧号完全相同；提交不代表建成 |
| 超时跳过 | #110 reactor barracks，#165–167 siegetank；两局一致 |
| 自动进攻 | 首波20，后续10、10、11；事件及最近记录帧均相同，后者不是精确触发帧 |
| 工人/军队 | 09:20的同帧快照两者均0；09:43为工人0、军队人口1 |
| 最后BO | 09:36开始#172，反复`[Wait] Action blocked: **Action Failed**: Missing facility.` |
| 终局 | 11:26，0工人/0军队，4000矿/1207气，reward -50，自然Defeat而非时间上限 |
| Planner | 两局0个generate_plan，0回答；无首次query可比，消费动作ID为null |

529个trace记录帧中，328组数值快照全部相同；BO时间线和命令相同。原始trace
并非逐字相同：从iteration2311（约06:52）起，52组`Visible enemy structures`
存在显示单位编号差异，如相同位置`(24,101)`的OrbitalCommand分别编号747/101。
**仅屏蔽`[数字]单位名称`中的标签后，完整trace相同**，未屏蔽坐标、类型、数量、
健康或命令中的ID。两局前670帧范围的86个记录帧也都与各自两分钟实验相同。
这支持实际固定BO游戏轨迹复现，不声称所有原始文本字节相同；也没有验证这些
显示标签对后续模型的影响，因为此次根本未到达Planner。

冻结`base_player.py:156–158`在单位或townhall为空时不再运行Planner/Executor循环；
因此正常游戏可继续到自然Defeat，而BO不再推进。该规则未改。此次09:43后周期
快照停止，终局来自SC2；不把缺失的中间快照补成观测，不把最后等待单独定为败因。

**判定：实际轨迹复现检查有支持，完整BO到Planner的接管验收未通过。** 本次只
证明该seed、g2/Level6组合在模型接管前就会失败，不能归因于Qwen/SpB或KL，
也不能据两局推导所有seed必败。首次Planner截点记录仍只有CPU测试、未到达实测。
下一步应在这些已有回放中定位固定BO/自动战斗如何丢失兵力与工人；不得因此
直接改战术、提前接管或续训。显示编号差异保留为事实，不作为新的修改任务。

[完整阶段JSON](r4-python-rng-preplanner.json)含真实终局、跳过项、首处原始差异、
清理状态、回放和trace哈希；[审计脚本](audit_r4_preplanner.py)仅读取产物。
两局自然结束，均保存了原生replay；owned-process清理记录均无残留。审计先区分
含观测文本的trace行与纯数值字段，并按源码真实`bo_skipped_`键核对跳过项；
这些是分析脚本口径核验，不是新的项目bug修复。

09-07追加[回放损失定位](r4-preplanner-replay-losses.md)：两份tracker事件流相同。
共74 Marine和41 SCV全部死亡；自动波次中两波未正式出击就取消/全灭。
三次Factory Tech Lab本地标完成但回放无开工，Tank连续超时；另有工人防御的
全场军队人口门槛。见对应事件账本，未修改运行代码或增加实验。
