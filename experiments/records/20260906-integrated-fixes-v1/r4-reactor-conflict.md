# Marine 排兵占用与 Reactor 超时：单点修复

2026-09-07。用户批准查清并修改。**CPU 与同 seed Level6/g2 单局机制验收通过：
3 个 Reactor 实际完成、0 BO 超时，保留3个完整Factory Tech Lab和至少4辆坦克。
到首次Planner即止，不能称为胜率或RL验收通过。**

## 证据与判断

依据是已完成的同 seed Level 6/g2 挂件选址验收，不是当前工作树猜测：
runtime `836460a8533b6f0429f9e9d401ac9715adc55f5b`，
RUN_ID `20260906-integrated-fixes-v1-r4-python-rng-pending-addon-g2-l6-s2621640384-repeat1-r1`。
完整位置和结果见[上一项验收](r4-pending-addon-fix.md)。
[提取记录](r4-reactor-conflict.json)保存原始观测、命令、失败帧和文件哈希。

| 实际时点 | 观测与提交 | 含义 |
| --- | --- | --- |
| frame 2095，BO #93 | `[849]` 已排 Marine×2，`[170]` 等带 Tech Lab 兵营 idle；仍选 `[849]` | 并非没有其他能造 Marine 的建筑；引擎能力表同时列出它们 |
| frame 2111，#110 前 | 唯二无挂件兵营 `[849]`、`[425]` 均 Marine×5；另有五个 Tech Lab 兵营和一个正在造 Reactor 的兵营 | 两个可供新挂件的生产者都被排兵占用 |
| #110，2112→2332 | `BUILD_REACTOR_BARRACKS not currently available`，11 次计数后跳过 | 等待约 39.29 游戏秒；未提交 Reactor 命令 |
| frame 2333/2334，#111/#112 | `[849]` 尚余 2/3 个 Marine，仍连续选它；`[387]` 已有完整 Reactor 且 idle | 首次超时后，排兵又延长了无挂件兵营的占用 |
| #114，2336→2556 | 相同能力不可用并超时；下一帧两个无挂件兵营仍分别有 2/1 个 Marine | 又等约 39.29 秒，没有恢复到可建挂件的空闲状态 |

原始输出示例：

```text
[849]Barracks
Production list: Marine, Marine

[387]Barracks
State: idle
This structure has an add-on: [122]BarracksReactor

llm_decision_final_action_111_2333:
[{'action': 'BARRACKSTRAIN_MARINE', 'units': [849]}]
```

冻结 `BasePlayer.units_to_text_wo_contructing`（941 起）按距离和名字排序；
`MicroAgent._fallback_select_executor`（98 起）直接取首个 ID，完全不看订单。
`LLMPlayer.run`（823 起）只检查引擎是否提供能力，**能继续排队不代表有空
生产槽**。当前 `SC2_LOCAL_EXECUTOR_FALLBACK=1`，选择走该本地路径；
`SC2_AUTO_TRAIN_MARINES=0`，上表新增订单来自固定 BO，不是自动补兵。

这确认了忙碌生产者被反复选择、空闲生产者被闲置的实现缺口，强烈支持它造成
Reactor 占用冲突。两段等待内没有完整单位观测和引擎 ActionResult，不能声称
逐次证明了唯一失败原因、排除了所有挂件空间问题。#91 的 Reactor 已实际建成，
也不能将其描述为 Reactor 映射/能力全局失效。

## 最小修改与公开依据

只改 `players/llm_player.py::LLMPlayer.run` 的候选过滤（900 起）：在
**本地 fallback + BARRACKSTRAIN_MARINE** 时，保留引擎合格候选中
`len(orders) < 1 + int(add_on_tag in reactor_tags)` 的兵营。
即普通/Tech Lab 兵营最多占一个生产槽，Reactor 兵营可占两个。
空槽候选之间仍用原距离排序；不增加 Reactor 优先级或未来 BO 预留算法。

该槽位条件直接对应实际依赖 **burnysc2 7.1.3** 的
`sc2/bot_ai.py::BotAI.train` 953、958、1040–1044 行：默认
`train_only_idle_buildings=True`，同一条件判断是否有可用槽位。
[7.1.3 官方发行包](https://files.pythonhosted.org/packages/46/3c/62a54a34eae13906672ab5a0c101cae71687b92a2f8b77da3d4d8249c76c/burnysc2-7.1.3-py3-none-any.whl)
与 44 实际 `bot_ai.py` 哈希均已核验，见 CPU 记录；
[公开源码文档](https://burnysc2.github.io/python-sc2/_modules/sc2/bot_ai.html#BotAI.train)
仅作辅助查阅，不冒充冻结版本。这里只复用槽位条件，不替换整个 BotAI.train。

无空槽时沿用现有重试/超时；本次等待不提交命令、不提前完成 BO，下一观测再查。
**没有取消已有排兵、延长超时、重排或增删 BO，也不保证永远预留空兵营。**
不改 SCV、Tank、Marauder 等其他能力的选择；Reactor/Tech Lab 指令、模型侧
prompt、非本地 Executor、奖励、战斗/微操、训练配置均冻结。
上一项待开工挂件空间修复原样保留。本修复会改变排兵时序，必须用实局验收收益。

## 验证与边界

[CPU 记录](r4-reactor-cpu.json)：44 的 Python 3.12.3 / burnysc2 7.1.3，
在内存中对照新方法与冻结原方法，无模型/GPU/游戏、无服务器文件改写，硬限20秒。
候选 **8/8 方法通过，无跳过**；旧方法 5 个方法不满足新断言（含4个真实快照，
共8条断言失败），不是8个独立 bug。四个相同观测的新旧选择分别为
`849→170`、`170→17`、`849→387`、`849→387`。

测试运行真实 run/文本排序/本地选择/最终动作组装方法，以及库的 Units、UnitOrder。
单位位置/占用取自记录；引擎能力查询、提交和未来观测为测试替身，不能当作实局。
还覆盖 Reactor 第二槽、全忙等待再执行、能力与资源检查不绕过、无取消、非目标
能力和非本地选择保持原样。首次测试夹具遗漏存活工人，on_step 正确提前返回；
补齐夹具后重测。这是测试自身返工，不是项目新 bug 或修复成果。

本地命令：

```bash
/private/tmp/sc2-planner-executor-venv/bin/python -B -m unittest discover \
  -s tests -p test_barracks_production_slots.py -v
```

只读证据重现：`python -B experiments/records/20260906-integrated-fixes-v1/audit_r4_reactor_conflict.py <原run副本目录>`。
本地运行依赖7.3.0与实际服务器7.1.3明确分开；不升级任何环境。
本地同一 loader 还执行挂件选址9、空BO6、冻结配置/启动器26、回放记录7、
子进程seed2、首次Planner截点2项，加新增8项，共 **60/60通过、0跳过**。
各测试模式、原始测试输出存于CPU记录；`git diff --check`通过，未声称全仓回归。

## 备份与下一项

沿用独立 `codex/fix-unit-id-exhaustion`，修改前已推送的完整备份
`bbcccfea07f4ed8427c9ed96237c7c308150867d`。原 r4/r6、deployment、历史产物和
主工作树未改写。候选由本次原子提交冻结；哈希清单仅更新 llm_player 项。
回滚从 `bbcccfe` 建立新的运行目录，不改写旧 run。

CPU 阶段仅完成上述回归，不据此称 Reactor 实局已修好或合入正式基线。
当时建议下一项获批后仅跑一局相同 Level6/g2/seed2621640384，无模型/GPU，墙钟
最多600秒，到首次Planner或自然终止结束；与上一项已有 run 对照，不重复基线。
检查三项 Reactor BO 是否真正开工/完成、是否新增排兵超时、Tech Lab/Tank 是否
仍正常；保留同口径观测和命令，不用“BO 标完成”冒充实际生产，不直接启动 RL。

## 同 seed 单局验收（09-07 追加批准）

启动前状态 `planned`。用户批准上述一局、无GPU、墙钟上限600秒。游戏源码冻结为
`c15d02b08085ddfcf20cff5517d10ceae9fa2af0`，不将后续记录提交混作runtime SHA。
44 新目录 `/root/new-scii-r4-python-rng-20260907/code-c15d02b`，不改原目录。
RUN_ID：`20260906-integrated-fixes-v1-r4-python-rng-reactor-slots-g2-l6-s2621640384-repeat1-r1`。
运行仍位于既有 integrated-fixes-v1/runs 根；使用原依赖/SC2 Base75689/Flat128，
Terran vs Terran Air、Harder、g2 237项、seed2621640384、Python随机数同seed。

唯一游戏机制差异是本地 Marine 空槽过滤；上一项挂件保护保留，其余冻结。
对照用已有 `836460a` 单局，不重复基线，不接模型、不做PPO或长训，不自动重试。
自然终止或首次Planner请求即结束；SC2时间上限1200秒。原入口：

```bash
SC2_RNG_PREFIX_REPEATS=1 SC2_RNG_PREFIX_RUN_TAG=reactor-slots-g2-l6 \
  bash experiments/records/20260906-integrated-fixes-v1/run_r4_rng_prefix_node44.sh first-planner
```

验收要求：观察三项 Barracks Reactor 实体开工与完成、相关 BO 超时数；检查是否
新增 Marine 超时、Factory Tech Lab 和Tank实体是否仍出现。原始观测/完整回放
优先于提交标记；若在首次Planner截点没有完整回放，按同口径观测报告，不编造
出生事件。单局和截点测试不作胜率提升、RL或架构结论；保留配置、日志、退出码。

### 完成结果

状态 `completed`，**exit0**。44 节点、tmux `sc2_r4_reactor_slots`，北京
**22:42:50–22:47:06**，256秒。实际runtime为上文 `c15d02b`，计划记录提交
`4dadcdb` 只更新文档，没有进入或改变游戏源码。与对照的effective config逐项
一致，仅源码路径、两个日志路径和run标签共5个字段变化；Python3.12.3 /
burnysc2 7.1.3、BO/地图/SC2哈希相同。

| 指标 | 对照：已修挂件选址，836460a | 本次：再加Marine空槽过滤，c15d02b |
| --- | --- | --- |
| 实际观察到完成的 Barracks Reactor | 1 | **3** |
| Reactor BO #110 / #114 | 均超时，没有提交 | 均提交并有对应完整挂件实体 |
| 全部 BO 超时数 | 2 | **0**，没有新增Marine超时 |
| Marine命令数 | 103 | 103 |
| 提交Marine时已满生产槽仍继续堆单 | 93/103 | **0/103** |
| 提交Marine前所选建筑最大已有生产订单数 | 7 | 1（Reactor可用第二槽） |
| 实际观察到完成的 Factory Tech Lab | 3 | 3 |
| 实际观察到的不同坦克 | 至少3辆 | 至少4辆 |
| 走完237项BO、到首次Planner | 10:00.18 | 10:11.96 |

Reactor #91/#110/#114 分别提交给兵营 `[858]`/`[938]`/`[114]`，对应实体
`[538]`/`[426]`/`[892]`。首次施工观测为05:26、06:13、06:22；首次完整观测
为06:04、06:49、06:58。来源是SC2的Own structures，不是计划文本：

```text
[426]BarracksReactor(building 0%)
Position: (20, 37)

[426]BarracksReactor
Position: (20, 37)
Health: 400/400 (100%)
State: idle
```

时间均为首次记录到该状态，不是精确开工/完成事件。两个截点时间不同，4辆与
3辆坦克本身不能单独证明产能或胜率提升。Tank的9条命令仍选同一工厂，未在
本次修改其选择；BO本地提前完成语义、微操/战斗与RL数值问题也未一并解决。

**判定：这一个seed上的占用冲突修复验收通过**，支持冻结此候选继续验证。
没有扩大到其他seed，没有合入正式基线。0模型回答/0PPO，10:11截点主动关闭；
没有自然胜负或完整回放，不能算1/1胜。cleanup确认无残留进程，原/新源码均clean。

[机器记录](r4-reactor-game.json)保存原始实体、提交命令、对照摘要、配置差异、
10项服务端/本地匹配哈希和准备阶段失败说明。[审计脚本](audit_r4_reactor_game.py)
实际对两组运行exit0；交叉核对旧组1/3/3实体和两个超时、新组3/3/4实体及零超时
均通过。`bash -n`、`git diff --check`通过；本轮没有再次改游戏代码。
本地全量产物：`/private/tmp/sc2-reactor-acceptance.dgj77F/20260906-integrated-fixes-v1-r4-python-rng-reactor-slots-g2-l6-s2621640384-repeat1-r1`。
仅跑这一局，没有自动重试、续局、模型评测或RL任务。
