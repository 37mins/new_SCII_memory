# 待开工生产建筑的挂件保护：单点候选

2026-09-07，用户批准实施。**CPU 回归及同 seed 的 Level 6/g2 单局生产链
验收已通过：3 个 Factory Tech Lab 完成，3 辆不同坦克实际出现，走完固定 BO。
证据为 SC2 实际观测；首次 Planner 截点未产生完整回放，不是胜率或 RL 验收。**

## 修改与回滚

在现有独立工作树 `/private/tmp/new-scii-unit-id-exhaustion`、分支
`codex/fix-unit-id-exhaustion` 修改。修改前完整备份是已推送的 Git 提交
`e18fe515a173111edc8a1796c66b6c6d5c95344e`；其中游戏源码与两份失败回放的
冻结 `5b24db566007b514686f61242ad999b5d296725b` 一致。原 r4/r6、服务器
冻结 deployment 和既有 run 均未改写。回滚使用该提交的新工作树，不覆盖当前源码。

唯一运行时修改位于
`agent_system/environments/env_package/sc2_online_restored/SC2ArenaDyStrategy/players/base_player.py`
的 `BasePlayer.find_placement`：新增 9 行，从当前 SCV 订单中识别
`TERRANBUILD_BARRACKS/FACTORY/STARPORT`，把 Point2 建造目标的挂件中心纳入
原有保护列表。已有建筑保护、偏移 `(2.5,-0.5)`、距离 `4.5`、矿路过滤、
引擎选址查询及随机选择规则均保持原样。

没有新配置开关或持久化预留表。此候选沿用原参数
`avoid_production_addons=True`；显式 False 时仍跳过保护。订单取消、工人死亡
或订单消失后不会留下历史预留；已开工建筑继续由 structures 保护。
找不到位置时仍返回 None，由现有调用者处理，不强行放置。

只改变选址输入来源；**不改变执行者选择、BO 完成/扣减语义、固定 BO、
Planner/prompt、战斗/微操、reward、生成和训练配置**。同步更新
`configs/system-v1.sha256` 的 base_player 条目，不改其余冻结哈希。

## 依据

[原始证据](r4-factory-techlab-cause.md)定位到：Factory 下令时还未进入
structures，随后 SupplyDepot 抢占其挂件区。不是因单局败北猜测代码有错。

实际依赖 [burnysc2 7.1.3](https://pypi.org/project/burnysc2/7.1.3/) 的
`sc2/bot_ai_internal.py::_worker_orders`（457 行起）也通过
`self.workers → worker.orders → order.ability.exact_id` 识别待建订单；
`sc2/unit.py::UnitOrder.from_proto`（88 行起）把位置目标解为 Point2，
单位目标解为整数 tag。两份实际库文件的哈希仍匹配前次核验记录。
公开[订单实现](https://burnysc2.github.io/python-sc2/_modules/sc2/bot_ai_internal.html)
可作辅助查阅，版本依据是实际安装的 7.1.3，不把滚动文档当作冻结版本。

本修改沿用这些订单语义和项目原有选址规则；没有引入新的战术或选址算法，
也不声称它逐字复制了某个公开 bot 的完整实现。

## 已执行验证

[CPU 原始结果](r4-pending-addon-cpu.json)记录输入源码/测试哈希和测试项。
先在本地 7.3.0 复现，再在 **44 实际 Python 3.12 / burnysc2 7.1.3** 运行
同一组测试、对照同一冻结原方法：旧版 4 个测试方法不满足新断言
（包括 3 个兵种子用例，共 6 条失败）；候选 **9/9 通过、0 跳过**。
这些失败均是同一个待开工保护遗漏的不同断言，不算 6 个独立 bug。

核验覆盖：原 Factory/Depot 坐标冲突；三种生产建筑及排队订单；搜索其他安全
位置；订单消失后的释放；非生产订单和单位 tag 目标；现有建筑保护；显式
关闭保护；无待建订单时原位置和随机状态；引擎拒绝时不绕过检查。
测试使用库的真实 UnitOrder、AbilityData、Point2；引擎选址和矿路查询是 stub。

本地重现命令（已有含 burnysc2 的环境，不安装/升级共享依赖）：

```bash
/private/tmp/sc2-planner-executor-venv/bin/python -B -m unittest discover \
  -s tests -p test_pending_addon_placement.py -v
```

同一命令分别将 `-p` 改为 `test_system_release.py`、`test_empty_bo_frame.py`、
`test_r4_replay_loss_record.py`、`test_sc2_child_seed.py`，实际分别通过
26/26、6/6、7/7、2/2；加新增 9 项，本地合计 50/50，无跳过。
`git diff --check` 通过；这不是全仓库或分布式训练回归。

44 核验把上述测试与候选文件通过 tar/stdin 读入内存，再由真实依赖执行。
仅替换测试读取的 PLAYER 源码，基线从原冻结目录只读取得并核验哈希；不将
候选部署进旧目录。CUDA 不可见、硬限 20 秒，退出 0；没有模型加载、游戏
或服务器文件写入。首次 SSH 被本地沙箱拒绝，随后一次 tar 传输包含 macOS
扩展元数据而在解码前失败；排除扩展元数据后完成。二者均为测试传输失败，
不是游戏结果或项目原有 bug 的修复成果。

## 仍未验证

只保护当前观测里已有的 worker.orders；不新增同帧尚未发送命令的预留机制。
不会移走已经堵住挂件的补给站，也不修复三次选择同一工厂和本地提前标完成。
选址变化会改变后续游戏轨迹，不能由 CPU 回归推断坦克产量或胜率。

CPU 阶段提出的一局同配置固定 g2 / Level 6 / seed 2621640384 实局验收，
此后已获追加授权并完成，见下节；不将它混入上面的 CPU 回归结果。

## 同 seed 实局验收（09-07 追加授权）

启动前状态 planned。用户批准一局 Level 6 固定 g2 验证。以 `5871ab6` 的修复源码为
候选，对照仍用两份已有 `5b24db5` 失败回放，不重复基线。新增提交仅为实验
编排/记录；与旧版相比的运行机制变化仍只有待开工挂件空间保护。

44 节点，CPU-only、无模型/GPU/PPO；沿用实际依赖与 SC2 Base75689、Flat128、
Terran vs Terran Air、Harder、g2 共 237 项、seed `2621640384`、
`SC2_SEED_PYTHON_RANDOM=1`。墙钟硬限 600 秒、游戏时间上限 1200 秒，不自动
重试；自然结束或首次 Planner 请求即结束诊断，不提供模型回答。

RUN_ID：`20260906-integrated-fixes-v1-r4-python-rng-pending-addon-g2-l6-s2621640384-repeat1-r1`。
使用现有 integrated-fixes-v1/runs 下的新目录；冻结源码部署在已登记的
`/root/new-scii-r4-python-rng-20260907/code-<SHA>` 新路径，不改旧目录。

复用已有入口，仅增加重复编号列表和标签覆盖（默认仍是旧两局），以防覆盖旧 run：

```bash
SC2_RNG_PREFIX_REPEATS=1 SC2_RNG_PREFIX_RUN_TAG=pending-addon-g2-l6 \
  bash experiments/records/20260906-integrated-fixes-v1/run_r4_rng_prefix_node44.sh first-planner
```

主要验收：SupplyDepot 不再占已下令工厂的挂件区，至少一次 Factory Tech Lab
真实开工并完成，至少一辆 SiegeTank 真正产出。优先以 replay tracker 事件为
依据，辅以 trace/原始观测；不把 BO 标完成或命令提交当成生产成功。
若先到 Planner 截点可能没有完整 replay，必须报告这个证据边界，不伪造终局。
胜负/存活时间作为次要观察，单局不证明稳定胜率或 RL 学习改善。

### 完成结果

冻结 SHA `836460a8533b6f0429f9e9d401ac9715adc55f5b`，private origin 已保存。
44 源码 `/root/new-scii-r4-python-rng-20260907/code-836460a`；tmux
`sc2_r4_pending_addon`。北京 **21:44:07–21:48:17**，250 秒，exit 0。
运行机制与 `5871ab6` 无差异；与旧组的 effective config 比较只有源码/输出路径
及单次运行编排字段不同。固定 BO、地图与 SC2 二进制哈希均核对。

| 验收项 | 旧同 seed，两局相同 | 修复后这一局 |
| --- | --- | --- |
| Factory Tech Lab 实际施工 | 回放中 0 | 07:38–07:39 的观测出现 3 个施工实体 |
| Factory Tech Lab 完成 | 回放中 0 | 07:57 观测到 3 个 400/400、idle 的实体 |
| 坦克实际出场 | 回放中 0 | 08:34、09:08、09:34 先后观测到 3 个不同坦克；最多同时 2 辆 |
| 固定 BO 进度 | 172/237 标完成/跳过，随后自然败 | 237/237 标完成/跳过，10:00 到达首次 Planner |
| 三项初始 Tank BO | #165–167 均 Tech Lab required 超时 | 均提交生产；这些提交本身不作为产出证据 |

上述时间是**首次记录到该状态的游戏时间**，不是精确开工/出生事件时间。
原始片段（分别来自 iteration 2567、2673、2882）：

```text
[548]FactoryTechLab(building 1%)
Position: (26, 35)

[548]FactoryTechLab
Position: (26, 35)
Health: 400/400 (100%)
State: idle

[27]SiegeTank
Position: (37, 47)
Health: 175/175 (100%)
State: moving to [(38, 52)]
```

这是 `Own structures` / `Own units` 中的真实实体，不是 production list、
可用动作表或模型回答。冻结 `BasePlayer.unit_to_text` 只有在
`unit.build_progress == 1.0` 才输出不带 building 的名字及完整生命值。
同局 `tag_to_id` 映射不删除、不重新分配，因此 `[27]`、`[579]`、`[251]`
确属三个不同坦克；“至少 3 辆出场”不冒充完整累计出生/死亡事件账本。

BO#123/#129/#130 这次分别选到 Factory `[235]`、`[562]`、`[554]`；对应
挂件 `[548]`、`[267]`、`[698]` 均完成。没有修改选择器；选址变化改变了
后续游戏轨迹，不能称为把旧回放同一个工厂原位修好后逐帧继续。

**判定：本次同 seed 生产链验收通过，支持这项最小修复有实际作用。**
仍有两个 reactor barracks 超时（#110/#114），也没有改 BO 提前完成、
生产队列分配或波次战斗。本次有进展，不等于所有下游问题已解决。

按预定首次 Planner 截点停止：保存完整 `boundary_trace.json` 和原始 query，
0 模型回答、0 PPO 更新，无自然胜负及完整 replay。不能报为 1/1 胜，
也不能提供不存在的 SUnitInit/SUnitDone 事件。进程清理记录无残留，原/新
deployment 均 clean，原始 run 未覆盖，没有排第二局或 RL。

[机器可读结果](r4-pending-addon-game.json)保存逐项原始观测、生产命令、截点、
配置对比及十项产物哈希；服务器与本地副本十项逐一匹配。
[审计脚本](audit_r4_pending_addon_game.py)实际 exit 0，启动器 `bash -n`、
新增选址回归 9/9 与截点回归 2/2 均通过；未修改冻结游戏源码。
本地原始产物位于
`/private/tmp/sc2-pending-addon-deploy.sB11Ui/20260906-integrated-fixes-v1-r4-python-rng-pending-addon-g2-l6-s2621640384-repeat1-r1`。
复核命令为 `python -B experiments/records/20260906-integrated-fixes-v1/audit_r4_pending_addon_game.py <原始产物目录>`。
