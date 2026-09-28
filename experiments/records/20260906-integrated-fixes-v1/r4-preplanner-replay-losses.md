# r4 固定 BO／自动战斗损失定位

2026-09-07，只读分析完成；没有新开游戏、占用 GPU 或修改运行代码。

**结论：这两局在模型接管前，已经出现“挂件未落地→Tank 生产阻塞”和
“小波次在前线集结／交战中被消耗→敌军推进矿区→工人被清空”。**
不是模型回答导致的进攻，也不是生产完全没有产出。尚不能量化这两条链各自
贡献多少败率，更不能把它们当成 r4 KL 漂移的解释。

## 样本与口径

来源是[已完成的两次完整固定 BO 验证](r4-python-rng-prefix.md#完整固定-bo-扩展09-07追加授权completed未到达planner)，
RUN_ID 为 `20260906-integrated-fixes-v1-r4-python-rng-preplanner-s2621640384-repeat{1,2}-r1`。
44 节点、Level6/Harder、Air、Flat128、g2 共237项、同 seed；两局均11:26自然败，
没有调用 Planner。不是 Qwen/SpB 对照，也不是两个独立随机样本。

冻结源码 `5b24db566007b514686f61242ad999b5d296725b`，部署
`/root/new-scii-r4-python-rng-20260907/code-5b24db5`。相对 r4 `ca7885d`，
游戏代码只多子进程 Python RNG 开关；本次开启它，其他游戏行为保持原样。
以下行号均核对该冻结源码，而非推测历史运行使用最新工作树。

用[暴雪官方 s2protocol](https://github.com/Blizzard/s2protocol)的 build75689
解码器读取出生、开工、死亡、击杀者及聊天事件。时间采用与运行日志一致的
22.4 game loops/秒；不使用 replay metadata 的16 loops/秒时长。
两份 replay 文件 SHA 不同，但 tracker、message 两个事件流分别逐字相同；
只分析一次共同轨迹。game events 未宣称逐字相同。

## 兵力如何损失

| 游戏时间 | 直接证据 | 对应行为 |
| --- | --- | --- |
| 05:59→06:13 | 首波20兵先往 `(28,73)` 集结，再向敌方分矿 `(24,101)` 出击 | 自动阈值触发，不是 BO 内攻击指令触发 |
| 06:29→06:48 | 主交战死亡点集中在敌分矿附近；06:48日志宣告第1波全灭 | 第一支主力没有保留下来 |
| 07:23→08:15 | 第2波10兵集结，未记录正式出击；最终只剩3兵，取消 | 集结状态并不等于安全等待 |
| 07:53→08:13 | 第3波10兵集结，08:07只剩8兵仍正式出击；08:13全灭 | 最低放行阈值是5，不是完整10兵或科技兵团 |
| 08:15→08:24 | 第2波取消后下一帧创建第4波11兵，目标 `(28,79)`；仍在集结状态就全灭 | 没有实际退回安全区后重建；新一波不到9秒即消失 |
| 08:22以后出生的新兵 | 21个 Marine 全部在出生后0.49–12.72秒死亡；例如08:47.01出生，08:47.50即死 | 后续生产确有产出，但已经无法形成持久兵力 |

全局共生产 **74 Marine、0 Marauder、0 Tank、0 Medivac**，74 Marine 全死：
58死于落地 Viking、12死于 Marauder、3死于 Marine、1死于 SCV。含侦察兵，
不能把74个死亡全部登记为某个攻击波；这里没有伪造逐兵的波次归属。

关键日志原文：

```text
[Auto-Attack] 首次总攻触发！当前兵力 20 >= 20
总攻波次 2: 集结后仅剩 3 单位，低于最低出击 5，取消本波等待补兵。
[Auto-Attack] 后续波次触发！当前积累 11 >= 10
已创建总攻波次 4，状态: GATHERING，集结点: (28, 79)，最低出击: 5。
总攻波次 4 已全灭。
```

源码位置（以下均相对 `agent_system/environments/env_package/sc2_online_restored/SC2ArenaDyStrategy/`）：

- `players/llm_player.py:731`：ATTACK BO 只完成记账，实际攻击交给脚本。
  所以 g2 原文的 `Attack with 89 marines, 4 marauders, 1 sieged tank...`
  **没有作为成军条件执行**。
- `players/script_player.py:4220`：首波20、后续10；达到门槛即选全部可用战斗单位，
  不为基地单独保留守军，也不以敌我兵力或基地受袭状态作为这里的放行条件。
- `players/script_player.py:3196`：集结点取小队中心和敌方目标的中点；
  `:2966` 的集结实际调用 `unit.attack(target_point)`，不是后方安全等待。
- `players/script_player.py:3289`：集结后存活数只需达到最低5即可正式出击；
  取消波次没有同时下达撤回安全区指令。现有驻防和自动防御函数确实运行，
  不能写成“完全没有防御系统”；但没有阻止这次前线消耗。

## 工人如何损失

总计41个 SCV（初始12＋新生产29）。03:08先死1个侦察 SCV；其余40个从
**08:54.69到09:20.71，约26秒内全部死亡**。死亡点由分矿附近 `(20,49)`
向 `(21,35)` 移动；34个被落地 Viking 击杀，6个被 Marauder 击杀。
不是 Battlecruiser 清工人，也不是固定 BO 中“17 SCV 前压”的指令实际执行。

09:10.71最后一个 Marine 死亡；同帧日志 `Automatic Defense: Assigned 11 new units to defend.`，
回放中可见 SCV 对敌军的显式命令。此时40名基地工人已经死亡20名。
不能用这些命令宣称覆盖了全部工人：game events存在压缩／复用命令，本次只保存
明确出现的 SCmdEvent，不伪造完整逐帧指令记录。

`players/script_player.py:2433` 按**全场 `supply_army > 0`**排除工人防御；
军队人口归零后才让工人参与，`:2510` 对最近威胁下达攻击。这个分支没有工人
撤离规则，也没有区分“全场有兵”与“矿区有可用守军”。这是确切的控制逻辑与
死亡时间对应，但没有反事实实验，不能声称提前拉工人或撤离一定能救下这局。

## 为什么没有 Tank：一处已证实的执行断链

09-07后续[挂件原因审计](r4-factory-techlab-cause.md)已确认固定建筑占地重叠，
并定位到待开工Factory未被选址保护覆盖；原ActionResult仍缺失。以下保留首轮分析口径。

g2 **有**三条 `techlab factory`，不是 BO 或名称表漏写。实际日志：

```text
[07:30] Start Processing BO #123: techlab factory
Local BO ability fallback: techlab factory -> BUILD_TECHLAB_FACTORY
llm_decision_final_action_123_2521: [{'action': 'BUILD_TECHLAB_FACTORY', 'units': [266]}]
[07:30] BO #123 Completed/Skipped.
#129、#130 同样选 [266]，同样标记完成。
[07:37] Start Processing BO #165: siegetank
[Wait] Action blocked: **Action Failed**: Tech Lab required.
```

回放有3座 Factory，**没有一个 FactoryTechLab 的开工事件**。三条 Tank BO
#165–167均因该前置条件等待后超时，07:37–09:35期间顺序 BO 被连续阻塞。
后续 Marauder 在#175以后，根本未走到。地图实际升级事件仅有装饰性 Spray，
没有 Stim/Combat Shield/攻防升级；不能把未解锁兵种说成模型选择了纯 Marine。

`agents/micro_agent.py:98` 的本地 Executor 从候选文本取首个编号，三次都选
同一座 Factory `[266]`，位于约 `(26,30)`。其右侧已有约 `(29,31)` 的
SupplyDepot；**挂件区域阻挡是有坐标支持的怀疑，不是已证实的引擎拒绝原因**。
保存了对应三帧完整 obs 和 replay game events；没有引擎 ActionResult 或完整
placement 查询结果，尚不能确定具体是空间、发送还是其他引擎校验环节拒绝。

已确定的反馈缺口：`players/base_player.py:720` 调用单位命令但不检查其
`run_state`，没有在该函数接回引擎结果；`players/llm_player.py:943` 在
`run_actions` 返回后直接 `_finish_current_bo()`，不等开工。因而“本地记录
完成”与“游戏内实际开工”在这三条命令上确实不一致。

## 结论和唯一优先项

已经定位的是**模型以下的真实失效链**，不是发现了新的 PPO 根因；这两局
0次 Planner 调用，不能用来判断训练是否把模型训坏，也不能否定其他 seed 的
历史成功结果。调整20/10阈值或关闭进攻是否能赢，尚无本轮对照证据。

最具体的下一项应只查 **Factory Tech Lab 未开工**：拿到这三条命令的引擎返回和
挂件位置判定，区分选址/选执行者/提交反馈。现有文件没有这些字段；若需新实局
采集，应先报最小预算取得批准，不直接换战术、修改 BO 扣减或同时修自动进攻。

## 复核与产物

[离线脚本](audit_r4_replay_losses.py)、[事件账本](r4-preplanner-replay-losses.json)
保存两份输入哈希、解码器版本、全死亡事件、出生/存活快照、波次原始日志、
三次 Factory 挂件的完整观测及部分显式工人命令。
脚本只读 replay/trace/driver，版本 `s2protocol 5.0.16.97563.0 / mpyq 0.2.5`；
使用已有本地工具环境，不安装依赖、不改服务器环境。

实际执行解码命令退出0，输入的 replay/trace/driver 六个哈希全部核验通过：

```bash
/private/tmp/sc2-planner-executor-venv/bin/python -B \
  experiments/records/20260906-integrated-fixes-v1/audit_r4_replay_losses.py \
  /private/tmp/r4-preplanner-replay.Z1yZ2Q
```

这是产物分析，不是新训练，也不是修复后的胜率验收。全部旧 run、deployment、
失败记录和 checkpoint 保持不动。

产物回归 `python -B -m unittest discover -s tests -p 'test_r4_replay_loss_record.py' -v`
实际4/4通过：输入身份、出生/死亡身份与击杀者、后段单位寿命、三次挂件执行者。
`git diff --check`通过；游戏源码/configs与冻结`5b24db5`无差异。
