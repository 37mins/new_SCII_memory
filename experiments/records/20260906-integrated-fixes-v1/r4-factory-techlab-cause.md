# 三次 Factory Tech Lab 未开工：原因审计

2026-09-07。结论：**同一座工厂的挂件占地被补给站挡住；选址保护漏掉了
“已下建造命令、尚未开工”的工厂，Executor 后续三次仍选择它，且没有回收执行失败。**
占地冲突与漏保护时序已核实；原引擎具体 ActionResult 未保存，不能补造错误码。
本轮没有修改游戏代码、启动新游戏、训练或占用 GPU。

沿用[两份回放](r4-preplanner-replay-losses.md)及原 RUN_ID，冻结 SHA
`5b24db566007b514686f61242ad999b5d296725b`；44部署重新确认 clean。
两份 replay/trace/driver 六个哈希均通过，本次另保存[精确证据](r4-factory-techlab-evidence.json)。
不将最新工作树当历史源码；引用文件均核对冻结版本。

## 因果链

| 游戏时间／帧 | 已有产物中的事实 | 含义 |
| --- | --- | --- |
| 06:10.536／loop8300 | BO#95下达 Factory，命令实际坐标 `(26.5,30.5)` | SCV还在走向工地，不代表建筑已经出现在场上 |
| 06:12.143／loop8336 | BO#104下达 SupplyDepot，命令坐标 `(28.5,30.5)` | 此帧 Own structures 没有任何 Factory；Own units 中SCV已有去 `(26,30)` 的建造订单 |
| 06:13.482／loop8366 | 该 Factory 才出现 SUnitInitEvent | 比补给站选址晚1.34秒，保护代码没看到它 |
| 06:16.205／loop8427 | 补给站开工，回放中心 `(29,31)` | 抢占了该工厂未来挂件的位置 |
| 07:30.179、07:31.250、07:31.429 | BO#123/#129/#130都向显示编号266的Factory下达BUILD_TECHLAB_FACTORY | replay选择事件均指向同一真实tag `(347,1)`，不是显示编号碰巧相同 |
| 07:37以后 | Tank持续 `Tech Lab required`，三项最终均超时 | 未执行的挂件使后续科技兵种被阻塞 |

两份回放在这些事件上逐项一致。工厂与补给站在挂件尝试前均已建成、未死亡，
工厂未升空；三次观测都是满血、idle、没有挂件，资源分别为2515/667、
2310/671、2320/675（矿/气）。没有把“资源不足”或“工厂还在施工”当作原因。

## 挂件空间确实重叠

实际依赖 burnysc2 **7.1.3** 的 `sc2/unit.py:1142` 明确规定，生产建筑挂件中心是
建筑中心加 `(2.5,-0.5)`，挂件2×2；`:855`规定SupplyDepot占地半径1。
同一函数定义也可见[官方文档](https://burnysc2.github.io/python-sc2/_modules/sc2/unit.html#Unit.add_on_position)。
这里以服务器实际文件为准，三个关键库文件均与[7.1.3官方发布包](https://pypi.org/project/burnysc2/7.1.3/)
逐字节哈希一致，详见[CPU核验记录](r4-factory-techlab-cpu.json)，没有安装或升级依赖。

```text
Factory 中心       (26.5,30.5)
Tech Lab 应有中心  (29,30)，占地 x=[28,30], y=[29,31]
SupplyDepot 中心   (29,31)，占地 x=[28,30], y=[30,32]
交叠区域                     x=[28,30], y=[30,31]，即2×1格
```

这不是仅凭“看起来挨得近”判断。Factory坐标取原生命令，Depot中心取回放开工
事件，并与后续存活观测一致；没有把日志中的目标占位符 `[1,1]` 当真实落点。

## 哪段代码漏了

以下路径相对 `agent_system/environments/env_package/sc2_online_restored/SC2ArenaDyStrategy/`。

1. **选址遗漏待开工建筑**：`players/base_player.py:562–567`只从
   `self.structures.of_type(...)` 生成挂件预留点，没有读已发出的SCV建造订单。
   BO#104选址时Factory尚未开工，所以没有预留这片空间。不是完全没写保护，
   而是保护覆盖范围不完整。
2. **重复选中受阻工厂**：`agents/micro_agent.py:98–103`的local fallback取候选
   文本中的首个编号。`players/llm_player.py:897`在候选生成时只核对可用ability，没有单独
   检查每座Factory的挂件空间。三次都选266，没有转到其余Factory；“有建造
   ability”不等于“挂件位置可放置”。
3. **本地完成早于引擎结果**：`players/base_player.py:720`只是把命令加入
   帧末发送队列；`players/llm_player.py:943–945`立即标完成。实际库
   `sc2/bot_ai_internal.py:797–810`在帧末发送，未保存返回；原`driver.log`、独立
   player日志、trace和文本observation均未保存这三次原始引擎错误。

补充排除：`_query_building_placement_fast`在实际7.1.3中返回bool列表，
现有`if r`筛选没有把错误枚举当成True；Factory挂件映射为BUILD_TECHLAB_FACTORY
也是正确的。所选工厂idle，正常去重分支不因已有同类order过滤它。

## CPU核验和证据边界

[结构核验脚本](probe_r4_pending_addon.py)在44实际7.1.3环境中运行，硬限20秒、
CUDA不可见，退出0；直接提取冻结`find_placement`，没有另写一套选址算法。
同一个补给站候选点，保留SCV待开工订单时：Factory不在structures中就被放行；
把Factory放进structures后便被现有保护拒绝。两个断言通过。
这证明控制流确实漏待开工订单；`can_place_single`和矿路检查是受控stub，
**不是引擎重放，也不是修复后游戏验收**。

实际命令（本地脚本通过stdin只读执行，不部署新文件）：

```bash
ssh sc2-a100-44 'CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=/root/new-scii-spb2-level5-control/python timeout 20s /usr/bin/python3 -B - \
/root/new-scii-r4-python-rng-20260907/code-5b24db5' \
< experiments/records/20260906-integrated-fixes-v1/probe_r4_pending_addon.py
```

另用[audit_r4_factory_techlab.py](audit_r4_factory_techlab.py)解码两份既有回放，
实际退出0；核对六个输入哈希、完整建成/死亡时间、三次真实执行者和占地重叠。
保存JSON与重新解码逐字段一致；`python -B -m unittest discover -s tests
-p 'test_r4_replay_loss_record.py' -v`实际7/7通过，`git diff --check`通过。
`git diff 5b24db5 -- agent_system configs`为空；未新增游戏或模型调用。

**可以确定**：存在足以阻止该位置建挂件的固定建筑占地冲突，并查清冲突如何
绕过现有保护、为何连续三次没换执行者。**仍不能宣称**：原引擎具体返回了哪个
错误码、没有其他并发失败条件、修复此处就能赢Level6。原ActionResult无法从
这些日志恢复；要核验错误码须另行批准最小只加观测的CPU实局。

下一项修改应只选一个边界，例如补齐“已下令但未开工生产建筑”的挂件空间预留；
这只是待确认方案，不在本轮同时改Executor扣减、生产调度和进攻策略。
