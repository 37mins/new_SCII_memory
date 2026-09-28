# 12 架 Viking：指令与损失只读核对

2026-09-08，SpB Level6对空备选单局，seed2621640384。
RUN_ID：20260908-planner-air-options-v1-spb2-l6-g2-s2621640384-r1。
冻结SHA：3adaaf70ac879a7f4d7bdecbfb8be60b75bf4fa6。
仅分析已完成产物，没有改游戏代码、部署、配置或旧结果，没有启动GPU/游戏。

## 结论

**生产出了Viking，但旧自动战斗没有把它们保留成一支对空队伍。**
不是12架都被派去侦察：前6架分到四个混合兵种自动波次；
第7架响应基地防御；第8、9架明确担任侦察；最后3架在基地附近很快被击杀。

最明确的实现冲突是：**驻防选兵没有排除已经属于进攻波次的空闲单位，
把已抵达集结点的Viking拉回驻防点；波次维护随后又把它拉回集结点。**
回放与观测同时证实第1、2、3架发生这种往返，不能扩大成全部12架的死因。
其他损失涉及混合波次、侦察和基地失守；修复一个冲突能否改善胜负仍未验证。

## 逐架记录

编号按实际出生顺序；时间为游戏时间，位置为SC2绝对坐标。最后击杀者不等于
唯一伤害来源。完整tag、command sequence、目标及观测见 [机器记录](viking-order-audit.json)。

| # | 出生 → 死亡 | 已确认指令/角色 | 死亡位置 | 最后击杀者 |
| --- | --- | --- | --- | --- |
| 1 | 11:43.8 → 13:30.8 | 自动波次11；入组后5次驻防Move，反复被拉回 | (31,63) | 敌Viking |
| 2 | 12:13.7 → 13:33.9 | 自动波次12；入组后2次驻防Move | (36,56) | 敌Viking |
| 3 | 12:43.7 → 13:46.2 | 自动波次12；入组后1次驻防Move，后到集结点交战 | (29,72) | 敌Viking |
| 4 | 13:13.6 → 13:48.5 | 自动波次13；集结途中与Battlecruiser交战 | (31,65) | Battlecruiser |
| 5 | 13:43.6 → 14:34.5 | 先被自动防御指向敌SCV，后入波次16 | (30,74) | Battlecruiser |
| 6 | 14:13.5 → 14:36.3 | 入组前同帧两次驻防Move；后入波次16，途中打Raven | (49,44) | Battlecruiser |
| 7 | 14:43.5 → 14:57.5 | 14:45.0自动防御，单架Attack敌Battlecruiser | (40,45) | Battlecruiser |
| 8 | 15:13.4 → 15:33.1 | 15:19.8侦察Move(49.5,126.5)，未抵达即死亡 | (42,82) | 敌Viking |
| 9 | 15:43.4 → 16:00.3 | 先侦察Move(24.5,22.5)，再Move(25.5,45.5) | (25,44) | Battlecruiser |
| 10 | 16:13.3 → 16:21.1 | 无直接选中该机的新命令；观测为Move(35,33)、受击 | (35,33) | Battlecruiser |
| 11 | 16:43.3 → 16:48.7 | 无直接选中该机的新命令；观测为Move(56,34) | (35,27) | Battlecruiser |
| 12 | 17:13.3 → 17:18.3 | 无直接选中该机的新命令；缺少存活期间观测，不补造路径 | (33,27) | Battlecruiser |

第10–12架仅存活7.72、5.40、5.04秒。星港(14,20)的生产集结指令在11:11.4
设为(35.11,33.11)，16:33.9改为(56.26,34.89)。第10、11架实际移动状态
与各自出生时的生产集结指令吻合。第12架只有出生、死亡及当时星港指令证据；
不能把“没有单独指令”解释成“完全没有移动/战斗”。

## 最直接的冲突证据

第1架，replay tag 17301515，观测ID814：

~~~text
12:08.9  seq818  自动波次11：Attack (29.21,73.63)
12:15.7  trace4120：已在(29,73)，idle，135/135血
12:21.6  seq827  驻防：Move (24.66,62.25)
12:23.8  trace4165：位于(26,67)，moving to [(24,62)]
12:25.4  seq833  波次维护：Attack (29.21,73.63)
12:33.6  seq847  驻防：Move (24.66,62.25)
12:37.3  seq852  波次维护：Attack (29.21,73.63)
~~~

入组后的5次驻防移动发生在12:21.6、12:33.6、12:45.5、12:57.5、13:09.5，
对应冻结脚本每67 iteration检查一次驻防。第2架入组后2次、第3架1次，
共8条已确认的波次内驻防覆盖。Move目标与同局基地驻防点公式一致；
观测证实单位确实离开已到达的集结点，不只是日志里有两种指令。

第6架还暴露同一驻防函数的重复分配：14:21.2、game loop19292，
seq1040先Move(38.56,52.87)，seq1046再Move(56.26,34.89)；
trace4832显示它正往后一个点飞。这在入波次16之前，不能计入“波次内覆盖”。

## 为什么没有形成对空集结

四次均有原日志：“[Auto-Attack] 后续波次触发！当前积累 10 >= 10”。
10是所有兵种合计，不是10架Viking：

| 自动波次 | 实际组成 | 集结点 | 后续状态 |
| --- | --- | --- | --- |
| 11 | 1 Viking + 3 Tank + 3 Marauder + 3 Marine | (29,73) | 仅剩1单位，取消 |
| 12 | 2 Viking + 2 Tank + 6 Marine | (29,72) | 全灭 |
| 13 | 1 Viking + 9 Marine | (30,74) | 全灭 |
| 16 | 2 Viking + 2 Tank + 6 Marine | (30,74) | 仅剩1单位，取消 |

四个波次均没有记录到GATHERING → ATTACKING，但集结本身用Attack-move，
途中就会交战。“还在集结”不等于“还没打起来”。14:30.9同组的第5、6架
相距约35.5：一架在(30,74)打敌Viking，另一架在(49,44)打Raven，随后均被
Battlecruiser击杀。这显示同组但未集中交战；不能据此保证换集结点就能赢。

侦察只确认两架、三条派遣指令。脚本优先抽空闲飞机，直接Move，
并将当前侦察机排除在自动进攻选兵之外。第8架向北方远点飞行中死亡；
第9架是在本方基地/扩张点附近侦察时死亡，不应都说成“冲敌方主矿送掉”。

另有局部分派问题：14:02.9 seq1031包含第5架及4个Marine，Attack目标为敌SCV
71827462；iteration4720自动防御日志恰好记录指派5单位。
trace4764显示第5架跟着该地面目标移动，没有形成对空集结。它后来入波次16
才死亡，不能把这次分派直接当作最后死亡原因。

## 冻结源码位置

文件：agent_system/environments/env_package/sc2_online_restored/SC2ArenaDyStrategy/players/script_player.py。

| 位置 | 实际逻辑 | 本局证据 |
| --- | --- | --- |
| manage_garrison L2621、筛选L2637、移动L2749附近 | idle选兵不排除波次；每基地重新使用同一候选集合 | 第1–3架往返；第6架同帧两次不同Move |
| rally_units_to_point L2966；launch_total_attack L3200 | 集结点取队伍中心与敌方目标中点，向那里Attack-move | 四波未完成集结便交战、损失 |
| manage_total_attack_groups L3289、L3342 | 离集结点远且idle便重新Attack-move；剩余成员靠近后才检查最低人数 | 被驻防拉走后又拉回；两波仅剩1人再取消 |
| _find_and_dispatch_scout L2850；_send_scout L2939 | 优先空闲飞机，直接Move目标 | 第8、9架被派走 |
| automatic_defense L2503 | 按最近威胁分派，未按对空/对地能力筛选目标 | 第5架指向SCV；第7架单独迎击BC |
| run_maintenance_tasks L4373附近 | 驻防在进攻维护之后；本局新micro关闭 | 未启用新micro的Viking精细走位保护 |

这些模块在本次“增加对空备选”中未改动。已核验记录提交
049c5176a6e5acfe3f2d63b1898660b90797d4af的gameplay/config文件与冻结SHA无差异。
该文件SHA256：5e44bedf015cc8d6397f0a27d1c795cb01cbf58c91f4326227c449b877404dd2。
没有据此断言这些逻辑最初是谁编写或何时引入。

## 边界与下一项

已确认：指令往返冲突、四组混编分流、两架侦察、后期近基地快速损失。
尚未证明：各机制分别造成多少额外损失，修复后能否提高Level6胜率。
死亡事件只给最后击杀者；没有完整逐帧伤害/路径，不能虚构每发攻击归因。

下一项最小候选应只验证：**驻防不再重新分配已属于进攻波次的单位**。
不同时改侦察、集结点、波次数量或micro；先做最小回归，再获授权做同seed对照。
这只验证冲突是否消除，不承诺独立解决基地失守和对空战术。本次没有实施。

## 离线复现与校验

[audit_vikings.py](audit_vikings.py)，依赖s2protocol5.0.16.97563.0、mpyq0.2.5、
sc2reader1.9.0。用SC2 build75689解码器，仅复用sc2reader选择集运算，
按其command-manager事件语义还原压缩指令。参考
[Blizzard s2protocol](https://github.com/Blizzard/s2protocol)与
[sc2reader SelectionTracker](https://github.com/ggtracker/sc2reader/blob/upstream/sc2reader/engine/plugins/selection.py)。
sc2reader整包加载此API回放因空cache_handles失败，未用其完整Replay loader；
这是离线工具限制，不是项目游戏bug。

~~~bash
/private/tmp/sc2-air-results.HBI61m/decoder/bin/python -B \
  experiments/records/20260908-planner-air-options-v1/audit_vikings.py \
  /private/tmp/sc2-air-results.HBI61m/candidate /tmp/viking-order-audit.json
~~~

实际执行exit0：12架、154条Viking观测；14条生产命令与trace的game loop逐条一致；
侦察3条命令与日志一致，选择集无错误。第1–11架观测ID按首次出现时唯一未匹配
存活Viking对应，后续观测全部落在其生命区间；第12架无观测。命令说明提交内容，
采样观测独立说明引擎实际状态。没有启动新游戏来补帧。

原产物均与已登记run哈希吻合：

| 产物 | SHA256 |
| --- | --- |
| replay | 064f78a3ea896c9bf590999d17343fd0b0d735c712c65e2dd495b22157d083ab |
| console | f81b1a80eadc029928894d1f07bb102d8a02e07d95940a6986dc3499900ada6a |
| trace | 518047262c3cd5122e86a3a1f37a5248dfcefd8abe47515856d0bf77be8a0574 |
