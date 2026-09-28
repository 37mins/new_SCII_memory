# 命令来源记录：Depot 关联通过单场景实局验收

2026-09-13最新：r3新实局已完成，66次开工、64次完成实时关联与独立审计一致；
只是单场景记录器验收，没有启用奖励或训练。[结果](#r3结果记录关联通过学习未验证)
以下日期段落保留各阶段实际状态，不将后续候选追写为原r2运行结果。

2026-09-12。用户批准的下一项是“只记录、不改行为”，不是启用
[人口纠错奖励候选](supply-reward-candidate.md)。基于
`dc1b22e96aef9046531e97bac79f4d4801a2e6cb`，在独立分支
`codex/diagnostic-command-provenance` 接入；原分支、r6、B4、deployment 和结果不动。
本次未启动 GPU、SC2 或 PPO，不改 prompt、奖励、待办扣减、过滤、补兵、微操或优化器。

## 实现的边界

`agent_system/sc2_command_trace.py` 只观察 SupplyDepot 建造和 Marine 生产。
Planner 保存收到的回答及解析前后序列，Commander 保存实际队列的独立来源副本；
BasePlayer 仅在原来的同步下令处携带 BO 来源。LLMPlayer 记录本地命令、底层请求、
引擎返回及施工/出生回调。实际 BO 数据结构、选择结果和发送给 SC2 的命令均不改写。

| 记录 | 可以说明什么 | 不能说明什么 |
| --- | --- | --- |
| `received_answer / parsed_plan / executor_queue` | 哪条收到的回答、解析条目或规则对应当前队列 | 回答一定是模型生成，或解析接受就是完成 |
| `local_command` | 最终下令单位、Executor 实际选择的位置、是否进入本地队列 | 引擎已经接受或单位已经产出 |
| `wire_request / wire_result` | SDK 过滤/合并后实际请求、原始返回码及异常 | 命令 ACK 等于施工/生产完成 |
| `depot_started / depot_completed` | 当前候选使用唯一SCV订单目标及唯一未完成、ACK成功命令，再按建筑tag关联完成 | 引擎直接提供了命令到建筑的因果ID；旧r2已运行该候选 |
| `marine_created` | 新 Marine tag、位置和兵营队列快照 | Marine 唯一由哪条命令生产；本版始终记未知 |

Depot 关联是保守的**观测匹配**：多候选、缺订单/ACK、回包无法对齐都不猜归属，
完成回调不使用“此刻最新 Planner 回答”作来源。重复完成带 `duplicate=true`。
不观察取消/摧毁来补造成功记录，不给这些事件分数。新候选实局匹配覆盖率仍未知。

来源区分 `validated_answer`、`postprocess_or_salvage`、`commander_rule`、
`fixed_bo` 和未知。只有既有整答合法直通路径中的匹配动作才标 `validated_answer`，
无效回答 salvage 不追认为合法模型选择。Commander 未匹配的插入项单列，
直接调用 Client 绕过 BO 的命令不借用最近回答来源。

`SC2Env.step` 可接收可选元数据 `trace_origin` 并原样转发，例如人工干预标 `human`。
没有此字段时记 `unverified_transport`；**该字段只是调用方声明，不是来源鉴定**。
当前训练 manager 不自动补这个字段。未来实局需与生成/干预原始记录核对，不能
把人工替换回答因通过 validator 就算作模型决策。原文字、格式奖励和返回 transition 不变。

## 依据与没有修的 SDK 行为

核对本地实际安装的 `burnysc2 7.1.3`，没有修改依赖：
`bot_ai_internal.py:475` 的 `do` 只入队；`:568` 在 `_do_actions` 过滤后调用 Client；
`client.py:206` 将命令组合成 `RequestAction`，再读取 `ResponseAction.result`。
它的成功无错误列表和被捕获的 `ProtocolError` 都可能返回 `[]`，因此记录器在
原 `_execute` 边界观察原始响应/异常，而不拿 `Client.actions()==[]` 当成功证明。
原库的返回和异常处理继续保留；多命令合并有歧义或结果数量不对应时记录未知。

[python-sc2 官方回调说明](https://burnysc2.github.io/python-sc2/_modules/sc2/bot_ai.html)
与 [Blizzard raw.proto](https://github.com/Blizzard/s2client-proto/blob/master/s2clientprotocol/raw.proto)
给出了 Unit/tag、订单和命令字段，但没有“新 Marine 对应本项目哪次命令”的 ID。
因此不使用最近兵营、FIFO 或最新回答猜测出生归属。在线文档会更新；本次实际测试
依据是上述本地 7.1.3 源码，而非宣称当前网页逐字等同冻结依赖。

本地 SDK 源码 SHA-256：

```text
client.py          4cca3d6b15530367b45579e8f67c20776f1ab3d082ec270df96ea1a9b22f3c21
bot_ai_internal.py 4ba60ce23d8573fd5cee0782796ec7dc41e064809796c69ca84728df6d56c9ff
unit.py            da5c1824787f2cae0164a614c85223ebc2f25ec7312be49b3e5f500399cd0589
```

## 最小用法与失败行为

默认关闭。经授权的未来独立实局中，将 `SC2_COMMAND_TRACE_DIR` 设置为本次运行
**已存在且可写**的独立输出目录；Ray 启动需要将它显式传入游戏进程的 runtime env，
例如 `+ray_init.runtime_env.env_vars.SC2_COMMAND_TRACE_DIR=<本次目录>`。
这不是本轮启动命令或新增资源授权，不要预建会被现有 launcher 拒绝的 RUN 根目录。

每个游戏生成独占文件 `game-<PID>-<UUID>-command-trace.jsonl`，事件 ID 在文件内递增，
记录 game loop、游戏时间、资源、人口及最新回答 ID。文件身份区分游戏，seed 不能
代替它；后续以 PID/回答 ID 与既有 env/seed 日志关联。UUID 不消费 Python 采样 RNG。
命令记录先缓冲，每帧和结束时写出；强制退出可能丢失最后一批未刷记录，不能补猜。
无权限、目录不存在或写入失败显式报错，不静默吞掉诊断故障。未加入自动重试。

## 已执行的 CPU 验证

本地 Python：`/private/tmp/sc2-air-results.HBI61m/decoder/bin/python`。
真实使用 SDK 的 `Unit.__call__ / BotAI.do / Client.actions` 和 protobuf，只有 RPC
返回用模拟对象；**没有加载模型或游戏**。

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:tests python -m unittest \
  test_sc2_command_trace test_execution_failure_feedback \
  test_r6_format_reward test_r6_natural_termination test_valid_bo_passthrough \
  test_empty_bo_frame test_g2_planner_handoff test_r6_planner_interface \
  test_original_bo_segment -q
```

95 项通过，退出码 0，4.486 秒；其中本次新增16项。原日志
`/private/tmp/sc2-command-trace-tests.tG242Y` SHA-256：
`682eba746501270ca31694e724722d60917345b06eca4836c525fc14019bad96`。
测试初版误用了 SDK 只读属性，已修测试夹具，不计为项目原有 bug 修复。

冻结源码与候选记录关闭/开启对照：四种回答的 query、reward、格式判定和最终计划
一致；Executor 用例的资源统计、历史动作和发送的 protobuf **逐字节相同**。
跨回答完成用例中，最新回答已为2，Depot完成仍连接回答1的原命令。
同时覆盖拒绝、异常、重复回调/命令、未收到 ACK、规则/人工来源以及未知 Marine。
这些有限 CPU 用例不证明真实 SC2 完全无时序影响，也不验证游戏效果。

**结论：可进入仅记录的实局验收，不能接入奖励候选。** 下一项需确认一次最小实局的
配置与资源预算，查看真实请求、ACK、Depot tag 和来源覆盖率；不先开新 RL，
也不为凑齐奖励链条强行推断 Marine 来源。奖励本身的首次造堵激励仍未解决。

## 实局记录验收 r1（已完成，来源关联未通过）

用户明确批准43 GPU0–7、一个原始SpB模型实例、Level6/g2、实际seed52620，
沿用 [已完成顺序对照](supply-order.md) 的 A/B 各4局，合计最多45分钟。
无PPO、无新checkpoint、不复制大权重、不自动重试。只有一个独立游戏场景。
这次验收的是记录器，不重新宣称顺序干预为新方法，也不使用人工回放赚模型纠错分。

核心记录代码 `e41f213fbc607503be730f9ba33d0804a5aece4c`；相对原实局
`4d39d06658364689518d5d6f44a2af4bb5f73b49`，agent_system/verl/gigpo/configs
差异仅本页已登记的来源观察及文件哈希。本次只增加入口/评测元数据，核心源码冻结。
原始 `Panyuqi/SpikingBrain-2.0-instruct` 的manifest为
`40d6f1417568d5fda1107cdafd66c7441a679b03fb148851c8f226ecc9cf884c`。
FP32参数/温度归一化、BF16 autocast、HF/eager/cache、分层FSDP、SWA dropout0，
T0.4/top_p0.9/top_k0、4096/192tokens、约15条及xN、累计配额、history/shaping off、
阶段/待办来源、过滤、规则补齐、微操和reward全部沿用原对照。

复用 `supply_order_eval.install`；只在记录模式中，原validator/projection之后的
SC2Env输入附加 `trace_origin`。前三答分别标历史回放/人工顺序干预，后继答标
`online_model_transport`，未消费终局标占位。它不等于逐动作均由模型产生：仍需
联读记录器的合法直通/规则来源和原始生成、转发记录。没有改变任何回答文本。

入口 `run_command_trace_node43.sh` 复用旧八卡入口及其全部游戏配置，只切换43已有
本地资产/依赖位置、新日志/临时目录，并向全部游戏进程传入记录目录。
既有0/1/2/3/4/5/6/7卡、八局显式override保留；val_only=True、resume=disable、
save_freq=-1。共享入口中total_training_steps=96只是登记字段，不会执行训练。

RUN_ID：
`20260911-execution-failure-feedback-v1-spb2-original-l6-g2-commandtrace-ab8-8gpu-s52620-r1`。
输出为43本地 `/root/new-scii-execution-feedback-local-20260911/shared/yangjinyue/
new_SCII/experiments/20260911-execution-failure-feedback-v1/runs/<RUN_ID>`。
新冻结目录为 `/root/new-scii-execution-feedback-20260911/code-<SHA>`，
控制目录`commandtrace-r1`，临时目录`/dev/shm/ct_0912_r1`；全部拒绝覆盖旧路径。
外层2685秒+15秒强制退出，内层2640秒+30秒退出预留；从实际启动计时。

20:50北京时间预检43八卡均0MiB、无compute/tmux任务，主机可用448GiB；本地盘
余94GiB、/dev/shm余424GiB。复用既有依赖/模型/SC2，不依赖共享盘进行模型加载。
启动前再核验归属；此快照不等于已启动。冻结提交和实际开始/退出状态随后追加。

本地23项CPU测试通过（0.898秒、退出0），包括原三次query哈希/46项队列、来源
标签与payload不变、两个入口effective config逐字段对照。差异仅RUN/输出/临时目录、
已登记依赖路径和记录开关。测试初版漏把新临时目录列为预期差异，修正了比较夹具，
不是运行代码bug或改变模型cache开关。没有用CPU通过冒充游戏已经启动。

验收时先核对8局的PID/seed/原始query与24条历史干预，再检查请求/原始ACK与
Depot开工/完成tag的一致性、跨回答完成是否保留旧命令来源、模型/规则/人工是否
分开，以及Marine未知是否如实保留。与旧事件日志按tag/类型核对，报告匹配覆盖率
及所有歧义；不可关联不冒充失败或成功。输入分岔、记录异常、超时都保留产物并停止，
不放宽哈希、不自动重试、不接续训练。游戏勝负/reward只辅助排查行为变化，不作为
记录器通过或学习成功的唯一标准。

奖励取舍保持不变：本轮不启用0.5事件bonus；“先堵再解”激励尚未排除，且Marine
唯一命令来源仍缺证据。实局若不能补齐，不扩展FIFO/最近兵营猜测来凑齐奖励链条。

### 实际结果

冻结运行 SHA `33dc18244015e5d5e546db2ebf6521afdead8e43`，43的`code-33dc182`
保持clean。完整私有源码bundle哈希及全部小型产物哈希见
[结果记录](commandtrace-r1-results.json)。服务器CUDA隐藏的42项回归通过，2.095秒；
实际依赖Torch2.7.1+cu128、Transformers4.55.2、burnysc2 7.1.3。

北京时间21:00:00启动tmux `sc2_commandtrace_r1`，21:01:57进入八局开局Planner；
21:08:06完整结束，控制`.done`及result退出0，0次PPO，没有新checkpoint。
21:09:16八卡均0MiB，无compute/tmux任务。结束附近有Ray worker SIGTERM警告；
本次8局自然终局、终局ACK、结果及回放均完整，未据此判为丢局或训练失败。

| 已核对的记录 | 实际结果 |
| --- | --- |
| 八局实际seed、前三答输入/转发 | 均52620；24/24完整prompt SHA和转发文本一致 |
| 消费回答 | 28条：20历史回放、4人工改序、4在线模型；32条总日志另有4条未消费终局占位 |
| 目标命令及引擎ACK | 382条，全部有唯一的本地请求对应和成功ACK；包括66条Depot命令 |
| 来源标签 | 328条命令来自历史回放、52条人工改序、2条在线模型；均走合法回答路径，不能把前380条算模型选择 |
| 施工/出生事件 | 66次Depot开工、65次完成、304次Marine出生；tag集合与原观察器逐局一致 |
| Depot命令→开工关联 | **0/66**；Marine命令→出生仍显式未知，未伪造来源 |
| 胜负和整局累计reward | A/B各4/4胜、每局20.5，与旧对照相同；不作为学习或记录器通过证据 |

06:13以后，B每局完成8个Depot、新增2个Marine；A为1/0/0/0个Depot、均无新增
Marine，与旧结果一致。B前三答全为人工干预，没有后继在线回答，不能拿它的新增
产出验证“模型纠错奖励”。A每局第四答是在线回答，但只有env0实际下出两条Depot命令。

缺口已定位到**新增记录器的坐标相等假设**，不是原系统没有执行命令。例如
`game-4063695-…-command-trace.jsonl`：event5在loop264下令，目标`[28.5,22.5]`；
event6发送对应请求，event7收到`[1]`成功；event8在loop324看到新Depot
tag4351852545、位置`[29,23]`，因此当前严格相等条件给出未知。
全部66次开工都能在更早且ACK成功的命令中找到唯一`(+0.5,+0.5)`位置候选。
这是**本场景的事后匹配现象**，尚未核验为普遍引擎坐标规则；没有硬编码加0.5、
改为最近邻或修改已有run的command_id。65次完成也不能因此追认为已通过来源验收。

请求包含其他动作时，原始返回会有多个结果；实际366次为`[1]`、8次`[1,1]`、
8次`[1,1,1]`，按已记录action_index核对目标命令均成功。离线探针最初误假设每次
只有一个返回码，纠正后通过；这是分析脚本假设，不是新增执行/引擎问题。

**本轮结论：实局完成，回答来源与命令ACK记录局部通过；Depot来源关联未通过。**
下一项先用这66条已有记录核验建造坐标规则，提出一处记录匹配修正并做CPU对照，
不立即再占GPU，不改Executor位置选择，不扩展Marine猜测或奖励机制。
这次诊断返工不能计为修复了项目原有学习问题；奖励候选仍不采用，未启动新RL。

### 坐标假设 CPU 对照（9月12日，未采用取整修复）

基于结果提交 `be6261680f294bdfd5b5e3c66177f29981a22e47`，只读复用上述8份
JSONL及冻结manifest。新增 `audit_depot_coordinates.py`，按事件先后恢复实际请求、
逐action ACK和匹配候选；严格相等分支必须逐条复现原始 `command_id`，不能利用
未来ACK、跳过歧义或改写旧日志。只输出离线假设结果，不接入运行时或奖励。

| 坐标处理 | 唯一已ACK候选/66 | 是否可采用为引擎规则 |
| --- | ---: | --- |
| 原严格相等 | 0 | 已复现漏关联 |
| SDK `Point2.rounded`（floor） | 0 | 不是建造吸附规则 |
| Python `round` | 0 | 不能解决这批记录 |
| `floor(x+0.5)` 假设 | 66 | 证据不足，未采用 |
| `ceil(x)` 假设 | 66 | 证据不足，未采用 |

**关键限制：66条命令只有12个不同目标坐标，132个坐标分量全部以 `.5` 结尾。**
因此后两种规则在现有输入上不可区分；例如28.25分别得到28和29，而本批没有这种
输入。两个规则同得66/66不等于证明其中一个是引擎规则。匹配覆盖率也不是独立
因果真值，不能据此追认65个完成事件已经正确归属于某个模型回答。

已核对的源码：冻结 `base_player.py:537–640,710–716` 的选址以基地坐标加整数
偏移产生候选，成功后原样返回，并未将Depot中心对齐到整数；本次没有改选址。
实际依赖burnysc2 7.1.3的 `position.py:178–179` 明确使用floor，`unit.py:855–864`
说明Depot footprint radius为1，但这两条都不定义引擎的吸附方向/中点处理。
`game_info.py` 的0.5偏移用于坡口网格点构造，不能当作任意建造命令统一加0.5。
SDK本地文件哈希：position.py为
`e96190b31bc80acd650d71d41951235232dfe78b516f97827f6e33890170ba67`，
game_info.py为 `a0f2419328d98b281010012457c9b05b637c133969bb1efc48dcef2921049196`。

公开依据：[SDK坡口源码](https://burnysc2.github.io/python-sc2/_modules/sc2/game_info.html)
描述该局部偏移；[Blizzard AbilityData](https://github.com/Blizzard/s2client-proto/blob/master/s2clientprotocol/data.proto#L9-L31)
只将footprint radius定义为估计尺寸，不给出精确吸附函数。本轮查到的公开依据不足以
支持硬编码通用 `+0.5` / ceil / 四舍五入；不是声称所有公开源码都不存在相关依据。

**唯一下一项：补开工时原下令SCV的实际订单目标证据。**
[Blizzard UnitOrder](https://github.com/Blizzard/s2client-proto/blob/master/s2clientprotocol/raw.proto#L47-L54)
提供目标坐标或目标tag；SDK `bot_ai.py:893–908` 也读取它区分施工SCV。
现有记录只有下令前的ability/progress，没有开工时的SCV订单目标，CPU分析无法恢复。
拟只补这一观察字段，核对原issuer tag、订单目标和新建筑tag/位置；目标缺失、多候选、
没有对应ACK仍为未知。先CPU验证记录不改变请求，再另行申请最小无PPO实局预算。
此处是后续方案，**本轮没有新增该在线字段、修改匹配函数或启动游戏**。

已执行：

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:tests python -m unittest \
  test_depot_coordinate_audit test_sc2_command_trace test_supply_order_eval -q
python experiments/records/20260911-execution-failure-feedback-v1/audit_depot_coordinates.py \
  /private/tmp/sc2-commandtrace-r1.sVnPdN \
  --output /private/tmp/sc2-commandtrace-r1-coordinate-audit-20260912.json
```

沿用前述本地Python，28项通过（新增5项），0.896秒、退出0；全66条CPU审计退出0。
完整逐事件结果258234字节，SHA-256为
`3d95ced315ab0b1c0a61292a083c9c3e23b2fcc5b51fd9947c03c33b16b1d600`，
其中保存每份输入哈希、事件ID、候选命令/位置/来源。输出拒绝覆盖旧路径。
记录器源码哈希仍为 `442783359ef4ea24ad460de6853905888c23468ab92a727ec301e23ecd9c35aa`；
prompt、Executor、训练配置、reward、既有deployment和run均未修改。

### 开工时 SCV 订单快照（9月12日，CPU阶段）

用户随后批准继续。以 `8042871c5b2b3c4b45c0992d141c480906b10e2c` 为父提交，
复用当前诊断分支，只修改记录器与 LLMPlayer 开工回调的一行传参，不开新分支。
原提交及现有run保持不变。仍须显式设置 `SC2_COMMAND_TRACE_DIR` 才启用记录。

`depot_started.issuer_observations` 保存**此前已发送、尚未关联建筑的 Depot 命令**
对应下令tag的当帧快照；不按距离或推测后的坐标筛选，不将这些工人称为已确认建造者。
每个下令tag只记录一次，`command_ids` 保留对应全部历史命令ID，ACK及来源从原命令
和原wire记录核对。拒绝或未知ACK也保留观察，不把有订单当作ACK成功。

| 字段 | 实际含义 |
| --- | --- |
| `issuer_tag / command_ids` | 原下令工人及已记录命令，不借用最近Planner回答 |
| `observed / position` | 该tag是否在当前workers中及其位置；缺失不推断为死亡 |
| `orders` | 原顺序的订单快照；`[]`为观察到空订单，`null`为未观察到工人 |
| `ability_id / progress` | 原始API值；订单进度不等于建筑已完成 |
| `target_kind / target_position / target_tag` | 明确区分坐标、单位tag和未提供目标；零坐标不是缺失 |

读取的是burnysc2 7.1.3 `Unit._proto.orders`，对应上述Blizzard `UnitOrder`字段；
不调用额外RPC、不改SDK、不发送动作。使用raw字段是为了保留oneof的缺失状态：
本地 `UnitOrder.from_proto` 在目标缺失时会读到protobuf默认tag0，不能仅靠转换后的
`order.target` 区分缺失和实际提供的零值。此处只绕开诊断序列化歧义，不修SDK行为。
快照在现有父类开工回调之后读取同一帧观测，事件沿用已有game_loop。

**原匹配条件完全保留。** 即使命令点为(28.5,22.5)、建筑及SCV订单点均为(29,23)，
本版仍记录 `command_id=null`；订单直接指向建筑tag也不会自动归因。
这是补证据，不是坐标归因修复，更不是学习改善。Marine、reward及其他模块不动。

CPU新增6项覆盖：半整数案例的订单与原命令来源、tag/零坐标/缺失目标、未观察工人与
空订单、拒绝ACK、记录关闭时不读工人、与父提交的开工/完成记录逐字段对照。
真实SDK/protobuf夹具通过LLMPlayer回调，模拟订单只用于CPU测试，不能当作实局输出。
比较前后工人原始protobuf不变、动作队列不变、RPC次数不增加；除新增字段外，
父提交的匹配结果和完成记录相同。现有Planner/Executor不改行为的测试继续保留。

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:tests python -m unittest \
  test_sc2_command_trace test_depot_coordinate_audit test_supply_order_eval -q
```

使用本页既有本地Python，34项通过，1.003秒、退出0。尚未占GPU、启动SC2或PPO，
没有用上轮GPU授权重跑。下一项仅申请沿用r1配置的无PPO记录验收，以真实SCV订单
核对命令点与建筑中心；不能先把待验证订单规则用于奖励。

随后扩大CPU回归至本页原95项及坐标/入口测试，共113项通过，5.117秒、退出0：

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:tests python -m unittest \
  test_sc2_command_trace test_depot_coordinate_audit test_supply_order_eval \
  test_execution_failure_feedback test_r6_format_reward test_r6_natural_termination \
  test_valid_bo_passthrough test_empty_bo_frame test_g2_planner_handoff \
  test_r6_planner_interface test_original_bo_segment -q
shasum -a 256 -c configs/system-v1.sha256
```

24个已登记文件哈希全部通过，`git diff --check`通过。运行源码只有记录器与开工
回调传参有变化，更新对应两个哈希；没有部署到服务器或写入历史实验目录。

### r2 订单记录实局验收（已授权，启动前计划）

用户批准43 GPU0–7、原始SpB、Level6/g2、实际seed52620，沿用r1的A/B各4局，
一个八卡实例、最多15分钟；无PPO/新checkpoint/自动重试。
22:00:43北京时间只读预检八卡均0MiB、无compute/tmux任务，主机可用448GiB，
本地盘余94GiB、/dev/shm余424GiB。旧长训有16个SC2进程但不占GPU，不清理它们。
模型、Python和头文件均复用43已有本地资产，不复制大权重或部署新依赖。

唯一运行行为变化为 `f71f17c7217587fdfac25f09a0f01315e03df672` 的订单记录扩展。
匹配、prompt、reward、游戏/生成配置及r1前三答回放/干预完全冻结，历史干预不能
算模型选择。新入口 `run_depot_issuer_node43.sh` 只改变RUN/临时目录及外层885秒
INT+15秒KILL预算；其余复用原入口及空卡/clean SHA/资产校验，不删除检查。
没有新增重试，既有环境 `SC2_RESET_RETRIES=1` 不变。

RUN_ID：`20260911-execution-failure-feedback-v1-spb2-original-l6-g2-commandtrace-ab8-8gpu-s52620-r2`。
结果仍位于43本地 `new-scii-execution-feedback-local-20260911/shared/yangjinyue/new_SCII/
experiments/20260911-execution-failure-feedback-v1/runs/<RUN_ID>`；新冻结目录位于
`/root/new-scii-execution-feedback-20260911/code-<SHA>`，控制目录 `commandtrace-r2`，
tmux `sc2_commandtrace_r2`，临时目录 `/dev/shm/ct_0912_r2`，均拒绝覆盖。

验收先复核24条历史输入/转发、实际seed和原始模型加载，再逐个Depot开工将
订单快照与原命令issuer/ACK、建筑tag/中心核对。仅观察者读取通过不等于归因通过；
如订单缺失、多候选或无法解释，保留未知，不修改匹配/奖励来追求覆盖率。
结束记录真实退出码、回放、8局胜负/reward及订单覆盖，不以八局胜率证明学习。
冻结SHA、实际启动和结果在执行后追加；本段不声称评测已经启动。

### r2 实局结果（2026-09-12 22:13完成）

**订单记录验收通过，命令到建筑的归因未通过；没有修改奖励或做PPO。**
运行SHA为 `3f06211ca3179e5b9fb3348d20594fd7ed3c0d60`，独立冻结目录
`sc2-a100-43:/root/new-scii-execution-feedback-20260911/code-3f06211`。
北京时间22:05:26实际启动，22:13:17正常退出0，7分51秒，未触发15分钟上限。
22:14:42核验八卡均0MiB、无compute/tmux，冻结目录仍clean，未终止旧SC2进程。
`result.json`及控制目录`.done`记录完成；manifest保存启动快照，不用其`running`
字样覆盖结束状态。没有重试、训练、checkpoint保存或模型下载。

原始SpB的11个权重分片加载完成，类为`SSESWAMoBAForCausalLM`，不是日志目录名
`Qwen3-30B`对应的模型。模型manifest哈希及资产路径见[r2结果](commandtrace-r2-results.json)。
与r1实际effective config比较，仅替换已知RUN、临时目录及冻结目录后完全相同。
24条回放/干预的最终输入哈希和转发回答逐条等于r1；八局实际游戏seed均52620，
是1个独立场景、4个采样配对，不能按8个独立游戏种子给胜率置信区间。

| env / 组 | 游戏结果 | 整局累计reward | 消费回答数（其中在线） | Depot开工/完成 | Marine出生 |
| --- | --- | --- | --- | --- | --- |
| 0 / A | 胜 | 20.5 | 4（1） | 5 / 4 | 37 |
| 1 / B | 胜 | 20.5 | 3（0） | 12 / 12 | 39 |
| 2 / A | 胜 | 20.5 | 4（1） | 4 / 4 | 37 |
| 3 / B | 胜 | 20.5 | 3（0） | 12 / 12 | 39 |
| 4 / A | 胜 | 20.5 | 4（1） | 4 / 4 | 37 |
| 5 / B | 胜 | 20.5 | 3（0） | 12 / 12 | 39 |
| 6 / A | 胜 | 20.5 | 4（1） | 5 / 4 | 37 |
| 7 / B | 胜 | 20.5 | 3（0） | 12 / 12 | 39 |

28条消费回答为20条历史、4条人工顺序干预、4条在线SpB；格式/语义有效28/28，
不能把24条干预算成模型选择。日志有32行：B结束后还有4行非活跃占位，
`policy_action_applied=false`且不可评估，不能再累计它们的终局reward。
按各env首次done收尾的累计reward均20.5，与实际评测汇总一致；冻结collector
`rollout_loop.py`按active mask累加，没有将占位重复计入整局结果。

**订单证据：**382条被记录的Depot/Marine命令均有唯一映射的wire ACK成功，
来源分别为历史328、人工52、在线2条；ACK成功不等于建成。66次开工均有且仅有
一个已观察SCV，其原始ability319订单的坐标恰等于该建筑中心（66次均为坐标目标，
没有tag目标案例）。全部历史候选在开工前已取得ACK，未借用未来结果。
64个Depot有唯一完成tag；A env0/6各1个在日志最后帧才开工，未记录完成，
不能补记为成功。Marine出生304个，仍不归属具体训练命令。

一个实际例子（env0，事件5→6→7→8）：

```text
命令5：SCV 4347920385，Depot目标(28.5,22.5)，来源历史回答1
请求6：确实发出该命令；回复7：ActionResult.Success=1
开工8：建筑4351852545，中心(29,23)
同帧该SCV原始订单：ability319，target_world_space_pos=(29,23)
冻结匹配结果：command_id=null（未采用坐标取整或新关联规则）
```

50次开工的对应SCV只有1条历史Depot命令，另外12次有2条、4次有3条。
例如env0事件22中SCV4347920385的当前订单目标是(25,27)，但记录仍保留命令5、18。
这说明**观察到具体SCV在建该建筑，不等于已唯一关联原BO命令**。旧严格坐标条件
继续得到0/66，不能将记录器自身缺失关联的返工作为项目原有bug修复成果。

相对r1，24条干预输入/回答相同，但4条在线回答中env0/6不同；r1完成65个Depot，
r2为64个。在线输入哈希均相同，不能声称整局命令逐字节相同，也不能仅由这次差异
认定记录导致模型变化。该次验收只支持订单快照可用，不支持游戏提升或确定性结论。

### r2 CPU复核与保存

新增离线脚本只按事件顺序核对原下令tag、完整历史命令集合、原ACK及原始订单，
报告多候选，不采用最近/FIFO/取整匹配，不写回运行日志。4项测试覆盖多历史命令、
拒绝/不确定ACK、缺失工人和订单目标类型，合并既有测试共38项通过，1.141秒。

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:tests python -m unittest \
  test_depot_issuer_audit test_sc2_command_trace test_depot_coordinate_audit \
  test_supply_order_eval -q
python experiments/records/20260911-execution-failure-feedback-v1/audit_depot_issuer.py \
  /private/tmp/sc2-commandtrace-r2.Cw8kFL \
  --output /private/tmp/sc2-commandtrace-r2-issuer-audit-20260912.json
```

沿用本页本地Python。全66条审计退出0，完整审计179679字节，SHA-256
`afc734927428c866ebefaf22fd27332a4856edf904586736974f7b380891bf18`，
包含每份JSON/JSONL输入哈希和逐开工事件。原始轻量证据包SHA-256
`e225957585ea01d24447efaaea4136fbf4441838a7afddb983c92a84adb14dae`，
远端为 `/root/new-scii-execution-feedback-20260911/commandtrace-r2-evidence-20260912.tgz`，
本地为 `/private/tmp/sc2-commandtrace-r2.Cw8kFL/evidence.tgz`。原日志、回放、退出状态
仍在唯一run/control目录，不覆盖r1或其他历史结果。启动源码完整bundle本地/43各一份，
SHA-256 `476197c6673cc09a58f47a6faa5fb7f8d4c98aa98ad4e22835f103a4e8f8fde0`。

**唯一下一项：**利用已有开工/完成tag和订单时序，离线核对那16次多历史命令，
区分已完成的旧命令与真正有歧义的待执行命令。先给关联规则证据与最小CPU对照，
不新增GPU、不先把未验证归因接进奖励。记录通过不等于纠错信号或RL学习验收。

收尾回归复用上文113项命令并加入`test_depot_issuer_audit`，117项通过，5.279秒；
24个登记源码哈希及`git diff --check`通过。结果JSON的6个产物哈希、逐局首次done
累计reward再次核对通过。收尾只新增离线审计/测试与结果文档，没有修改运行源码。

### r2 多历史命令核对（9月12日，CPU完成）

用户批准继续只读分析。复用`22a138e`的审计和原始r2产物，逐份复核上轮保存的
JSON/JSONL哈希及证据包哈希，运行源码仍为`3f06211`，未启动新游戏或PPO。

**16次多候选均可由已完成旧命令解释，不是观察到了16次并行未完成施工冲突。**
下表命令/事件ID均在单局内使用；相同模式在同seed的多个env重复，不是独立场景。

| 出现范围 | 开工事件 | 原候选命令 | 在新命令发出前已完成的链 | 剩余命令 |
| --- | --- | --- | --- | --- |
| 全8局 | 22 | 5、18 | 命令5→开工8→完成17 | 18 |
| B的4局 | 199 | 9、183 | 命令9→开工12→完成21 | 183 |
| B的4局 | 200 | 5、18、189 | 5→8→17；18→22→24 | 189 |

例如同一个SCV：命令5于11.786秒发出，14.464秒开工，35.893秒完成；命令18
到43.750秒才发出，46.429秒开工。不能把已经结束的命令5和新的18当作两个仍未解决
的施工请求。全部20次被排除的旧命令引用，其完成事件都早于本次新命令的提交。

代码原因已具体定位：冻结`sc2_command_trace.py:201–206`要求提交坐标与建筑中心
严格相等，当前半整数命令点不满足，因此`building_tag`一直为空；181–184行又按
`building_tag is None`保留历史命令。这是**本次新增记录器的关联缺口**，不是证明
原Executor重复执行，也不是RL不学习的新根因。

离线规则只在记录范围内建立可检查的解释链：当帧只有一个SCV的建造订单目标等于
该建筑坐标/tag；回看其此前命令，只有已经唯一对应更早开工、且同tag同坐标完成
事件已出现的命令才排除；剩余恰一条且此前ACK成功、未对应另一建筑时才记录推定
关联。没有取最近、FIFO、坐标取整、未来完成补证或按最近Planner回答借用来源。
50次原本单历史命令可建立首段链，16次经完成历史排除后得到唯一后续链。
结果是66/66在记录内可唯一解释，原运行日志依然是0/66；这两个数不能互相替代。

数据依据是[Blizzard原始协议的UnitOrder/Unit字段](https://github.com/Blizzard/s2client-proto/blob/master/s2clientprotocol/raw.proto#L47-L54)：
订单包含ability、目标坐标或tag，建筑另有tag和build_progress；没有本项目command ID。
本地固定burnysc2 7.1.3的`bot_ai.py:893–908`也用施工SCV的订单目标与建筑坐标/tag
核对施工占用。**公开代码支持这些字段的用途，不等于公开项目已经实现本诊断的
完整命令归因算法**。唯一关联仍依赖本次命令记录完整性、时序及一对一约束，不能
声称是引擎直接返回的因果ID；跨seed、取消/接建等场景未做新实局验收。

该解释链下，Depot开工来源为历史32、人工32、在线2；64个完成事件全部来自历史
或人工回答，在线2个仅开工未完成。因此本次数据**不能提供模型纠错完成的正样本**，
更不能给模型补上这32个人工Depot的奖励。Marine归因和奖励候选仍未验证。

复用`audit_depot_issuer.py`新增可选`--completed-history`，不改默认审计口径。
反例覆盖：未来或未完成/错位置的完成事件、两条未解决命令、两个SCV指向同建筑、
拒绝/不确定/未来ACK、同命令或建筑重复归属。为避免离线分析借用后来ACK，候选
保存当时字典副本（分析工具自身修正，不计项目修复成果）；实际r2原审计字段逐项
与上轮一致。新增6项测试，与既有相关测试合计44项通过，1.092秒。

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:tests python -m unittest \
  test_depot_issuer_audit test_sc2_command_trace test_depot_coordinate_audit \
  test_supply_order_eval -q
python experiments/records/20260911-execution-failure-feedback-v1/audit_depot_issuer.py \
  /private/tmp/sc2-commandtrace-r2.Cw8kFL --completed-history \
  --output /private/tmp/sc2-commandtrace-r2-completed-history-20260912.json
```

审计退出0，结果230746字节，SHA-256
`14e2920b1d9977cb1a0baed1d301f4a990211ad97cecee0e0ef28c99f4308d07`。
原r2及上轮审计文件未覆盖。汇总见[生命周期复核](commandtrace-r2-lifecycle.json)。

下一项最小方案是**只修可选记录器的关联逻辑**，采用已核对的订单与完成时序证据，
缺失/多候选继续未知；先将原66次事件和上述反例作为同输入CPU A/B，要求只改变
诊断归因字段，不改任何游戏命令、BO扣减、prompt或reward。须确认后才改运行模块；
本轮没有实施。暂不申请GPU，不把过程奖励接入RL。

最终扩展回归按上文117项的相同模块执行，共123项通过，5.181秒；24个登记源码
哈希、结果哈希/计数和`git diff --check`通过。相对运行SHA，`agent_system/`、
`verl/`、`gigpo/`、`configs/`均无差异。仅在现有诊断分支保存分析、测试和记录。

### 记录器关联候选（2026-09-13，CPU通过）

用户确认只改可选记录器、先同输入CPU A/B。基于clean `26695c6`，复用当前分支，
不创建新分支、不修改旧deployment/run。生产代码仅`agent_system/sc2_command_trace.py`
有变化，并同步其登记哈希；Planner、Executor、LLMPlayer、reward、训练配置均不动。

关联不再仅要求命令点等于建筑中心，而是要求：有下令记录的SCV中，恰有一个
当帧建造订单目标等于该建筑坐标/tag；该SCV历史命令排除**同tag、同开工坐标、
progress=1且非重复完成回调**确认的旧命令后，恰剩一条ACK成功且未绑定其他建筑
的命令。已开工但未完成的旧命令仍参与排歧义，未知/拒绝ACK也不被偷偷排除。
缺订单时，即使坐标相等也不回退猜关联；多候选、不在场工人、重复绑定继续未知。

为保留审计依据，`issuer_observations.command_ids`保留该SCV全部已发送Depot历史；
`candidate_command_ids`保存此次实际未完成候选。内部新增完成标记和开工坐标，
不会进入BO数据、reward或模型输入。完成事件仍引用原command ID，不能借最新回答
的来源。无新增RPC、工人命令、采样或重试，Marine出生关联继续未知。

复用r2哈希确认的8局、66次开工、64次完成，调用**原SHA和候选的实际记录器方法**
进行CPU A/B，不是只运行离线规则。`replay_depot_recorder.py`将录制命令和请求作为
输入，执行`wire_result`及施工回调；用真实SDK/protobuf重建已保存的订单，不启动SC2。
它不重跑模型、Commander或整局模拟。未记录的开工progress用占位值，仅供创建Unit，
两版开工匹配均不读取该值；完成progress、坐标/tag、订单和ACK均来自真实记录。

| 同一批录制事件 | 原r2记录器 | 候选记录器 |
| --- | --- | --- |
| 关联开工 | 0/66 | 66/66 |
| 关联完成 | 0/64 | 64/64 |
| 原始非归因字段 | 与历史逐字段相同 | 与原版逐字段相同 |

原版先逐条重现r2原事件；候选只有开工/完成的`command_id`、候选列表、匹配类型和
来源字段可变化，其余全部比较相等，SCV protobuf也未被修改。候选关联逐一等于
上轮独立生命周期分析。两条在线模型Depot仍仅开工，没有伪造完成事件或模型奖励。
6项新增运行类反例测试覆盖完成时序、错tag/位置、未完成、未来/拒绝/未知ACK、
多命令/多工人、重复绑定和缺失工人；既有真实SDK命令及prompt/reward不变测试保留。

本轮初次测试因旧临时Python包文件缺失而未导入，非生产代码失败。在独立
`/private/tmp/sc2-depot-cpu-20260913.Gq5Waa`恢复burnysc2 7.1.3、s2clientprotocol
5.0.16.97563.0、protobuf 3.20.3，未修改旧/服务器环境或安装GPU框架。首次回放夹具
漏传SDK Unit所需game_loop，已修正夹具并成功复跑；不将诊断脚本返工作为项目bug修复。

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:tests python -m unittest \
  test_sc2_command_trace test_depot_recorder_replay test_depot_issuer_audit -q
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. python \
  experiments/records/20260911-execution-failure-feedback-v1/replay_depot_recorder.py \
  /private/tmp/sc2-commandtrace-r2.Cw8kFL \
  --audit /private/tmp/sc2-commandtrace-r2-completed-history-20260912.json \
  --output /private/tmp/sc2-depot-recorder-ab-20260913.json
```

使用上述新环境Python，38项通过（0.461秒），回放退出0。完整结果51314字节，SHA-256
`8324ce977b1542bf9b8b6163dc4b58479a0c2dd94a25711de6249f5f4e145504`；
候选源码SHA-256 `b5f733c394348bdb53fa2e43faef8bc63a7dba24d3d4a79a0f8501f23cdad6e1`。
原始产物及旧匹配结果保留。该结果只验收候选在既有输入上的记录关联，不证明新实局
无副作用、模型已学会纠错或RL有效；新实局需要另行资源批准，奖励仍不接入。

扩大回归在原123项模块中加入`test_depot_recorder_replay`，129项通过，5.078秒；
24个登记哈希及`git diff --check`通过。摘要见[候选CPU结果](depot-recorder-candidate-cpu.json)。

### r3 新实局准备（9月13日，planned，预算待确认）

用户要求继续下一步。本轮先准备仅记录的新实局验收，不复用r2资源授权。
候选核心SHA为`796e77742b859b5d34b1ce0fc70d9bd8b7a08091`；本次仅新增薄入口
`run_depot_association_node43.sh`及其配置一致性测试，未改运行模块。
拟在43 GPU0–7用一个原始SpB实例重放已有Level6/g2顺序A/B各4局，实际游戏seed
均为52620，沿用r2的全部数值/游戏配置、前三答来源及192-token预算。
这是一个场景的记录器验收，不是新的战术对照，也不是八个独立种子的学习评估。

拟议总预算为从实际启动起15分钟（885秒加15秒退出预留），不做PPO、不开新奖励、
不保存checkpoint、不复制大权重、不自动重试。CPU配置对照、准备或资源空闲均不
构成启动成功；必须用户确认本轮预算后才上卡。新入口保留空卡、clean SHA、已登记
源码哈希、授权标记和独占目录检查。

新RUN为`20260911-execution-failure-feedback-v1-spb2-original-l6-g2-commandtrace-ab8-8gpu-s52620-r3`，
输出沿用43本地反馈实验根。控制目录为`/root/new-scii-execution-feedback-20260911/commandtrace-r3`，
临时目录`/dev/shm/ct_0913_r3`，冻结源码仍用同一项目根下新的`code-<SHA>`。
9月13日11:12北京时间只读检查：八卡均0MiB、无compute进程，主机可用448GiB，
本地盘可用94GiB；上述新目录未使用。仍有16个旧SC2进程，路径属于9月10/11日
历史长训；本轮没有终止它们，启动前须重新检查归属与资源，不将本快照视作永久空闲。

验收读取实际请求/ACK、SCV订单、开工tag、同tag完成、原回答来源及首次done整局
结果。逐个关联须满足候选的唯一订单、唯一未完成ACK成功命令和时序条件；缺证据
继续未知，不能为了覆盖率放宽规则。独立离线审计与实时关联逐项核对，重复完成不
重复计数，历史/人工/在线来源分别统计。旧r2原始关联结果仍保留0/66与0/64。
本轮不会因8/8胜或关联覆盖提高宣布学习有效；Marine归因、模型纠错奖励及RL
改善仍不在此次验收范围。

准备验证：`PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:tests`配合上述独立CPU环境
执行`python -m unittest test_supply_order_eval -q`，7项通过（0.661秒）；其中dry-run
逐字段核对原顺序入口、r1、r2、r3配置，除记录字段、目录/运行身份及既有依赖位置
外一致。24个登记源码哈希及`git diff --check`通过。此处未加载模型或启动SC2。

### r3 实际启动（9月13日11:20，北京时间）

用户本轮确认上述八卡15分钟预算。使用clean冻结SHA
`95f6e2185acb37d792970010429e8828e0c40164`，独立部署
`sc2-a100-43:/root/new-scii-execution-feedback-20260911/code-95f6e21`。
Git bundle已在本地及43保存并核验，SHA-256
`f7d8d139f713e45288d0e214dd477769000416fdcbb0575f1e91d8e3b97377af`。
服务器CPU运行`test_sc2_command_trace test_depot_recorder_replay test_supply_order_eval`
35项通过（3.650秒），24个源码哈希通过。旧SC2进程和旧结果未动。

会话`sc2_commandtrace_r3`于11:20:44启动；原始SpB的11个权重分片加载完成，
八局11:22:32已进入开局并请求Planner。控制日志/开始时间保存在预定`commandtrace-r3`，
不是前台SSH任务。实际预算截止11:35:44，无自动重试、续训或PPO。
`supply_evaluation.json`确认`val_only=true`、`resume_mode=disable`、
`loaded_global_step=0`、`optimizer_updates=0`，实际游戏/采样种子与r2逐一相同。
实际effective config经仅替换代码/运行/临时目录后与r2逐字段相等。

运行中只在本地准备离线验收：复用`audit_depot_issuer.py`的独立历史关联规则，
显式`--association-candidate`仅接受本r3 SHA/RUN，核对候选新增的全量下令历史快照，
逐事件比较实时开工/完成的command ID及来源。不把实时关联值当作推断依据。
旧r2默认审计行为保留；未改冻结源码或在服务器替换运行脚本。

### r3结果：记录关联通过，学习未验证

11:28:34完成，控制器和`result.json`均退出0，实际470秒，预算900秒。
11:29:37已无GPU compute进程，11:31:50确认八卡均0MiB；无新checkpoint、无PPO、
未重试游戏。收尾日志有Ray worker SIGTERM提示，但八局均已自然结束、完整保存，
TaskRunner和控制器退出0；不把该提示单独当成训练/游戏失败。

| 证据 | 原r2运行 | 本r3运行 |
| --- | --- | --- |
| 实时开工关联 | 0/66 | 66/66 |
| 实时完成关联 | 0/64 | 64/64 |
| 独立订单/完成历史复核 | 66次可解释 | 与66条实时关联及64次完成逐条一致 |
| 自然胜负、整局累计reward | 8/8，每局20.5 | 8/8，每局20.5 |

382条记录命令均有成功ACK。仍有16次含已完成旧命令的历史，20个旧命令引用被
先前完成证据排除；没有用最近命令或最新Planner回答猜归属。按env0–7，Depot
开工为`5,12,4,12,4,12,5,12`，完成为`4,12,4,12,4,12,4,12`，Marine出生为
`37,39,37,39,37,39,37,39`，均与r2相同。Marine命令归属依然未知。
实际配置经已登记路径替换后一致；前三答共24份prompt哈希/转发回答逐项一致，
后续4份在线回答也与r2相同。这些核对未发现游戏行为变化，但不证明全部执行时序
逐字节相同，也不外推到跨seed、取消或接建场景。

真正消费28答：历史20、人工4、在线4，均通过原validator。32条回答日志中有4条
已结束游戏占位；逐局累计reward只累加到首次done，不能把B的终局重复累加为41。
64个Depot完成的原命令来源为历史32、人工32；在线2个只开工未完成。示例：env0
在线回答为`- supplydepot x 1 / - barracks x 1 / - marine x 1`（实际为answer标签
内三行）；不能把它的其他未执行项或32个人工Depot当作在线纠错完成。

完整证据在`/private/tmp/sc2-commandtrace-r3.kKRGqU/verified`；远端原run及replay
保留。已核验的证据包为项目根下`commandtrace-r3-evidence-v3-20260913.tgz`，SHA-256
`12cd28614b26e7628424ccaa3a7790e2d1ffd6f1a8d97caabe42112e3d42dcbd`。
离线审计`/private/tmp/sc2-commandtrace-r3-association-audit-20260913.json`的SHA-256
为`28fd56a97b2f982e186e4af1fd96173b8e1a6e7e7003e15b366e94e34969b252`。

```bash
python experiments/records/20260911-execution-failure-feedback-v1/audit_depot_issuer.py \
  /private/tmp/sc2-commandtrace-r3.kKRGqU/verified --association-candidate \
  --output /path/to/new-audit.json
```

收尾工具的两次打包失败（参数位置、误判回答文件所在目录）保留为前两版证据包，
未覆盖原run；v3打包/解包成功。新审计起初误读完成事件应含origin，已按冻结schema
改为通过command ID查原命令来源；这不是生产bug或本次游戏失败。CPU扩展130项
通过（5.012秒），r2旧审计输出逐字节相同。没有将收尾工具返工算成项目修复成果。

**采用范围：仅保留该可选记录器，单场景实局关联验收通过。** 不晋级奖励或RL。
下一项先利用已有失败输入和回答，界定真正在线纠错的可归因正/负样本及奖励触发
边界，尤其排除人工回放和先造堵再解堵的激励；不继续为凑覆盖率修改记录器，
也不补猜Marine来源。新奖励机制和新GPU预算仍需用户另行确认。
摘要与哈希见[本次结果](commandtrace-r3-results.json)。

9月13日后续只读复核：四条在线回答中，两条Depot优先确实下令并开工，但自然胜利
时进度仅82.5%；另两条Marine优先未发出新的Depot/Marine命令。四局累计均20.5。
前序失败来源是历史回放，不能作为严格的模型自发纠错奖励正样本。详见
[在线纠错与奖励边界](supply-reward-candidate.md#9月13日记录器验收后复核真实在线纠错)。
记录器维持冻结，不扩展、不开启奖励；新评测仍须单独确认预算。
