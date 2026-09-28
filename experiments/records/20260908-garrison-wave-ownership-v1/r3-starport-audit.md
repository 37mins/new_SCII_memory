# r3 星港未完工：施工 SCV 阵亡后无人接续

2026-09-08，只读分析已结束的 SpB Level6/g2、seed2807211443 单局。
RUN_ID：`20260908-garrison-wave-ownership-v1-spb2-l6-g2-s2807211443-planner-r3`。
冻结 runtime `e1e25ce1d14362b209e34ec73871ebaeb63538a6`；manifest 显示正式、clean、
纯评测。当前隔离树的整个 SC2ArenaDyStrategy 与该 SHA 无差异；未修改游戏源码。
原 replay、console 的 SHA256 与 result 登记一致，原始事件见 [JSON](r3-starport-audit.json)。

## 已确认的因果链

| 游戏时间／loop | 实际证据 |
|---|---|
| 10:20.54／13900 | SCV 收到唯一星港建造命令，sequence755、raw ability129/11，实际位置(117.5,38.5)。Executor 随后推进 BO。 |
| 10:25.00／14000 | Starport `SUnitInitEvent`，确实开工。 |
| 10:25.98／14022 | 同一施工 SCV 在(118,37)被敌方 Banshee 击杀，距开工仅0.982游戏秒。此刻仍有37个其他SCV存活。 |
| 11:09—11:49 | 模型合法选择的 Viking 到达执行项，10次 `Facility under construction.`，第11次检查按原阈值跳过。 |
| 12:50.85／17267 | 星港被敌方 Marine 摧毁。整个 replay 无其 `SUnitDoneEvent`，没有 Viking 训练命令或产出。 |

SCV日志tag `4345036803`、局部ID803，replay tag `50069507`（日志tag低32位）；
星港replay tag `99352595`。SCV出生后的两条我方命令仅为采矿、建星港；建造命令
之后直到阵亡没有Move、采矿、Stop或其他改派。星港出现后到死亡，没有任何我方
命令以该星港为TargetUnit，未见替代SCV接续。建筑在工人阵亡后又存续144.87秒。

因此本局不是“命令完全没执行”，也不是施工工人被驻防／采矿逻辑改派：
**施工工人被击杀 → 无人接班 → 未完工设施持续阻断Viking。**
37个剩余工人只证明当时不是全图无SCV，不证明它们安全、空闲或来得及救回星港。

## 对应代码与公开依据

以下位置均核对冻结 e1e25ce，不用当前分支名代替历史源码：

- `players/llm_player.py:813`：缓存建造分支调用 `run_actions` 后立即
  `_finish_current_bo()`；该函数418行推进索引、清除当前状态，没有持续施工跟踪。
  “下发后继续下一BO”本身不等于错误；实际缺口是后续没有中断施工接班。
- `players/script_player.py:2081` 的工人分配处理采矿／采气；4294行维护入口
  处理这些任务、MULE、防御、侦察、进攻、驻防，没有未完工建筑的SCV补位调用。
  1443行repair仅属另一路战斗单位微操，不是建筑接续，且本局new-micro关闭。
- `players/llm_player.py:368`：生产建筑不存在ready对象就提示施工中、建议等待，
  不区分正在施工和失去工人的停工，最终走原超时跳过分支。

项目使用的 burnysc2 **7.1.3** 已提供 `BotAI.structures_without_construction_SCVs`
（本地同版本 `sc2/bot_ai.py:889`）：检查未完成、需要SCV、无人以位置/tag施工、
且相邻观测进度不变的建筑；它只是查询属性，不会自动派工。本项目未调用它。
[上游公开源码文档](https://burnysc2.github.io/python-sc2/_modules/sc2/bot_ai.html#BotAI.structures_without_construction_SCVs)
也展示这一接口及接续施工时order目标为建筑tag的语义；网页版是滚动文档，版本
对应依据是本地实际安装7.1.3源码，不能把滚动文档当作冻结runtime。

另外，原日志的预走位目标(131.5,56.5)不是最终建造位置：预走位Move被本地检查
拒绝，`base_player.py:701`建造分支又重新选址到(117.5,38.5)。这两点确实存在，
但星港已开工且工人随后被击杀，现有证据不能把它们定为停工根因；本次不扩展修改。

## 复核与边界

沿用已安装的轻量回放解码环境，不加载模型、不运行SC2、不占GPU：

```bash
/private/tmp/sc2-air-results.HBI61m/decoder/bin/python -B \
  experiments/records/20260908-garrison-wave-ownership-v1/audit_r3_starport.py \
  /private/tmp/sc2-planner-r3-results-20260908
```

命令已执行exit0，重用既有命令解码器并校验原文件哈希，结果保存到上述JSON。
console行号按原始LF计数，保留日志内CR，不混用终端展示行号。
缺少逐帧建筑百分比／血量和工人order观测，不捏造停工百分比；tracker只确定
最后击杀者，不代表它是唯一攻击者。本局不能证明补派工人就能造出Viking或获胜。

唯一建议的下一项候选：复用上述现成查询，只为已开工、无人继续施工的建筑
派可用SCV接续，保留原BO和其他模块。需先获用户确认，再做CPU回归与同seed
单局验收；本轮没有实施该候选、启动实验或声称Level6战斗问题已解决。
