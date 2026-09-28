# BO 段协议恢复对照：A/B 纯评测完成

2026-09-09。用户批准 guard 的最小长度联动后，B 已在独立分支
`codex/experiment-original-bo-segment` 实施，95 项 CPU 回归通过；**A/B 于北京时间
15:59:08全部完成、退出0，A为4/4胜、B为3/4胜；未做PPO，无RL学习结论。**
A 基线、旧 deployment、任务和旧结果未修改。
本地候选目录 `/private/tmp/sc2-bo-segment-20260909`，A/前置审计仍保留在原 worktree。

## 原包与 A 来源

原包 `/Users/yangjinyue/Desktop/codex/星际争霸/code/SC2_online.zip`，完整 SHA256：
`e40fb7dad98f4c7d336031f3b2047ee3e15113ce14925830cfb6540aaa5888ab`。
直接读取 `SC2_online/SC2ArenaDyStrategy/` 内文件，没有整包替换或导入执行依赖。
包内未找到 LICENSE/COPYING 文件；本次仅提取 `get_bo` AST 在 CPU 作解析参照，
不据此宣称具有额外的再分发许可。

| 原包位置 | 实际行为 |
| --- | --- |
| `agents/planner.py:13` | 明确要求约 15 条列表条目，不是总单位数；攻击指令算一条 |
| `agents/planner.py:105–126` | `get_bo()` 逐行读取 `x N` 并展开，没有 15/96/512 个展开动作硬上限 |
| `agents/planner.py:195–220` | 组装阶段/观测/待办 → 原始回答 → `get_bo` → 返回；没有硬截 15 行 |
| `agents/commander.py:188–202` | 遍历全部返回动作、扣减待办、包装 action/intent，无队列段长截断 |
| `players/llm_player.py:64–69,613–624` | 固定 BO 加载被注释；空队列调用 Planner，之后逐项执行 |

本次 A 固定为 `0d7ba09ddbdf94d85d67b553bf7978f402cf2052`，仍为 6/6。
它比真实 RL runtime `e3ffac9e2da577679b9b980d66678e4cc72b06aa` 多了已批准的
两行 Factory 前置候选修复；**该修复仅 CPU 验证，A/B 均应包含，不能只放 B。**
本次测试输入不触发这两行，且其他协议/Commander/validator 源码相同。
源文件 Planner SHA256 `a3668a9cee19ec2077158a5dd4a758955050b509467f671d5619c2379a91cf84`。
修改前 A 完整 Git 历史备份为 `/private/tmp/sc2-bo-segment-baseline-0d7ba09.bundle`，
`git bundle verify` 通过；SHA256 为
`37cffb39b32cc6b59710f1c101e6f9c0486bbdb271efdff3e87e1f3e2ca335ec`。
前置审计提交为 `23a2d5945f34aed8501799a86082d9bfe0fe565b`，生产代码同 A；
本次 B 从该提交新建独立候选分支，未修改 A 所在工作树。

2026-09-09 14:56:51 北京时间，44 八卡均为 0MiB；只做读取，没有占卡或停任务。
随后直接核验旧 `code-e3ffac9` 为上述 clean SHA，旧 Planner 哈希为
`b6994c43da70ab21407e3ec20fb4538f1dbd2923172e8a5051a51e0e0944f305`。
原训练 result 保持 failed/exit1、0 次更新，结束于 14:28:54，不改写失败记录。
这次资源快照不构成新的资源授权。

## 当前真正生效的限制

以下行号以 A 为准。Planner、Commander 均位于
`agent_system/environments/env_package/sc2_online_restored/SC2ArenaDyStrategy/agents/`。

| 位置 | 实际限制及恢复时的要求 |
| --- | --- |
| `configs/system-v1.env:10–11` | 默认 segment=6、bullet=6；两者不能直接改成 15/15 |
| `planner.py:23–27,1191–1196` | prompt 明确禁止 `x N`，要求最多 6 行/6 动作；B 需恢复约 15 条、支持数量 |
| `strategic_sc2_env_manager_v2.py:889–898,924–938` | 原始整段回答先检验精确名称，再检验行数/展开数/剩余配额；B 要接受精确名称后缀数量，解除段长硬限，保留名称和配额约束 |
| `planner.py:379–428,1263` | 原始回答先截 6 行，strict parser 不解析 `x N`，展开还受 6 上限；B 需联动，不能仅改 prompt |
| `planner.py:1287–1301` | 合法回答直通；无效回答进入原规则补齐/重排。不得连同全局 segment 值调大规则补兵量 |
| `planner.py:1302–1307` | 带攻击时额外把生产动作截到 segment−1；B 需解除这个长度截断，保留现有攻击筛选与 before/after 位置策略 |
| `commander.py:132–173` | 开局补 Marine 的规则会先删动作到 5 个，再加 Marine，结果最多 6 个——这是冻结边界冲突 |
| `commander.py:303–325`、`players/llm_player.py:678` | 接收最终列表、包装、整体交给 Executor；未发现另一个统一 6/15 队列切片 |

`format_bo_todo` 的 80 项调度序列和 `constrain_bo_to_todo` 的 per-item caps 属于当前
规则后处理，不是 B 合法模型计划应经过的数量上限。保留这些规则，不把取消它们
混进本次对照。当前未调用的局部 `check_format` 写着 15，不能当成有效协议证据。

## CPU 结果及样本身份

脚本 [audit_cpu.py](audit_cpu.py) 使用原包真实 `get_bo`，以及 A 的真实
Planner/get_action/validator/开局 guard，模型 socket 由已有 CPU 测试方式替代。
没有加载模型、运行游戏或向 SC2 提交命令。完整原始回答、校验原因、截断文本、
解析结果及 A 的最终动作队列见 [cpu-audit.json](cpu-audit.json)。

真实回答来自已停止 r1 的 `prompt_response.jsonl`，第 38、46 行（env5）；
来源 RUN：`20260908-g2-planner-handoff-v1-spb2-l6-opening-rl96-gate8-8gpu-s0-r1-train8`，
节点44，共享根 `/mnt/data0601/yangjinyue/new_SCII/experiments/20260908-g2-planner-handoff-v1/runs/`。
日志完整 SHA 保存在 JSON。本次日志没有找到合法的 7–15 条或 `x N` 原始回答；
因此边界用例明确标记为从真实重复列表截取/数量编码或合成测试，**不冒充生成结果**。
CPU回填的是日志中的观测、阶段文本与已过滤候选，不声称从query恢复了Commander
原始未过滤的waiting_bo。现场命令是否执行不由这个CPU替身判断。

| 输入 | A 校验/解析及最终队列 | 原包 get_bo 解析参照 |
| --- | --- | --- |
| 原始 6 条（第46行） | 合法，原顺序保留 6 个 | 6 个 |
| 7 / 15 条 Marine，真实上下文剩余42（派生） | 行数超限，先解析前6个，再规则生成 `supplydepot,barracks,refinery,marine,barracks,reactor barracks` | 7 / 15 个 Marine |
| `- marine x 29`，剩余42（等价数量编码） | strict parser 不接受，解析[]，进入上述规则队列 | 29 个 Marine，非15个 |
| `- supplydepot x 2`、`- barracks`、`- marine x 15`（合成） | 数量写法无效，仅解析 barracks，进入规则队列 | 3条展开18个动作 |
| 原始29条 Marine，剩余8（第38行） | 先报行数超限，解析6个，规则最后保留3个 | 解析29个，但仍超当前配额；恢复长度不能让它合法 |
| `- barrack`（合成负例） | 名称不在表内，拒绝 | 原包可提取文本，但 B 仍须保留当前精确名称过滤 |

原包 get_bo 不检查当前剩余配额，故本表的“解析成功”不等于 B 语义有效。
前置审计中的 B 最终队列字段为 null：当时没有用“原包 parser → 当前系统”的不完整拼接
冒称 B 全链路通过。token 截断率也不能从本次纯文本 CPU 测试推断。

独立 guard 边界用例输入 3 个 supplydepot + 4 个 barracks、开局且仍需 Marine：
真实代码删除 2 个 barracks，再补 1 个 Marine，只剩 6 项。此例为合成的规则边界
验证，不是假称真实模型回答。说明只改 prompt/parser 不能完成恢复。

前置审计实际执行命令（在 `23a2d59` worktree，退出0，7个解析用例及guard断言通过）：

```bash
PYTHONDONTWRITEBYTECODE=1 /private/tmp/sc2-air-results.HBI61m/decoder/bin/python \
  experiments/records/20260909-bo-segment-protocol-v1/audit_cpu.py \
  '/Users/yangjinyue/Desktop/codex/星际争霸/code/SC2_online.zip' \
  /private/tmp/sc2-opening-r1-protocol-audit-20260909.jsonl
```

## 已批准的最小联动

另运行既有相关回归：`PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:tests /private/tmp/sc2-air-results.HBI61m/decoder/bin/python -m unittest test_factory_prereq_candidate test_planner_air_options test_valid_bo_passthrough test_r6_planner_interface test_r6_format_reward test_g2_planner_handoff test_system_release test_opening_rl_launch -q`。
75项通过、退出0、7.159秒。审计脚本重跑输出与记录逐字相同。生产源码没有变化，
这些结果不替代尚未实施的 B 全链路验收。

用户随后明确批准：B 保留开局 guard 的触发条件、插入位置及“补一个 Marine”行为；**只对通过
B 整段校验的模型回答，取消该 guard 为凑 6 项而删除原动作的步骤**，新增 Marine
仍单独归为规则动作。无效回答继续完整走 A 原有截断、补齐、重排和 guard，不增加
规则配额。这改变了 guard 的长度处理，不能称“规则逐字未动”。

已使用默认关闭的 `SC2_ORIGINAL_BO_SEGMENT=1` 开关实施：Planner 的段长提示、
数量示例、精确名称数量解析、行/动作截断及攻击分支长度；manager 的数量/长度
校验；Commander 上述唯一长度联动；launcher 只传递开关。名称/前置/配额过滤、
无效回答规则补兵及重排、攻击资格/位置、reward计算、微操均保留。
这是 **BO 段协议恢复对照**，不是单一参数变化或整个压缩包恢复。

## B 实现与 CPU 验收

原有 `SC2_PLAN_BULLET_MAX=6` / `SC2_PLAN_SEGMENT_MAX=6` 配置保留，用于 A 和 B
无效回答的冻结规则路径。B 合法回答不读取它们作为长度上限；不设置 15/15，
也没有暗换成 96/15 或 512 上限。名称必须精确、正整数数量不超剩余配额，仍拒绝
`barrack`、`siege tank`、`#1` 等。原包可提取 `x 0` 为零动作；本候选不恢复这种
无动作条目，维持有效动作要求，`x 0` 判无效。当前可用表、阶段/观测和攻击资格不变。

改动位置：`planner.py`（提示/示例、经验证的计数解析与长度）、
`strategic_sc2_env_manager_v2.py`（对应数量和长度校验）、`commander.py`（已批准
guard 联动），`scripts/train_sc2.sh` 仅向 Ray 传递开关；manifest 同步源码哈希。
未改 Executor、微操、自动战斗、模型/数值路径、生成预算、reward公式或训练器。
格式有效性随协议变化可能影响原格式奖励，这是同一协议变化的结果，非修改reward公式。

新 [compare_cpu.py](compare_cpu.py) 调用真实整段校验、Planner、Commander.renew_bo
及 guard，输出 [cpu-ab.json](cpu-ab.json)。A 在内存中直接加载冻结 `0d7ba09`
源码，不用当前关闭开关的源码假冒 A；另测开关关闭与原 A 的 query、队列、扣减及
fallback 完全相同。日志/派生用例身份沿前置审计，不将合成边界说成模型生成。

| 用例 | A | B |
| --- | --- | --- |
| 原始6动作 | 有效，6个原动作 | 相同 |
| 7 / 15 条 Marine，余量42 | 长度无效，规则6项 | 有效，原样7 / 15个 |
| `marine x 29`，余量42 | 数量写法无效，规则6项 | 有效，完整29个 |
| 3条展开18动作 | 数量写法无效，规则6项 | 有效，完整18个，原顺序 |
| 原始29条 Marine，余量8 | 无效，规则3个Marine | 仍因配额无效，完全相同队列 |
| barrack错名 | 无效，规则补齐 | 仍无效，完全相同队列 |
| 开局3SupplyDepot+4Barracks | 数量写法无效 | 7个模型动作全部保留，再补1个规则Marine |
| 合法20Marine+攻击 | 原段长限制 | 20个全部保留；攻击位置仍按原配置 |

扩展边界测试97/513个展开动作也完整通过，证明不存在15/96/512隐含上限；只是
CPU合成测试，并非模型应在游戏中输出那么长。还验证非法回答的完整旧路径、
前置候选、memory-off/projection透传、同组实际种子派发，以及启动配置差异。

实际运行：

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:tests /private/tmp/sc2-air-results.HBI61m/decoder/bin/python -m unittest test_original_bo_segment test_factory_prereq_candidate test_planner_air_options test_valid_bo_passthrough test_r6_planner_interface test_r6_format_reward test_g2_planner_handoff test_system_release test_opening_rl_launch test_handoff_paired_eval -q
PYTHONDONTWRITEBYTECODE=1 /private/tmp/sc2-air-results.HBI61m/decoder/bin/python experiments/records/20260909-bo-segment-protocol-v1/compare_cpu.py '/Users/yangjinyue/Desktop/codex/星际争霸/code/SC2_online.zip' /private/tmp/sc2-opening-r1-protocol-audit-20260909.jsonl
```

95项通过，退出0，11.333秒；8组完整 A/B CPU 输出断言通过。准备时两个测试问题
（把配置插值当CLI键、AST假对象未注入os）已修正，不计为项目原有bug修复。
这些测试止于 Executor 的 action/intent 队列，不证明引擎接受、单位产出或模型改善。

## 纯评测授权（2026-09-09，独立新预算）

B 的完整 CPU 队列验证已通过。用户明确批准44八卡、A/B各四卡四局、50分钟
共同墙钟上限；不做PPO、不自动重试。A runtime为 `0d7ba09ddbdf94d85d67b553bf7978f402cf2052`，
B runtime为 `a1f15af1553b36cc691c0a19b6be65ea9cd25bda`，本次只新增编排和记录，
不改变两个冻结runtime。B完整bundle已本地验证，SHA256
`4f89e72e9724093d7cf4cb701cdb939b77fbb9659433b514dc59a1b8aa4832e3`。
拟用同一原始 SpikingBrain-2.0-instruct（无项目SFT/RL，manifest SHA
`40d6f1417568d5fda1107cdafd66c7441a679b03fb148851c8f226ecc9cf884c`），
Level6/Harder、Flat128、Terran/Air、相同 g2 资产。保持当前开局 Planner 接管
`SC2_FIXED_BO_PREFIX_LIMIT=-1`，**不是重新执行完整固定 g2**，不转换阶段待办。
两组实际游戏种子均为 `[2732,2732,43567,43567]`（2个独立seed、各2局），同采样
种子安排、各4卡4局、无PPO。可用一个空闲八卡节点分为两组并行；候选预算为
共同墙钟上限50分钟（含加载，最多400 GPU分钟），不自动重试。该预算已由本次用户确认。

两组保持 FP32参数/温度归一化、BF16 autocast、HF/eager/cache、SWA dropout0，
temperature0.4/top_p0.9/top_k0、history0/off、prompt4096/response96、原reward，
阶段待办/微操/自动战斗及其他开关相同。当前Factory候选也必须两组相同。

逐局统计完整累计reward（包括无Planner局）、胜负、真实调用次数及游戏时间间隔、
回答token截断、格式/语义有效率、原始选择/规则动作/最终队列。以JSONL、console
和replay联合核对，不把 projected_action 等同于引擎实际命令。结束保留配置、
退出码及失败记录。若96token与B长回答不匹配，只能报告预算不匹配；单独申请
新预算，不据此说原协议更差。少量纯评测胜负不证明RL学习，本轮不能给短RL晋级结论。

入口为本记录的 [run_eval.sh](run_eval.sh)，复用原 `handoff_eval` 四局配对入口，
不新增种子或战术算法。`SOURCE_DIR`/`SOURCE_SHA` 明确指定 A 或 B 的独立源码；
两组相同 g2 和开局接管，唯一方法差异为协议开关。实际启动须先设共享
`PAIR_DEADLINE`（当前时间+3000秒），包含最后60秒结束宽限；无自动重试。
每组独立 RUN_ID、OUTPUT_DIR、Ray目录、状态文件，受控后台会话仅在授权后创建。
新入口已 dry-run 核对各4卡4局、val_only、resume禁用、不保存训练权重/不执行PPO、
原模型和96token，其余effective config相同。候选不自动续任何长训。

拟用 RUN_ID：`20260909-bo-segment-protocol-v1-spb2-l6-g2-opening-A-eval4-s0-r1` 和
`20260909-bo-segment-protocol-v1-spb2-l6-g2-opening-B-eval4-s0-r1`，运行目录在共享
`/mnt/data0601/yangjinyue/new_SCII/experiments/20260909-bo-segment-protocol-v1/runs/`。
冻结源码拟放44独立 `/root/new-scii-bo-segment-20260909/code-<SHA前7位>`，不覆盖旧目录。
15:33:37北京时间复查44八卡均0MiB、无GPU进程，共享实验目录可读；未保留卡位，
启动前必须重新核验归属。此次未部署、未推送GitHub；旧推送限制未绕过。获得GPU
授权后，正式运行前还需在用户节点核验完整Git bundle备份与源码哈希。

15:38:14北京时间启动前检查：44八卡0MiB，无GPU进程，独立部署及experiment根未占用；
共享盘可读且7.2TB可用，系统盘21GB、/dev/shm491GB可用。现有依赖为
Torch2.7.1+cu128、Transformers4.55.2、Ray2.48.0、NumPy1.26.4，不复制/升级依赖。
[launch_pair_node44.sh](launch_pair_node44.sh) 在两个独立冻结目录核验clean SHA与
源码manifest后，再检查八卡空闲并启动受控pair，共用绝对截止时间。
实际启动状态和SC2证据另行追加；准备完成不算已运行。

## r1 实际启动核验

北京时间2026-09-09 15:55:38核验：两组模型11/11分片加载完成，均为真实
SSESWAMoBAForCausalLM、swa_dropout=0，已进入SC2并消费Planner回答。
pair于15:46:21启动，共同截止16:36:21（含退出宽限），受控会话
`sc2_bo_segment_pair_r1`；A GPU0–3，B GPU4–7。两组manifest均formal=true、dirty=false、
mode=evaluation，分别记录上述0d7ba09/a1f15af冻结SHA；val_only=true、resume禁用，
不做PPO。effective config归一化源码/run/Ray路径后只差协议开关。

RUN_ID为上列A/B各自的`eval4-s0-r1`，没有重试。实际
`evaluation_seeds.jsonl`均为`[2732,2732,43567,43567]`，四卡采样seed均为rank0–3；
不是只设置seed=0就声称配对。首条实际回答A为6条单动作，B为3条`x 2`；B随后
env3/step_index2生成`- marine x 20`，整段语义校验通过。随后核验同一
SC2Worker PID1104413在07:50:13 UTC的`BO ori`解析为20个Marine，
`refresh build order`也完整保留20个；紧接着Executor报告矿不足并等待。
这证明真实队列未被后续截成6/15个，不等于20条生产命令已成功提交或20个单位已产出。

15:55快照：A已有env0/2/3三局自然胜利，B已有env2/3两局自然胜利；其余未完成，
不得将它们当成最终胜率。有效回答统计只计policy_action_applied=true；结束后的
占位回答不计为新模型决策。96token预算不变，完整截断率和结论待结束记录。

启动前八卡无进程；07:47:05/07:47:22 UTC另一任务随后进入A的GPU1/0，分别为
PID1106399/1108127、共同父进程1074732，cwd为其他用户项目`/mnt/data0601/cjy/horizon`，
各约9.5GB显存。未停止或修改它们；这使A/B吞吐不再是资源隔离对照，必须记录此限制。
当前没有OOM证据，也不因此擅自换卡、重启或延长预算。

编排提交`5395d99ab309f4364c76fd4960d21c61d1723020`，完整历史bundle在本地及
44独立目录`/root/new-scii-bo-segment-20260909/source-5395d99.bundle`均已核验，SHA256
`08ac1049c5a75de561aa8a387c6b1f58f819318bc7865c1e0f064bcce325a252`；
含A/B源码及编排历史。远端两个code目录clean且源码manifest全部通过，不修改旧deployment。
GitHub未推送，未绕过之前的推送权限限制。

## r1 完整结果

A于15:57:46、B于15:59:08北京时间结束；两个`result.json`和`.done`均为completed、
退出0、training_updates=0，pair也为A_EXIT_CODE=0/B_EXIT_CODE=0。八局均自然终止，
无无决策局。总墙钟12分47秒，未触及50分钟上限，未重试；本实验八个GPU worker
已释放，只剩前述其他任务。源码A/B仍分别冻结为0d7ba09/a1f15af。

| 指标 | A：6行/6动作 | B：约15条、数量展开 |
| --- | --- | --- |
| 自然胜负 | 4胜0负 | 3胜1负 |
| 平均整局累计reward | 20.5000 | 2.8875 |
| 实际消费的Planner回答 | 39 | 39 |
| 完整回答有效 | 32/39（82.1%） | 19/39（48.7%） |
| 回算文本达到96token | 0/39 | 17/39（43.6%），全无效 |

“有效”是整段回答经当前名称/数量/配额/格式联合校验，不是建成单位比例；本次
格式/语义字段相同，不能把两列相等当成独立验证。仅计policy_action_applied=true，
排除终局占位和已结束环境的重复输出。完整reward按每环境从第一条记录累加到
首次done，含终局奖励；不拿最后一轮`val/sc2_action_valid`（A0.75/B0.25）冒充全程有效率。

| env / 实际seed | A 胜负 / 累计reward / 有效回答 | B 胜负 / 累计reward / 有效回答 |
| --- | --- | --- |
| 0 / 2732 | 胜 / 21.0 / 6/9 | 胜 / 21.0 / 6/14 |
| 1 / 2732 | 胜 / 20.5 / 11/12 | 负 / -49.95 / 4/10 |
| 2 / 43567 | 胜 / 20.0 / 7/9 | 胜 / 20.0 / 5/8 |
| 3 / 43567 | 胜 / 20.5 / 8/9 | 胜 / 20.5 / 4/7 |

四局首次Planner均在00:00。按各局query时间得到调用间隔中位数：A为
50.5/37/38.5/41游戏秒，B为21/60/71/71.5游戏秒；实际时间序列保存在
[r1-results.json](r1-results.json)。B不是所有局都调用更少或间隔更长。

真实B回答`- marine x 20`已完整展开、交给Executor，原数量协议不再被后置6/15上限
截断。另一方面B出现重复`- marine x 2`，最后停在`- marine x`；另有
`Launch OFFENSE when 12 marine are available`、`techlab`、`siege tank`等不符合保留
协议/名称的回答。A也有`barrack`、超剩余配额和1次超过6行，不是完全无缺陷。
B的20条无效回答中17条回算到96token，但长度与无效共现不证明所有错误由截断造成。

长度仅用模型原tokenizer.json重新编码已消费的decoded text（不加special tokens），
**不是从原始生成token IDs/EOS标志得到的精确截断率**；这些标志本次没有保存在
逐回答日志。A文本最长44token，B有17条正好96token，加上半截数量表达，支持
“恢复协议与当前96token预算明显不匹配”，不能据此说原始协议更差。

结论：协议恢复的CPU和真实队列接口验证通过；当前预算下B未显示优于A的系统效果，
不足以支持直接短RL晋级。仅2个不同游戏seed、每个重复2局，不报告伪精确的
跨seed置信区间，也不把3/4对4/4解释为显著差异。A还有额外GPU进程竞争，因此
耗时不作吞吐结论。每组结果目前只索引到1个回放路径，不能声称四份逐局回放齐全；
胜负与调用数依据四条独立outcome及JSONL，不用该单回放代表全部游戏。本次不改归档机制。

下一项最小建议是单独比较B的96与更长response预算，保持本次其余配置；必须另行
报告局数、种子、资源预算并取得授权，当前未启动。没有修改prompt、战术或开长训。
用户随后批准[B192八卡四局验证](budget192.md)：复用B96作参照，不重复旧评测，
最多30分钟；八卡部署差异与预算变化单独列明，不称严格单变量因果对照。
该评测runtime为71fae36，于17:02:11启动、17:21:04完成，退出0、0次PPO；3/4胜，
整段有效22/41（53.7%），仍有15/41条文本回算达到192token且全部无效。
与B96胜负相同，未证明扩大预算解决重复或提升游戏表现；详见上述记录及
[逐局结果](budget192-r1-results.json)，不自动启动RL。

CPU分析脚本[summarize_eval.py](summarize_eval.py)实际在44以现有tokenizers依赖执行，
未加载模型；两组成功汇总，JSON保存run路径、完整SHA、模型manifest、逐局结果、
token长度直方图及manifest/config/outcome/原始回答/console哈希。运行方式：
`PYTHONPATH=/root/new-scii-spb2-level5-control/python /usr/bin/python3 summarize_eval.py <A_RUN_DIR> <B_RUN_DIR>`。
本地AST语法检查和git diff --check通过。此前py_compile因macOS默认cache目录写权限
失败，随后改用不写cache的AST检查；这是本地分析命令问题，不是项目运行bug。
