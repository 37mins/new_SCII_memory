# 人口阻塞：合理选择能否真正执行

2026-09-11；状态：completed，16:14:15北京时间结束、退出0，八局自然获胜。
目标干预覆盖8/8，B解除人口阻塞；A/B每局reward相同，未证明学习提升。
与43的原始SpB反馈短RL并行；不占用43、不修改其runtime `6effc640767891da50c4c70e111c9d510f47b587`。
此处只诊断系统是否容纳纠错选择，不训练、不重做已完成的Refinery顺序实验。

## 对照

从r3 B的env0/seed52620重放前两条回答，到首次真实人口阻塞后的同一完整query。
历史模型是Level7 step24，但本次在线模型为原始SpB；前缀来自已记录回答，不冒充
本次模型自行作出的决定。三个完整输入、原回答及源文件SHA见
[夹具](supply-order-inputs.json)，源码来源 `d155c96eaccf9aa809544945b387f0e17e1ca5bd`。
干预前三次逐项校验完整query SHA；不符合历史输入就终止，不自动放宽条件或重试。

首两轮两组完全相同。第三轮A转发历史回答，B仅将同一条SupplyDepot列表项移到最前：

```text
A：marine×29 → supplydepot×8 → barracks×3 → reactor barracks×3 → scv×2 → refinery×1
B：supplydepot×8 → marine×29 → barracks×3 → reactor barracks×3 → scv×2 → refinery×1
```

均46个展开动作，不增减数量、不改其他项相对顺序。不是新增自动排序机制。
第三轮以后全部恢复原始SpB在线生成，不再干预。两组都打开既有失败反馈，
否则不能在这个原场景收到同一失败后的query；这不是43反馈开关A/B的重复。
人工转发回答会进入既有评测日志的`llm_response`，必须联读`interventions.jsonl`，
其中另存本次实际生成但被替换的文本。前3条不得计算为模型独立选择或学习收益。

## 冻结条件与资源

同一八卡模型实例并行8局，env0/2/4/6为A，1/3/5/7为B；全部实际游戏seed52620，
只有1个独立场景。采样初始Torch seed按rank为0/0/1/1/2/2/3/3，初始化一次，
后续场景分岔不声称随机数逐token配对。精度和采样与43短RL相同，
但八卡评测不冒充四卡训练轨迹复现。

模型为原始`Panyuqi/SpikingBrain-2.0-instruct`，无SFT/RL恢复；原模型manifest
`40d6f1417568d5fda1107cdafd66c7441a679b03fb148851c8f226ecc9cf884c`。
FP32参数/温度归一化、BF16 autocast、HF/eager/cache、分层FSDP、SWA dropout0，
T0.4/top_p0.9/top_k0、4096/192tokens不变。Level6 Harder、Terran/Air、Flat128，
canonical g2资产/开局Planner、marine_pressure阶段、约15条与数量展开/累计配额、
名称和前置过滤、合法计划保留、规则补齐、微操、自动战斗、history off、shaping off不改。

44已有依赖/原始权重/SC2只读复用，不下载、不复制大权重、不加载旧大checkpoint。
新增部署 `/root/new-scii-execution-feedback-20260911/code-<SHA>`，私有Git bundle备份；
新控制目录`supply-order-r1`，缓存和临时文件在`/dev/shm/sc2_supply_order_0911_r1`。
日志/回放在共享本experiment下唯一RUN，不写模型checkpoint，不删除旧产物。

RUN_ID：`20260911-execution-failure-feedback-v1-spb2-original-l6-g2-supply-order-ab8-8gpu-s52620-r1`。
从clean SHA运行`run_supply_order_node44.sh`，外层`timeout --signal=INT --kill-after=15s 2685`
保证总上限2700秒；入口子任务2640秒/退出预留30秒。无自动重试/后续训练。
启动前重新检查八卡任务归属；44剩余系统盘17GiB，仅存小型源码/控制日志，
模型原地只读，Ray/HF/Triton临时数据放共享内存，不与43的本地保存争盘。

## 判断标准

先核对历史输入相同、两个回答均合法且46项完整顺序进入Executor。
只读记录生产事件的tag/类型/位置/游戏时间/人口，以及最终队列，回调仍执行原逻辑。
区分建造命令提交、开工、完成，核对新增Depot完成后人口上限、后续Marine产出、
是否仍同一人口阻塞、下一次Planner时间及自然终局。所有局均记整局reward/胜负。

B成功建成并恢复生产而A仍阻塞，支持“现系统存在可执行纠错选择”；不证明奖励
足够区分这些选择，也不证明RL学会。若B仍无法施工，则沿具体命令/资源/位置证据
定位下一处阻塞，不顺手改微操或战术。未到目标输入、事件证据缺失或预算终止，
记证据不足，不能只凭少量胜负或退出0宣布系统/学习验收通过。

本次仅新增诊断入口、夹具、观察hook、测试及记录，不改生产模块。
首次本地CPU配置回归发现薄入口尚未覆盖旧EVAL_ONLY单卡默认，未启动GPU；
补齐八卡/八局显式覆盖后重新验证。这是本次诊断准备返工，不是新发现项目bug。

本地实际运行`test_supply_order_eval test_execution_failure_feedback test_original_bo_segment
test_valid_bo_passthrough`：47项全部通过，6.235秒、退出0；新诊断5项包括真实
validator/parser完整46项顺序、原回调仍被调用、仅一次八种子评测和显式八卡/不恢复
checkpoint配置。`bash -n`与`git diff --check`通过；44固定依赖验证/实局另记。

## 实际启动

冻结SHA `4d39d06658364689518d5d6f44a2af4bb5f73b49`，独立部署
`sc2-a100-44:/root/new-scii-execution-feedback-20260911/code-4d39d06`保持clean。
相对43的6effc64，agent_system/verl/gigpo/scripts/configs没有diff。
完整私有bundle已本地/44校验一致，SHA256
`f5b0e3109e99cf1d6a25f03c15fae6078f33c2f80707dd893834076025e08d5f`，未推送GitHub。
44 CUDA隐藏的`test_supply_order_eval test_execution_failure_feedback`共24项通过，
1.233秒、退出0；23项source哈希通过，固定依赖及setup hook延迟CUDA初始化检查通过。

16:06:15北京时间启动受控tmux `sc2_supply_order_ab_r1`，硬上限16:51:15。
启动前GPU0–7没有compute任务；没有停止其他任务。既有权重11/11分片加载后，
八rank初始化完成并进入纯评测，实际seed为8个52620。16:08:11已打印开局Planner；
保存八个独立游戏PID/首次query哈希。配置为val_only、resume disabled、global_step0，
不加载Level7 step24、不做PPO。manifest的training_goal96来自共享入口记录，
不能解释为本次启动96步训练。

[启动证据](supply-order-launch.json)保存模型provenance、配置哈希、实际种子和
八局开局证明。该快照尚无转发回答记录，不冒称已到人口阻塞或通过游戏效果验收。
43的反馈RL继续原样运行；16:03核验A完成第1global step，首批A/B均7/8胜，
B当时仍无更新完成指标。不能用这一批训练胜率宣布反馈机制学会纠错。

## 完成结果（17:00后只读核验）

控制`.done`与run的`result.json`均退出0，training_updates=0；八个replay已自动
记录哈希，console SHA256为`e502a7b7ff3c6c81a6b6c0a865649644cc869b83bbbcc1b35b034beb5c5715ca`。
`intervention_coverage.json`确认env0–7全部到目标，24条干预记录覆盖每局前三次回答。
第三轮两组最终Executor队列都为46项，数量一致，仅Depot项前移；没有修改生产模块。

| 观察 | A：Marine在前 | B：Depot在前 |
| --- | --- | --- |
| 06:13之后同一人口阻塞再次触发 | 4/4局，06:53再次反馈 | 无再次反馈 |
| 新Depot首次完成 | 仅env0在07:18完成1个；另3局无完成 | 每局06:37首个完成，07:12共完成8个 |
| 人口上限 | env0最终55，另3局47 | 每局最终111 |
| 干预后新Marine产出 | 每局0 | 每局2，07:12出现 |
| Planner消费回答数 | 每局4，其中3条人为转发 | 每局3，均人为转发 |
| 自然胜负／整局累计reward | 4/4胜，每局20.5 | 4/4胜，每局20.5 |

A第四轮恢复在线模型后，env0/6开始造Depot，env0的1个最终完成，不能将A写成
“全程不造Depot”。B八个Depot全部前置使开工/完成可见，但本对照不证明该数量
或计划是最优资源利用。证据来自`game_worker_mapping.jsonl`、各PID的
`supply-events.jsonl`、`boundaries.jsonl`及完整prompt_response；B实际四局
事件路径相同，只有1个独立游戏seed，不当作4个独立场景。

整局reward以实际消费动作和终局口径核对，每局20.5，与评测指标一致。
B结束后还有4条未消费终局占位日志重复显示20.5，不能再次累加为41。
28条消费记录=24条干预+4条在线回答；32条日志不等于32次模型独立决策。

结论：在这个人口阻塞场景，现有选择表/过滤/执行链确实允许纠错动作落地，
并不需要额外改prompt或Executor规则才可解除阻塞。但改善与未改善计划得到
同样的整局reward，不能由本次胜率宣布RL能区分/学会这些选择，更不能外推
全系统已健康。保留结果，不自动晋级训练或重复评测。
