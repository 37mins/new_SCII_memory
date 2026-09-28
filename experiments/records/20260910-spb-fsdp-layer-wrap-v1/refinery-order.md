# Refinery／Reactor顺序因果对照（游戏运行中）

用户在温度A/B分析后批准执行。只验证同一合法计划的资源顺序，不新增自动重排
机制，不改模型、prompt、策略、名称/前置过滤、补兵、战斗、reward、cache或精度。
43 Level6长训继续原样运行；本次不进行PPO或启动新长训。

## 对照与边界

代码基准为温度A/B冻结`939c4ca956adabf1728e0cdd4a87c879593d7cbe`；生产文件
保持逐字节一致，只在独立评测hook中干预前两轮回答。输入取自温度B的env4，
原日志SHA256 `f49674f0863394497a4edffef754b3968b8a5ab90e88de7ab56a57a266b91edb`，
完整两轮query/回答保存在[夹具](refinery-order-inputs.json)，启动核验实际query全哈希。

第一轮两组重放相同depot4/barracks4/marine24。第二轮：

```text
A：marine → supplydepot → barracks → reactor barracks×2 → scv → refinery
B：marine → supplydepot → barracks → refinery → reactor barracks×2 → scv
```

动作及数量相同，其他动作相对顺序不变。使用冻结validator及parser，要求两条都
合法且交给Executor的7动作顺序保留。若开局/第二轮query与原场景不同则失败，
不在不同观测上悄悄干预。第三轮起恢复同一SpB在线规划，不再重排。

前两轮是人工回答干预，不算模型自行做出的选择；interventions.jsonl同时保存
被替换的实际生成文本、转发文本、arm、输入哈希。沿用日志llm_response字段
在这两轮表示转发文本，分析必须联读干预记录。此后计划可能随新观测分岔，
整局结果是“一次顺序干预＋后续在线策略”的效果，不能视为全局行动序列都固定。

## 资源与冻结配置

44 GPU4–7，同一四卡模型实例并行4局，env0/2为A、env1/3为B；全为seed17089。
两次配对生成随机种子为0/1，rank0/1同seed0、rank2/3同seed1；只初始化一次，
后续轨迹分岔不保证token随机数逐项配对。只有一个诊断场景，不能估计泛化胜率。
总墙钟上限30分钟、不自动重试；预计约15–25分钟（含checkpoint加载）。

权重是同一个Level7续训step24，使用既有四rankFSDP loader准确恢复后纯评测。
temperature1.0/top_p0.9/top_k0、192tokens、FP32参数/温度归一化、BF16 autocast、
HF cache、SWA dropout0、history0/off、Level6/Harder、Terran/Air、Flat128、
canonical g2及开局Planner接入不变。与前次B相比评测局数8→4；A/B在同批中共享
这一配置，不宣称复现原八局的所有生成RNG状态。优化器即使由loader恢复也不更新。

唯一RUN_ID：`20260910-spb-fsdp-layer-wrap-v1-spb2-l7s24-l6-g2-refinery-order-ab4-4gpu-r1`。
输出在本experiment共享runs；控制目录为44
`/root/new-scii-spb-fsdp-20260910/refinery-order-r1`，新commit独立code目录，旧run不覆盖。
游戏日志/回放按子进程PID分目录，防止同秒覆盖；只改输出路径，不改游戏逻辑。
每次Planner请求额外保存资源、单位、建筑进度及挂件tag，关联env与PID，均为只读观察。

## 验收

先核验同输入、合法性、原样队列和实际checkpoint恢复，再看是否采到50气、Reactor
开工/超时、下次反馈时刻、军队存活及自然结局/整局reward。减少超时不等于赢得
游戏，单局改善不等于学会规划。未调用模型/被人工替换的回答单独归因。
未到目标输入、进程/存储异常或缺关键证据时只记录失败，不自动重试或修改机制。
未获因果支持的候选不加入正式训练。

## CPU验证

`PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:tests /private/tmp/sc2-air-results.HBI61m/decoder/bin/python
-m unittest test_refinery_order_eval test_temperature_eval -v`：6项通过，0.579秒。
包括原始query全哈希、固定回答来源、真实validator/parser输出、下一轮不干预、
四种子只取一次，以及相对温度B的完整effective config差异仅评测8→4和新hook。
最初两次launcher dry-run分别遇到macOS无timeout、旧Bash空数组nounset问题，
已在诊断入口修正；这些不是原项目bug或正式GPU实验结果。bash -n和diff检查通过。

## 冻结部署与启动

冻结SHA `e0568cc7a7a00a58c59ef1b40ae15d2aa7462134`，44部署
`/root/new-scii-spb-fsdp-20260910/code-e0568cc`。生产目录相对939c4ca无diff，工作树clean。
增量私有备份`source-e0568cc.bundle`，SHA256
`b7fcb7ae9dc0fa21631b540bb433c0913dc9af362a5e697fead99655770f191e`；
依赖既有939c4ca及其备份，没有推送GitHub、修改旧部署或复制模型。
44隐藏CUDA的相同6项测试全部通过（0.812秒），实际setup hook导入/安装通过；
依赖为Torch2.7.1+cu128、Transformers4.55.2、Ray2.48.0、Gymnasium1.2.3。

北京时间2026-09-10 22:24:45启动受控tmux `sc2_refinery_order_ab_r1`；上限约22:55。
启动前4–7均无计算任务，GPU0与2为外部任务，未停止。manifest确认evaluation、
world_size4、准确source SHA；进入权重加载。这一记录不等于checkpoint恢复/游戏
干预验收通过，需以随后checkpoint_evaluation、种子和两轮输入/执行证据确认。

22:28只读确认：checkpoint loader已返回global_step24、val_only=true，实际四个
种子均17089，4个独立SC2子进程已进入Flat128/Harder/Terran/Air。四份开局输入
全哈希均与历史00:00输入一致；没有从原始模型误跑，也不把配置中的total_steps96
当作正在训练。后续干预和胜负仍待验收。实际小型产物哈希、资源/开局快照及采样
种子保存在[启动证据](refinery-order-launch.json)。
