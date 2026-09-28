# 固定41份输入：累计配额prompt A/B

2026-09-09用户批准：44 GPU4–7，全部41份已消费query，A/B各41次真实回答，
总计82次；从启动起最多30分钟，包括加载与60秒退出宽限。不启动SC2、不做PPO，
不自动重试或扩大预算。第二阶段整局评测未获本次启动授权。

对照来自冻结71fae36的B192原prompt。候选只将示例前的说明替换成：同名动作跨行
累计不超余量；示例不是额外任务；完成近段后结束，不为约15条重复凑数。
保留约15条、x N、重复单位/顺序以及无15个展开动作硬限，不改示例具体动作。
本轮变换只在[固定输入入口](fixed_prompt_ab.py)生效，未修改生产Planner。

源run为`20260909-bo-segment-protocol-v1-spb2-l6-g2-opening-B192-8gpu-eval4-s0-r1`；
prompt_response.jsonl SHA256为`5067499ab9efd79f88bb5c11a333780dea003417bb6d53a89fdc6a7ba8ecdebf`。
保留全部41份输入，包括历史22有效和19无效，不按生成结果筛选。
新RUN_ID：`20260909-bo-segment-protocol-v1-spb2-fixed41-quota-prompt-ab-4gpu-s0-r1`。

复用原run/model已准备的原始SpB及其模型代码补丁，只读，不重新下载/复制权重。
同一四卡FSDP模型只加载一次，先A后B，每个query采样seed为10000+原消费顺序index，
两组同query、同rank、相同seed重置。这个配对种子不是历史游戏RNG回放；历史22/41
只作样本来源，实际因果比较用本次新生成A与B。HF生成/cache、FP32参数、BF16 autocast、
FP32 normalization配置、temperature0.4/top_p0.9/top_k0、192token、原chat template不变。
直接使用原Worker与TrajectoryCollector的预处理及生成方法，不建立新模型执行器。

末批1条真实输入按原四卡方式补3条推理占位，每组3条占位输出单独标记、排除统计，
不形成新游戏/PPO样本。实际报告82条真实回答，另6条推理补齐，成本包含在30分钟内。
Worker沿原actor_rollout初始化但不调用任何optimizer step，不加载reference；未初始化
Adam动量，结束检查optimizer状态仍为空。train配置只作复用加载入口，名义world/env.n
共同设4，未使用的total_training_steps设1供scheduler初始化，不执行训练或reward计算。

保存逐query新旧prompt、原始token IDs/EOS、回答、配对seed/RNG哈希、全卡日志、
有效性/完整条目超配额、重复行、输出长度及配额覆盖；同时报告无效→有效和有效→无效。
必须检查合法输出是否退化成极少动作；不能把低token数自动视为改进。
少量相邻query并非41个独立游戏，结果只判断这个提示候选是否值得做整局评测，不作
胜率、学习、跨模型或架构结论。超时、异常或未完成全部82条保留失败状态，不自动补跑。

17:47:51北京时间预检44 GPU4–7空闲，GPU0为其他任务；现有依赖与旧SHA可用。
启动前再次确认空闲；候选提交后复制到独立冻结目录，旧deployment和run均不修改。

CPU已实际通过28项相关测试（1.868秒、退出0），shell语法与diff检查通过。
44禁用CUDA的实际配置/tokenizer预检通过：41份query齐全，原22份有效包含在内；
候选输入977–1115tokens，低于4096，旧输入844–982。两组解析契约对历史全部
回答的判定相同，正文中的约15条、x N、可用表、阶段、观测、待办和具体示例未改。
详见[CPU记录](fixed-prompt-ab-cpu.json)。这些CPU结果不是生成或游戏验收。

## 实际启动与状态

2026-09-09北京时间18:02:16启动，状态`running`。运行冻结SHA为
`385829c48099c4cbcf749d0ae83c4b7761117235`，44节点GPU4–7，受控tmux
`sc2_prompt_quota_r1`。18:02:47四个rank全部记录模型加载完成，实际架构
`SSESWAMoBAForCausalLM`、swa_dropout=0、参数FP32、BF16 autocast、cache开启。
18:06:44快照：A组完成20/41份真实回答，正在生成第21–24份；B组尚未开始。
不是仅创建会话或CPU通过，已有真实生成产物；尚无最终结果或`.done`。

冻结源码在`/root/new-scii-bo-segment-20260909/code-385829c`。
运行产物在`/mnt/data0601/yangjinyue/new_SCII/experiments/20260909-bo-segment-protocol-v1/runs/20260909-bo-segment-protocol-v1-spb2-fixed41-quota-prompt-ab-4gpu-s0-r1`。
配置、41对输入及rank日志已写入该run，最终汇总为`result.json`，退出码为`.done`。
预计北京时间18:20左右完成；包含退出宽限的硬截止为18:32:16，不自动续预算。

启动前另用真实TrajectoryCollector/DataProto做CPU检查：A/B张量均为`[1,4096]`，
首例有效输入分别857/990tokens，`validate=True, do_sample=True`。部署使用完整Git
bundle，本地与44副本SHA256均为
`9a297fc54830228253ea4494e3c1d08181c5e9050c30e7503048225724b13d5d`；
校验完整历史及冻结clean SHA后启动。当前有本地Git与服务器bundle备份，未推送GitHub。
本段是启动后状态记录，不改变运行中的冻结源码或配置。

## 完成结果

2026-09-09北京时间18:16:17退出0，耗时14分01秒；`result.json`和四个rank均完成，
82份真实回答齐全，另6份推理占位明确排除。41对生成前CUDA RNG哈希全部一致。
没有游戏、没有参数更新。18:16:42核验GPU4–7已释放。

| 指标 | A 原prompt | B 配额/示例/结束澄清 |
| --- | --- | --- |
| 整段回答通过冻结validator | 22/41（53.7%） | 28/41（68.3%） |
| 完整且名称合法的条目累计超配额 | 19/41 | 11/41 |
| 生成满192token而未出现EOS | 16/41 | 10/41 |
| 有效回答待办展开数中位数 / 范围（不计攻击指令） | 6 / 2–54 | 8.5 / 2–73 |

配对变化为14份无效→有效、8份有效→无效，14份均有效、5份均无效。
没有空回答；有效率上升不是简单靠所有回答缩成空计划，但单份有效计划仍可能变短。
例如index13，A是合法`- marine x 2`，B重复该行至32次、累计超配额；
index32，A为Marine/SupplyDepot/Barracks各2个，B重复同一组三行而超Barracks配额。
因此只能说候选在这批固定输入上净改善，不能说重复已经修好，也没有游戏或RL收益证据。
这41份是四局内相关输入，不作为41个独立游戏计算胜率或伪精确置信区间。

`result.json` SHA256：`93b57a7c99b8d1656b7dd227af6fe051b703b2e1a41470fd974c409540562137`；
`.done`：`362c6bcd270211a42ff8fa97b7a1aa0266d3bd9a839c7fd88882b090ad4b521c`；
`cases.json`：`61021f855c64aa38c2d5f36fed242b3fe5dacfd26b8ba457b00f5efa4a92fa26`；
`effective_config.json`：`8b3bf52c41e12244ea7713f43c4a0e04ec86056e0309ac6ab905b69163b6fbfd`。
原始token、回答与逐rank日志留在上述唯一run，不覆盖旧实验。

下一项只建议[同种子整局prompt A/B](quota-game-ab.md)，资源预算待单独确认，尚未启动。
