# B 协议：96 → 192 token 预算验证

状态：44八卡四局评测于2026-09-09 17:21:04北京时间完成，退出0、0次PPO；
17:02:11启动，耗时18分53秒，未超过30分钟预算。3/4胜，完整回答有效22/41。
原四卡方案未启动，不复用上一轮A/B的50分钟授权。

依据是B96已消费的39条回答中17条文本回算达到96token、全部无效，且存在末尾
`- marine x`。增加预算可能减少半截输出，也可能只延长重复或超配额回答；不能预设
一定改善。192是本次单次倍增探测，不是重新调prompt或保证所有回答足够长。

参照为已完成的`20260909-bo-segment-protocol-v1-spb2-l6-g2-opening-B-eval4-s0-r1`，
runtime `a1f15af1553b36cc691c0a19b6be65ea9cd25bda`：3/4胜、19/39完整有效、
平均整局reward2.8875。不重复该评测、不重跑A、不续RL。

| 项目 | 拟执行配置 |
| --- | --- |
| 计划验证变量 | `data.max_response_length`: 96 → 192；另有用户批准的四卡→八卡硬件差异，不称严格单变量因果对照 |
| 模型 | 原始Panyuqi/SpikingBrain-2.0-instruct，无项目SFT/RL；manifest `40d6f1417568d5fda1107cdafd66c7441a679b03fb148851c8f226ecc9cf884c` |
| 环境 | Level6/Harder、Flat128、Terran/Air，相同g2资产及阶段待办，Planner仍从00:00接管，不是完整固定g2直跑 |
| 配对 | 4局，实际seed `[2732,2732,43567,43567]`，采样种子按rank0–3 |
| 已批准资源 | 44 GPU0–7；一次，最多30分钟墙钟含加载及退出宽限，即240 GPU分钟；不自动重试 |
| 新RUN_ID | `20260909-bo-segment-protocol-v1-spb2-l6-g2-opening-B192-8gpu-eval4-s0-r1` |

保留FP32参数/温度归一化、BF16 autocast、HF/eager/cache、swa_dropout0、
temperature0.4/top_p0.9/top_k0、prompt4096、history0/off，以及B的prompt/解析器、
名称/配额/前置过滤、规则补兵、微操、战斗和reward。纯评测、禁用resume/save，0次PPO。

准备仅把既有handoff评测入口的硬编码96改为`${SC2_EVAL_RESPONSE_LENGTH:-96}`，
本次显式设192，旧默认不变；不改服务器上的旧entry或任何冻结deployment。
统计脚本从effective config读取实际预算，算法仍为相同tokenizer回算文本，不能
称为精确EOS截断统计。新增CPU测试比较完整effective config，归一化run路径后
只差`data.max_response_length`；相关96项回归11.030秒通过、退出0。CPU重算旧A/B，
除新增实际预算字段外，与已有r1-results.json逐字段相同，包括全部产物哈希。
`git diff a1f15af -- agent_system configs scripts gigpo verl`为空。

16:32:31北京时间只读检查：44 GPU4–7均0MiB，GPU2/3为其他任务；旧B部署clean，
现有Python环境和共享资产可用。不停止别人的任务，启动前再次检查；若资源变化
不擅自换卡或扩预算。使用新clean SHA、独立code目录和上述唯一run，失败不覆盖、
不自动重试。现有run_eval.sh提供截止时间与60秒退出宽限，授权后设30分钟绝对截止。

观察仍为完整回答有效率、达到实际预算的文本比例和半截/重复/超配额样例、
模型选择与最终队列、调用次数/游戏时间间隔、各seed胜负及整局累计reward。
若只减少长度受限现象但仍重复/超配额，应说明预算不足以解决问题；若有效率改善，
仅说明该预算在这四局局部有效，不能宣称RL学习或跨seed提升。当前不会自动晋级RL。

最小入口为当前记录的`run_eval.sh B`，新增环境变量`SC2_EVAL_RESPONSE_LENGTH=192`；
同时显式传入冻结SOURCE_DIR/SOURCE_SHA、新RUN_ID/OUTPUT_DIR/RAY_TMPDIR和已批准
PAIR_DEADLINE/ARM_STATUS_PATH。配置从同一record继承，不新建方法分支或平行实验目录。

## 八卡适配边界

用户2026-09-09随后明确要求用八卡跑。16:44:54检查44八卡0MiB、无GPU进程，
共享盘可读，系统盘21GB、/dev/shm491GB可用。不迁移或升级依赖。
既有评测entry增加默认4的`SC2_EVAL_N_GPUS`，本次设8；同时将`env.rollout.n=8`
满足`ray_trainer.py:_validate_config`对训练名义batch与world size的整除检查，
不删除校验。`val_kwargs.n=4`及`env.val_rollout_n=4`不变；实际评测只reset四个环境，
未调用的训练环境不启动SC2游戏，不做PPO。

`rollout_loop.py:generate_text_from_obs`原有流程是四条输入补到8→八卡推理→
去掉末尾4条输出→只给原四个环境，不能把推理补齐称为新增rollout或PPO样本。
前四个rank仍使用原采样seed0–3，末四个rank的输出不送入游戏。没有改padding函数、
采样hook或模型；但FSDP分片数及HF同步生成长度改变可能影响浮点或后续RNG推进，
不保证与四卡逐token相同，也不承诺八卡一定更快。结果仅作B192的八卡验证。
启动器[launch_budget192_node44.sh](launch_budget192_node44.sh)使用独立控制目录、
唯一run及30分钟截止，不触碰A/B已完成产物或旧deployment。

八卡准备实际执行原相关测试命令：97项通过、11.276秒、退出0；新增dry-run比较
四卡96、四卡192和八卡192，除response长度/world size/env.rollout.n及GPU路径外
effective config完全一致，评测局数始终4。另在44现有Torch2.7.1+cu128依赖上
禁用CUDA、调用真实DataProto的pad/chunk/unpad：输入4、补齐8、保留rank0–3四条，
退出0。生产目录对B原始a1f15af的diff仍为空，修改仅评测编排、测试和记录。

## 实际运行核验

runtime冻结SHA为`71fae36ccc112d4c9cd303ab084c1e0a0c7eebec`，独立源码
`sc2-a100-44:/root/new-scii-bo-segment-20260909/code-71fae36`，clean且源码manifest
全部通过。完整Git历史备份保存在本地`/private/tmp/sc2-bo-segment-B192-8gpu.bundle`
及44的`/root/new-scii-bo-segment-20260909/source-71fae36.bundle`；两处SHA256均为
`adf48cfd2c345c9441c53c0acf0b83ae4396c4d312bb294dfb2dc224ceecf354`。未推送GitHub，
没有绕过此前的推送权限限制。

上表RUN_ID输出于共享
`/mnt/data0601/yangjinyue/new_SCII/experiments/20260909-bo-segment-protocol-v1/runs/`。
tmux为`sc2_bo_budget192_r1`；独立控制目录为
`/root/new-scii-bo-segment-20260909/budget192-8gpu-r1`，外层console为同级
`budget192-8gpu-r1-console.log`。启动于2026-09-09 17:02:11北京时间，绝对截止
17:32:11；含加载、正常结束及60秒强制退出宽限，不自动重试或延长。

实际manifest记录formal=true、dirty=false、mode=evaluation、world_size=8、GPU0–7，
模型manifest与B96相同。实际effective config已验证192tokens、val_only=true、
resume禁用、save_freq=-1，val两处n=4；env.rollout.n=8仅为已说明的八卡适配。
服务器实际dry-run对旧B配置归一化路径后，仅上述预算/world size/env.rollout.n三键不同。
`evaluation_seeds.jsonl`记录四局实际seed `[2732,2732,43567,43567]`，不是八局。

17:04观察八个actor_rollout_generate_sequences进程，模型11/11分片加载完成，
SSESWAMoBAForCausalLM 5.17B、swa_dropout=0。17:05:21四个SC2Worker均在游戏0帧
消费首轮回答`- supplydepot x 2 / - barracks x 2 / - marine x 2`，解析成6个动作，
日志`refresh build order`原顺序保留，确认不是仅创建会话或CPU准备完成。
后续胜负、整局reward、回答有效率和预算占满比例待完整结果，当前不能宣称192改善。

启动前GPU全空；另一任务PID1176690随后于17:02:46启动，在GPU0占约9.6GB，
cwd为`/mnt/data0601/cjy/horizon/probe-dev`。未停止该任务，没有切换卡或改预算；
截至17:05:27未见OOM/Traceback。资源竞争另记，不能把此次耗时当作纯吞吐对照。

## 完成结果（17:22核验）

run的`.done`与`result.json`内容哈希相同，均completed/exit0，control/run.done也为0。
四局均自然终止、均调用Planner，没有PPO更新。最后验证指标输出后出现Ray worker
SIGTERM退出告警；主流程已完成四局且退出0，不据此称评测中途失败。17:22 GPU1–7
仅4MiB，本实验actor已退出；GPU0另有任务占用，未干预。
同一CPU汇总脚本读取完整日志，结果与产物哈希保存于
[budget192-r1-results.json](budget192-r1-results.json)。

| 指标 | B96（四卡） | B192（八卡） |
| --- | --- | --- |
| 胜负 | 3胜1负 | 3胜1负 |
| 平均整局累计reward | 2.8875 | 3.2625 |
| 全部实际消费回答 | 39 | 41 |
| 整段格式/语义有效 | 19/39（48.7%） | 22/41（53.7%） |
| 回算文本达到预算 | 17/39（43.6%） | 15/41（36.6%） |
| 达到预算且无效 | 17/17 | 15/15 |

有效率分母是全部实际消费回答，不是最后一轮四个环境状态的均值；console中
`val/sc2_action_valid=0.25`不能替代22/41。token统计仅解码文本回算代理，未保存的
原token IDs/EOS不能恢复，因此不把上表写成精确截断率。

| env / 实际seed | B96胜负/整局reward/调用 | B192胜负/整局reward/调用/有效 |
| --- | --- | --- |
| 0 / 2732 | 胜 / 21.0 / 14 | 胜 / 21.0 / 12 / 8 |
| 1 / 2732 | 负 / -49.95 / 10 | 负 / -49.45 / 11 / 5 |
| 2 / 43567 | 胜 / 20.0 / 8 | 胜 / 20.5 / 9 / 5 |
| 3 / 43567 | 胜 / 20.5 / 7 | 胜 / 21.0 / 9 / 4 |

实际回答仍存在重复直至半截结束：env0、policy_action_id=3先输出`- marine x 2`，
再反复输出`- marine x 1`，最后为`- marine x`；env3、id=3反复输出
`- marine x 2`并以`- marine x`结束。两者整段均被拒绝。19条无效中，首个失败原因
包括marine x 5条、supplydepot x 1条、格式3条、超剩余配额4条，另有名称/候选问题。
这只是validator首个失败原因，不意味着一个回答只有一种错误；projected_action
也不能当作最终Executor命令。

结论：加倍预算后仍有15条回答占满192且全部无效，重复、半截及名称/配额问题未解决。
有效率仅上升约5个百分点，胜负完全相同；目前不能据此接受为系统改进或启动短RL。
八卡/四卡差异、GPU竞争、仅两个独立seed均保留为限制；也不能断言预算完全无用。
本次不改prompt、不再扩大token预算、不自动续训。若继续，最小下一项为仅用已有
失败query/回答核对重复和配额提示的一致性，提出有证据的单点候选后再申请实验。
本run仍仅索引到1个回放，未声称四局回放齐全；完整逐局胜负以outcome/JSONL为准。

后续[只读输入/输出审计](budget192-diagnosis.md)：19条无效中17条完整条目已超
配额；15条占满预算的回答即使去掉末行仍全无效。未发现合法x N解析/长度丢失，
待验证的是prompt整段计数与结束约束，不是直接继续加预算。没有新增GPU实验。
