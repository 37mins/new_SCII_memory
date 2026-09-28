# SpB Level6/g2 首次接管 A/B

2026-09-08 用户批准：各四卡四局、八卡并行、最多60分钟，不做PPO。**r1 两组均在游戏之前 failed：A退出1，NCCL重复GPU；B退出1，Ray主机内存保护。0局游戏、0条回答、0次PPO，不是0/4胜率，也不能判定接管候选好坏。** [原始manifest/result与启动记录](r1-failed-results.json)已保留，未自动重试。

22:45：44的启动门禁退出1，GPU0已被其他任务占用；尚未创建tmux、pair-r1或正式run，没有启动失败模型、没有消耗游戏预算。44的 `code-4e51548`、私有bundle（SHA256 `d7ee68313b42e859ca9f2d081537b65a2df58795e887fd3ba6238c3520d4408e`）及两份CPU预检保留。两组Hydra解析与真实 `_validate_config` 均通过。

改用44 GPU4–7跑A、42 GPU4–7跑B，各自独立四卡；不是一个跨节点八卡worker group。42此前已部署同一份44依赖归档，直接读现有 `/root/new-scii-spb-level6-96-20260908/python` 及 `include/python3.12:include`，不复制新环境。两组均从新编排SHA运行 `run_eval.sh`，差异仍仅接管开关与节点/GPU/依赖等物理路径。节点负载不同，吞吐不作方法结论。`run_timed_arm.sh` 使用同一绝对UTC截止时间（含60秒终止宽限），确保总墙钟最多60分钟、GPU预算不超过8小时；不增加局数或自动重试。

两组使用同一个干净候选SHA，包含已通过106项CPU检查的 `52225eb` 游戏修改；新增内容仅为本次纯评测编排。A（GPU0–3）`SC2_FIXED_BO_PREFIX_LIMIT=0`，B（GPU4–7）为138。原始SpB2-instruct、Level6/Harder、Terran Air、Flat128/Base75689、canonical g2；air选项/有效BO直通开启，SWA dropout0、FP32参数与温度归一化、BF16 autocast、HF/eager/cache、history0/off、6BO/96token、temperature0.4/top_p0.9/top_k0及其余系统行为冻结。无项目SFT/RL、无checkpoint恢复、纯评测val_only、save_freq=-1。

游戏种子逐环境为 `[2732,2732,43567,43567]`。复用 `r4_pair_seed.install`，由 `handoff_eval.py` 向现有reset传入一次性四局种子，拒绝多采一批。两组HF各rank在**第一次生成之前**以 `0 + rank` 初始化Torch采样RNG，保存 `sampling-rankN.json` 中实际seed及CPU/CUDA状态哈希；后续回答不重新seed。此为两组共同的配对编排，不改变采样温度/概率路径，也不声称恢复了历史模型RNG。交接后生成长度、请求次数可能不同，不能据此假设后续每个token随机抽样仍一一对应。

RUN_ID：

```text
20260908-g2-planner-handoff-v1-spb2-l6-g2-A-eval4-s0-r1
20260908-g2-planner-handoff-v1-spb2-l6-g2-B-eval4-s0-r1
```

结果根 `/mnt/data0601/yangjinyue/new_SCII/experiments/20260908-g2-planner-handoff-v1/runs`。44节点新部署根 `/root/new-scii-g2-planner-handoff-20260908`，使用私有Git bundle、`code-<SHA>` 和独立 `pair-r1` 启动日志；Ray目录 `/dev/shm/g2handoff_A_r1`、`/dev/shm/g2handoff_B_r1`。旧环境 `/root/new-scii-spb2-level5-control/python`、Python3.12系统头和共享模型/SC2/FLA/MoBA仅复用，不安装依赖、下载权重或改旧deployment。

模型manifest `configs/spikingbrain2.sha256` 的SHA256为 `40d6f1417568d5fda1107cdafd66c7441a679b03fb148851c8f226ecc9cf884c`；原入口启动时逐文件核验模型、BO、SC2/map和已登记源码。22:32北京时间只读检查：44 GPU0–7均0MiB且无compute进程；Torch2.7.1+cu128、Transformers4.55.2、Ray2.48.0、NumPy1.26.4，128CPU、可用RAM509GiB、/dev/shm剩491GiB、系统盘剩22GiB。既有两个tmux不是本评测，不终止。启动前再次核验空闲和RUN不存在。

本次CPU编排测试3项通过：原reset实际派发四种子且只一批；每worker仅首次生成seed；实际A/B启动器dry-run记录四卡四局、原模型纯评测，排除输出/GPU路径后只有接管参数不同。还需在44执行Hydra解析和实际配置校验后启动，不跳过空卡门禁。

启动由受控tmux调用 `timeout --signal=INT --kill-after=60s 3540 bash run_pair_node44.sh`，整个pair共用最长60分钟（含终止宽限），不自动重试。每组独立Ray、输出和状态；任何失败保留，不增加游戏数量。若GPU被他人占用则不启动，不停止他人任务。

主判读是配对的自然胜负及整局累计reward（包括零模型决策局），辅以真实Planner消费量、严格有效率、原始回答、首次接管时间/阶段、单位产出和损失。以实际evaluation_seeds、sampling-rank哈希、manifest/effective config核验对齐，以.done和result核验结束，不能把目录存在当完成。先确认交接前固定轨迹未发生意外变化，再解释交接后的差异；少量涨跌不宣称学习或架构优势。既有按秒命名回放可能同秒覆盖，单独注明，不在本轮改日志/游戏机制。

## r1 实际结果及启动修正

运行SHA `0ff77c939a88b794e98c708eccd830333fb2b7d5`，两节点均由完整私有bundle部署，bundle SHA256 `b270f95f4fdbdd5a7b1be22d699e5662473183d9da1402f6b693d85a47c80ecf`。两组真实Hydra解析和 `_validate_config` 通过，source clean/formal，已完成输入资产核验。两个后台会话分别为44 `g2_handoff_A_r1`、42 `g2_handoff_B_r1`。

| 组 | 后台开始（北京时间） | 结束 | 退出码 | 结果 |
|---|---|---|---|---|
| A，44 GPU4–7 | 22:55:35 | 22:56:55 | 1 | FSDP初始化时 `Duplicate GPU detected : rank 2 and rank 0 both on CUDA device 89000` |
| B，42 GPU4–7 | 22:55:50 | 22:59:10 | 1 | `ray.exceptions.OutOfMemoryError: Task was killed due to the node running low on memory.` |

两组配对种子选择记录已产生，但尚无SC2 reset派发/整局结果；没有 `sampling-rankN.json`，故新增的实际seed函数尚未执行，不能把报错归因于seed数值。23:01/23:05复查两组GPU4–7均0MiB；没有停止他人任务、调整Ray内存保护或继续采样。共同原截止时间23:54:42未触发，因为两组更早失败。

A的有证据嫌疑是**本次新增编排**过早导入 `HFRollout`：Ray2.48的 `default_worker.py:312–321` 在进入任务循环之前运行setup hook，而rollout导入链会执行 `verl/utils/device.py` 的全局 `torch.cuda.is_available()`。原seed hook不显式导入该模块。此时尚未执行新seed函数，且NCCL报多个rank指向同一个物理GPU；这是初始化顺序候选解释，尚无GPU修正对照，不能宣布已经定位/修复全部原因。

仅在本地候选调整这一编排顺序：worker setup保持种子选择hook；在TaskRunner创建环境之后登记纯评测包装，等模型worker初始化完成、`_validate`开始时，使用冻结VERL已有 `actor_rollout_execute_with_func_generator` RPC逐卡安装原采样包装。`seed=rank`、首次生成才seed且仅一次的语义不变；不改模型/训练/runtime核心文件。5项CPU回归通过，覆盖延迟安装及原reset/配置/seed规则；**没有GPU验证，没有覆盖0ff77c9部署或原run**。这属于本次编排返工，不计为项目原有bug修复成果。

B启动前仅约52GiB主机可用内存，尽管GPU4–7空闲，仍触发Ray保护；将该节点判为可承载评测的把握不足。下一次须核验CPU内存余量和实际GPU分配，不能关闭/提高Ray阈值来强行启动，也不能拿CPU配置通过代替这些运行门禁。当前没有可用的A/B效果结论；新RUN重试需再次确认，且只修启动编排，不扩展到prompt、战术或RL。

## r2 重试授权与冻结

用户在清理旧短测权重后明确要求立即启动上述实验。23:47重新核对44只余GPU6–7，
不能继续安排四卡A；42 GPU4–7空闲、主机可用165GiB。23:48发现46八卡全空，
主机可用793GiB、/dev/shm剩312GiB，因此本次两组都放46，A=0–3、B=4–7。
不抢其他任务，不跨节点拼卡，不扩大各四卡四局/共同60分钟上限。

冻结runtime为 `a34e539274b4a976863e90a47c503d89ce9f35e9`，含上述延迟采样hook
修正；游戏、模型、prompt、概率路径与预算相对r1不变。A/B只有接管参数0/138
与GPU/输出路径不同。CPU编排5项测试、21项源码manifest核验再次通过。
使用46已有 `/root/new-scii-spb-unitid-20260906/python`，系统Python3.12头文件；
不重新部署依赖或复制模型。冻结部署为 `/root/new-scii-g2-planner-handoff-20260908/code-a34e539`。
完整Git bundle SHA256 `c5e33aa8291f00d0c6589e9bc8dc52d5fc5e03fa94e2f5f067c8592170764017`。

新RUN为原A/B前缀各改末尾 `r2`；原r1原样保留。实际编排见
[launch_r2_node46.sh](launch_r2_node46.sh)，调用冻结版本已有 `run_timed_arm.sh`。
后台会话 `g2_handoff_pair_r2`，独立控制日志 `pair-r2`，共同截止时间从实际启动
计算，包含终止宽限；不自动重试。此段是启动计划，模型加载和游戏开始必须另核验。

### r2 实际失败及单点修正

46受控会话于23:56:34启动，共同截止时间原定09-09 00:56:34。两份实际manifest
均为clean/formal、源码a34e539、原始SpB、evaluation、world_size4；A=GPU0–3/
prefix0，B=GPU4–7/prefix138。`trainer.val_only=True`、resume disable、SWA dropout0
及四个配对种子均核对一致。两组通过真实Hydra配置检查和模型初始化后，在进入
SC2之前失败：A23:59:26、B23:59:29，均退出1，训练更新0、回答0、游戏0。
不是0/4胜，亦没有候选效果结论。09-09 00:01复查八卡均0MiB、无SC2进程，
后台会话已退出。[非敏感manifest/result摘要及配置哈希](r2-failed-results.json)已取回；
完整原始JSON仍保存在服务器RUN中。完整配置取回被安全校验拒绝后，没有绕过，
改为只输出白名单字段，不将潜在凭据字段带入本地审计。

直接错误是本次新增采样RPC包装，而非游戏或模型：

```text
verl/single_controller/base/worker.py:319
ret_proto = func(self, *args, **kwargs)
TypeError: install_sampling_on_initialized_worker() takes 0 positional arguments but 1 was given
```

本地只把该回调签名改为 `install_sampling_on_initialized_worker(_worker)`；
不改变函数体、seed规则、模型或运行中的deployment。之前CPU double直接调用
无参回调，漏测了这个实际接口；现改为执行冻结源码的真实
`Worker.execute_with_func_generator` 方法，再检查原Rollout方法被包装，五项
CPU回归全部通过。此为本次编排返工，不计作项目原有bug修复，也尚无GPU验收。
不原地覆盖r2或自动重试；若再获批准，新SHA/新RUN仍可沿用原00:56:34截止时间。

## r3 原预算内重启

2026-09-09用户明确批准修正后重启：仍46八卡、A/B各四局、不做PPO，沿用原
00:56:34北京时间截止（epoch1788886594），不增加预算。runtime固定为
`ea0055ca6eff2a802ea5d07098e3f3d0e49b361f`；相对r2唯一运行代码变化是上述回调
接收worker参数，原模型/数值路径/生成/种子/游戏配置不变。
原计划脚本`launch_r3_node46.sh`（保留在commit4a903b3）调用新冻结目录`code-ea0055c`中的原
`run_timed_arm.sh`；A/B后缀改r3，输出、Ray目录及`pair-r3`会话日志均独立。
旧r1/r2与部署原样保留；无自动再次重试。此段为计划，实际开始和游戏状态另记。

09-09 09:39北京时间收到批准后重新核验：原00:56:34截止已过，46八卡均为其他
`yht_axbench`任务占用。因此本次**未部署、未创建r3 RUN、未启动评测**；不将
昨晚的空卡快照沿用到早上，不延长过期预算。CPU编排五项回归、r3 shell语法检查
再次通过。r3薄脚本保留原绝对截止时间并在入口拒绝过期启动；需要新的时间窗口
与可用节点后才能继续。源代码修正仍为ea0055c，没有增加任何游戏或训练候选。

09:41补查44八卡全空、主机可用513GiB；42仅GPU4–7空闲、主机可用138GiB。
建议申请44同节点两组四卡、从实际启动起最多50分钟（仍各四局、无PPO）的新窗口，
不自行把已过期的00:56:34替换为新的截止。新窗口批准前不启动、不排后台等待任务。

### r3 新窗口：44八卡、最多50分钟

用户明确改批44八卡，从实际启动起最多50分钟，SpB Level6/g2、A/B各四局、无PPO。
未执行的r3计划按新授权迁到[launch_r3_node44.sh](launch_r3_node44.sh)，两组共用
启动时间加3000秒的绝对截止（含60秒终止宽限）。不另开方法分支或增加游戏数量。
runtime仍ea0055c；原始SpB、SWA dropout0、FP32参数/温度归一化、BF16 autocast、
HF/eager/cache、6BO/96token、history0/off、prompt/reward/BO/微操不变。
A=GPU0–3/prefix0，B=GPU4–7/prefix138；种子均`[2732,2732,43567,43567]`。

44复用`/root/new-scii-spb2-level5-control/python`和系统Python3.12头，不部署依赖、
不复制模型。新冻结目录`/root/new-scii-g2-planner-handoff-20260908/code-ea0055c`，
后台会话`g2_handoff_pair_r3`；新RUN保留A/B原前缀并使用r3。旧r1/r2不动，
r3此前没有创建RUN，因此不覆盖失败产物。启动前再次核验八卡空闲；失败不自动重试。

实际启动：09-09 10:00:26北京时间，44八卡复核全空、主机可用514GiB，后台
`g2_handoff_pair_r3`主PID871930；两组共同截止 **10:50:26**（epoch1788922226）。
此前09:56 GPU0被其他任务占用，部署完成后09:58该任务自然退出，未停止它。
编排commit `aef45cc7611bcb185f6a68e0caaf4b8b46eba1be`，冻结runtime仍ea0055c；
完整bundle SHA256 `e51a4eaed7b9a0e16d54ff4e6fb657580529a3b10b5ef6a3037effecd78f0dc9`，
已在44独立目录备份并验证。旧部署不变，runtime工作树clean。

[实际启动白名单摘要](r3-startup.json)显示两组formal/evaluation、world_size4、
SpikingBrain-2.0-instruct原始权重、resume disable、save_freq-1、SWA dropout0、
temperature0.4、四条rollout；接管参数分别0/138。未输出完整配置中的其他字段。
CPU编排五项测试及21项源码清单通过，服务器依赖确认2.7.1+cu128/4.55.2/2.48.0/1.26.4。

10:02:36两组均已完成模型初始化并进入SC2固定BO前段；A的四个模型worker为
879562/880038/880039/880040，B为879424/879768/879769/879770，分别映射八张不同GPU。
SC2日志已经出现worker的真实开局建造和侦察记录，未重现r2回调TypeError。
这仅确认启动关卡通过，尚未验收两组首次生成/交接、胜负或方法效果；本次不做PPO。
日志旧名称`default_player_Qwen3-30B`不代表模型身份，实际加载以manifest原始SpB
及已核验权重哈希为准，本轮不修改这个日志标签。

10:03:47两组实际`SC2 validation seed record`均为
`[2732,2732,43567,43567]`，不是仅凭seed=0推断配对。
B实际日志确认保留138项、暂不直接执行原列表的`[138,237)`项，
原g2文件SHA256为`304b1f4ff12fe30c5e6d5f7b8eaef8ed6823ca65afed3a3706cae10e3639c494`。
当时8个SC2进程在运行，会话仍存活。
