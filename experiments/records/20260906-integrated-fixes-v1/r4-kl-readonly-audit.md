# r4 KL 漂移：只读更新链路审计

更新（09-07 18:39北京）：一次重放已完成，恢复、Adam22→23及全184条检查通过；
已补齐切片级loss梯度、裁剪和Adam证据，尚未发现实现bug。
详见[实际重放结果](r4-first-update-replay.md)。
下表“仍缺什么”保留本次重放之前的审计时点，新增证据以该结果记录为准。

后续授权更新：用户已批准文末的一次参数更新重放，并进一步允许在其他空节点
并行执行；准备与结果另见[r4一次更新重放](r4-first-update-replay.md)。
下文的“尚未申请执行”保留为本只读审计完成时的状态，不代表最新授权。

结论：**尚未定位最初漂移的根因，没有发现足以直接实施新修复的更新链路错误。**
已经确认实际计权、更新次数、Adam 参数和饱和分支行为；这些证据不等于参数更新已验收。
本次不停止真实 RL，不改冻结代码，不加入 early-stop，不新增 GPU 实验。

## 审计身份与复核结果

对象为44节点GPU0–7的
`20260906-integrated-fixes-v1-qwen-instruct2507-l6-8gpu-fp32-rl96-gate8-s0-r4-train8`。
冻结SHA `ca7885d886aaaafb39fc105b1d5d46cddf4bcc5a`，deployment为
`/root/new-scii-empty-bo-frame-20260907/code-ca7885d`；远端clean，七个关键源码
哈希与该Git对象一致，不使用最新工作树推断。manifest确认原始Instruct-2507、
Level6八卡、target_steps=8、总目标96、无SFT/项目RL续训。
2026-09-07 13:01北京核验：初评已退出0，训练仍在运行，已记录完成step1–4，
step8评测未开始。这个快照不是整次实验的最终状态。

第2个global step：原始时间窗内552条日志中180条真实模型动作、180个不同
`(env_idx, policy_action_id)`；PPO有184条回答、2087个有效token。
八卡各23条，23次mini-batch optimizer更新、ppo_epochs=1，绝非整批训练23轮。
保存的180组不同有效输入中四组出现两次，与copy补齐吻合；具体rank/offset和
每次更新的八条输入哈希、长度、advantage范围都保存在[证据JSON](r4-kl-readonly-audit.json)。

| 检查环节 | 实际证据 | 与参考实现是否一致 | 仍缺什么 |
| --- | --- | --- | --- |
| Mask／回答计权 | 184个mask均为0/1、非空连续，包含EOS 151645，排除右padding。每卡mini=1、micro=1、累积=1。长度3–31，其中83条长3 | 与[GiGPO固定版本actor](https://github.com/langfengQ/verl-agent/blob/f8698a982d731e0968f1859f1d983447fad0017a/verl/workers/actor/dp_actor.py#L316)的静态micro-batch计算一致；实际为回答等权，不是全局token等权 | 这种目标在本任务是否更合适，需单变量对照，不能仅凭名称判错 |
| 复制与组成 | 180→184；四条额外副本长度3、12、12、3，保留真实mask与loss权重。先balance，再按每卡23条顺序各取一条 | [官方copy路径](https://github.com/langfengQ/verl-agent/blob/f8698a982d731e0968f1859f1d983447fad0017a/agent_system/multi_turn_rollout/utils.py#L71)已有此行为；分母计算代码有历史调整，但本配置均为8 | 复制对这次漂移的因果影响。不能将四条副本直接认定为根因 |
| Backward／FSDP／裁剪 | 每mini先zero_grad，一次backward后FSDP全局clip_grad_norm_(1)，再step；step2日志裁剪前norm均值2.842 | actor主链路与官方对应；安装的[FSDP归约](https://github.com/pytorch/pytorch/blob/v2.7.1/torch/distributed/fsdp/_runtime_utils.py#L827)及[全局裁剪](https://github.com/pytorch/pytorch/blob/v2.7.1/torch/distributed/fsdp/fully_sharded_data_parallel.py#L1069)源码哈希与PyTorch2.7.1一致 | 没有逐次参数梯度张量、归约前后对照及裁剪后张量。均值不能证明每次都被裁剪或实际更新正确 |
| AdamW／状态 | 八卡checkpoint小元数据均为lr=1e-6、betas=.9/.999、eps=1e-8、wd=.01、decoupled=true；计数22→45，正好增加23；一组FP32平坦参数动量元数据存在 | 安装的[AdamW](https://github.com/pytorch/pytorch/blob/v2.7.1/torch/optim/adamw.py#L21)/[Adam](https://github.com/pytorch/pytorch/blob/v2.7.1/torch/optim/adam.py#L246)逐字匹配2.7.1；offload/load工具与官方GiGPO整文件同哈希，只搬tensor，不重建optimizer | 未读大动量内容；没有每mini的m/v与参数差，不能证明动量数值未损坏。console lr=0.000是三位小数显示，不是真实零学习率 |
| KL口径 | pre actor-old最大差0；过程micro均值0.486871；最终micro均值1.491812、token加权2.176038 | 都是采样token上的old-current K1；过程值来自不同参数时点，最终值来自同一终态参数，不能混用。不是reference KL或精确全词表KL | r4没有cached rollout概率记录，不能推导cached/full一致性 |
| 饱和后的直接梯度 | 最终111个token下降>20；对冻结loss做CPU求导，111个的PPO和reference分支直接梯度均0；其中34个advantage正、77个负 | reference K3封顶10[上游已有](https://github.com/langfengQ/verl-agent/blob/f8698a982d731e0968f1859f1d983447fad0017a/verl/trainer/ppo/core_algos.py#L515)；指数输入±20是项目既有数值保护，非原版逐字行为 | 只能解释饱和后该分支缺少拉回信号，不能解释首次漂移；entropy、其他token与共享参数路径仍可有梯度 |

参考GiGPO commit为`f8698a9…`（2025-06-03），是公开函数级对照点，不冒充
r4完整上游provenance。另交叉检查VERL v0.4.0静态actor主链路；没有升级依赖。

## 哪些值得怀疑，哪些不能下结论

**已确认的实现差异，不等于新发现的bug：** 项目有±20指数保护、mask-safe
处理及非有限梯度显式失败；copy计权、reference封顶10不是本项目新造。
另外`gigpo/core_gigpo.py:174–242`含既有的episode advantage按轨迹长度分摊，
与[官方episode_norm_reward](https://github.com/langfengQ/verl-agent/blob/f8698a982d731e0968f1859f1d983447fad0017a/gigpo/core_gigpo.py#L125)不完全相同。
因此只能说被核对的actor更新主链路一致，不能说整个GiGPO算法逐字一致。
这部分本轮冻结，未获得其引起漂移的证据。

**有证据的可疑因素：** 每次更新实际目标为
`(1/8) × Σ回答[(1/L) × Σ有效token loss]`。3-token回答中的一个token，在
相同advantage和分支状态下，其归一化系数是31-token回答的约10.3倍；83条
短回答占45.1%的回答份额，但只占11.9%的有效token。这个差异真实存在，
但回答等权可以是合法目标，不应未经A/B就改为全局token等权。

概率尾部明显失真：最终gap中位数0、p1=-48.154、p5=-30、最小=-75.858612。
最差样本rank2/offset2/token1：old=-0.07888794、post=-75.9375、ref=-2.01767731、
adv=-1.78721452。饱和限制可能使后续纠正变弱，不代表最初大更新由它产生。
第11次更新前，被取到的mini已有一个>20的gap；没有每次更新后的全批forward，
不能把这个时点说成全批第一次发生严重漂移。

**尚未验证：** 参数Jacobian／BF16反向误差、Adam动量方向、梯度归约的实际
张量结果、mini顺序与复制的因果影响，以及cached生成差异。目前不能选一个
当根因。post-update CPU求导不是那些token实际参与更新时的梯度，更不是
`d(loss)/d(parameter)`；这里只用于验证冻结分支在已观察终态是否饱和。

## 证据缺口与下一项最小验证

没有新确认的实现错误，本轮不提出盲改或启动GPU A/B。参数链条若要继续查，
应在step8之后另行批准一次短重放：从r4 step1完整model/optimizer/RNG状态，
恢复第2批同顺序，在八卡只执行其第一次mini-batch更新；前后核对原184条回答，
保存选定参数块的更新前值、归约／裁剪后的梯度、m/v/step和实际参数差，按
PyTorch2.7.1公式离线对账。这样保留step2的历史Adam状态，不能替换成原始
模型零动量再声称复现本次链路。建议资源上限八卡30分钟（4 GPU小时），
不启动游戏，不运行23/35次更新；仅建议，**尚未申请执行、未上卡**。
若它不能覆盖异常发生时点，就报告证据不足，不自行加次数。

现有global-step checkpoint虽保存该步末m/v，却不能唯一恢复23次中间梯度、
方向与参数差；标量norm也缺少方向信息。因此不能只用历史r32 CPU求导补上
r4缺失的参数证据。确认一处偏差后，再提出仅改该处的同输入A/B。

复核脚本为[audit_r4_kl_cpu.py](audit_r4_kl_cpu.py)与
[audit_r4_optimizer_metadata.py](audit_r4_optimizer_metadata.py)。前者单线程CPU读取
约19MB诊断张量，不加载模型；后者仅读16份ZIP元数据和4-byte Adam计数，
不读权重或大动量存储。两者实际执行成功。最初模块路径定位脚本的返工不计
项目bug修复。原r4源码、配置、日志、checkpoint均未修改。
