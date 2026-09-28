# 三批非零信号：实际偏向什么计划

2026-09-15，现有产物只读＋轻量CPU分析。没有新GPU实验、模型加载、训练修改或
新增诊断工具。A为scene-A-r1，冻结`75a6b5dc3621e9d670fbfb440ae917bfacd5e360`；
B为scene-B-r2，冻结`8c2f0f8b103dab054a612ce95a041e4dde9c7c8c`，不是旧Coverage组。
run路径、配置和独评见[训练记录](scene-training.md)及[独评收尾](scene-posteval-r1-final.json)。

## 结果

下表是**复制补齐前、逐回答标量task advantage**，由已有冻结函数重建，最大绝对值
逐批与训练时`checkpoints/signal_sampling.jsonl`精确相等。不是参数梯度或token概率。

| 批次／同一第二query内的选择 | 局数与结果 | 第二答联合advantage | 实际区分 |
|---|---|---:|---|
| A global6：Marine8或42，Depot8 | 7局全胜，均2次调用 | +0.249844 | 不区分8与42 |
| A global6：Marine8、Depot1 | 1局胜，4次调用 | −1.748907 | 回报出现的决策位置不同 |
| A global7：Marine42、Depot8 | 5局全胜 | +39.364311 | 与下行有胜负区分 |
| A global7：Marine8、Depot8 | 3局全败 | −65.607185 | 只有Marine数量不同 |
| B global5：Marine42、Depot8 | 7局全胜 | +12.906250 | 与下行有胜负区分 |
| B global5：Marine8、Depot2 | 1局败 | −80.062500 | Marine、Depot及后续轨迹均不同 |

A global7最清晰：8个第二query逐字一致，5条回答为下面内容，另3条只将42改为8；
其余字符也一致。原日志第二答行57–64，败局为env4/5/7（行61/62/64）。

```text
<answer>
Here are the Build Orders with attack instructions:
- marine x 42
- supplydepot x 8
- barracks x 3
- reactor barracks x 3
- scv x 2
- refinery x 1
</answer>
```

这证明本批具有可区分的“回答—结果”关联，不是只有格式奖励。它仍不是随机干预的
因果实验，不能单凭5胜/3败证明Marine数量是唯一胜负原因。
B global5的对应行9–16，env5行14为Marine8/Depot2，其第三答行22含不合法`techlab`。
因此B不能单独归因于Marine数量。

A global6八局累计reward均20.5，episode项全零。7局的第二答直接收到20.5；
env7第二答之后还有两次调用，折扣回报为18.501249。gamma=.95按决策次数折扣，
产生表中的正负step项。env7第三答行40截断且无效，但该输入只有一个样本，step项
中心化后为0，episode项也为0，补齐前联合项为0。不能把本批统称为“胜负学习”或
“无效答都受到了负advantage”。这与现有计算路径一致，本轮不新增惩罚机制。

## 与实际回答变化、独评连接

A global7出现5个Marine42/3个Marine8后，global8八局第二答全为Marine42/Depot8。
B global5出现7个Marine42/1个Marine8后，global6–8的24局第二答全为Marine42/Depot8。
后续这些批次均全胜且task advantage精确零。观察到的回答选择变化与信号方向一致，
但没有参数/概率对照，不把它当成单次更新因果效应的证明。

共同独评中，A4在seed1436145665第二答选择Marine8，A8/B8改成Marine42，**仍败**。
另两个共同失败seed363527354、3434611143，三份权重原本就都选Marine42，亦全部失败。
三组均5/8胜。这更符合“偏向了训练场景中有用的单一选择，但该选择不足以解决其他
失败场景”，不支持“所有task signal都无用”，也不支持“RL已学会通用规划”。

当前三批还没有提供改变生产顺序的成功样本：第二答均Marine在Depot之前。
不能由此断言系统禁止其他顺序，也不能据此直接改prompt、奖励或执行器。

## 核验范围

复用`audit_scene_signal.py`的`reconstruct/decompose`，不另写奖励算法。
43节点CUDA隐藏、OMP/MKL/Torch单线程；实际`cuda_devices=0`，退出0。
按`episode_outcomes.reset_index`及累计step_index划分批次：A6行17–48、A7行49–64、
B5行1–24；排除各局terminal后的占位行，分别18、16、17条真实回答。
三个最大绝对联合项精确匹配原记录：1.7489070892、65.6071853638、80.0625。

A部署9个相关文件与`git show 75a6b5d:<file>`逐字核验；A/B的GiGPO、奖励、collector、
trainer及相关配置源码无diff。核心位置：`gigpo/core_gigpo.py:compute_step_discounted_returns`、
`episode_norm_reward`、`step_norm_reward`；`ray_trainer.py:apply_invalid_action_penalty`。
有效token共享一个回答标量，不能将正/负advantage直接解释为某个BO或token必涨/跌。
首答完全相同仍会因后续胜负获得正负值，是轨迹归因现象，不单独定性为bug。

没有恢复复制索引、balance顺序和最终PPO逐token张量；A18/B17条之后有补齐路径，
故表中数值不能冒充补齐后的实际loss计权或参数梯度。这一边界不作为继续数值诊断的理由。

| 输入产物 | A SHA256 | B SHA256 |
|---|---|---|
| prompt_response.jsonl | 05e608a9e005028db2e00ea3cbb634844bed457afdf479843f6dfe4c5a016af1 | 582844c1b22e6cb59e537512e9769a69d5f4ced1f9536e7345c7d6f4b8816956 |
| episode_outcomes.jsonl | 5d80d145580a9f9e005363303833c2f41f633f12916c3b5b2b62b4c4692e28bb | 410ae5e8639b9bd5139de424c71b136b49c889c75912e3daa8a4011b493f98fc |
| checkpoints/signal_sampling.jsonl | d0b1c3c7a49b449b014913f80bfbfd578719f5d265a9885f0707415929e470fb | be2e54efd7f77c1f6ad5f85c3f3087a812079a90fb9077ce0211062eb566d561 |

## 判定与唯一下一项

### 9月15日晚：直接核对已完成 checkpoint

46节点只读CPU检查，无Torch导入、GPU、游戏、整模加载、转换或远端写入。
[原始数值记录](checkpoint-readonly-audit.json)覆盖同一A4及SceneA8/B8：

| 检查 | A4 | SceneA8 | SceneB8 |
| --- | ---: | ---: | ---: |
| Adam计数（四rank、每rank37组均一致） | 15 | 24 | 24 |
| rank0六个权重片段中相对A4改变的元素 | — | 6141/6144 | 6142/6144 |
| 既有共同条件独评胜局 | 5/8 | 5/8 | 5/8 |

两组实际新增9次optimizer update，不是9个global step。rank0保存的优化器配置
均为lr=1e-6、betas=(.9,.999)、eps=1e-8、weight_decay=.01；各rank的动量片段
也已读取，记录保留rank0样本。模型抽查第0/17/35层q_proj、k_proj各前1024个
FP32值，全部有限、六个片段均改变；不以变化幅度大小推断学习质量。

这排除了“新增四步完全没更新模型”，不证明每次参数更新都正确。B6–8原记录的
task advantage和pg_loss均为0，grad_norm仍为.005/.003/.002。冻结
`8c2f0f8b:verl/workers/actor/dp_actor.py:440–486`仍把熵项和reference KL加入
loss并调用Adam；A/B这两个核心文件无diff。因此零task signal不等于完全不更新。
已有末态不能拆出每次变化中任务梯度、正则项和历史动量各占多少，不据此宣称
“KL毁掉学习”或新增优化器修复。

当前只读优先级：串联已有回答、分组reward及实际advantage，区分真正胜负信号
与调用时序信号，并定位这些信号消失前后计划如何趋同；不为这一步等待显卡。
旧温度/过滤候选保留为待验证方案，不把checkpoint抽查当作其通过。
首次权重元数据读取因分析代码未处理DTensor的tuple state而退出1，处理后成功；
仅为本次轻量读取工具的格式适配，不计作项目bug修复。

### 同晚继续：强化了什么，为什么仍是同三个失败seed

只读串联两组训练64局、三组独评24局及九份失败游戏的已有边界/trace。
[逐场景证据](checkpoint-plan-reward-audit.json)保留源SHA、回答指纹、原文及trace哈希。
没有GPU/PPO、回放重解码或生产代码修改；各游戏用worker mapping匹配env和PID，
按首次自然终局排除后续占位行，不把不同seed的胜负当作同输入因果对照。

| 证据 | 实际观察 | 能说明什么 |
| --- | --- | --- |
| 64局训练第二答 | A为25/32个Marine42、7/32个Marine8；B为31/32和1/32；64/64都Marine先于Depot | 采样集中于数量变化，没有覆盖另一种生产顺序 |
| B再次遇到相同query | global5的1098341402有7个42/1个8；global7同query全部42；3446496864在global6/8亦全部42 | 局部选择趋同与后三批全胜、零task advantage同时发生，不是缺少参数更新 |
| 独评第二答 | A4已有7/8个Marine42；A8/B8为8/8；另一处回答变化是去掉胜局中的攻击bullet | 原本就接近该数量选择的上限，改变答案不等于解决新失败场景 |
| 失败seed1436145665 | A4第二段8 Marine/共25动作；A8/B8为42 Marine/共59动作；终局队列直接核验 | 该变化真实进入BO队列，并非被解析器还原成同一段 |
| 同seed实际结果 | A4 09:10败、reward−49.401；A8/B8 09:58败、−49.45；最后模型决策均04:09 | 延长了这局存活，不是胜负改善，也没有后续模型纠错样本 |
| 另两个失败seed | 三模型均42 Marine；最后决策04:44/04:13，终局11:26/09:48 | 失败窗口内均没有第三答，不能声称训练已覆盖失败后的重新规划 |

64局训练加24局独评的88个首答全部相同：Depot4→Barracks4→Marine24。
独评三组16次query逐一相同，14次回答逐字相同。独评第二query没有与本轮训练
第二query逐字重复；这只能说明精确输入覆盖不同，时间/资源/敌情的差异是否造成
泛化困难还没有单独验证，不能只用hash不同证明泛化失效。

第二答时三个失败seed的观测已包含敌方Starport/带TechLab的Starport，但本阶段
候选都是Marine、Depot、Barracks、Reactor、SCV、Refinery及一条进攻指令，
不含Viking/科技转型。冻结`strategic_sc2_env_manager_v2.py:860–952`只接受当轮
候选；`planner.py:422–436,1316–1362`在已验证原段协议下保留合法回答及数量展开。
因此不能要求RL在该轮学出列表禁止的单位；仍能选数量、子集和顺序，不能夸大成
“模型没有任何学习空间”。当前配置中的原始6/6字段被`SC2_ORIGINAL_BO_SEGMENT=1`
的有效路径绕过，59动作队列是直接证据，未发现旧6条截断重新生效。

当前更有证据支持的解释：本轮RL提供并强化了一个局部有用的数量选择，随后组内
计划/回报趋同，任务梯度来源耗尽；该选择不足以解决其他失败场景。64局筛选中
另有5/8组在Marine8/42仍并存时也同为全胜，说明不只是“提高回答多样性”就能
自动产生有效奖励差。少量非零信号也包含调用时序，不应混称为战术学习。

未定位新的PPO实现bug；没有把长BO段或候选集合直接判定为bug。本轮不改接管、
阶段切换、prompt、微操或reward。已准备的零信号过滤只可能避免无任务信号的
更新，不会创造新成功计划；其后续验收必须看同一游戏预算内是否实际获得非零
任务信号和共同条件收益，不能把跳过零组本身叫“RL修好”。

### 此前结论与待验证候选（本轮未启动）

未发现新的advantage实现偏差。本次场景采样候选没有通过学习收益验收，不采用为
已修复方案。证据支持优先检查**训练探索是否过早收缩、未产生能改善失败场景的计划**，
而不是再加优化器/数值修复或重复8/42已能获胜的场景。

下一项候选可只改变rollout采样温度，在同一已失败场景、同一冻结权重下观察合法
计划多样性及reward是否出现区分；先不做PPO。温度和资源预算需另行确认，尚未实施。
若不同计划仍得到相同结果，就不能称探索候选修好了奖励信号；本轮不扩大成战术优化。
