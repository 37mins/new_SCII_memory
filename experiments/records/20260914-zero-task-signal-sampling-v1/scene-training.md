# 训练场景采样 A/B

最新收尾（2026-09-15）：A-r1和B-r2均完成新增4步并保存完整step8；B-r1失败保留。
12:05共同独评完成，A4/A8/B8均5/8胜、同三个seed失败，见文末。
随后[三批非零信号核对](nonzero-plan-credit.md)确认存在Marine42/8的局部选择信号，
但它未改善独评失败场景；未发现新的实现偏差，未续长训。

2026-09-15：用户批准将“两新混合场景”门槛改为“一旧＋一新”，并批准43节点
GPU0–3串行A/B、总计最多3小时、约120GiB新产物。不重试、不续长训，独评另报预算。
状态：A完成、B失败，配对训练未完成；冻结SHA
`75a6b5dc3621e9d670fbfb440ae917bfacd5e360`。00:13:51启动，A于00:44:12
结束、00:45:28完成控制器检查；B于00:49:13失败，控制器退出1，未自动重试。
以下时间均为2026-09-15北京时间。原 [启动记录](scene-train-r1-launch.json) 保留为00:17快照，
不再用其中的running表示当前状态。09:37–09:47只读复核了原run完成状态和产物。

## r1实际结果

| A的global step | 实际游戏seed（每批8局） | 胜负 | 平均整局reward | 补齐前最大绝对task advantage | 记录的裁剪前梯度范数 |
| --- | --- | --- | ---: | ---: | ---: |
| 5 | 3974528993 | 8胜0负 | 20.500 | 0 | 0.009 |
| 6 | 1183530000 | 8胜0负 | 20.500 | 1.748907 | 1.701 |
| 7 | 3446496864 | 5胜3负 | -5.743 | 65.607185 | 120.836 |
| 8 | 2996859384 | 8胜0负 | 20.500 | 0 | 0.004 |

A新增4个global step，32局自然终止、29胜，退出0，step8完整保存；`.done`与
`result.json`一致，记录所列model/optimizer/extra/data文件尺寸全部匹配。没有空策略
批次，metrics有限。task advantage来自`checkpoints/signal_sampling.jsonl`的真实回答、
复制补齐之前；不要混用补齐后metrics的数值。step6全部整局reward相同，但一局调用
4次，其余2次，并有一次格式无效/回答触顶；其非零信号不能当成胜负区分。
两批精确零、一批胜负混合，尚不支持“持续提供任务学习信号”或“已学会”；未做独评。

B首次生成时发生Ray主机内存OOM：日志记录962.67/1007.51GB、95.5501%，超过
95%阈值。退出1，新增0步、完整游戏0局、没有新checkpoint。不是模型0胜率，
`metrics_finite=true`在空metrics情况下也不能当作数值验证通过。
失败时四个B worker PID565062/565262/565260/565261各约33.6GB；Top10还列出
六个约20.82GB的`actor_rollout_init_model`进程。前四个已匹配B自己的Ray日志；
后六个未在A/B的Ray日志目录匹配，归属尚未确认，不能据此断言是A未释放或他人任务。
当前43占八卡任务的PID与这次失败进程不同，不拿当前占用反推昨晚根因。

证据哈希（原run根路径见下文，checkpoint内容本次未重读）：

| 文件 | A SHA-256 | B SHA-256 |
| --- | --- | --- |
| `result.json` | `e1d44946c96c639edaa30ad8a2e5364be8b31d9ba8206a75aadf33802b37249a` | `166678525f4f43e808dc77beea1216ecd6023399417fcbc23a79602e7e3a96f3` |
| `metrics.json` | `65ea9505a71add524f23d2f12b60613f933df410364f10d357eaa98efdc9d68c` | `37517e5f3dc66819f61f5a7bb8ace1921282415f10551d2defa5c3eb0985b570` |
| `effective_config.json` | `465d77c65c02bda6bf4a2c03ebdb74cccd4791586098ec99b45b544ea9fda071` | `f9f9d5dee584da07209667f8e6f056033cbf2fdc8ccbb429d289cca9acfc04bc` |

## B单组补跑方案（用户已批准，部署/训练状态另记）

用户已批准下述47四卡两小时、资产迁移和约130GiB共享存储预算；不重跑A。
开始迁移时写入唯一`started_unix.txt`，截止时间固定为其后7200秒，部署耗时计入。
阶段性迁移失败不自动重试，训练不得另起两小时计时。
本次实际迁移10:08:17开始，统一截止12:08:17。冻结训练runtime为
`8c2f0f8b103dab054a612ce95a041e4dde9c7c8c`；与r1生产目录无diff，仅训练记录入口
允许显式传入依赖路径。9项本地CPU回归通过，实际47环境测试及GPU加载另行记录。
受控43会话`sc2_scene_B_stage_r2`只迁移，控制日志在
`/root/sc2-scene-B-staging-20260915-r2/staging.log`。

09:42只读快照：43/44/42/46无空卡，48仅GPU0–1空闲；47八卡空闲，
MemAvailable约929GiB、`/dev/shm`剩454.5GiB，但系统盘100%。47原Python目录
存在NumPy1.26.4和2.5.2两份dist-info，未证明实际导入版本与43一致；不直接沿用。
共享SpB/SC2/FLA/MoBA存在，但共享路径没有本次所需完整A4；A4当前可用来源在43本地。

拟迁移43已验证依赖约8.1GiB和A4完整状态约59.2GiB，不复制原始SpB，不下载模型。
依赖、临时文件和缓存放47独立内存盘目录，A4经独立共享暂存供47读取；源码和控制
记录使用独立目录，不写满盘，不修改47旧环境或43冻结源码。启动器只做必需路径适配，
新SHA及CPU/dry-run配置对比通过后才可启动；所有训练/游戏字段与r1 B一致。

拟使用47 GPU0–3、同一A4恢复、新增4个global step（5–8）、每步8局，种子仍为
`1098341402,3446496864,1098341402,3446496864`。step8保存完整状态。
总批准预算2小时（含部署、校验、加载和训练），新增共享存储预算约130GiB，
不自动重试、不续96步、不做独评、不重跑A；不关闭Ray内存保护、不清理其他进程。
启动前再次检查四卡、无并发大型Ray任务及主机内存余量，资源不满足则不启动。
新RUN_ID拟为`20260914-zero-task-signal-sampling-v1-spb2-A4-l6-scene-B-rl4-4gpu-r2`；
09:45确认尚未使用。旧r1失败产物和A8全部保留。

迁移入口`stage_scene_B_node43.sh`在43执行，仅复制完整A4及必要manifest/seed/状态，
逐文件对照源/目标SHA；依赖打包并校验。新暂存为
`/mnt/data0601/yangjinyue/new_SCII/staging/20260915-scene-B-r2`。
训练入口仅将原硬编码Python和include路径改为可显式传入的
`FEEDBACK_PYTHON_OVERLAY` / `FEEDBACK_INCLUDE_ROOT`，默认仍为43旧路径；
不修改生产源码、prompt或算法。CPU新增回归要求两种路径的effective config仅
PYTHONPATH不同，其余逐字段相同；CPATH沿原入口由driver环境继承，另在启动环境记录。
首次测试误将CPATH当作Hydra字段而失败，修正测试后再验收，不记为项目bug修复。
47的源码、依赖、临时控制socket和缓存
均使用独立`/dev/shm/sc2-scene-B-retry-20260915-r2`族目录，checkpoint写共享run。
`run_scene_B_node47.sh`是独立编排，调用冻结runtime的既有B入口，不调用A。
`check_scene_B_retry.py`对照原B的effective config，只归一化明确的路径/RUN_ID；
模型来源manifest、系统文件哈希、依赖版本、Python版本、四卡与起止步数逐项一致才放行。
两小时包含等待迁移、CPU测试、dry-run、实际模型加载、四步训练和checkpoint保存；
完整运行与学习效果仍分别验收。启动资源守卫要求GPU0–3空闲、无既有raylet且
MemAvailable大于600GiB；这是此次运行的安全余量，不是声称找到OOM根因。

## 原实验定义与冻结配置

问题是：同样32局预算，选择已有胜负区分的场景，能否比未按结果筛选的场景表
持续产生可区分的任务advantage并完成连续RL？它是训练采样候选，不是奖励或PPO修复。
混合场景也可能在首步后全部趋同；出现这种情况即报告候选不足，不能临时换种子。

| 新 global step | A：预先筛选顺序的前四个场景 | B：旧＋新混合场景交替 |
| --- | --- | --- |
| 5 | 3974528993 | 1098341402 |
| 6 | 1183530000 | 3446496864 |
| 7 | 3446496864 | 1098341402 |
| 8 | 2996859384 | 3446496864 |

每步同一场景8局，不是8个独立场景。A取筛选前固定列表前四项，不按胜负挑最差
对照；B依据已记录的一旧一新混合结果，是有选择的课程。两组均恢复同一SpB A4
完整模型、optimizer、scheduler与per-rank状态，使用原继续训练边界；新游戏从
reset_index=0、driver seed0开始，不称为SC2随机状态无缝恢复。

父run：`20260911-execution-failure-feedback-v1-spb2-original-l6-g2-A-rl4-4gpu-s0-r2`，
父SHA `361cb7ae7fd86fe8811d6fcb6946d0241bf93e11`，`checkpoints/global_step_4`。
原始模型资产为SpikingBrain-2.0-instruct，manifest SHA-256
`40d6f1417568d5fda1107cdafd66c7441a679b03fb148851c8f226ecc9cf884c`；不下载或复制权重。

仅实验hook增加显式种子表，旧表及旧入口默认行为不变。两组都关闭零组跳过，保留
既有4批上限和只读signal记录：每组必须真实更新4个global step，不能用少更新
获得低KL来宣称改进。生产源码不变，冻结：FP32 FSDP参数/温度归一化、BF16
autocast、HF/eager/cache、分层FSDP、SWA dropout0、response192、prompt4096、
temperature.4、top_p.9、top_k0，及Level6/g2/Air/Flat128、开局Planner、约15条/x N
配额prompt、现有过滤与补齐、微操/自动战斗、反馈关闭、history0/memory off、
step shaping off、GiGPO mean_norm/gamma.95/step权重1/无效惩罚.1、lr1e-6、loss/KL裁剪。
全量effective config、资产校验结果、回答、seed和终局日志由原启动器保留。

入口仍为 `run_pair_node43.sh`，设置`SC2_SCENE_SAMPLING=1`、
`SC2_SCENE_SAMPLING_AUTHORIZED=1`、`SOURCE_SHA=<clean SHA>`、唯一`SIGNAL_CONTROL_DIR`。
RUN_ID：`20260914-zero-task-signal-sampling-v1-spb2-A4-l6-scene-{A|B}-rl4-4gpu-r1`。
共享run根不变；控制目录`/root/new-scii-zero-signal-20260914/scene-train-r1`，
独立部署`/root/new-scii-zero-signal-20260914/code-<SHA>`，
Ray目录`/dev/shm/ss_{A|B}_0915_r1`。总截止时间从控制器开始算，包含预检和加载；
A失败、卡被他人占用、存储失败或超时则停止，不杀他人任务，不自动重试。

先报告每批真实返回、episode/step/task advantage、精确零及残差信号、回答多样性、
有效动作、真实optimizer update数、KL/梯度及两次完整step8 checkpoint。
回答数变化、调用时序产生的非零advantage必须与胜负区分分开。自然终止、无空策略、
完整恢复和保存是运行验收，不是学习验收。B训练胜率受场景选择影响，不能直接
与A训练均值相减作为学习收益；只有另行批准的共同条件独评，才能判断泛化改善。

保留所有失败和基线产物；本次不合入新的学习机制。该分支的外部GitHub同步仍待
明确权限，启动前保存可验证的私有服务器Git bundle，交付时不声称已推GitHub。

本地CPU验证：`python3 -B -m unittest discover -s tests -p test_signal_sampling_record.py -v`
8/8通过；bundled Python执行`test_a4_continuation.py`为6通过、2项Torch测试跳过。
三份启动脚本`bash -n`及`git diff --check`通过；生产目录无diff。
部署后用43既有Torch环境、隐藏CUDA执行上述两套测试，16/16通过、无跳过，退出0；
真实dataloader耗尽边界、四次种子消费及第5次拒绝已验证，冻结系统哈希全部匹配。

## B-r2启动核验（2026-09-15 10:35北京时间）

仅补B，未重跑A。迁移10:08:17开始、10:29:18退出0；原A4的23个文件共
63,568,322,890字节，源与目标逐文件SHA256全部相等，见
[复制凭据](scene-B-r2-checkpoint-copy.json)。依赖归档SHA256
`04543994c613e343d2ddd80a24f3356036a06ac8a1c7125a0fe56f0c6e107596`，47复核通过。
共享FLA的437个、MoBA的8个源码/运行库文件与43原本地资产逐文件相等。

47上CPU执行`test_signal_sampling_record.py`为9/9、`test_a4_continuation.py`为
8/8，均退出0、无跳过；冻结系统哈希通过。此次本地系统Python另跑continuation时
因缺NumPy有4项导入失败、2项Torch跳过，不将其称为通过；47使用复制的依赖完成了
完整验证。dry-run与r1 B的训练字段、生产文件哈希、Python及依赖版本一致，
仅按清单归一化迁移路径与RUN_ID。真实完成后的effective config仍由控制器再核验。

47四卡启动前资源检查通过，10:30:00进入加载，10:34:45已创建Level6/Flat128/
Terran Air游戏，10:34:53出现开局观测与首次Planner调用；10:35四卡均有模型占用。
截至本次核验尚无新增完整global step，不把已开局称为PPO已更新或学习验收通过。
10:36补核验：`a4_continuation.json`确认加载global4，模型/optimizer加载返回，
原rank状态恢复、旧epoch耗尽边界已处理；第一批实际8个游戏seed均为1098341402。
游戏从reset_index0重新开始，未声称恢复SC2随机状态。
预算含迁移和保存，仍统一在12:08:17截止；不自动重试，不增加评测或96步。
冻结runtime与独立编排SHA、完整路径及后续恢复证据见[启动记录](scene-B-r2-launch.json)。

## B-r2结果与结论（2026-09-15 11:17北京时间）

11:02:34训练结束、11:03:34控制器验收完成，均退出0。47 GPU0–3，冻结SHA
`8c2f0f8b103dab054a612ce95a041e4dde9c7c8c`；实际配置对照通过，仅路径和RUN_ID变化。
32局全部自然终止、31胜，无空策略批次，metrics有限。step8的model/optimizer/
extra/data文件完整性检查通过，本次复核尺寸全匹配；未额外重新加载整套checkpoint。
11:14快照47八卡显存均为0，未启动新任务。小型原始产物哈希、完整路径和验收证据
见[scene-B-r2-final.json](scene-B-r2-final.json)。

| global step | 实际游戏seed | 胜负 | 平均整局reward | 补齐前最大绝对task advantage | 有效动作率 | 裁剪前梯度范数 |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| 5 | 1098341402 | 7胜1负 | 11.688 | 80.0625 | 95% | 79.341 |
| 6 | 3446496864 | 8胜0负 | 20.500 | 0 | 100% | 0.005 |
| 7 | 1098341402 | 8胜0负 | 20.500 | 0 | 100% | 0.003 |
| 8 | 3446496864 | 8胜0负 | 20.500 | 0 | 100% | 0.002 |

上述reward/梯度/有效率来自逐步日志，按日志精度报告；advantage来自真实回答、复制
补齐前的JSONL，不是日志四舍五入为0。step6–8各16条真实回答、752有效token，
各局都调用两次、累计reward均20.5，task advantage全零；对应pg_loss为0。
reference KL记录0.010/0.011/0.012/0.012，不能与PPO old-policy K1混称；
四批response clipping均0。全零task advantage不代表KL分支、Adam状态或参数都不变。

结论：这次解决了补跑未完成的阻塞，验证同配置A4完整恢复、四步运行和保存；
没有修改或证明修复奖励/PPO根因。B只有1/4批有非零任务信号，A为2/4批非零
（其中仅1批有胜负混合），本次两场景交替不足以持续提供区分信号。
同一seed在step5的7/8变为step7的8/8，既可能包含训练效果，也可能是采样波动；
没有共同条件独评不能定论。B的31/32也不能与不同种子的A的29/32直接视作学习增益。

下一项先做已保留A4、A8、B8在共同种子/采样条件下的独立评测，判断有限的非零
更新是否带来迁移收益，再决定采样方案去留；需另报GPU预算。当前不新增过程奖励、
不修改战术、不继续96步，也不将本候选合入正式学习方案。

## 共同条件独评（2026-09-15，八卡60分钟已批准）

用户确认预算并要求八卡并行：47 GPU0–3先A4后SceneB8，GPU4–7同时SceneA8。
各8局Level6/g2，合计24局、共同最多60分钟含CPU门禁、加载及收尾；
不做PPO、不新增checkpoint、不复制大权重、不自动重试、不续长训。
11:21只读快照47八卡均空闲，无raylet，MemAvailable约920GiB，原独立Python/include
仍在。三份checkpoint的完成标记、来源SHA、四分片及文件尺寸均通过；实际加载
返回4/8/8仍须在启动后验证，不把尺寸检查当作加载成功。

评测仅切换checkpoint：A4为本记录的原父run，SceneA8为scene-A-r1/global8，
SceneB8为scene-B-r2/global8；三者路径及SHA已在本记录登记，不混用旧CoverageA8/B8。
原模型资产及47依赖只读复用，最多两个四卡模型实例；第三个等待A4退出释放四卡。
11:27复核八卡仍空、MemAvailable约920GiB、无raylet。启动门禁要求八卡空闲、
MemAvailable超过800GiB，保留Ray既有内存保护；双模型峰值仍需实际运行观察。
不为保持并行关闭内存保护，也不杀其他任务。旧部署与结果保持不变。

采用历史commit `2bf247d8c247695025404e9042e173df2a74ad1d` 的
`learning_eval.py:FRESH_GAME_SEEDS`完整面板，不按这次模型结果挑seed：
`2259791793,1436145665,363527354,462000319,3434611143,1968356657,4276540916,2065928601`。
它在本次两组训练之前已固定，与A/B新增四步的全部训练seed无交集（CPU检查通过）。
该面板曾用于历史评测，已有结果被看过，所以只能称共同条件独评/回归，不是新盲测，
也不声称它与所有更早训练均无交集。每模型每seed仅一局，不能证明稳定胜率或架构优势。

复用现有`feedback_eval.install`纯评测及采样状态记录，只补本实验标签、checkpoint
路径与上述已有seed表的薄入口；不迁移历史训练/战术候选。反馈关闭、history0/off、
FP32参数/归一化、BF16 autocast、HF/eager/cache、SWA dropout0、192token、
T0.4/top-p0.9、prompt/约15条及xN、g2、reward和全部执行层保持一致。
三组初始Torch采样seed都按rank0–3设置，不逐call重置；核对实际game seed和各卡
初始采样状态，不只比较一个driver seed。模型路径之外的effective config逐项比对。
新增标签`SceneA8/SceneB8`不覆盖旧`A8`意义。`learning_eval.install_scene_posteval`
只向原`feedback_eval.install`传入global4/8和上述seed表；既有shell入口通过显式
`FEEDBACK_SCENE_POSTEVAL=1`接入指定checkpoint，默认行为不变。
`run_scene_posteval_node47.sh`负责60分钟共同截止与两组四卡编排，
`scene_posteval_check.py`检查父权重来源/尺寸、三组配置和每组纯评测收尾。
三组实际初始CPU/CUDA采样状态哈希还需相等；不只核对seed数字。

本地已有bundled Python执行`PYTHONPATH=tests python3 -B -m unittest
test_scene_posteval test_scene_screening test_feedback_learning_eval -v`：20/20通过，
包含新三组dry-run、旧入口回归、一次8seed读取、拒绝额外采样及配置漂移测试。
shell语法和`git diff --check`通过；生产目录无diff。服务器启动前另跑同20项，
通过后才加载模型。本地CPU通过不代表GPU/SC2已启动。

独立部署/控制根`/dev/shm/sc2-scene-posteval-20260915-r1`，源码`code-<SHA>`，
依赖仍读`/dev/shm/sc2-scene-B-retry-20260915-r2/deps`；私有Git bundle备份放
`/mnt/data0601/yangjinyue/new_SCII/staging/20260915-scene-posteval-r1`。
共享本experiment runs下唯一ID为
`20260914-zero-task-signal-sampling-v1-spb2-<A4|SceneA8|SceneB8>-l6-panel-eval8-4gpu-r1`。
Ray目录`/dev/shm/pe_<label>_0915_r1`分别独立，沿用已有`ray_init.address=local`，
不连接另一组Ray；无预算重置、自动重试、训练或新checkpoint。

按同seed报告胜负、整局累计reward、真实Planner调用数、有效决策数/消耗决策数，
原始回答与解析/实际执行计划；终止后的填充行排除，未调用Planner的局也计入整局结果。
重点检查训练前失败局是否改善、原胜局是否退化，以及同输入下回答改变是否实际进入
执行。独评无改善则不能采用采样候选为学习修复；少量胜负差只作方向性证据，不启动96步。

### 启动核验

冻结SHA `8069b4fd6a8a8bb1b5308642fcf499e29a4bece1`，private bundle源/目标SHA256
`47050acee30b8031f8b1ce92cdea6a6cdd38f767ff12884643df2bd25dff33e2`一致，生产目录
相对B-r2无diff。11:41:56启动tmux `sc2_scene_post_r1`，截止12:41:56。
服务器20/20 CPU测试通过，父权重/配置/资源门禁通过。A4 GPU0–3、SceneA8 GPU4–7
均已成功恢复到4/8，并在SC2实际开局、调用Planner；两组实际八seed相同、val_only=true、
0次optimizer更新。SceneB8仍等待A4结束，未额外开局。11:45可用主机内存约644GiB。
未观察到启动错误或OOM，仍不能将启动成功当作评测完成或学习验收通过。
状态、完整RUN_ID及路径见[启动记录](scene-posteval-r1-launch.json)。

### 独评收尾：运行通过，未见学习收益

2026-09-15北京时间12:05:09三组全部结束，suite及各组退出码均0，24局全部自然
终止、0次PPO、无新增checkpoint。A4/A8/B8分别于11:53:36、11:54:26、12:05:08
完成，真实恢复global4/8/8。共同初始CPU/CUDA采样状态核验通过；12:08节点47
八卡显存均0。冻结源码仍为`8069b4f`，未修改运行模块或启动下一项。

| 权重 | 同面板胜负 | 平均整局累计reward | 合格/实际消费回答 |
|---|---:|---:|---:|
| A4父权重 | 5胜3负 | −5.722750 | 16/16 |
| SceneA8 | 5胜3负 | −5.728875 | 16/16 |
| SceneB8 | 5胜3负 | −5.728875 | 16/16 |

三组逐seed胜负完全相同：1436145665、363527354、3434611143仍失败，其余五个
seed均胜。每局恰有2条真实模型决策，没有无Planner局。reward按首次terminal
之前及当次所有返回值累计；合格率只含`policy_action_applied`的消费回答，不含占位。
每组8局单次评测，不以细小reward差或5/8胜率宣称改善/退化的统计结论。

回答并非完全没变：A8、B8与A4的16个query逐字相同，各有14/16原始回答和
projected回答相同。A4有`marine x 8`的回答，以及另一个包含攻击行的回答；
训练后两组这两个变体消失，所见第二答统一使用`marine x 42`，仍先Marine后Depot。
各组没有观察到不合格输出；但原败局未转胜，也没有原胜局转败。
projected回答不是引擎完成证据；本次未据此宣称每项BO实际完成。

结论：两组新增四步在本面板上**均未提供可见的游戏收益**，也没有观察到胜负层面的
退化。结合B后3批task advantage精确零，本次场景采样方案没有通过“持续提供有效
信号并改善学习”的验收，不作为已修复方案、不续96步。参数/概率可能已变化，不能
从14/16回答相同倒推没有更新；也未定位新的PPO实现bug。

唯一建议的后续是复用已有训练batch，核对非零advantage究竟落在上述计划选择还是
调用/格式差异上，解释为什么少量更新没有改变失败结果；不新增战术或过程奖励。
逐局reward、输入产物SHA256及比较计数见[收尾JSON](scene-posteval-r1-final.json)。
