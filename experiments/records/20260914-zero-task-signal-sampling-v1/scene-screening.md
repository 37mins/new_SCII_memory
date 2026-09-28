# 先寻找可区分的训练场景，不再盲换四个 seed

2026-09-14，43 GPU0–3的64局已于22:51:20全部完成，控制器及八组均退出0，
55/64胜。21:35:46启动，耗时75分34秒，未超90分钟预算；
[启动记录](scene-screen-launch-r1.json)、[完成结果与CPU信号归因](scene-signal.md)。
只找到一个新混合胜负场景，没有满足预定的两个新场景门槛，未自动启动训练。
不做PPO、新checkpoint、资产复制、自动重试或长训，不修改当前A/B。

## 已有证据够做什么

按**实际生成这些回答的权重**分组，不把A4训练之前的批次、后续A5–A8、
B4人工干预或反馈开启的评测混成同一个A4采样池。也不把相同采样种子的重复运行
当作多个独立场景。

| seed / 场景 | 已有证据 | 筛选结论 |
| --- | --- | --- |
| 1098341402 | A4纯评测7/8胜；相同第二答输入，Marine42的7局胜、Marine8的1局败，Depot均8；A4开始的首批训练另有6/8和7/8 | 已有候选，但只有一个独立场景；不能宣称单一数量因果或泛化已证明 |
| 42613 | A4温度对照T0.4为0/8、第二队列2种，T0.8为0/8、8种；该旧对照反馈开启。当前反馈关闭A4面板仅1局败 | 温度实验仅作旁证，不混入本次反馈关闭的采样池，也不冒称已有同配置A4的8局结果 |
| 17089、3236598124 | A4原面板各只有1局败；后续CoverageB更新后各8局全败 | 不把后续权重的0/8冒称A4的8次采样；后者仅为旁证 |
| 45891、21243 | CoverageA分别8/8，存在Marine8/42两种第二答；只有前者是在第一次新增更新之前由A4生成 | 同胜局无须强造胜负排序；21243不可当作A4重复评测 |
| 30403、32103 | CoverageA后续权重均0/8，最终reward略有差别 | 未证明小幅回报差表示能改善胜负的规划信号 |
| 原八seed面板与新八seed面板 | A4/A8/B8各5/8；多数seed每模型只评1局 | 有独评无收益证据，但不足以判断每个seed内能否采到胜负两类回答 |

B4的Marine8→42单点干预在同一1098341402场景为0/4→4/4，说明该场景中计划能
影响游戏；它用了B4和人工回答，不算A4自主学习成绩。已有A4→单场景step5独评
7/8→8/8，而原八seed仍5/8；因此不再重复该一个seed的拟合验证。

Coverage两组8个训练batch中只有1组有胜负混合；部分全败组的reward/调用次数不同
也产生非零优势。不能将“非零组数”当作“有价值的规划排序组数”。当前零组候选的
两批同分残差5.72e-6被接受，是这个候选未覆盖目标现象的证据；不据此改变训练精度、
奖励或自行增加判零阈值。

## 下一项最小补证：只评A4的新训练候选seed

提议8个新seed、每seed8局，共64局，Level6/g2、同一A4、失败反馈关闭。
每个seed独立一组，保持生成T0.4/top-p0.9/192tokens、约15条/xN协议、
history0/off、FP32参数/归一化、BF16 autocast/cache、SWA dropout0及全部系统行为。
不干预回答，不做PPO，不保存新checkpoint，不重复1098341402或已完成独评。

新seed在查看其任何结果前生成并固定：取UTF-8字符串
`20260914-task-signal-screening-v1/train/<i>`的SHA256前4字节，大端无符号整数，i=0…7：

```text
3974528993, 1183530000, 3446496864, 2996859384,
3847406327, 1824802814, 175019443, 3122016241
```

按8次现有单批纯评测串行复用已验证四分片加载/评测入口；每次首次生成仍按rank
设0–3，不逐call重置，记录真实游戏seed及采样状态。没有为此新造多批评测循环。
入口已实现，必须从新clean SHA和唯一RUN启动；不修改已冻结部署。

已批准资源：当前B结束且43 GPU0–3确认释放后，四卡串行，合计最多90分钟，
包括加载与收尾；不复制大权重，不自动重试、扩样或续训。新训练A/B不包含在此授权内。

记录每个seed的全部胜负/整局reward、真实Planner调用与有效率、逐字相同输入下
的不同原始回答及解析/转发计划。不同回答与胜负关联只作为候选证据；引擎噪声、
未执行后缀和不同输入不能冒充规划因果。全胜/全败也保留，不报过滤后的胜率。
至少找到两个不同seed出现自然胜负混合且有可核对的计划差异，才值得提交
训练采样分布A/B；八局只是筛选，不能证明真实成功概率介于0和1。
若不足两个，报告场景筛选不足，不用微小奖励差/数值残差强行补足，不自动加预算。

随后训练A/B只改seed采样分布、都关闭零组过滤，使用同一A4完整状态与相等总游戏预算。
该训练及后评另报预算。旧面板包含已训练seed，只能称回归；已有fresh8结果也已看过，
不能再称新盲测。训练候选确定前另冻结独立评测种子，评测结果不得用于筛训练场景。

## 可追溯证据

历史汇总来自只读commit `7c92c8f8628c3923637212bcb34f85d26536d923` 的
`experiments/records/20260911-execution-failure-feedback-v1/`，不是借用该分支后来的
阶段重规划runtime。Coverage实际训练SHA为`ce61865ac709fae44be22e705a8349720a19df52`，
后评为`2bf247d8c247695025404e9042e173df2a74ad1d`。

| 已读取的汇总 | SHA256 |
| --- | --- |
| coverage-final-result.json | 487b7df2d38d0afee34b39657e2ac449564753af23082bb567997302b8c8750d |
| a4-scene-eval-r1-result.json | ef0da90e0be88259068403b8a6135768d9960c4d44138398db476b0507d61697 |
| a4-multiseed-eval-r1-result.json | 46350e9598b797d6a726a97118cf5db0e9a458e61a4a5ebc764bcc55a48ba64d |
| temperature-r1-results.json | 20292a7b2c853e7e224b33cf0761cc4caaa4bdb7baf6940c9270f549e2f454a6 |
| a4-scene1-r1-result.json | c45dc60bb9bc4b721de52289249ff7e691a4aed062e740203cdbb97b519062ba |

本轮重新按coverage-final-result的登记哈希核验本地snapshot
`/private/tmp/sc2-coverage-final.ngiw5D/train-{A,B}`的16份输入文件，全部一致。
其余表格复用已有已归档审计，不声称本轮重新解析全部回放或加载模型。
上述只读筛选阶段仅新增记录。随后获批的评测入口仅改实验编排文件；
`agent_system/verl/gigpo/scripts/configs`相对当前A/B运行`0cccaa9`无diff。

本地CPU校验实际通过：16份Coverage输入SHA256、状态JSON计数、8个新seed的
确定性生成/范围/唯一性，以及与上述已登记训练和两张评测面板无交集；
`git diff --check`退出0。这些是记录校验，不是新候选入口或GPU验收。

## 获批入口与验证

`learning_eval.py`增加固定seed表和`install_screen_scene()`，调用现有
`install_scene(game_seeds=[seed]*8)`；`run_feedback_eval_node47.sh`仅增加显式
`FEEDBACK_SCENE_SET=screening`分支及scene-index传入。原单场景和多seed入口保持原样。
所有回答仍在线生成，不使用温度实验/干预实验hook。

控制器`run_scene_screen_node43.sh`先检查pair-r1完成、GPU0–3空闲及A4元数据/分片大小，
再按预定顺序运行8次已有纯评测入口，使用共同90分钟deadline，每轮核对exit0、
0更新、8局自然终止、真实游戏seed、首次采样seed和A4加载返回。
输出复用本experiment的共享runs，每轮唯一RUN：
`20260914-zero-task-signal-sampling-v1-spb2-A4-l6-screen-s<seed>-eval8-4gpu-r1`。
新控制目录`/root/new-scii-zero-signal-20260914/scene-r1`，Ray临时目录
`/dev/shm/ss_<index>_0914_r1`；模型/父checkpoint/依赖全部读43已有本地副本。

```bash
SC2_SCENE_SCREEN_AUTHORIZED=1 SOURCE_SHA=<冻结SHA> \
SCREEN_CONTROL_DIR=/root/new-scii-zero-signal-20260914/scene-r1 \
bash experiments/records/20260914-zero-task-signal-sampling-v1/run_scene_screen_node43.sh
```

本地命令`PYTHONPATH=tests <bundled-python> -B -m unittest
test_scene_screening test_feedback_learning_eval -v`：16/16通过，退出0。
包含8个场景的dry-run配置逐项对照：相对已验收A4单场景入口，仅hook和scene-index不同；
原入口回归、只读加载选择、禁止额外采样及预算测试通过。首次使用系统python因无numpy
8项导入错误，改用已有bundled Python通过；未安装依赖，不算项目bug或GPU重试。
`git diff --check`、控制器`bash -n`通过。43隐藏CUDA、固定依赖复核16/16通过，
日志`/root/new-scii-zero-signal-20260914/cpu-screen-r1.log`；runtime为clean
`c684176d41bc4debe830896572cf0a5c51e9a43a`，增量Git备份两端哈希一致。
首轮manifest为evaluation/4卡，checkpoint加载返回global4；`val_only=true`、
0次optimizer更新，实际8个游戏seed全为3974528993，SC2已到执行开局阶段。
以上为21:39的启动核验；最终64局完成与信号分析见本文顶部链接，未自动开训练。
