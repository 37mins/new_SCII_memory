# SceneB8 同场景温度复核

状态：2026-09-15，用户批准47节点GPU4–7、温度0.4/0.8各8局串行、合计45分钟。
15:07资源复核发现GPU4/6/7已有他人任务，尚未启动；不停止这些任务，不擅自迁移。

## 问题与边界

SceneB8训练后三个独评失败seed未改善。此次只问：在实际失败seed1436145665，
提高采样温度是否产生不同的合法选择，以及结果是否有所不同。不是新奖励、战术或
训练机制，也不预称能解决零advantage。只有游戏结果与折扣/格式记录齐全时，才讨论
是否值得另批短RL；更多不同文本不等于可用奖励信号或学习收益。

这是**当前权重复核**，不是第一次温度实验：
[9月10日T0.4/1.0](../20260910-spb-fsdp-layer-wrap-v1/temperature-ab.md)无胜率差；
[9月12日A4、反馈开、T0.4/0.8](../20260911-execution-failure-feedback-v1/temperature-ab.md)
均0/8，较高温度仅增加计划多样性并降低有效率。旧负结果不删除。

## 冻结配置

两组均加载`spb2-A4-l6-scene-B-rl4-4gpu-r2/checkpoints/global_step_8`完整四分片，
父run源码`8c2f0f8b103dab054a612ce95a041e4dde9c7c8c`；完整路径由
`scene_posteval_check.PARENTS['SceneB8']`定义。原始模型manifest SHA256：
`40d6f1417568d5fda1107cdafd66c7441a679b03fb148851c8f226ecc9cf884c`。
不使用旧CoverageB8，不从原始SpB重新训练。

Level6/Harder、g2、对手Air；每组实际游戏seed均1436145665×8，初始Torch/CUDA
采样种子按rank0–3匹配。温度之外保持FP32 FSDP参数/归一化、BF16 autocast、
HF/eager/cache、SWA dropout=0、192-token、top_p0.9、top_k0、失败反馈关闭、
history0/off、约15条/x N及配额澄清prompt、reward和执行/微操不变。
同种子重复8局是采样差异测试，不代表8个独立场景。

仅在已有纯评测入口增加显式SceneB8温度hook，复用原checkpoint恢复和种子记录。
effective config两组只允许两处temperature数值及对应审计环境变量不同；运行路径不同。
生产模块未改。默认入口和历史A4温度实验行为不变。

## 执行与验收

`run_scene_temperature_node47.sh`需`SOURCE_SHA`、未使用的
`SCENE_TEMPERATURE_CONTROL`和`SC2_SCENE_TEMPERATURE_AUTHORIZED=1`；从clean SHA
在独立冻结目录运行。一个模型实例串行，45分钟总截止时间包含检查、加载与收尾，
不因第二组开始重置。CPU回归/配置对比先通过，实际恢复step8且SC2开始后才报告跑上。
不做PPO、不写新checkpoint、不复制大权重、不自动重试、不续96步。

RUN_ID预留：

- `20260914-zero-task-signal-sampling-v1-spb2-SceneB8-l6-temp04-eval8-s1436145665-r1`
- `20260914-zero-task-signal-sampling-v1-spb2-SceneB8-l6-temp08-eval8-s1436145665-r1`

run根为`/mnt/data0601/yangjinyue/new_SCII/experiments/20260914-zero-task-signal-sampling-v1/runs`。
新source/control使用`/dev/shm/sc2-scene-temperature-20260915-r1`，Ray分别使用
`/dev/shm/st04_0915_r1`和`/dev/shm/st08_0915_r1`。依赖复用47已验证目录，不修改旧deployment。

验收实际step8恢复、每组8个种子、相同初始采样状态、自然终止、0次训练更新及退出码；
报告全部16局胜负/累计reward、Planner调用、截断/有效率、原始与执行计划差异。
无模型决策局仍计入整局结果。若更高温度仅增无效回答/噪声，拒绝作为改进；
小样本胜负差只支持进一步测试，不宣布学习改善。

## 启动前核验（9月15日）

15:07，47 GPU0–3是既有vLLM服务；批准的GPU4、6、7上有其他用户的Python/vLLM任务，
仅GPU5空闲。15:09，43/44八卡均占用，48仅GPU1空闲；15:13，46八卡均占用，
42 SSH失败。资源是时点快照，不代表永久不可用。没有停进程、改卡数、创建评测run或
消耗已批GPU预算；启动仍需重新检查47 GPU4–7，不在后台无限等待或自动重试。

本地Python为Codex bundled runtime，`PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=tests`：

```text
python3 -B -m unittest test_scene_temperature test_scene_posteval test_feedback_learning_eval -v
18 tests OK
python3 -B -m unittest test_feedback_temperature_eval test_temperature_eval test_scene_screening -v
10 tests OK
```

首轮新测试用`04/08`作短run名，配置比较器将PYTHONPATH中的日期片段也替换了，
导致假差异；改为`temp-04/temp-08`后18项通过。仅测试夹具返工，不算项目bug修复。
shell语法检查、`git diff --check`通过；`agent_system/verl/configs/scripts`相对准备前
SHA`83e7007`无修改。服务器模型加载与SC2验收尚未执行，不以CPU通过代替实验跑上。

准备版本已冻结为`838886068dc1aa15fed99d5633a9c9fd30bbb8e5`，28项合并CPU回归通过。
本地完整历史备份`/private/tmp/sc2-scene-temperature-8388860.bundle`，`git bundle verify`
通过；SHA256为`923572a56d3fe18fe03d60be6e19d66b59ed0f043fb647f96370fe02a531c084`。
15:17再次连接47返回SSH退出255，未上传或创建服务器新目录。远端备份与空卡检查
仍须在正式启动前完成；不能把本地准备完成称为评测已经运行。

16:53–16:56继续核验：43/44/46/48/197均无空闲卡，42/47/198 SSH返回255；
47另用不复用连接的只读检查仍超时，不能据此推断其GPU状态。未停止他人进程或改变
批准的节点/卡组。本轮未创建评测run，45分钟GPU实验未开始。

通过健康的43已完成共享盘备份：
`/mnt/data0601/yangjinyue/new_SCII/staging/20260915-scene-temperature-r1/runtime-8388860.bundle`，
远端SHA256与上方一致，bundle的HEAD为冻结SHA`8388860`；不是GitHub推送或新部署。
随后只读核验SceneB8父run的`.done/result`一致、退出0、step8完成、来源正确，
13个checkpoint文件大小均匹配记录，合计63,552,435,735字节。未读取权重张量、
未加载模型；实际加载与SC2验证仍待批准的47 GPU4–7可用后执行。
