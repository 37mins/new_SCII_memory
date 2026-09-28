# 失败输入的示例顺序对照

2026-09-12，completed，用户批准43节点GPU0–3、总上限20分钟、不自动重试。
RUN_ID：`20260911-execution-failure-feedback-v1-spb2-A4-exampleorder-fixed1x8-4gpu-r1`。

问题来自[首攻回放审计](first-attack-replay-audit.md)：05:49已人口阻塞，失败反馈明确
建议先建Depot，但示例及实际第三答仍把Marine放在前面。这里检验示例顺序是否影响
这一个失败输入的纠错选择，不预设prompt是唯一根因，不测试胜率或学习效果。

## 唯一差异与输入

| A原示例 | B候选示例 |
| --- | --- |
| Marine2 → Depot2 → Barracks2 | Depot2 → Marine2 → Barracks2 |

只交换示例的两个完整bullet，不改变数量、待办顺序、阶段指令、观测、反馈或输出
约束。输入直接取`firstattack40-eval4-s42613-r1/prompt_response.jsonl`的env0、
step_index2、policy_action_id3，不重构prompt；该处四局query相同。
源runtime=`e3c0695adf0a2e36c97afba838ec3b508f000e34`，日志SHA256
`b1729caf9394a50072feda8f6836ae0907eb19b0b65f2d56df363373a9d61bf1`，原querySHA256
`18e7797126e1af1d9b48a27eb82293bfa457e9be081c72ba1647cef870955627`。

模型为SpB A4，来自`spb2-original-l6-g2-A-rl4-4gpu-s0-r2/checkpoints/global_step_4`，
父训练runtime=`361cb7ae7fd86fe8811d6fcb6946d0241bf93e11`，不是原始SpB、B4或A8。
两组共用一次实际四分片checkpoint恢复；保存加载后及生成结束的Adam计数，绝不调用更新。

## 运行与判断

复用`fixed_feedback_compare.py --example-order`及既有FSDP/HF生成、完整checkpoint
加载、`TrajectoryCollector`分词和现有validator/parser。不修改正式Planner或deployment。
配置从源run effective_config还原：FP32参数/温度归一化、BF16 autocast、cache、
dropout0、temperature0.4/top_p0.9/top_k0、最多192生成token，history0/memory off。
没有SC2或Trainer；游戏/BO/微操/reward不被运行或改变。

A/B各8答，同一输入，采样seed12000–12007逐对相同并重置。它们是8次生成抽样，
不是8个独立游戏种子，也不是恢复旧游戏在05:49的生成随机数状态。四卡每卡每次1答，
每组两次分布式生成。记录每答输入token哈希、随机数状态哈希、原始回答和token、
EOS/截断、校验原因、展开队列及首动作。检查分词后无输入截断。

主指标是**合法且Depot位于任何Marine/SCV之前**；同时报告Depot首项、严格合法率、
是否选择其他会阻塞的动作、数量和EOS。仅“形式合法”不算纠错成功；解析顺序也不
冒充真实执行或建成。若候选只增加无效回答/截断，不支持采用；若纠错选择改善，
最多支持此输入的局部示例敏感性，实际游戏价值另行报最小预算，不能自动合入主线。

入口：`SOURCE_SHA=<clean SHA> EXAMPLE_ORDER_AUTHORIZED=1 bash
experiments/records/20260911-execution-failure-feedback-v1/run_example_order_node43.sh`。
从独立冻结目录、受控tmux启动；只用已有本地权重，不复制大权重、不保存新checkpoint。
启动前复核GPU空闲和已有RUN，完成后记录退出码和全部16答。新SHA由manifest实际记录。

## CPU记录

`test_example_order_probe.py`3项及`test_fixed_feedback_compare.py`1项已通过。
真实输入CPU检查须带仓库`PYTHONPATH`；首次漏设该路径而导入失败，未加载模型或上卡，
这是诊断入口调用问题，不是项目原有bug。补齐调用环境后重验原始输入与唯一文本差异。
补齐后实际退出0：A/B的原回答校验及展开队列完全相同；文本diff只有Marine2/Depot2
两行换位。B query SHA256为`b29a12f6b167904995341ff3f5766089a3ecb0ad3772c492c9482b3805b752f1`。
`bash -n`、`git diff --check`通过；相对`e3c0695`的生产模块diff为空。

## 启动核验

冻结SHA=`0ac4c73afacf94914a1ddfa1015b152cc764e5fe`，43部署
`/root/new-scii-execution-feedback-20260911/code-0ac4c73`。
源码bundle备份`/root/new-scii-execution-feedback-20260911/example-order-deploy-r1.bundle`，
SHA256=`2c24ad8a415ea9ff0f0967b7dbb0c9b4aa46691583736290e4e47edda05782b5`。
新目录准备时首次fetch使用branch ref而bundle仅导出HEAD，检出前即失败；改用已列出的
HEAD完成准备，未变更代码或启动GPU。旧部署未修改。

18:23北京时间43八卡0MiB、无compute进程/tmux，主机可用449GiB，系统盘余94GiB。
服务器CPU4/4回归、全部System-v1文件哈希、源A4 completed/.done和四分片尺寸检查通过；
实际分词A/B均1478token，未截断，生成参数和history与源run相同。

18:25:47在`sc2_example_order_r1`启动，18:26:38四rank实际checkpoint加载返回并
开始生成，Adam计数均15；非仅目录存在。总截止18:45:47，GPU0–3，无SC2、无PPO。
原始产物在43本地`/root/new-scii-execution-feedback-local-20260911/shared/yangjinyue/
new_SCII/experiments/20260911-execution-failure-feedback-v1/runs/<RUN_ID>`。
18:30:23完成，退出0，用时4分36秒；18:30:25八卡均0MiB，任务已释放四卡。

## 完成结果：证据不足，暂不采用

| 同一个05:49输入、8个配对采样seed | A原示例 | B交换示例 |
| --- | ---: | ---: |
| 现有validator接受 | 8/8 | 7/8 |
| 合法且Depot先于Marine/SCV | 5/8 | 7/8 |
| 自然EOS并闭合回答 | 8/8 | 6/8 |
| 同时完整、合法、Depot优先 | 5/8 | 5/8 |
| 达192token且无EOS | 0/8 | 2/8 |

主指标有局部改善，但不能忽略预先要求核对的截断与配额错误。两组均只在**同一个**
失败输入上抽样，非独立8局，不据此推断总体效果。A本身有5条先补Depot，说明
该权重并非完全不会纠错；旧四局的失败输出也不代表这个输入必然只生成错误顺序。

真实配对（省略固定标签和标题）：

```text
seed12006 A: Marine2 → Depot2 → Barracks2 → Reactor2
          B: Depot1 → Barracks1 → Reactor1 → SCV1 → Marine2
```

B seed12002出现两个`barracks x 2`，总数4超过待办3，被现有规则拒绝。
B seed12000/12007回答相同：`Depot1 → Barracks1 → Reactor1 → SCV1`之后
反复输出`- marine x 1`，最终停在`- marine`，192token、无EOS及`</answer>`。
现有parser展开成26个Marine；没有超过待办30，因此validator仍接受。冻结
`strategic_sc2_env_manager_v2.py:879–880`移除answer标签后检查各行，`:944–947`
检查累计配额，**不检查标签闭合或生成是否到达EOS**。这里只澄清指标口径，未修改
validator，也不能把“接受7/8”说成“7/8完整正确规划”。

仅按合法纠错：3对改善、1对退化；再要求完整回答：2对改善、2对退化。没有改动
验收阈值或隐藏主指标，两个口径都保留。示例换序影响选择得到了这组样本的支持，
但尚未支持稳定纠错改善，更未证明Executor完整恢复生产、游戏获胜或RL学会。
本轮**不合入正式Planner，不扩大token预算，不新增规则或继续GPU测试**。
如继续，最有针对性的是用同一失败输入比较原始SpB与A4，区分初始可靠性与训练后
变化；它仍需单独资源授权，不重跑战术或长训。

16答的prompt/输入token哈希、配对CUDA RNG、score重算、4rank加载及生成前后
Adam计数15均核验通过。没有更新或新游戏。完整小型产物本地副本
`/private/tmp/sc2-example-order-results.UvLUuA`；配对指标及原始文件哈希见
[example-order-r1-results.json](example-order-r1-results.json)。原始生成、失败样本、
旧部署和权重全部保留。
