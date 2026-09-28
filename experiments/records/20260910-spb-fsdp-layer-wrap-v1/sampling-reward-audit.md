# 输出趋同与奖励区分：2026-09-10 只读审计

结论：当前采样逐渐集中到少数计划；奖励并非与计划脱钩，但对同样获胜的不同
计划常没有区分。另已验证：`gamma=0.95`按Planner调用次数折扣，在同样获胜、
终局奖励相等时，偏向少调用的计划。这是训练目标的实际偏向，尚非实现bug，
也未证明它是趋同的唯一原因。未改运行代码、配置，未增加GPU或游戏。

## 来源与范围

运行根为
`/mnt/data0601/yangjinyue/new_SCII/experiments/20260910-spb-fsdp-layer-wrap-v1/runs`。

| 对象 | RUN_ID（共同前缀为20260910-spb-fsdp-layer-wrap-v1-） | 冻结SHA | 分析范围 |
| --- | --- | --- | --- |
| L6，43八卡 | spb2-l6-g2-opening-quota-rl96-8gpu-s0-r1 | 6a1657f7e045dcf998495a3d0adfb331572bf60d | step1–23 |
| L7，44 GPU4–7 | spb2-l7-g2-opening-quota-rl4-4gpu-s0-r1 | cfdd3b468cf9bbe3a1f85ce8dc9a5a9ea530ffbd | step1–4 |
| L7续训，同节点卡数 | spb2-l7-g2-opening-quota-resume4to96-4gpu-s0-r1 | 90b3fdc2fb7fadc1230861306b20c27540d9b736 | step5–25 |

读取manifest、effective config、console、prompt_response、episode_outcomes、
training_seeds，逐env/结局step_index分割。48批、384局，均指global step。
实际仍是SpikingBrain-2.0-instruct、HF采样、temperature0.4、top_p0.9、top_k0、
192token、history0/off、swa_dropout0。开局prompt的完整SHA256均为
`95f4e1650fe0f587aba982557916cbb0e59d9bcbf1c10092fd40664578a5d424`。

运行中日志继续追加，本次分析只取上述完整批次，不代表最终96步结论。
[逐步统计与原始回答摘录](sampling-reward-audit.json)保存完整RUN_ID、所查数据与限制。

## 1. 趋同具体发生在哪里

L6开局曾先集中到小计划，再切到大计划，不是一直单调复制全部待办。

| L6批次 | 同一开局输入的8个回答 | 当批胜局 |
| --- | --- | --- |
| step7–16 | 8个均为depot2 / barracks2 / marine2 | 随游戏seed变化 |
| step17，seed39512 | 7个小计划；仅env1选择depot4 / barracks4 / marine24 | 仅env1获胜 |
| step18–23 | 8个均为depot4 / barracks4 / marine24 | 包含step18的8胜、step20的0胜 |

step17唯一胜局reward20.5，其他七局为−49.5或−50；下一步全部选择该大计划。
这支持训练强化少数成功模板的解释，但后续回答也不同，不能把获胜全部归因于
第一条计划，更不能由advantage符号断言具体token参数更新方向。

L6 step1的整局原始回答序列有8种，step18、20各只剩1种；L7也反复出现同类
集中现象。L6日志actor/entropy由step1的0.103降至step18的0.005、
step23的0.002，仅作旁证：这是不同batch上的全词表指标，不是同一输入、
top-p采样后的严格熵对照。

补充上一份审计：L7 step6虽有3种原始文本序列，但差别只是是否带标题，
动作名称和数量相同；不能把它当作三种语义规划。

## 2. 不同计划能否得到不同reward

**能，但不是每种不同计划都有不同分数。** 同一游戏seed、相同历史计划、
相同当前query也存在以下现成样本：

| L7样本 | 本轮选择与后续 | 结果／整局环境reward |
| --- | --- | --- |
| step5，seed45891，env0 | 第二轮marine42、depot8、barracks3、reactor3、scv2、refinery1；全局2次回答 | 胜／20.5 |
| 同批env4，同一第二轮query | marine2、depot2、barracks2、reactor2、scv2、refinery1；继续规划，全局4次回答 | 胜／20.5 |
| step7，seed30403，env0–5及7 | 第二轮marine42、depot8、barracks3、reactor3、scv3、refinery1；全局2次回答 | 7败／各−49.481 |
| 同批env6，同一第二轮query | 先小计划，再继续规划，全局4次回答 | 胜／20.5 |

step5两个第二轮query哈希前缀均为`c64f73642031`；step7均为`801f5a01fdcd`。
step7获胜轨迹第三次回答含不在候选表的`techlab starport`，触发原有无效处理；
不能把它说成纯模型小计划的独立因果胜利。表内是整局环境reward，未冒充
额外无效动作惩罚后的PPO最终score。

源码口径：
- `agents/planner.py:1076–1093`：胜+20、败−50，正确格式+0.5；非胜局才加终局状态分。
- `agents/planner.py:961–1021`：终局状态分看部分工人、军队、建筑等；不是完整战斗效率评分。
- `agents/commander.py:267–300`：仍有阶段速度／attack等条件奖励；
  `reward_planner_bo`调用已注释。不能说奖励完全只有胜负。
- `verl/trainer/ppo/ray_trainer.py`另有无效动作惩罚；本次step5均合法，不受它影响。

上述两个`agents/`路径均位于
`agent_system/environments/env_package/sc2_online_restored/SC2ArenaDyStrategy/`。
在step5观察到的奖励序列分别是[0,20.5]与[0,0,0,20.5]，同样胜利时没有
额外评分区分这些计划的战损、资源转化或后续适应性。

## 3. 已核实的训练目标偏向

`gigpo/core_gigpo.py:87–126`沿真实policy transition反向递推：
`G_t = r_t + 0.95 * G_(t+1)`。此处一步是一次Planner交互，不是固定游戏秒数。

对step5同一第二轮query：

| 方案 | 整局reward | 本轮折扣回报 | 本轮joint advantage方向 |
| --- | --- | --- | --- |
| 整批下发，当轮后终局 | 20.5 | 20.5 | 正 |
| 分段规划，再调用两次后终局 | 20.5 | 18.501249 | 负 |

CPU执行冻结GiGPO原函数重算。18条有效回答需复制2条补至20，原复制索引未记录；
枚举全部153种不重复的两项补齐组合，整批回答advantage为+0.199875～+0.444166，
分段回答为−1.798876～−1.554585，符号始终不变。不是假装恢复了实际补齐选择。

仅在CPU反事实中把gamma换为1，两种折扣回报都为20.5，全部advantage归零。
这确认该批差异由调用次数折扣产生；没有在训练里改gamma。
重算用一个有效token表示每回答的广播标量，不是模型梯度或参数重放。

这类折扣递推也存在于[GiGPO公开源码](https://github.com/langfengQ/verl-agent/blob/master/gigpo/core_gigpo.py)
（2026-09-10读取，compute_step_discounted_returns；upstream master并非本项目冻结SHA）。
不能把它本身称作新增bug。当前项目episode计权与upstream并非逐字相同，但
step5全局reward相等，episode项为0，不影响以上符号结论。

少调用不等于真实游戏更快或战术更好；step7说明整批计划也可能全败。
gamma改为1也不会凭空产生探索或区分同奖励的计划，不能直接当作修复建议。

## 4. 已排除与仍缺证据

训练入口和HF分支均为do_sample=True，不是误用评测greedy；温度与top-p未变。
L6 step4/20的8rank、L7 step16的4rank小型extra_state分别有不同CUDA RNG哈希。
因此这些边界上不存在所有rank共用完全相同RNG状态的证据；没有逐调用RNG记录，
不能宣称排除了所有采样路径问题。actor checkpoint配置与启动override均确认
swa_dropout=0；原始模型目录config里0.5不代表本次训练实际配置。

低温和top-p可能放大既有集中，但缺生成时的逐token logits，无法量化其贡献，
也不能证明候选被top-p裁到只剩一个。没有本批参数梯度/Adam更新证据，
未确认优化器实现偏差，不把以前的数值审计代替当前证据。

**当前解释是：少数成功模板被强化；同样获胜时存在少调用偏向；采样与结果
随后趋同，组内相对奖励失去区分信号。前两项有具体样本／CPU计算支持，
整个反馈过程仍不是单变量因果实验。** 输出合理集中与有害探索不足需结合
独立表现判断；不能因为回答重复就判定模型训坏。

## 复现与交付边界

使用44已部署依赖，CUDA_VISIBLE_DEVICES为空、Torch/BLAS单线程、timeout35s：
`python3 - < sampling-reward-cpu-audit.py`，实际退出0、CUDA未初始化。
[CPU审计脚本](sampling-reward-cpu-audit.py)仅读固定run的小日志/配置并重建回报；
不加载模型、不执行PPO、不启动SC2。另读取20个约15KB的checkpoint extra_state
核验RNG；未读取模型或optimizer大分片。
运行日志分析脚本及CPU输出保留在JSON／相邻文件中，旧审计不覆盖。

下一项若要验证因果，应先批准单独的对照配置和预算；本次不更改prompt、
reward、折扣、采样或运行状态，不新建GPU实验。尚未找到“一个实现bug导致全部趋同”。
