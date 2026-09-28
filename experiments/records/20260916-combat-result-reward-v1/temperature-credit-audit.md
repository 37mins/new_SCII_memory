# C组：非零优势在奖励什么

2026-09-16 16:03北京时间，只读核验。C的global5–7已完成，global8在采样；
没有新增GPU实验、修改冻结配置或中断C。此次是已有日志分析，不是新bug修复。

## 结论

温度0.8令原零信号的第6/7批出现计划和回报差异，但最大的正优势来自
**整局回报更差、Planner调用更多的失败轨迹**。按调用次数折扣的step分量
压过了方向相反的episode分量。非零advantage不能据此验收为有效学习改善。
这与冻结代码的公式一致，尚不是“实现偏离算法”的证据，也不能解释所有历史退化。

| 同批同输入下的第二答 | 整局回报 | 总Planner调用 | episode分量 | step分量 | 联合优势（补齐前） |
| --- | ---: | ---: | ---: | ---: | ---: |
| step6/env0：Marine42 | −53.257 | 2 | +0.111 | −1.988 | −1.878 |
| step6/env6：Marine8 | −54.463 | 6 | −0.164 | +6.852 | **+6.688** |
| step7/env0：Marine42优先 | −54.075 | 2 | +0.322 | −2.821 | −2.499 |
| step7/env4：Depot2优先，随后Marine42 | −55.551 | 7 | −0.119 | +7.848 | **+7.729** |

以上四条第二答均有效；同批第二答的最终输入逐字相同。
不能据此声称Marine8或Depot优先一定不好：这里只比较这次实际轨迹及其回报排序。
step6/env2的第二答把Marine8与Marine42合写，超过累计配额、判无效，
整局−53.820、4次调用，却也得到补齐前+2.450。不是全部新增差异都可用。

## 复核链

冻结runtime：`cf3ff4c26c9cb3f4f744b6cf4eeae909ccf92565`。
RUN：`20260916-combat-result-reward-v1-spb2-A4-l6-C-temp08-rl4-4gpu-r1`，
47 GPU0–3，共享根为
`/mnt/data0601/yangjinyue/new_SCII/experiments/20260916-combat-result-reward-v1/runs`。
来源是该run的manifest、effective config、prompt_response、episode_outcomes、
training_seeds、console和checkpoints/signal_sampling；未加载模型或Adam大文件。
本地只读快照：`/private/tmp/sc2-combat-C-audit-20260916-1602`。
完整逐答重算与各输入文件hash：同级`sc2-combat-C-audit-20260916-1602-credit.json`。
仓库保留[关键逐答数值和完整hash清单](temperature-credit-1602.json)，不依赖口头结论。

`audit_temperature_credit.py`核对9个核心文件与冻结commit逐字节一致，再复用已有
`policy_transitions`审计，只保留真实决策并归入未消费的终局reward。
每批8局整局奖励均值与console一致；第5/6/7批补齐前任务优势最大值重算为
60.257845、6.687885、7.729117，与运行日志60.257843、6.687880、7.729116
一致到1e−4以内。这是标量公式复算，不是恢复原始FP32张量。

冻结代码位置：

- `gigpo/core_gigpo.py:87`：按真实决策序列算`G_t = r_t + 0.95 * G_(t+1)`，不是按游戏秒数折扣。
- `gigpo/core_gigpo.py:189`：整局相对优势再除该轨迹的回答数；`mean_norm`不除标准差。
- `gigpo/core_gigpo.py:338`：相同anchor的折扣回报减组均值；两分量在第170行等权相加。
- `verl/trainer/ppo/ray_trainer.py:201`：无效回答另减0.1；不保证无效答的联合优势一定为负。
- `verl/trainer/main_ppo.py:152`：EpisodeRewardManager显式`normalize_by_length=False`。

step6/env6归因后的逐答奖励为
`[0, -0.5, 0, 0, +0.85, -54.813]`。
第二答未折扣后续奖励和是−54.463，按调用次数折扣后为
`-0.5 + 0.95^3 * 0.85 - 0.95^4 * 54.813 = -44.416762`。
同输入env0第二答之后立即终局，折扣回报仍−53.257。
因此前者相对step优势更高，尽管整局结果更差。这里“立即”指没有下一次Planner
决策，不代表游戏马上结束；**更多调用不等于更长游戏时间或更好生存**。
无效回答所在时间点的正环境增量也不能解释成“无效指令造成战果”。

仅在同一日志上把gamma代入1重算，step6/env6联合优势变为−1.124，
step7/env4变为−0.938；对应两条env0分别+0.356、+0.980。
这是排序反转来源的代数分解，不是新的游戏/PPO对照，更不证明改gamma能够学好。
历史[gamma-only A/B](../20260910-spb-fsdp-layer-wrap-v1/gamma-ab.md)已经完成，
两组均39/64胜、未采用gamma1；不能忽略这条负结果再宣称参数修改就是修复。

## 运行状态与边界

| global step | 胜局/8 | 实际有效回答/真实决策 | 平均整局reward | PPO K1 | reference KL | 裁剪前grad norm |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 5 | 4 | 18/20 | −12.923 | −0.001 | 0.012 | 167.527 |
| 6 | 0 | 18/23 | −53.478 | −0.001 | 0.013 | 7.322 |
| 7 | 0 | 23/27 | −54.720 | 0.000 | 0.024 | 14.954 |

三批response clip均0。第6/7批各复制补齐一答，日志有效率为19/24、24/28，
不能替代上表真实回答分母。第5批无复制；平衡后首个episode代表可能改变无效惩罚。
原复制ID和平衡顺序未保存，故不冒充逐答最终PPO张量；枚举允许的代表/复制数量
所得保守范围，step6/env6仍为+5.912～+6.936、env0为−2.676～−1.638；
step7/env4仍为+6.845～+8.072、env0为−3.479～−2.156，关键正负不依赖该未知量。
正优势也不保证对应回答每个token的概率都提高，不等于实际梯度归因。

目前只支持“探索产生了差异，但优势排序与整局目标出现不一致”，不支持学习修好。
C保持原90分钟预算跑完并归档，不追加采样温度或精度试验、不续96步。
之后应根据完整C结果决定是否值得测试训练信号的目标对齐；需要新GPU预算时另报，
不能将历史gamma1无提升结果、当前战损奖励未验收状态或缺独立评测略去。

CPU验证：`PYTHONPATH=tests python3 -B -m unittest test_temperature_credit_audit -v`
3项通过；覆盖同轨迹零优势、更多调用的更差败局发生符号反转、无效答/singleton。
审计首次误以为normalize_by_length会打印在Hydra日志，改为核对冻结构造函数后重算通过；
这是审计脚本自身校验修正，不计为项目bug修复。
