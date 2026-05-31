import numpy as np, copy
from data_generation_multi_scenario import MTTDataGeneratorMultiScenario
from phd_filter import evaluate_scenario_phd
from mht_tracker import evaluate_scenario_mht

gen = MTTDataGeneratorMultiScenario(task_type=1, seed=42, n_cross_traj=3)
sc  = gen.generate_by_type('crossing')

print("=== PHD 自主起始 ===")
r1 = evaluate_scenario_phd(copy.deepcopy(sc), tau=4, use_gt_init=False)
print(f"  平均关联正确率: {np.mean(r1['frame_assoc_acc'])*100:.1f}%")

print("\n=== PHD 真值热启动 ===")
r2 = evaluate_scenario_phd(copy.deepcopy(sc), tau=4, use_gt_init=True)
print(f"  平均关联正确率: {np.mean(r2['frame_assoc_acc'])*100:.1f}%")

print("\n=== MHT 自主起始 ===")
r3 = evaluate_scenario_mht(copy.deepcopy(sc), tau=4, max_hypotheses=5, use_gt_init=False)
print(f"  平均关联正确率: {np.mean(r3['frame_assoc_acc'])*100:.1f}%")

print("\n=== MHT 真值热启动 ===")
r4 = evaluate_scenario_mht(copy.deepcopy(sc), tau=4, max_hypotheses=5, use_gt_init=True)
print(f"  平均关联正确率: {np.mean(r4['frame_assoc_acc'])*100:.1f}%")
