import copy, numpy as np
from data_generation_multi_scenario import MTTDataGeneratorMultiScenario
from jpda_tracker import evaluate_scenario_jpda

gen = MTTDataGeneratorMultiScenario(task_type=1, seed=42, n_cross_traj=3)
sc  = gen.generate_by_type('crossing')

print('=== JPDA 自主起始 ===')
r1 = evaluate_scenario_jpda(copy.deepcopy(sc), tau=4, use_gt_init=False)
acc1 = float(np.mean(r1['frame_assoc_acc']))
print(f'  平均关联正确率: {acc1*100:.1f}%')

print()
print('=== JPDA 真值热启动 ===')
r2 = evaluate_scenario_jpda(copy.deepcopy(sc), tau=4, use_gt_init=True)
acc2 = float(np.mean(r2['frame_assoc_acc']))
print(f'  平均关联正确率: {acc2*100:.1f}%')
