python train.py --config config_multi_scenario.json --use-multi --device cuda

python train.py --config config_multi_scenario.json --use-multi --device cuda --resume checkpoints_multi/best_model.pth


python evaluate_3d.py --checkpoint checkpoints_multi\best_model.pth --scenarios-pkl checkpoints_multi\train_scenarios_crossing.pkl --num-scenarios 10 --output-dir eval_crossing

python evaluate_3d.py --checkpoint checkpoints_multi\best_model.pth --scenarios-pkl checkpoints_multi\train_scenarios_many_targets.pkl --num-scenarios 10 --output-dir eval_many_targets

python evaluate_3d.py --checkpoint checkpoints_multi\best_model.pth --scenarios-pkl checkpoints_multi\train_scenarios_high_maneuver.pkl --num-scenarios 10 --output-dir eval_high_maneuver

python evaluate_3d.py --checkpoint checkpoints_multi\best_model.pth --scenarios-pkl checkpoints_multi\train_scenarios_spindle.pkl --num-scenarios 10 --output-dir eval_spindle

# 方式一：val_split 模式（自动合并所有类型，随机顺序）
python evaluate_3d.py --checkpoint checkpoints_multi/best_model.pth --scene-source val_split --num-scenarios 10 --output-dir eval_val

# 方式二：按场景类型分别评估（推荐）
python evaluate_3d.py --checkpoint checkpoints_multi/best_model.pth --scenarios-pkl checkpoints_multi/val_scenarios_crossing.pkl --output-dir eval_val_crossing --num-scenarios 10
python evaluate_3d.py --checkpoint checkpoints_multi/best_model.pth --scenarios-pkl checkpoints_multi/val_scenarios_spindle.pkl --output-dir eval_val_spindle --num-scenarios 10



被控制的随机操作	举例
目标数量
poisson.rvs(lambda_0) → 泊松抽样几个目标
目标初始位置
np.random.uniform(r_min, r_max) → 随机距离/方位
初始速度大小和方向
np.random.uniform(*velocity_range)
交叉点位置和时刻
星形交叉的 cross_xyz、t_cross
各轨迹的进入/离开方向
azimuth_approach、phi_leave
CA 加速度大小和方向
high_maneuver 的 a_mag、rand_dir
纺锤形间距参数
sep_near、sep_far、speed
过程噪声
np.random.multivariate_normal(..., Q)
杂波数量和位置
poisson.rvs(lambda_c) + 随机杂波坐标
测量噪声
_measure_3d 中的球坐标加噪
场景打乱顺序
np.random.permutation(len(raw_scenarios))