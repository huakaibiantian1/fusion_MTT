"""
evaluate_gui.py  —  多算法交互评估 GUI

支持算法：
  · BAIT  —— 深度学习关联滤波模型（需要 .pth 检查点）
  · MHT   —— N-scan 多假设跟踪器（无需模型文件）
  · PHD   —— GM-PHD 高斯混合概率假设密度滤波器（无需模型文件）

功能：
  · 统一算法选择面板（BAIT / MHT / PHD），切换时自动显示/隐藏对应参数面板
  · 选择 4 种场景之一（星形交叉 / 多目标 / 高机动 / 纺锤形）
  · 配置公共参数：速度范围、加速度范围、总帧数
  · 场景特定参数：交叉轨迹数（crossing），纺锤间距（spindle）
  · 一键生成场景 → 运行算法 → 输出统计图

用法：
  python evaluate_gui.py
"""

import os
import sys
import copy
import io
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch

# ── 项目模块 ─────────────────────────────────────────────────
from bait_model import BAIT
from data_generation_multi_scenario import MTTDataGeneratorMultiScenario
from evaluate_3d import (
    evaluate_scenario,
    plot_3d_trajectories,
    plot_error_curves,
    plot_ospa_curve,
    plot_association_accuracy,
    COORD_SCALE,
)
from phd_filter   import evaluate_scenario_phd
from mht_tracker  import evaluate_scenario_mht
from jpda_tracker import evaluate_scenario_jpda

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

OUTPUT_DIR = 'eval_gui'

SCENARIO_LABELS = {
    'crossing':      '星形交叉（3~15 条轨迹汇聚一点）',
    'many_targets':  '多目标（3~5 个 CV 目标）',
    'high_maneuver': '高机动（3 段 CA 加速度）',
    'spindle':       '纺锤形（接近→平行/交叉→分离）',
}

# 模型默认结构参数（与 checkpoints_multi 训练配置一致）
MODEL_KWARGS = dict(
    d_model=256, nhead=8,
    num_encoder_layers=6,
    num_associate_decoder_layers=3,
    num_filtering_decoder_layers=6,
    dim_feedforward_encoder=2048,
    dim_feedforward_associate=1024,
    dim_feedforward_filtering=2048,
    max_targets=20,
)


# ================================================================
# 主 GUI 类
# ================================================================

class EvaluateGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title('多算法交互评估（BAIT / MHT / PHD / JPDA）')
        self.root.geometry('860x800')
        self.root.resizable(True, True)

        self._model  = None
        self._device = None
        self._thread = None
        self._result_queue: list = []

        self._apply_style()
        self._build_ui()

    # ────────────────────────────────────────────────────────────
    # 样式
    # ────────────────────────────────────────────────────────────

    def _apply_style(self):
        s = ttk.Style()
        s.theme_use('clam')
        s.configure('Title.TLabel',  font=('Microsoft YaHei', 13, 'bold'))
        s.configure('Head.TLabel',   font=('Microsoft YaHei', 10, 'bold'))
        s.configure('Run.TButton',   font=('Microsoft YaHei', 11, 'bold'), padding=8)
        s.configure('TLabelframe.Label', font=('Microsoft YaHei', 9, 'bold'))

    # ────────────────────────────────────────────────────────────
    # UI 构建
    # ────────────────────────────────────────────────────────────

    def _build_ui(self):
        pad = dict(padx=10, pady=4)

        # ── 标题 ──────────────────────────────────────────────
        ttk.Label(self.root, text='多算法交互评估（BAIT / MHT / PHD / JPDA）',
                  style='Title.TLabel').pack(pady=(10, 3))
        ttk.Separator(self.root, orient='horizontal').pack(fill='x', padx=10)

        # ── 算法选择 ──────────────────────────────────────────
        algo_frame = ttk.LabelFrame(self.root, text='算法选择', padding=8)
        algo_frame.pack(fill='x', **pad)

        self._algo_var = tk.StringVar(value='BAIT')
        for algo in ('BAIT', 'MHT', 'PHD', 'JPDA'):
            ttk.Radiobutton(
                algo_frame, text=algo, variable=self._algo_var,
                value=algo, command=self._on_algo_change,
            ).pack(side='left', padx=12)

        # ── BAIT 专属面板：检查点 ────────────────────────────
        self._bait_frame = ttk.LabelFrame(self.root, text='BAIT — 模型检查点', padding=8)
        self._ckpt_var = tk.StringVar(
            value=os.path.join('checkpoints_multi', 'best_model.pth'))
        ttk.Entry(self._bait_frame, textvariable=self._ckpt_var, width=58).pack(
            side='left', padx=(0, 6))
        ttk.Button(self._bait_frame, text='浏览…', command=self._browse_ckpt).pack(side='left')

        # ── MHT 专属面板：参数 ───────────────────────────────
        self._mht_frame = ttk.LabelFrame(self.root, text='MHT — 算法参数', padding=8)
        mht_col1 = ttk.Frame(self._mht_frame)
        mht_col1.pack(side='left', padx=(0, 20))
        mht_col2 = ttk.Frame(self._mht_frame)
        mht_col2.pack(side='left')

        self._mht_sigma_r    = self._flt_row(mht_col1, '测量噪声 sigma_r (m)',  50.0,   1.0, 500.0, 0)
        self._mht_sigma_q    = self._flt_row(mht_col1, '过程噪声 sigma_q',       0.5,  0.01,  10.0, 1)
        self._mht_P_d        = self._flt_row(mht_col1, '检测概率 P_d',           0.95,  0.5,   1.0, 2)
        self._mht_P_s        = self._flt_row(mht_col1, '存活概率 P_s',           0.99,  0.5,   1.0, 3)
        self._mht_lambda_c   = self._flt_row(mht_col2, '杂波率 lambda_c',       10.0,   1.0, 100.0, 0)
        self._mht_gate       = self._flt_row(mht_col2, '门限 gate_sigma',         4.0,   1.0,  10.0, 1)
        self._mht_n_scan     = self._int_row(mht_col2, 'N-scan 深度',               2,     1,    10, 2)
        self._mht_max_hyp    = self._int_row(mht_col2, '假设数 K',                 20,     1,   100, 3)

        # 真值热启动选项
        self._mht_gt_init = tk.BooleanVar(value=True)
        gt_chk_frame = ttk.Frame(self._mht_frame)
        gt_chk_frame.pack(fill='x', padx=4, pady=(4, 0))
        ttk.Checkbutton(
            gt_chk_frame,
            text='使用真值热启动（前 tau 帧只喂真实目标量测，与 BAIT 一致）',
            variable=self._mht_gt_init,
        ).pack(anchor='w')

        # ── PHD 专属面板：参数 ───────────────────────────────
        self._phd_frame = ttk.LabelFrame(self.root, text='PHD — 算法参数', padding=8)
        phd_col1 = ttk.Frame(self._phd_frame)
        phd_col1.pack(side='left', padx=(0, 20))
        phd_col2 = ttk.Frame(self._phd_frame)
        phd_col2.pack(side='left')

        self._phd_sigma_r    = self._flt_row(phd_col1, '测量噪声 sigma_r (m)',  50.0,  1.0,  500.0, 0)
        self._phd_sigma_q    = self._flt_row(phd_col1, '过程噪声 sigma_q',       0.5, 0.01,   10.0, 1)
        self._phd_P_d        = self._flt_row(phd_col1, '检测概率 P_d',           0.95,  0.5,    1.0, 2)
        self._phd_P_s        = self._flt_row(phd_col1, '存活概率 P_s',           0.99,  0.5,    1.0, 3)
        self._phd_lambda_c   = self._flt_row(phd_col2, '杂波率 lambda_c',       10.0,  1.0,  100.0, 0)
        self._phd_birth_w    = self._flt_row(phd_col2, '新生权重 birth_weight',  0.05, 0.001,   1.0, 1)
        self._phd_prune      = self._flt_row(phd_col2, '剪枝阈值 prune_thresh', 1e-4,  1e-8,   0.1, 2)
        self._phd_merge      = self._flt_row(phd_col2, '合并距离 merge_dist',    4.0,  0.5,   20.0, 3)

        # 真值热启动选项
        self._phd_gt_init = tk.BooleanVar(value=True)
        phd_chk_frame = ttk.Frame(self._phd_frame)
        phd_chk_frame.pack(fill='x', padx=4, pady=(4, 0))
        ttk.Checkbutton(
            phd_chk_frame,
            text='使用真值热启动（前 tau 帧只喂真实目标量测，与 BAIT 一致）',
            variable=self._phd_gt_init,
        ).pack(anchor='w')

        # ── JPDA 专属面板：参数 ──────────────────────────────
        self._jpda_frame = ttk.LabelFrame(self.root, text='JPDA — 算法参数', padding=8)
        jpda_col1 = ttk.Frame(self._jpda_frame)
        jpda_col1.pack(side='left', padx=(0, 20))
        jpda_col2 = ttk.Frame(self._jpda_frame)
        jpda_col2.pack(side='left')

        self._jpda_sigma_r   = self._flt_row(jpda_col1, '测量噪声 sigma_r (m)',  50.0,  1.0,  500.0, 0)
        self._jpda_sigma_q   = self._flt_row(jpda_col1, '过程噪声 sigma_q',       0.5, 0.01,   10.0, 1)
        self._jpda_P_d       = self._flt_row(jpda_col1, '检测概率 P_d',           0.95,  0.5,    1.0, 2)
        self._jpda_lambda_c  = self._flt_row(jpda_col1, '杂波率 lambda_c',       10.0,  1.0,  100.0, 3)
        self._jpda_gate      = self._flt_row(jpda_col2, '门限 gate_gamma (χ²)',   16.0,  1.0,   50.0, 0)
        self._jpda_N_confirm = self._int_row(jpda_col2, '确认帧数 N_confirm',        2,    1,     10, 1)
        self._jpda_N_delete  = self._int_row(jpda_col2, '删除帧数 N_delete',         3,    1,     10, 2)

        # 真值热启动选项
        self._jpda_gt_init = tk.BooleanVar(value=True)
        jpda_chk_frame = ttk.Frame(self._jpda_frame)
        jpda_chk_frame.pack(fill='x', padx=4, pady=(4, 0))
        ttk.Checkbutton(
            jpda_chk_frame,
            text='使用真值热启动（前 tau 帧只喂真实目标量测，与 BAIT 一致）',
            variable=self._jpda_gt_init,
        ).pack(anchor='w')

        # ── 场景类型 + 参数（左右并排）───────────────────────
        mid_frame = ttk.Frame(self.root)
        mid_frame.pack(fill='both', expand=False, **pad)

        scene_frame = ttk.LabelFrame(mid_frame, text='场景类型', padding=10)
        scene_frame.pack(side='left', fill='y', padx=(0, 8))

        self._scene_var = tk.StringVar(value='crossing')
        for key, label in SCENARIO_LABELS.items():
            ttk.Radiobutton(
                scene_frame, text=label, variable=self._scene_var,
                value=key, command=self._on_scene_change,
            ).pack(anchor='w', pady=2)

        param_outer = ttk.Frame(mid_frame)
        param_outer.pack(side='left', fill='both', expand=True)

        common_frame = ttk.LabelFrame(param_outer, text='公共参数（所有场景）', padding=8)
        common_frame.pack(fill='x', pady=(0, 6))

        self._v_min    = self._spin_row(common_frame, '速度最小值 (m/s)',    100, 0,   1000, 0)
        self._v_max    = self._spin_row(common_frame, '速度最大值 (m/s)',    500, 0,   1000, 1)
        self._a_min    = self._spin_row(common_frame, '加速度最小值 (m/s²)',   5, 0,    100, 2)
        self._a_max    = self._spin_row(common_frame, '加速度最大值 (m/s²)',  15, 0,    100, 3)
        self._n_frames = self._spin_row(common_frame, '总帧数（轨迹长度）',   30, 5,    300, 4)

        self._spec_frame = ttk.LabelFrame(param_outer, text='场景特定参数', padding=8)
        self._spec_frame.pack(fill='x')

        self._cross_frame = ttk.Frame(self._spec_frame)
        self._n_traj = self._spin_row(self._cross_frame, '交叉轨迹条数', 4, 3, 15, 0)
        self._cross_frame.pack(fill='x')

        self._spindle_frame = ttk.Frame(self._spec_frame)
        self._sep_far_min = self._spin_row(self._spindle_frame, '最远间距最小值 (m)', 3000, 500, 20000, 0)
        self._sep_far_max = self._spin_row(self._spindle_frame, '最远间距最大值 (m)', 7000, 500, 20000, 1)

        self._spindle_cross_var = tk.StringVar(value='random')
        ttk.Label(self._spindle_frame, text='平行段是否交叉', width=22, anchor='w').grid(
            row=2, column=0, sticky='w', padx=(0, 6), pady=2)
        cross_sel = ttk.Frame(self._spindle_frame)
        cross_sel.grid(row=2, column=1, sticky='w')
        for val, text in [('random', '随机'), ('yes', '是'), ('no', '否')]:
            ttk.Radiobutton(cross_sel, text=text, variable=self._spindle_cross_var,
                            value=val).pack(side='left', padx=2)

        self._on_scene_change()
        self._on_algo_change()   # 初始化面板显示

        # ── 运行按钮 ──────────────────────────────────────────
        self._run_btn = ttk.Button(
            self.root, text='生成场景并运行评估',
            style='Run.TButton', command=self._on_run)
        self._run_btn.pack(pady=8, ipadx=20)

        # ── 日志区 ──────────────────────────────────────────
        log_frame = ttk.LabelFrame(self.root, text='运行日志', padding=6)
        log_frame.pack(fill='both', expand=True, padx=10, pady=(0, 8))

        self._log = scrolledtext.ScrolledText(
            log_frame, height=10, wrap='word',
            font=('Consolas', 9), state='disabled',
            background='#1e1e1e', foreground='#d4d4d4',
            insertbackground='white')
        self._log.pack(fill='both', expand=True)

        # ── 底部结果摘要 ──────────────────────────────────────
        self._summary_var = tk.StringVar(value='等待运行…')
        ttk.Label(self.root, textvariable=self._summary_var,
                  font=('Microsoft YaHei', 9),
                  foreground='#2c7be5').pack(pady=(0, 8))

    # ────────────────────────────────────────────────────────────
    # 控件辅助
    # ────────────────────────────────────────────────────────────

    def _spin_row(self, parent, label, default, lo, hi, row):
        var = tk.IntVar(value=default)
        ttk.Label(parent, text=label, width=24, anchor='w').grid(
            row=row, column=0, sticky='w', padx=(0, 6), pady=2)
        ttk.Spinbox(parent, from_=lo, to=hi, textvariable=var,
                    width=8).grid(row=row, column=1, sticky='w', pady=2)
        return var

    def _flt_row(self, parent, label, default, lo, hi, row):
        var = tk.DoubleVar(value=default)
        ttk.Label(parent, text=label, width=26, anchor='w').grid(
            row=row, column=0, sticky='w', padx=(0, 4), pady=2)
        ttk.Spinbox(parent, from_=lo, to=hi, textvariable=var,
                    increment=(hi - lo) / 100, format='%.4g',
                    width=9).grid(row=row, column=1, sticky='w', pady=2)
        return var

    def _int_row(self, parent, label, default, lo, hi, row):
        var = tk.IntVar(value=default)
        ttk.Label(parent, text=label, width=26, anchor='w').grid(
            row=row, column=0, sticky='w', padx=(0, 4), pady=2)
        ttk.Spinbox(parent, from_=lo, to=hi, textvariable=var,
                    width=9).grid(row=row, column=1, sticky='w', pady=2)
        return var

    # ────────────────────────────────────────────────────────────
    # 事件处理
    # ────────────────────────────────────────────────────────────

    def _on_algo_change(self):
        algo = self._algo_var.get()
        for w in (self._bait_frame, self._mht_frame, self._phd_frame, self._jpda_frame):
            w.pack_forget()
        if algo == 'BAIT':
            self._bait_frame.pack(fill='x', padx=10, pady=4)
        elif algo == 'MHT':
            self._mht_frame.pack(fill='x', padx=10, pady=4)
        elif algo == 'PHD':
            self._phd_frame.pack(fill='x', padx=10, pady=4)
        elif algo == 'JPDA':
            self._jpda_frame.pack(fill='x', padx=10, pady=4)

    def _on_scene_change(self):
        scene = self._scene_var.get()
        self._cross_frame.pack_forget()
        self._spindle_frame.pack_forget()
        if scene == 'crossing':
            self._cross_frame.pack(fill='x')
        elif scene == 'spindle':
            self._spindle_frame.pack(fill='x')

    def _browse_ckpt(self):
        path = filedialog.askopenfilename(
            title='选择模型检查点',
            filetypes=[('PyTorch checkpoint', '*.pth'), ('All files', '*.*')],
        )
        if path:
            self._ckpt_var.set(path)

    def _on_run(self):
        if self._thread and self._thread.is_alive():
            messagebox.showwarning('提示', '评估正在运行中，请等待完成。')
            return

        algo    = self._algo_var.get()
        v_min   = self._v_min.get()
        v_max   = self._v_max.get()
        a_min   = self._a_min.get()
        a_max   = self._a_max.get()
        n_frames = self._n_frames.get()
        scene   = self._scene_var.get()

        if v_min >= v_max:
            messagebox.showerror('参数错误', '速度最小值必须小于最大值。')
            return
        if a_min >= a_max:
            messagebox.showerror('参数错误', '加速度最小值必须小于最大值。')
            return

        # 场景特定参数
        extra = {}
        if scene == 'crossing':
            extra['n_cross_traj'] = self._n_traj.get()
        elif scene == 'spindle':
            sfmin = self._sep_far_min.get()
            sfmax = self._sep_far_max.get()
            if sfmin >= sfmax:
                messagebox.showerror('参数错误', '最远间距最小值必须小于最大值。')
                return
            extra['sep_far_range'] = (sfmin, sfmax)
            cv = self._spindle_cross_var.get()
            extra['spindle_crossing'] = (None if cv == 'random' else cv == 'yes')

        # BAIT 需要检查点文件
        if algo == 'BAIT':
            ckpt = self._ckpt_var.get().strip()
            if not os.path.isfile(ckpt):
                messagebox.showerror('文件不存在', f'找不到检查点文件：\n{ckpt}')
                return
        else:
            ckpt = None

        self._run_btn.config(state='disabled')
        self._result_queue.clear()
        self._log_clear()
        self._summary_var.set('运行中…')

        # 收集算法参数
        if algo == 'MHT':
            algo_params = dict(
                sigma_r      = self._mht_sigma_r.get(),
                sigma_q      = self._mht_sigma_q.get(),
                P_d          = self._mht_P_d.get(),
                P_s          = self._mht_P_s.get(),
                lambda_c     = self._mht_lambda_c.get(),
                gate_sigma   = self._mht_gate.get(),
                n_scan       = self._mht_n_scan.get(),
                max_hypotheses = self._mht_max_hyp.get(),
                use_gt_init  = self._mht_gt_init.get(),
            )
        elif algo == 'PHD':
            algo_params = dict(
                sigma_r      = self._phd_sigma_r.get(),
                sigma_q      = self._phd_sigma_q.get(),
                P_d          = self._phd_P_d.get(),
                P_s          = self._phd_P_s.get(),
                lambda_c     = self._phd_lambda_c.get(),
                birth_weight = self._phd_birth_w.get(),
                prune_thresh = self._phd_prune.get(),
                merge_dist   = self._phd_merge.get(),
                use_gt_init  = self._phd_gt_init.get(),
            )
        elif algo == 'JPDA':
            algo_params = dict(
                sigma_r      = self._jpda_sigma_r.get(),
                sigma_q      = self._jpda_sigma_q.get(),
                P_d          = self._jpda_P_d.get(),
                lambda_c     = self._jpda_lambda_c.get(),
                gate_gamma   = self._jpda_gate.get(),
                N_confirm    = self._jpda_N_confirm.get(),
                N_delete     = self._jpda_N_delete.get(),
                use_gt_init  = self._jpda_gt_init.get(),
            )
        else:
            algo_params = {}

        self._thread = threading.Thread(
            target=self._worker,
            args=(algo, ckpt, scene, v_min, v_max, a_min, a_max, n_frames, extra, algo_params),
            daemon=True,
        )
        self._thread.start()
        self.root.after(200, self._poll)

    # ────────────────────────────────────────────────────────────
    # 后台线程：生成场景 + 评估
    # ────────────────────────────────────────────────────────────

    def _worker(self, algo, ckpt_path, scene_type,
                v_min, v_max, a_min, a_max, n_frames, extra, algo_params):
        buf        = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = _Tee(old_stdout, buf, self._log_append)

        plot_paths = []
        summary    = {}
        error_msg  = None

        try:
            print(f'算法: {algo}  场景: {scene_type}  帧数: {n_frames}')

            # ── 生成场景 ──────────────────────────────────────
            gen = MTTDataGeneratorMultiScenario(
                task_type=1,
                velocity_range=(v_min, v_max),
                high_maneuver_accel_range=(a_min, a_max),
                crossing_probability=0.7,
                T=float(n_frames),
                seed=None,
                **extra,
            )
            scenario = gen.generate_by_type(scene_type)
            print(f'场景生成完毕，目标数: {len(scenario[0])}，帧数: {len(scenario[1])}')

            # ── 运行算法 ──────────────────────────────────────
            if algo == 'BAIT':
                result = self._run_bait(ckpt_path, scenario)
            elif algo == 'MHT':
                result = evaluate_scenario_mht(copy.deepcopy(scenario), **algo_params)
            elif algo == 'PHD':
                result = evaluate_scenario_phd(copy.deepcopy(scenario), **algo_params)
            elif algo == 'JPDA':
                result = evaluate_scenario_jpda(copy.deepcopy(scenario), **algo_params)
            else:
                raise ValueError(f'未知算法: {algo}')

            # ── 绘图 ──────────────────────────────────────────
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            plot_3d_trajectories(result, 0, OUTPUT_DIR)
            plot_error_curves(result, 0, OUTPUT_DIR)
            avg_ospa = plot_ospa_curve(result, 0, OUTPUT_DIR)
            plot_association_accuracy(result, 0, OUTPUT_DIR)

            plot_paths = [
                os.path.join(OUTPUT_DIR, 'scenario_1_3d_trajectory.png'),
                os.path.join(OUTPUT_DIR, 'scenario_1_error_curves.png'),
                os.path.join(OUTPUT_DIR, 'scenario_1_ospa.png'),
                os.path.join(OUTPUT_DIR, 'scenario_1_association_acc.png'),
            ]

            # ── 统计 ──────────────────────────────────────────
            avg_acc  = float(np.mean(result['frame_assoc_acc']))
            all_dist = []
            for fi in range(len(result['frame_pred_xyz'])):
                pred = result['frame_pred_xyz'][fi]
                true = result['frame_true_xyz'][fi]
                n    = min(len(pred), len(true))
                if n > 0:
                    diffs = (pred[:n] - true[:n]) * COORD_SCALE
                    for d in np.linalg.norm(diffs, axis=1):
                        if not np.isnan(d):
                            all_dist.append(float(d))
            avg_dist = float(np.mean(all_dist)) if all_dist else 0.0

            summary = {
                'algo':     algo,
                'avg_acc':  avg_acc,
                'avg_ospa': avg_ospa,
                'avg_dist': avg_dist,
                'n_traj':   len(scenario[0]),
            }
            print(f'\n[{algo}] 关联正确率: {avg_acc*100:.1f}%  '
                  f'OSPA: {avg_ospa:.2f}m  '
                  f'平均3D误差: {avg_dist:.2f}m')

        except Exception as e:
            import traceback
            error_msg = traceback.format_exc()
            print(f'\n[错误] {e}\n{error_msg}')

        finally:
            sys.stdout = old_stdout

        self._result_queue.append({
            'plot_paths': plot_paths,
            'summary':    summary,
            'error':      error_msg,
        })

    def _run_bait(self, ckpt_path: str, scenario):
        """加载 BAIT 模型并运行评估。"""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'设备: {device}')
        print(f'加载检查点: {ckpt_path}')
        ckpt  = torch.load(ckpt_path, map_location=device, weights_only=False)
        model = BAIT(**MODEL_KWARGS).to(device)
        model.load_state_dict(ckpt['model_state_dict'])
        model.eval()
        print(f'模型加载成功（训练步骤 {ckpt.get("step", "?")}）')

        eval_max_measurements = max(
            30, max((len(m) for m in scenario[1]), default=0))
        return evaluate_scenario(
            model, copy.deepcopy(scenario),
            tau=4, max_targets=20,
            max_measurements=eval_max_measurements,
            device=device, scenario_idx=0,
        )

    # ────────────────────────────────────────────────────────────
    # 轮询：等待线程完成
    # ────────────────────────────────────────────────────────────

    def _poll(self):
        if self._result_queue:
            res = self._result_queue.pop(0)
            self._run_btn.config(state='normal')
            if res['error']:
                self._summary_var.set('运行失败，请查看日志。')
                messagebox.showerror('运行错误', res['error'][:500])
            else:
                s = res['summary']
                self._summary_var.set(
                    f"[{s['algo']}]  关联正确率: {s['avg_acc']*100:.1f}%    "
                    f"OSPA: {s['avg_ospa']:.2f} m    "
                    f"平均3D误差: {s['avg_dist']:.2f} m    "
                    f"目标数: {s['n_traj']}"
                )
                if res['plot_paths']:
                    self._show_plots(res['plot_paths'])
        else:
            self.root.after(300, self._poll)

    # ────────────────────────────────────────────────────────────
    # 图片展示窗口
    # ────────────────────────────────────────────────────────────

    def _show_plots(self, paths: list):
        try:
            from PIL import Image, ImageTk
        except ImportError:
            messagebox.showinfo(
                '提示',
                f'图片已保存到 {OUTPUT_DIR}/，请手动查看。\n'
                '（安装 Pillow 可在此窗口直接显示：pip install Pillow）',
            )
            return

        win = tk.Toplevel(self.root)
        win.title('评估结果图')
        win.geometry('1100x820')

        titles = ['3D 轨迹', '误差曲线', 'OSPA 曲线', '数据关联正确率']
        self._photo_refs = []

        canvas_frame = ttk.Frame(win)
        canvas_frame.pack(fill='both', expand=True, padx=8, pady=8)

        for idx, (path, title) in enumerate(zip(paths, titles)):
            row, col = divmod(idx, 2)
            cell = ttk.LabelFrame(canvas_frame, text=title, padding=4)
            cell.grid(row=row, column=col, padx=6, pady=6, sticky='nsew')
            canvas_frame.rowconfigure(row, weight=1)
            canvas_frame.columnconfigure(col, weight=1)

            if os.path.isfile(path):
                img = Image.open(path)
                img.thumbnail((520, 370), Image.LANCZOS)
                photo = ImageTk.PhotoImage(img)
                self._photo_refs.append(photo)
                ttk.Label(cell, image=photo).pack()
            else:
                ttk.Label(cell, text='图片未找到').pack()

        ttk.Button(win, text='保存目录：' + os.path.abspath(OUTPUT_DIR),
                   command=lambda: os.startfile(os.path.abspath(OUTPUT_DIR))
                   ).pack(pady=6)

    # ────────────────────────────────────────────────────────────
    # 日志工具
    # ────────────────────────────────────────────────────────────

    def _log_clear(self):
        self._log.config(state='normal')
        self._log.delete('1.0', 'end')
        self._log.config(state='disabled')

    def _log_append(self, text: str):
        def _do():
            self._log.config(state='normal')
            self._log.insert('end', text)
            self._log.see('end')
            self._log.config(state='disabled')
        self.root.after(0, _do)


# ================================================================
# 辅助：stdout 同时写入终端 + GUI 日志
# ================================================================

class _Tee:
    def __init__(self, terminal, buf, gui_cb):
        self._term   = terminal
        self._buf    = buf
        self._gui_cb = gui_cb

    def write(self, msg):
        self._term.write(msg)
        self._buf.write(msg)
        if msg:
            self._gui_cb(msg)

    def flush(self):
        self._term.flush()


# ================================================================
# 入口
# ================================================================

if __name__ == '__main__':
    root = tk.Tk()
    app  = EvaluateGUI(root)
    root.mainloop()
