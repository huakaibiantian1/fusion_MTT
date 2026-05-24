"""
evaluate_gui.py  —  BAIT 模型交互评估 GUI

功能：
  · 选择 4 种场景之一（星形交叉 / 多目标 / 高机动 / 纺锤形）
  · 配置速度、加速度范围（m/s / m/s²）
  · 场景特定参数：
      - 星形交叉：指定轨迹条数（3~5）
      - 纺锤形：指定最远横向间距范围（m）
  · 一键生成场景 → 运行 BAIT 模型 → 输出统计图

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
matplotlib.use('Agg')   # 无头渲染，避免与 tkinter 事件循环冲突
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

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

OUTPUT_DIR = 'eval_gui'

SCENARIO_LABELS = {
    'crossing':      '星形交叉（3~5 条轨迹汇聚一点）',
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
        self.root.title('BAIT 模型交互评估')
        self.root.geometry('720x680')
        self.root.resizable(True, True)

        self._model  = None
        self._device = None
        self._thread = None
        self._result_queue: list = []   # 线程完成后放入结果

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
        pad = dict(padx=10, pady=5)

        # ── 标题 ──────────────────────────────────────────────
        ttk.Label(self.root, text='BAIT 模型交互评估',
                  style='Title.TLabel').pack(pady=(12, 4))
        ttk.Separator(self.root, orient='horizontal').pack(fill='x', padx=10)

        # ── 检查点 ──────────────────────────────────────────────
        ckpt_frame = ttk.LabelFrame(self.root, text='模型检查点', padding=8)
        ckpt_frame.pack(fill='x', **pad)

        self._ckpt_var = tk.StringVar(
            value=os.path.join('checkpoints_multi', 'best_model.pth'))
        ttk.Entry(ckpt_frame, textvariable=self._ckpt_var, width=55).pack(
            side='left', padx=(0, 6))
        ttk.Button(ckpt_frame, text='浏览…', command=self._browse_ckpt).pack(side='left')

        # ── 场景类型 + 参数（左右并排）───────────────────────────
        mid_frame = ttk.Frame(self.root)
        mid_frame.pack(fill='both', expand=False, **pad)

        # 左：场景类型选择
        scene_frame = ttk.LabelFrame(mid_frame, text='场景类型', padding=10)
        scene_frame.pack(side='left', fill='y', padx=(0, 8))

        self._scene_var = tk.StringVar(value='crossing')
        for key, label in SCENARIO_LABELS.items():
            ttk.Radiobutton(
                scene_frame, text=label, variable=self._scene_var,
                value=key, command=self._on_scene_change,
            ).pack(anchor='w', pady=2)

        # 右：参数配置
        param_outer = ttk.Frame(mid_frame)
        param_outer.pack(side='left', fill='both', expand=True)

        # 公共参数
        common_frame = ttk.LabelFrame(param_outer, text='公共参数（所有场景）', padding=8)
        common_frame.pack(fill='x', pady=(0, 6))

        self._v_min     = self._spin_row(common_frame, '速度最小值 (m/s)',    100, 0,   1000, 0)
        self._v_max     = self._spin_row(common_frame, '速度最大值 (m/s)',    500, 0,   1000, 1)
        self._a_min     = self._spin_row(common_frame, '加速度最小值 (m/s²)',   5, 0,    100, 2)
        self._a_max     = self._spin_row(common_frame, '加速度最大值 (m/s²)',  15, 0,    100, 3)
        self._n_frames  = self._spin_row(common_frame, '总帧数（轨迹长度）',   30, 5,    300, 4)

        # 场景特定参数（动态显示/隐藏）
        self._spec_frame = ttk.LabelFrame(param_outer, text='场景特定参数', padding=8)
        self._spec_frame.pack(fill='x')

        # crossing 专用
        self._cross_frame = ttk.Frame(self._spec_frame)
        self._n_traj = self._spin_row(self._cross_frame, '交叉轨迹条数', 4, 3, 15, 0)
        self._cross_frame.pack(fill='x')

        # spindle 专用
        self._spindle_frame = ttk.Frame(self._spec_frame)
        self._sep_far_min = self._spin_row(self._spindle_frame, '最远间距最小值 (m)', 3000, 500, 20000, 0)
        self._sep_far_max = self._spin_row(self._spindle_frame, '最远间距最大值 (m)', 7000, 500, 20000, 1)

        self._spindle_cross_var = tk.StringVar(value='random')
        cross_lbl = ttk.Label(self._spindle_frame, text='平行段是否交叉', width=22, anchor='w')
        cross_lbl.grid(row=2, column=0, sticky='w', padx=(0, 6), pady=2)
        cross_sel = ttk.Frame(self._spindle_frame)
        cross_sel.grid(row=2, column=1, sticky='w')
        for val, text in [('random', '随机'), ('yes', '是'), ('no', '否')]:
            ttk.Radiobutton(cross_sel, text=text, variable=self._spindle_cross_var,
                            value=val).pack(side='left', padx=2)

        self._on_scene_change()  # 初始化显示状态

        # ── 运行按钮 ────────────────────────────────────────────
        self._run_btn = ttk.Button(
            self.root, text='生成场景并运行评估',
            style='Run.TButton', command=self._on_run)
        self._run_btn.pack(pady=10, ipadx=20)

        # ── 日志区 ──────────────────────────────────────────────
        log_frame = ttk.LabelFrame(self.root, text='运行日志', padding=6)
        log_frame.pack(fill='both', expand=True, padx=10, pady=(0, 8))

        self._log = scrolledtext.ScrolledText(
            log_frame, height=12, wrap='word',
            font=('Consolas', 9), state='disabled',
            background='#1e1e1e', foreground='#d4d4d4',
            insertbackground='white')
        self._log.pack(fill='both', expand=True)

        # ── 底部结果摘要 ────────────────────────────────────────
        self._summary_var = tk.StringVar(value='等待运行…')
        ttk.Label(self.root, textvariable=self._summary_var,
                  font=('Microsoft YaHei', 9),
                  foreground='#2c7be5').pack(pady=(0, 8))

    def _spin_row(self, parent, label, default, lo, hi, row):
        """创建一行 label + Spinbox，返回 IntVar。"""
        var = tk.IntVar(value=default)
        ttk.Label(parent, text=label, width=22, anchor='w').grid(
            row=row, column=0, sticky='w', padx=(0, 6), pady=2)
        ttk.Spinbox(parent, from_=lo, to=hi, textvariable=var,
                    width=8).grid(row=row, column=1, sticky='w', pady=2)
        return var

    # ────────────────────────────────────────────────────────────
    # 事件
    # ────────────────────────────────────────────────────────────

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

        # 参数校验
        v_min = self._v_min.get()
        v_max = self._v_max.get()
        a_min = self._a_min.get()
        a_max = self._a_max.get()

        if v_min >= v_max:
            messagebox.showerror('参数错误', '速度最小值必须小于最大值。')
            return
        if a_min >= a_max:
            messagebox.showerror('参数错误', '加速度最小值必须小于最大值。')
            return

        ckpt = self._ckpt_var.get().strip()
        if not os.path.isfile(ckpt):
            messagebox.showerror('文件不存在', f'找不到检查点文件：\n{ckpt}')
            return

        scene = self._scene_var.get()

        n_frames = self._n_frames.get()

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

        self._run_btn.config(state='disabled')
        self._result_queue.clear()
        self._log_clear()
        self._summary_var.set('运行中…')

        self._thread = threading.Thread(
            target=self._worker,
            args=(ckpt, scene, v_min, v_max, a_min, a_max, n_frames, extra),
            daemon=True,
        )
        self._thread.start()
        self.root.after(200, self._poll)

    # ────────────────────────────────────────────────────────────
    # 后台线程：生成场景 + 评估
    # ────────────────────────────────────────────────────────────

    def _worker(self, ckpt_path, scene_type, v_min, v_max, a_min, a_max, n_frames, extra):
        buf = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = _Tee(old_stdout, buf, self._log_append)

        plot_paths = []
        summary    = {}
        error_msg  = None

        try:
            # ── 设备 ──────────────────────────────────────────
            device = torch.device(
                'cuda' if torch.cuda.is_available() else 'cpu')
            print(f'设备: {device}')

            # ── 加载模型 ──────────────────────────────────────
            print(f'加载检查点: {ckpt_path}')
            ckpt  = torch.load(ckpt_path, map_location=device, weights_only=False)
            model = BAIT(**MODEL_KWARGS).to(device)
            model.load_state_dict(ckpt['model_state_dict'])
            model.eval()
            print(f'模型加载成功（训练步骤 {ckpt.get("step", "?")}）')

            # ── 生成场景 ──────────────────────────────────────
            print(f'\n生成场景类型: {scene_type}，帧数: {n_frames}')
            gen = MTTDataGeneratorMultiScenario(
                task_type=1,
                velocity_range=(v_min, v_max),
                high_maneuver_accel_range=(a_min, a_max),
                crossing_probability=0.7,
                T=float(n_frames),   # dt=1.0，帧数 == 持续秒数
                seed=None,
                **extra,
            )
            scenario = gen.generate_by_type(scene_type)
            print(f'场景生成完毕，目标数: {len(scenario[0])}，帧数: {len(scenario[1])}')
            eval_max_measurements = max(30, max((len(m) for m in scenario[1]), default=0))
            print(f'Max measurements in this scenario: {eval_max_measurements}')

            # ── 模型评估 ──────────────────────────────────────
            result = evaluate_scenario(
                model, copy.deepcopy(scenario),
                tau=4, max_targets=20, max_measurements=eval_max_measurements,
                device=device, scenario_idx=0,
            )

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
            avg_acc = float(np.mean(result['frame_assoc_acc']))
            all_dist = []
            for fi in range(len(result['frame_pred_xyz'])):
                pred = result['frame_pred_xyz'][fi]
                true = result['frame_true_xyz'][fi]
                n = min(len(pred), len(true))
                if n > 0:
                    all_dist.extend(
                        np.linalg.norm((pred[:n] - true[:n]) * COORD_SCALE,
                                       axis=1).tolist())
            avg_dist = float(np.mean(all_dist)) if all_dist else 0.0

            summary = {
                'avg_acc':  avg_acc,
                'avg_ospa': avg_ospa,
                'avg_dist': avg_dist,
                'n_traj':   len(scenario[0]),
            }
            print(f'\n关联正确率: {avg_acc*100:.1f}%  '
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

    # ────────────────────────────────────────────────────────────
    # 轮询：等待线程完成
    # ────────────────────────────────────────────────────────────

    def _poll(self):
        if self._result_queue:
            res = self._result_queue.pop(0)
            self._run_btn.config(state='normal')
            if res['error']:
                self._summary_var.set('运行失败，请查看日志。')
                messagebox.showerror('运行错误', res['error'][:400])
            else:
                s = res['summary']
                self._summary_var.set(
                    f"关联正确率: {s['avg_acc']*100:.1f}%    "
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
        """在独立 Toplevel 窗口中分 2×2 展示 4 张统计图。"""
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
        self._photo_refs = []   # 防止 GC 回收

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
                lbl = ttk.Label(cell, image=photo)
                lbl.pack()
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
        """线程安全地向日志区追加文字。"""
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
