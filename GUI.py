import copy
import io
import os
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from bait_data_io import load_scenario_file
from bait_model import BAIT
from data_generation_multi_scenario import MTTDataGeneratorMultiScenario
from evaluate_3d import (
    COORD_SCALE,
    evaluate_scenario_kf_managed,
    plot_3d_trajectories,
    plot_association_accuracy,
    plot_error_curves,
    plot_ospa_curve,
)


plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

OUTPUT_DIR = "eval_gui"

MODEL_KWARGS = dict(
    d_model=256,
    nhead=8,
    num_encoder_layers=6,
    num_associate_decoder_layers=3,
    num_filtering_decoder_layers=6,
    dim_feedforward_encoder=2048,
    dim_feedforward_associate=1024,
    dim_feedforward_filtering=2048,
    max_targets=20,
)

SCENARIO_LABELS = {
    "crossing": "星形交叉",
    "many_targets": "多目标",
    "high_maneuver": "高机动",
    "spindle": "纺锤形",
}


def resource_path(relative_path):
    base = getattr(sys, "_MEIPASS", os.path.abspath(os.path.dirname(__file__)))
    return os.path.join(base, relative_path)


class GUI:
    def __init__(self, root):
        self.root = root
        self.root.title("GUI - BAIT")
        self.root.geometry("860x780")
        self.root.resizable(True, True)
        self._thread = None
        self._result_queue = []
        self._apply_style()
        self._build_ui()

    def _apply_style(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Title.TLabel", font=("Microsoft YaHei", 13, "bold"))
        style.configure("Run.TButton", font=("Microsoft YaHei", 11, "bold"), padding=8)
        style.configure("TLabelframe.Label", font=("Microsoft YaHei", 9, "bold"))

    def _build_ui(self):
        pad = dict(padx=10, pady=4)
        ttk.Label(self.root, text="BAIT 多目标跟踪评估", style="Title.TLabel").pack(pady=(10, 3))
        ttk.Separator(self.root, orient="horizontal").pack(fill="x", padx=10)

        algo_frame = ttk.LabelFrame(self.root, text="算法", padding=8)
        algo_frame.pack(fill="x", **pad)
        ttk.Label(algo_frame, text="BAIT（KF 航迹管理 + BAIT 数据关联）").pack(anchor="w")

        ckpt_frame = ttk.LabelFrame(self.root, text="模型检查点", padding=8)
        ckpt_frame.pack(fill="x", **pad)
        self._ckpt_var = tk.StringVar(value=resource_path(os.path.join("checkpoints_multi", "best_model.pth")))
        ttk.Entry(ckpt_frame, textvariable=self._ckpt_var, width=72).pack(side="left", padx=(0, 6))
        ttk.Button(ckpt_frame, text="浏览...", command=self._browse_ckpt).pack(side="left")

        source_frame = ttk.LabelFrame(self.root, text="数据来源", padding=8)
        source_frame.pack(fill="x", **pad)
        self._source_var = tk.StringVar(value="file")
        ttk.Radiobutton(
            source_frame, text="从文件加载真值和测量", variable=self._source_var,
            value="file", command=self._on_source_change,
        ).pack(side="left", padx=8)
        ttk.Radiobutton(
            source_frame, text="随机生成测试场景", variable=self._source_var,
            value="generated", command=self._on_source_change,
        ).pack(side="left", padx=8)

        self._file_frame = ttk.LabelFrame(self.root, text="输入数据文件", padding=8)
        self._data_file_var = tk.StringVar(value="")
        ttk.Entry(self._file_frame, textvariable=self._data_file_var, width=72).pack(side="left", padx=(0, 6))
        ttk.Button(self._file_frame, text="浏览...", command=self._browse_data_file).pack(side="left")
        self._file_frame.pack(fill="x", **pad)

        kf_frame = ttk.LabelFrame(self.root, text="BAIT 航迹管理参数", padding=8)
        kf_frame.pack(fill="x", **pad)
        kf_col1 = ttk.Frame(kf_frame)
        kf_col1.pack(side="left", padx=(0, 20))
        kf_col2 = ttk.Frame(kf_frame)
        kf_col2.pack(side="left")
        self._kf_gate = self._flt_row(kf_col1, "关联门限 gate (m)", 600.0, 50.0, 2000.0, 0)
        self._kf_sigma_r = self._flt_row(kf_col1, "测量噪声 sigma_r (m)", 80.0, 1.0, 500.0, 1)
        self._kf_sigma_q = self._flt_row(kf_col1, "过程噪声 sigma_q (m)", 50.0, 1.0, 500.0, 2)
        self._kf_confirm = self._int_row(kf_col2, "确认帧数", 4, 2, 10, 0)
        self._kf_delete = self._int_row(kf_col2, "删除漏检帧数", 3, 1, 10, 1)

        self._gen_frame = ttk.LabelFrame(self.root, text="随机场景参数", padding=8)
        gen_left = ttk.Frame(self._gen_frame)
        gen_left.pack(side="left", padx=(0, 20))
        gen_right = ttk.Frame(self._gen_frame)
        gen_right.pack(side="left")
        self._scene_var = tk.StringVar(value="crossing")
        ttk.Label(gen_left, text="场景类型", width=18, anchor="w").grid(row=0, column=0, sticky="w")
        ttk.Combobox(
            gen_left, textvariable=self._scene_var, values=list(SCENARIO_LABELS.keys()),
            width=16, state="readonly",
        ).grid(row=0, column=1, sticky="ew", pady=2)
        self._v_min = self._int_row(gen_left, "速度最小值 (m/s)", 100, 0, 1000, 1)
        self._v_max = self._int_row(gen_left, "速度最大值 (m/s)", 500, 0, 1000, 2)
        self._a_min = self._int_row(gen_right, "加速度最小值", 5, 0, 100, 0)
        self._a_max = self._int_row(gen_right, "加速度最大值", 15, 0, 100, 1)
        self._n_frames = self._int_row(gen_right, "总帧数", 60, 5, 300, 2)
        self._seed = self._int_row(gen_right, "随机种子", 260613, -1, 999999999, 3)

        run_frame = ttk.Frame(self.root)
        run_frame.pack(fill="x", **pad)
        self._run_btn = ttk.Button(run_frame, text="运行 BAIT", style="Run.TButton", command=self._on_run)
        self._run_btn.pack(side="left")
        self._summary_var = tk.StringVar(value="等待运行")
        ttk.Label(run_frame, textvariable=self._summary_var).pack(side="left", padx=12)

        log_frame = ttk.LabelFrame(self.root, text="运行日志", padding=6)
        log_frame.pack(fill="both", expand=True, padx=10, pady=(4, 10))
        self._log = scrolledtext.ScrolledText(log_frame, height=17, state="disabled", font=("Consolas", 9))
        self._log.pack(fill="both", expand=True)

        self._on_source_change()

    def _flt_row(self, parent, label, default, lo, hi, row):
        var = tk.DoubleVar(value=default)
        ttk.Label(parent, text=label, width=20, anchor="w").grid(row=row, column=0, sticky="w", pady=2)
        ttk.Spinbox(parent, from_=lo, to=hi, increment=1.0, textvariable=var, width=12).grid(row=row, column=1, pady=2)
        return var

    def _int_row(self, parent, label, default, lo, hi, row):
        var = tk.IntVar(value=default)
        ttk.Label(parent, text=label, width=20, anchor="w").grid(row=row, column=0, sticky="w", pady=2)
        ttk.Spinbox(parent, from_=lo, to=hi, increment=1, textvariable=var, width=12).grid(row=row, column=1, pady=2)
        return var

    def _browse_ckpt(self):
        path = filedialog.askopenfilename(
            title="选择 BAIT 检查点",
            filetypes=[("PyTorch checkpoint", "*.pth"), ("All files", "*.*")],
        )
        if path:
            self._ckpt_var.set(path)

    def _browse_data_file(self):
        path = filedialog.askopenfilename(
            title="选择输入数据文件",
            filetypes=[
                ("Scenario files", "*.pkl *.pickle *.npz *.json"),
                ("Pickle", "*.pkl *.pickle"),
                ("NumPy", "*.npz"),
                ("JSON", "*.json"),
                ("All files", "*.*"),
            ],
        )
        if path:
            self._data_file_var.set(path)

    def _on_source_change(self):
        if self._source_var.get() == "generated":
            self._file_frame.pack_forget()
            self._gen_frame.pack(fill="x", padx=10, pady=4)
        else:
            self._gen_frame.pack_forget()
            self._file_frame.pack(fill="x", padx=10, pady=4)

    def _on_run(self):
        if self._thread and self._thread.is_alive():
            messagebox.showwarning("提示", "评估正在运行中，请等待完成。")
            return

        ckpt = self._ckpt_var.get().strip()
        if not os.path.isfile(ckpt):
            messagebox.showerror("文件不存在", f"找不到检查点文件:\n{ckpt}")
            return

        data_file = self._data_file_var.get().strip()
        if self._source_var.get() == "file" and not os.path.isfile(data_file):
            messagebox.showerror("文件不存在", f"找不到输入数据文件:\n{data_file}")
            return

        self._run_btn.config(state="disabled")
        self._result_queue.clear()
        self._log_clear()
        self._summary_var.set("运行中...")

        kf_params = dict(
            gate_m=self._kf_gate.get(),
            sigma_r_m=self._kf_sigma_r.get(),
            sigma_q_m=self._kf_sigma_q.get(),
            confirm_hits=self._kf_confirm.get(),
            max_missed=self._kf_delete.get(),
        )

        self._thread = threading.Thread(
            target=self._worker,
            args=(ckpt, data_file, self._source_var.get(), kf_params),
            daemon=True,
        )
        self._thread.start()
        self.root.after(200, self._poll)

    def _worker(self, ckpt_path, data_file, source, kf_params):
        buf = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = _Tee(old_stdout, buf, self._log_append)
        plot_paths = []
        summary = {}
        error_msg = None

        try:
            print("算法: BAIT")
            if source == "file":
                print(f"输入文件: {data_file}")
                scenario = load_scenario_file(data_file)
                print(f"文件加载完毕，目标数: {len(scenario[0])}，帧数: {len(scenario[1])}")
            else:
                scenario = self._generate_scenario()
                print(f"场景生成完毕，目标数: {len(scenario[0])}，帧数: {len(scenario[1])}")

            self._print_lifecycles(scenario)
            result = self._run_bait(ckpt_path, scenario, kf_params)

            os.makedirs(OUTPUT_DIR, exist_ok=True)
            plot_3d_trajectories(result, 0, OUTPUT_DIR)
            plot_error_curves(result, 0, OUTPUT_DIR)
            avg_ospa = plot_ospa_curve(result, 0, OUTPUT_DIR)
            plot_association_accuracy(result, 0, OUTPUT_DIR)
            plot_paths = [
                os.path.join(OUTPUT_DIR, "scenario_1_3d_trajectory.png"),
                os.path.join(OUTPUT_DIR, "scenario_1_error_curves.png"),
                os.path.join(OUTPUT_DIR, "scenario_1_ospa.png"),
                os.path.join(OUTPUT_DIR, "scenario_1_association_acc.png"),
            ]

            avg_acc, avg_dist, miss_rate = self._summarize_result(result)
            summary = {
                "algo": "BAIT",
                "avg_acc": avg_acc,
                "avg_ospa": float(avg_ospa),
                "avg_dist": avg_dist,
                "miss_rate": miss_rate,
                "diagnostics": result.get("diagnostics", {}),
                "n_traj": len(scenario[0]),
            }
            print(
                f"\n[BAIT] 关联正确率: {avg_acc*100:.1f}%  "
                f"OSPA: {avg_ospa:.2f}m  命中3D误差: {avg_dist:.2f}m  "
                f"漏失率: {miss_rate*100:.1f}%"
            )
            self._print_diagnostics(summary["diagnostics"])
        except Exception as exc:
            import traceback
            error_msg = traceback.format_exc()
            print(f"\n[错误] {exc}\n{error_msg}")
        finally:
            sys.stdout = old_stdout

        self._result_queue.append({
            "plot_paths": plot_paths,
            "summary": summary,
            "error": error_msg,
        })

    def _generate_scenario(self):
        seed = self._seed.get()
        seed_value = None if int(seed) < 0 else int(seed)
        gen = MTTDataGeneratorMultiScenario(
            task_type=1,
            velocity_range=(self._v_min.get(), self._v_max.get()),
            high_maneuver_accel_range=(self._a_min.get(), self._a_max.get()),
            crossing_probability=0.7,
            T=float(self._n_frames.get()),
            seed=seed_value,
        )
        return gen.generate_by_type(self._scene_var.get())

    def _run_bait(self, ckpt_path, scenario, kf_params):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"设备: {device}")
        print(f"加载检查点: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model = BAIT(**MODEL_KWARGS).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        print(f"模型加载成功（训练步骤 {ckpt.get('step', '?')}）")
        eval_max_measurements = max(30, max((len(m) for m in scenario[1]), default=0))
        return evaluate_scenario_kf_managed(
            model,
            copy.deepcopy(scenario),
            tau=4,
            max_targets=20,
            max_measurements=eval_max_measurements,
            device=device,
            scenario_idx=0,
            kf_params=kf_params,
        )

    def _print_lifecycles(self, scenario):
        print("轨迹生命周期:")
        for tr in scenario[0]:
            b = int(tr.get("birth_frame", 0))
            d = int(tr.get("death_frame", len(tr.get("states", [])) - 1))
            print(f"  traj{int(tr['label']):2d}: birth={b:2d}, death={d:2d}, len={d - b + 1:2d}")

    def _summarize_result(self, result):
        avg_acc = float(np.mean(result["frame_assoc_acc"])) if result["frame_assoc_acc"] else 0.0
        all_dist = []
        missed_targets = 0
        total_targets = 0
        for pred, true in zip(result["frame_pred_xyz"], result["frame_true_xyz"]):
            total_targets += len(true)
            if len(true) > len(pred):
                missed_targets += len(true) - len(pred)
            n = min(len(pred), len(true))
            if n <= 0:
                continue
            valid = ~np.any(np.isnan(pred[:n]), axis=1)
            if np.any(valid):
                errs = np.linalg.norm((pred[:n][valid] - true[:n][valid]) * COORD_SCALE, axis=1)
                all_dist.extend(errs.tolist())
            missed_targets += int(np.sum(~valid))
        avg_dist = float(np.mean(all_dist)) if all_dist else 0.0
        miss_rate = float(missed_targets / total_targets) if total_targets > 0 else 0.0
        return avg_acc, avg_dist, miss_rate

    def _print_diagnostics(self, diagnostics):
        if not diagnostics:
            return
        manager = diagnostics.get("manager", "KF")
        recall = diagnostics.get("manager_confirm_recall", diagnostics.get("phd_confirm_recall", 0.0))
        false_confirmed = diagnostics.get("manager_false_confirmed", diagnostics.get("phd_false_confirmed", 0))
        manager_err = diagnostics.get("manager_mean_pos_error_m", diagnostics.get("phd_mean_pos_error_m", float("nan")))
        post_cov = diagnostics.get("manager_post_confirm_coverage", 0.0)
        post_miss = diagnostics.get("manager_post_confirm_miss_rate", 0.0)
        confirm_delay = diagnostics.get("manager_mean_confirm_delay", float("nan"))
        early_breaks = diagnostics.get("manager_early_break_count", 0)
        early_break_rate = diagnostics.get("manager_early_break_rate", 0.0)
        delayed_deaths = diagnostics.get("manager_delayed_death_count", 0)
        delayed_death_rate = diagnostics.get("manager_delayed_death_rate", 0.0)
        delayed_death_frames = diagnostics.get("manager_delayed_death_frames", 0)
        mean_delayed_death = diagnostics.get("manager_mean_delayed_death", float("nan"))
        print(
            "[分层诊断] "
            f"{manager}确认覆盖率={recall*100:.1f}% | "
            f"{manager}确认后覆盖={post_cov*100:.1f}% | "
            f"{manager}确认后漏失={post_miss*100:.1f}% | "
            f"{manager}提前断轨={early_breaks}({early_break_rate*100:.1f}%) | "
            f"{manager}延迟结束={delayed_deaths}({delayed_death_rate*100:.1f}%) | "
            f"{manager}延迟结束帧={delayed_death_frames} | "
            f"{manager}平均延迟结束={mean_delayed_death:.1f}帧 | "
            f"{manager}平均确认延迟={confirm_delay:.1f}帧 | "
            f"{manager}虚警确认={false_confirmed} | "
            f"{manager}确认位置误差={manager_err:.1f}m | "
            f"BAIT条件关联={diagnostics.get('bait_assoc_on_confirmed', 0.0)*100:.1f}% | "
            f"BAIT预测误差={diagnostics.get('bait_mean_pred_error_m', float('nan')):.1f}m"
        )

    def _poll(self):
        if self._result_queue:
            res = self._result_queue.pop(0)
            self._run_btn.config(state="normal")
            if res["error"]:
                self._summary_var.set("运行失败，请查看日志。")
                messagebox.showerror("运行错误", res["error"][:500])
                return
            summary = res["summary"]
            self._summary_var.set(
                f"[{summary['algo']}]  关联正确率: {summary['avg_acc']*100:.1f}%    "
                f"OSPA: {summary['avg_ospa']:.2f} m    "
                f"命中3D误差: {summary['avg_dist']:.2f} m    "
                f"漏失率: {summary.get('miss_rate', 0.0)*100:.1f}%    "
                f"目标数: {summary['n_traj']}"
            )
            if res["plot_paths"]:
                self._show_plots(res["plot_paths"])
        else:
            self.root.after(300, self._poll)

    def _show_plots(self, paths):
        try:
            from PIL import Image, ImageTk
        except ImportError:
            messagebox.showinfo("完成", "评估完成。PIL 未安装，图片已保存到 eval_gui。")
            return
        win = tk.Toplevel(self.root)
        win.title("BAIT 评估结果")
        nb = ttk.Notebook(win)
        nb.pack(fill="both", expand=True)
        for path in paths:
            if not os.path.exists(path):
                continue
            frame = ttk.Frame(nb)
            nb.add(frame, text=os.path.basename(path).replace("scenario_1_", ""))
            img = Image.open(path)
            img.thumbnail((1000, 760))
            photo = ImageTk.PhotoImage(img)
            label = ttk.Label(frame, image=photo)
            label.image = photo
            label.pack(fill="both", expand=True)

    def _log_clear(self):
        self._log.config(state="normal")
        self._log.delete("1.0", "end")
        self._log.config(state="disabled")

    def _log_append(self, text):
        def _do():
            self._log.config(state="normal")
            self._log.insert("end", text)
            self._log.see("end")
            self._log.config(state="disabled")
        self.root.after(0, _do)


class _Tee:
    def __init__(self, terminal, buf, gui_cb):
        self._terminal = terminal
        self._buf = buf
        self._gui_cb = gui_cb

    def write(self, msg):
        self._terminal.write(msg)
        self._buf.write(msg)
        if msg:
            self._gui_cb(msg)

    def flush(self):
        self._terminal.flush()


if __name__ == "__main__":
    root = tk.Tk()
    app = GUI(root)
    root.mainloop()
