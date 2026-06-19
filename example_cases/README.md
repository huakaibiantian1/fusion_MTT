# BAIT 输入示例

这些文件可以直接在 `GUI.py` 里选择“从文件加载真值和测量”后打开。

## 示例文件

- `case_crossing.json`
  两条轨迹在中心交叉，用来测试交叉时的数据关联。

- `case_birth_death.json`
  三条轨迹，其中一条中途出生，一条提前结束，用来测试生命周期管理。

- `case_clutter.json`
  两条轨迹加若干杂波点，用来测试新生抑制和虚警确认。

- `case_crossing_seed260614_30f.json`
  使用现有 crossing 生成器、seed=260614、30 帧随机生成。包含 5 条随机生命周期轨迹。

- `case_spindle_seed260615_30f.json`
  使用现有 spindle 生成器、seed=260615、30 帧随机生成。包含 3 条随机生命周期轨迹。

## GUI 运行

```powershell
python C:\Users\yy\Desktop\fusion_MTT\GUI.py
```

然后选择：

```text
数据来源: 从文件加载真值和测量
输入数据文件: example_cases\case_crossing_seed260614_30f.json
```

## 命令行运行

```powershell
python C:\Users\yy\Desktop\fusion_MTT\run_bait_file.py ^
  --input C:\Users\yy\Desktop\fusion_MTT\example_cases\case_spindle_seed260615_30f.json ^
  --checkpoint C:\Users\yy\Desktop\fusion_MTT\checkpoints_multi\best_model.pth
```
