# BAIT GUI 数据接口

新 GUI 入口：

```powershell
python GUI.py
```

命令行入口：

```powershell
python run_bait_file.py --input your_scenario.pkl --checkpoint checkpoints_multi\best_model.pth
```

## 支持格式

支持 `.pkl` / `.pickle` / `.npz` / `.json`。

内部统一读取为：

```python
scenario = (trajectories, measurements, gt_associations)
```

其中坐标单位使用米，和原训练/评估代码一致。

## trajectories

推荐结构：

```python
trajectories = [
    {
        "label": 1,
        "states": states,          # numpy/list, shape [T, 6] 或 [T, 3]
        "birth_frame": 0,
        "death_frame": 59,
    },
]
```

`states` 至少需要前三列 `x,y,z`。如果只有 `[T,3]`，速度列会自动补零。

也支持直接给数组：

```python
truth.shape == [N, T, D]
```

或：

```python
truth.shape == [T, N, D]
```

数组中未出生/已死亡帧可以用 `NaN` 填充，读取时会自动推断 `birth_frame/death_frame`。

## measurements

推荐结构：

```python
measurements = [
    np.ndarray(shape=[M0, 3]),
    np.ndarray(shape=[M1, 3]),
]
```

也支持 dense 数组：

```python
measurements.shape == [T, M, 3]
```

无效填充值可以用 `NaN`。

## gt_associations

可选。每帧一个数组，长度等于该帧测量数：

```python
gt_associations[t][i] = label
```

`0` 表示杂波或未关联。

如果文件里没有 `gt_associations`，程序会用真值轨迹和测量按最近距离自动生成，默认门限是 `300m`。

## JSON 示例

```json
{
  "trajectories": [
    {
      "label": 1,
      "birth_frame": 0,
      "death_frame": 2,
      "states": [
        [0, 0, 0],
        [100, 0, 0],
        [200, 0, 0]
      ]
    }
  ],
  "measurements": [
    [[5, 0, 0]],
    [[102, 2, 0]],
    [[198, -1, 0]]
  ]
}
```

