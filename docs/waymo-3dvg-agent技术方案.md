# Waymo 3DVG Agent 技术方案

Waymo 3DVG Agent 为 Waymo Perception 的原生三维目标生成并核验英文指代表达。安装和启动见 [README](../README.md)。

## 数据来源与身份

每条样本使用完整单帧点云、英文描述及实际来源相机的一张同帧原图和标定，监督为同帧原生三维框。用 `context` 标识驾驶片段，用 `frame_timestamp_micros` 标识片段内的帧，用 `track_id` 标识片段内的物体轨迹。轨迹用于选帧、去重和调用额度管理。

| Waymo 组件 | 用途 |
| --- | --- |
| `lidar_box`、`projected_lidar_box` | 原生三维框、类别、轨迹身份及官方相机投影 |
| `camera_box`、`camera_to_lidar_box_association` | 独立二维标注及与雷达目标的官方关联 |
| `camera_image`、`camera_calibration` | 原图、相机标定和采集时序 |
| `lidar`、`lidar_camera_projection` | 五部激光雷达的双回波及点到图像的投影索引 |
| `lidar_pose`、`lidar_calibration`、`vehicle_pose` | TOP 雷达逐像素位姿补偿、雷达标定及自车位姿 |
| `lidar_camera_synced_box`、`stats` | 同步标签和场景统计 |

官方投影提供原生目标的图像位置。具体可见性由实际来源原图核对，独立二维框的关联使用官方身份信息。两类框可能重叠，重叠本身不足以确认同一物体。官方 `training` 和 `validation` 分别存放在 `waymo_trainset/` 和 `waymo_valset/`。候选、任务、转换产物和导出保留可核对的官方划分。 [Waymo Perception 数据说明](https://waymo.com/intl/en-ca/open/data/perception/)

本项目使用 Waymo LLC 提供的 Waymo Open Dataset。原始数据及派生数据的使用和分发遵循 Waymo Dataset License Agreement for Non-Commercial Use。 [Waymo Open Dataset 使用条款](https://waymo.com/open/terms/)

## 转换环境

主应用使用 Python 3.11 或以上。官方点云转换使用独立的 Linux x86_64 环境，Windows 上通过 WSL 执行。在该 Linux 环境中进入项目根目录，从锁文件安装 Python 3.10 的转换依赖。

```bash
uv sync --locked --project preprocessing
```

服务按需调用 `preprocessing/.venv` 中的转换器，复用同帧已登记的场景。SDK 保留五部雷达的两次回波及 TOP 雷达位姿补偿。未转换的帧仍可进入候选目录。

## 标注流程

1. **准备证据。** 按轨迹选择代表帧与备用帧，关联原生框、实际来源相机及完整点云。目标须有框内点云支持。准备和运行中的换帧凭据共同绑定最终来源帧，模型调用前、结果保存、审核和导出均核对身份。
2. **观察。** 核实原图中的目标及相关竞争对象。局部遮挡、部分出画或仅可见背面时，只描述可确认的部分。未观察、零点或身份未决的对象仍参与消歧。
3. **生成。** 用已确认的显著外观、自车方位或有依据的地标关系写一句自然英文。新任务的空间关系依据原生三维几何，来源相机用于限定图像证据。车辆地标自身的前后方向依据其原生朝向。定指地标须唯一，存在式地标关系须使完整描述唯一命中目标。外观本身足够唯一时即可单独使用。
4. **核验。** 检查文本声明、外观与身份，再在两个独立上下文中核验定位。候选编号各轮随机，输入不含预期目标。两轮应支持原句并唯一命中同一原生目标，未知竞争者仍参与判定。证据不足时可在原额度内补看、换视角、同轨迹换帧或改写。

程序保留失败原因、模型请求与返回、调用额度及冻结执行配置。低额度下由程序提出的草稿也须完成同样的核验。原图用于观察外观和身份，画面排序、像素边缘及单帧运动推测均不作为新描述证据。历史记录沿用其冻结规则。

## 执行与恢复

- 后端使用单服务进程和持久化任务队列。目标和模型请求默认各并发 4 路，上限均为 4。原生证据准备最多 2 路，官方转换串行。
- SQLite 事务保存任务、证据引用及调用预留。已确认的暂时故障可在原额度内有界重试，远端结果未知的请求不重放。
- 继续操作在原任务中完成。配置改变时记录新的执行修订，已完成结果及累计调用保留。异常退出后，只有符合租约和配置检查的遗留运行可自动恢复。
- 删除按选定任务及引用关系执行，保留存续任务使用的场景、证据和账本。文件变更可回滚。操作步骤见[工作台使用与验收](标注Agent开发.md)。

## Waymo-3DRef 导出契约

日常 ZIP 的目录结构见 [README](../README.md#导出结构)。导出只接受已完成且执行器已退出、官方划分相同的任务。在所选范围内，同一 `(context, track_id)` 保留最新有效标注。样本须来自真实模型运行，通过自动核验或当前版本的显式人工审批，并通过身份、资源和官方划分检查。

| 字段或资源 | 含义 |
| --- | --- |
| `samples.jsonl` | 每行一条样本，包括稳定的 `sample_id`、片段、帧、轨迹、描述、来源相机、场景引用和目标框 |
| `scene.json` | 场景身份、官方划分、点云、标定、框清单及引用相机。共享场景保存被样本引用的相机并集 |
| `target_box`、`objects.json` | 样本的目标框与场景级原生框清单，属于监督数据 |
| `calibration.json` | 原生标定、帧与相机位姿及采集时序。相机模型同时提供内参 `K`、畸变参数和 `ego2cam` |
| `loader.py`、`merge.py` | 独立加载与合并工具，只需 NumPy 和 Pillow |

三维框顺序为 `[cx, cy, cz, length, width, height, heading_rad]`，长度单位为米，角度单位为弧度。帧自车坐标系的 +x 向前、+y 向左、+z 向上，朝向从 +x 绕 +z 逆时针计量。`ego2cam` 将该帧的自车坐标转换为 OpenCV 相机坐标，使用齐次列向量，输出 +x 向右、+y 向下、+z 向前。投影按图像位姿时间计算，畸变单独应用，滚动快门时序随标定保留。

| `points.npz` 数组 | 类型 | 形状 | 内容 |
| --- | --- | --- | --- |
| `points` | `float32` | `(N, 6)` | `x, y, z, intensity, elongation, range` |
| `laser_id`、`return_id` | `uint8` | `(N,)` | 雷达编号与回波编号 |
| `range_image_pixel` | `int32` | `(N, 2)` | 距离图像的行和列 |
| `camera_projection` | `int32` | `(N, 6)` | 原生相机投影 |
| `nlz` | `int8` | `(N,)` | 无标签区域标记 |

`N` 为完整单帧点数，导出保留原有坐标和数值。公开包使用统一精简结构，省略 `schema_version`、内部质量状态、执行方法和证据来源等元数据。合并按 `sample_id` 去重，核对同轨迹版本和场景资源，并检查训练集与验证集之间的片段重叠。格式以[导出实现](../app/datasets/dataset_export.py)和[加载器](../app/datasets/export_loader.py)为准。
