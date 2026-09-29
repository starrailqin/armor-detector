# Armor Detector

RoboMaster 装甲板检测与定位 ROS2 节点。


## 功能

- 订阅 `/sensor_img` 图像话题
- YOLOv8 检测装甲板
- 传统视觉精修灯条角点
- PnP 解算 3D 坐标
- 坐标变换（相机 → 机器人）
- 发布 `/aim_target`（自定义消息 `AimInfo`）
- 发布可视化图像到 `/armor_debug_image`
- 未检测到装甲板时，发布空消息（`type = -1`）

## 依赖

- ROS2 Humble
- OpenCV
- ONNX Runtime 1.24.4
- Ultralytics YOLOv8

## 编译
```bash
colcon build
source install/setup.bash
```
##运行
ros2 run armor_detector detector_node

```bash
ros2 run armor_detector detector_node
```


##消息定义
int16[] coordinate  # 机器人坐标系下的坐标（毫米）
int16 type          # 装甲板图案类型（哨兵输出 7）

## 致谢

- 数据集：[TAber-W](https://github.com/TAber-W) 自采的 RoboMaster 装甲板数据集
- YOLOv8：[Ultralytics](https://github.com/ultralytics/ultralytics)
- 推理框架：[ONNX Runtime](https://github.com/microsoft/onnxruntime)
- 参考项目：[SZURPVision](https://github.com/SZURPVision/26_NNDeployment_Lib_and_Detection_Models)（OpenVINO 部署方案，因硬件不兼容未采用）

## 作者
starrailqin
