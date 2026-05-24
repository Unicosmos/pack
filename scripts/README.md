# Pack 项目脚本工具集

## 目录结构

```
scripts/
├── data/              # 数据处理工具
│   ├── coco2yolo_seg.py    # COCO格式转YOLO分割格式
│   └── filter_images.py    # 图片质量筛选
├── detect/            # 检测处理工具
│   └── batch_detect_crop.py  # 批量检测并裁剪
└── visualize/         # 可视化工具
    └── visualize_annotations.py  # 标注可视化
```

## 使用方式

### 方式一：直接运行脚本

```bash
# 数据处理
python scripts/data/coco2yolo_seg.py --input annotations.json --output labels/
python scripts/data/filter_images.py --input images/ --output filtered/

# 检测处理
python scripts/detect/batch_detect_crop.py --input images/ --output crops/

# 可视化
python scripts/visualize/visualize_annotations.py --image image.jpg --label label.txt
```

### 方式二：作为模块导入

```python
from scripts.data import coco2yolo_seg
from scripts.detect import batch_detect_crop
from scripts.visualize import visualize_annotations

# 使用模块功能
coco2yolo_seg.convert(input_path, output_path)
```

## 脚本功能说明

### 数据处理 (data/)

| 脚本 | 功能 | 输入 | 输出 |
|------|------|------|------|
| `coco2yolo_seg.py` | COCO分割标注转YOLO格式 | COCO JSON | YOLO TXT |
| `filter_images.py` | 过滤低质量/模糊图片 | 图片目录 | 筛选后图片 |

### 检测处理 (detect/)

| 脚本 | 功能 | 输入 | 输出 |
|------|------|------|------|
| `batch_detect_crop.py` | 批量检测并裁剪目标 | 图片目录 | 裁剪图片 |

### 可视化 (visualize/)

| 脚本 | 功能 | 输入 | 输出 |
|------|------|------|------|
| `visualize_annotations.py` | 可视化标注框/分割区域 | 图片+标注 | 可视化图片 |

## 依赖关系

```
scripts/
├── 依赖 core/utils/
│   ├── image_utils.py (图像处理)
│   └── logger.py (日志)
└── 依赖 core/detector/ (检测功能)
```

## 开发规范

1. **命名规范**: 使用小写下划线命名，如 `batch_detect_crop.py`
2. **参数解析**: 使用 `argparse` 模块处理命令行参数
3. **日志输出**: 使用 `core.utils.logger` 统一日志格式
4. **错误处理**: 捕获异常并提供清晰的错误信息
