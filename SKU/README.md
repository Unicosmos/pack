
# SKU建库流程

本目录包含 SKU 图片库的构建工具和流程脚本。

## 📁 目录结构

```
SKU/
├── source/           # 原始箱体图片（待处理）
├── crops/            # 剪切后的箱体图片（按原图名组织）
├── sku_output/       # 人工审核后的初始库（按SKU分类）
│   ├── {sku_id}/     # 每个SKU一个文件夹
│   └── sku_database.json
├── sku_auged/        # 数据增强后的图片库
│   ├── {sku_id}/     # 增强后的图片
│   ├── sku_library.csv
│   └── metadata.json
├── sku_review.py     # 人工审核工具
├── sku_augmentation.py  # 数据增强脚本
├── build_library.py     # 特征提取建库脚本
├── split_train_val.py   # 训练/验证集切分
├── sku_model_trainer.py # 模型微调训练
└── feature_extractor.py # ViT-S16 DINO 特征提取器
```

## 🔄 建库流程

### 步骤1：准备箱体图片

将检测到的箱体图片放入 `source/` 目录，然后使用 `box_detector.py` 进行批量剪切：

```bash
# 从 source 剪切到 crops
cd d:\A_pack\pack\utils
python box_detector.py --input ../SKU/source --output ../SKU/crops
```

### 步骤2：人工审核分类

运行 Gradio 人工审核工具：

```bash
cd d:\A_pack\pack\SKU
python sku_review.py
```

- 打开浏览器访问 http://localhost:7860
- 将 `crops/` 中的图片分类到各个 SKU 文件夹
- 审核完成后，`sku_output/` 目录包含初始 SKU 库

### 步骤3：数据增强

对初始库进行数据增强（每图生成10张变体）：

```bash
python sku_augmentation.py --input ./sku_output --output ./sku_auged
```

**增强方案（10种）：**
| 编号 | 增强类型 | 说明 |
|-----|---------|------|
| 001 | 左侧视角 | 透视变换模拟左侧拍摄 |
| 002 | 右侧视角 | 透视变换模拟右侧拍摄 |
| 003 | 暗光环境 | 亮度-25%，饱和度-15% |
| 004 | 亮光环境 | 亮度+25%，饱和度+10% |
| 005 | 轻微模糊 | 高斯模糊 |
| 006 | 旋转90度 | 顺时针旋转 |
| 007 | 旋转180度 | 倒置放置 |
| 008 | 旋转270度 | 逆时针旋转 |
| 009 | 对比度增强 | 对比度×1.15 |
| 010 | 轻微噪声 | 高斯噪声 |

### 步骤4：特征提取建库

提取 ViT-S16 DINO 特征并构建 SKU 库：

```bash
python build_library.py --input ./sku_auged --output ../sku_library --use-aug-csv
```

输出文件：
- `sku_library/sku_features.npy` - [N, 384] 特征矩阵
- `sku_library/sku_library.csv` - 图片索引文件
- `sku_library/images/{sku_id}/` - 复制的图片文件

### 步骤5（可选）：模型微调

如果需要微调特征提取模型：

```bash
# 切分训练/验证集
python split_train_val.py --input ../sku_library/sku_library.csv --output_dir ../sku_library/ --val_ratio 0.2

# 训练模型
python sku_model_trainer.py --epochs 10 --lr 1e-5 --batch_size 4 --n_labels 4
```

微调后的模型保存在 `models/sku_trained_vits16_dino.pth`

## 📝 文件说明

| 文件 | 用途 |
|------|------|
| `sku_review.py` | Gradio 人工审核界面 |
| `sku_augmentation.py` | 数据增强（透视、光照、旋转等） |
| `build_library.py` | 特征提取和库构建 |
| `split_train_val.py` | 按 label 切分训练/验证集 |
| `sku_model_trainer.py` | OML 度量学习微调 |
| `feature_extractor.py` | ViT-S16 DINO 封装 |

## 🔧 配置说明

### 路径配置

| 路径 | 说明 |
|------|------|
| `source/` | 原始箱体图片输入 |
| `crops/` | 剪切后的箱体图片 |
| `sku_output/` | 人工审核后的初始库 |
| `sku_auged/` | 数据增强后的库 |
| `../sku_library/` | 最终特征库 |

### 依赖安装

```bash
pip install -r requirements.txt
# 如需要模型微调
pip install open-metric-learning
```

## 📊 数据格式

### sku_database.json 格式

```json
{
  "000001": {
    "name": "商品名称",
    "images": ["img1.jpg", "img2.jpg"],
    "image_count": 2
  }
}
```

### sku_library.csv 格式

```csv
image_name,sku_id,label,sku_name
001_001.jpg,000001,001,商品名称
```

## 🚀 完整工作流示例

```bash
# 1. 剪切图片
cd d:\A_pack\pack\utils
python box_detector.py -i ../SKU/source -o ../SKU/crops

# 2. 人工审核（打开浏览器）
cd ../SKU
python sku_review.py

# 3. 数据增强
python sku_augmentation.py -i ./sku_output -o ./sku_auged

# 4. 特征建库
python build_library.py -i ./sku_auged -o ../sku_library --use-aug-csv

# 5. 可选：模型微调
python split_train_val.py -i ../sku_library/sku_library.csv -o ../sku_library/
python sku_model_trainer.py --epochs 10
```

## 📈 统计信息示例

完成建库后，会输出类似以下统计：

```
SKU数量: 32
总原始图片(面数): 40
总输出图片: 400
特征矩阵: (400, 384)
```

## 📌 注意事项

1. **度量学习数据集划分**：训练/验证集按 label 整体切分，避免同一 label 横跨两个集合
2. **数据增强**：每张原图生成10种增强变体，覆盖不同视角和光照条件
3. **特征维度**：ViT-S16 DINO 输出 384 维特征向量
4. **图片格式**：支持 jpg, jpeg, png, bmp, webp 格式

---

*项目：毕设项目*
*日期：2026年4月*
