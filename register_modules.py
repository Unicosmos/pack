"""
注册自定义模块到 ultralytics
在加载模型前调用此函数
"""

import ultralytics.nn.tasks as tasks


def register_custom_modules():
    """注册自定义模块到 ultralytics"""
    from ultralytics.nn.modules import CBAM
    
    # 将 CBAM 添加到 tasks 模块的 __dict__ 中
    tasks.__dict__['CBAM'] = CBAM
    
    # 同时添加到 parse_model 函数的 globals
    if hasattr(tasks, 'parse_model'):
        tasks.parse_model.__globals__['CBAM'] = CBAM
    
    print("✓ CBAM 模块已注册到 ultralytics")


if __name__ == '__main__':
    register_custom_modules()
    
    from ultralytics import YOLO
    
    print("\n测试 yolov8n_cbam.yaml...")
    model = YOLO('yolov8n_cbam.yaml')
    params = sum(p.numel() for p in model.model.parameters())
    print(f'  参数量: {params:,}')
    print("成功!")
