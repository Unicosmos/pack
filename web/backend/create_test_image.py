# 生成测试图片

from pathlib import Path
from PIL import Image, ImageDraw

# 创建一个简单的测试图片
img = Image.new('RGB', (640, 480), color='white')
draw = ImageDraw.Draw(img)
draw.rectangle([100, 100, 300, 300], outline='red', width=3)
draw.rectangle([350, 150, 550, 350], outline='blue', width=3)

# 保存
test_img_path = Path(__file__).parent / "test_image.jpg"
img.save(test_img_path)

print(f"测试图片已保存到: {test_img_path}")
