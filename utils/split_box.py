import os
import cv2
import argparse

def crop_image_by_yolo_label(img_path, label_path, save_dir):
    img = cv2.imread(img_path)
    if img is None:
        print(f"❌ 读取失败：{img_path}")
        return

    h, w = img.shape[:2]
    img_name = os.path.splitext(os.path.basename(img_path))[0]

    try:
        with open(label_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except:
        print(f"❌ 标注读取失败：{label_path}")
        return

    for idx, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue

        data = list(map(float, line.split()))
        if len(data) < 5:
            continue

        xc, yc, bw, bh = data[1], data[2], data[3], data[4]
        x1 = max(0, int((xc - bw / 2) * w))
        y1 = max(0, int((yc - bh / 2) * h))
        x2 = min(w, int((xc + bw / 2) * w))
        y2 = min(h, int((yc + bh / 2) * h))

        crop_img = img[y1:y2, x1:x2]
        if crop_img.size == 0:
            continue

        save_path = os.path.join(save_dir, f"{img_name}_{idx}.jpg")
        cv2.imwrite(save_path, crop_img)

    print(f"✅ 已裁剪：{img_name}")

def process_image(img_path, label_dir, output_dir):
    img_name = os.path.splitext(os.path.basename(img_path))[0]
    label_path = os.path.join(label_dir, img_name + ".txt")

    if not os.path.exists(label_path):
        print(f"⚠️  跳过 {img_name}，未找到标注")
        return

    save_folder = os.path.join(output_dir, img_name)
    os.makedirs(save_folder, exist_ok=True)
    crop_image_by_yolo_label(img_path, label_path, save_folder)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO标注裁剪工具")
    parser.add_argument("-i", required=True, help="图片路径 / 图片文件夹")
    parser.add_argument("-l", required=True, help="标注文件夹")
    parser.add_argument("-o", required=True, help="输出目录")
    args = parser.parse_args()

    input_path = args.i
    label_dir = args.l
    output_dir = args.o

    if os.path.isfile(input_path):
        process_image(input_path, label_dir, output_dir)

    elif os.path.isdir(input_path):
        img_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
        for f in os.listdir(input_path):
            if f.lower().endswith(img_exts):
                img_path = os.path.join(input_path, f)
                process_image(img_path, label_dir, output_dir)
    else:
        print("❌ 输入路径不存在")