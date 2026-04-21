"""
透视校正脚本

功能：将斜拍的商品图片透视变换为正面视角
适用场景：SKU库建设中，原图拍摄角度不正的情况

使用方法：
    # 交互模式（自动输出到原文件夹）
    python perspective_correction.py -i image.jpg --interactive
    
    # 指定输出路径
    python perspective_correction.py -i image.jpg -o corrected.jpg --interactive
    
    # 指定角点坐标
    python perspective_correction.py -i image.jpg \
        --tl 100,50 --tr 1500,80 --br 1480,1580 --bl 120,1600

参数说明：
    -i, --input: 输入图片路径
    -o, --output: 输出图片路径（可选，默认自动命名：原文件名_corrected.jpg）
    --interactive: 交互模式，显示图片手动调整角点
    --tl: 左上角坐标 (top-left)
    --tr: 右上角坐标 (top-right)
    --br: 右下角坐标 (bottom-right)
    --bl: 左下角坐标 (bottom-left)
    --width: 输出图片宽度（可选）
    --height: 输出图片高度（可选）

作者：毕设项目
日期：2026年4月
"""

import os
import sys
import argparse
import json
from pathlib import Path

import numpy as np
import cv2


def perspective_correction(image: np.ndarray,
                           src_points: np.ndarray,
                           dst_width: int,
                           dst_height: int) -> np.ndarray:
    """
    透视变换
    
    Args:
        image: 输入图片
        src_points: 源四角点，形状(4,2)，顺序为[左上,右上,右下,左下]
        dst_width: 输出宽度
        dst_height: 输出高度
    
    Returns:
        校正后的图片
    """
    # 目标四角点（矩形）
    dst_points = np.float32([
        [0, 0],                    # 左上
        [dst_width - 1, 0],        # 右上
        [dst_width - 1, dst_height - 1],  # 右下
        [0, dst_height - 1]        # 左下
    ])
    
    # 计算透视变换矩阵
    matrix = cv2.getPerspectiveTransform(src_points.astype(np.float32), dst_points)
    
    # 应用透视变换
    result = cv2.warpPerspective(image, matrix, (dst_width, dst_height))
    
    return result


def get_auto_output_path(input_path: str) -> str:
    """
    自动生成输出路径
    
    规则：在原文件名后加 '_corrected' 后缀
    例如：image.jpg -> image_corrected.jpg
    """
    input_path = Path(input_path)
    output_name = f"{input_path.stem}_corrected{input_path.suffix}"
    return str(input_path.parent / output_name)


def interactive_correction(image_path: str, output_path: str):
    """
    交互式透视校正
    
    操作说明：
    1. 鼠标点击图片上的四个角点（顺序：左上、右上、右下、左下）
    2. 按 'r' 重置
    3. 按 's' 保存并退出
    4. 按 'q' 退出不保存
    """
    # 读取图片
    image = cv2.imread(image_path)
    if image is None:
        print(f"错误：无法读取图片 {image_path}")
        return
    
    # 缩放图片以适应屏幕
    h, w = image.shape[:2]
    max_display_size = 1200
    scale = min(max_display_size / w, max_display_size / h, 1.0)
    display_w, display_h = int(w * scale), int(h * scale)
    display_image = cv2.resize(image, (display_w, display_h))
    
    # 存储角点
    points = []
    
    def mouse_callback(event, x, y, flags, param):
        nonlocal display_image, points
        
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(points) < 4:
                # 转换回原图坐标
                orig_x = int(x / scale)
                orig_y = int(y / scale)
                points.append((orig_x, orig_y))
                
                # 在显示图上标记
                cv2.circle(display_image, (x, y), 5, (0, 255, 0), -1)
                cv2.putText(display_image, str(len(points)), (x + 10, y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # 连线
                if len(points) > 1:
                    prev_x = int(points[-2][0] * scale)
                    prev_y = int(points[-2][1] * scale)
                    cv2.line(display_image, (prev_x, prev_y), (x, y), (0, 255, 0), 2)
                
                # 最后一个点连接到第一个点
                if len(points) == 4:
                    first_x = int(points[0][0] * scale)
                    first_y = int(points[0][1] * scale)
                    cv2.line(display_image, (x, y), (first_x, first_y), (0, 255, 0), 2)
                
                cv2.imshow('Perspective Correction', display_image)
    
    # 创建窗口
    cv2.namedWindow('Perspective Correction')
    cv2.setMouseCallback('Perspective Correction', mouse_callback)
    
    print("=" * 50)
    print("交互式透视校正")
    print("=" * 50)
    print(f"原图尺寸: {w} x {h}")
    print(f"显示尺寸: {display_w} x {display_h}")
    print("=" * 50)
    print("操作说明：")
    print("  1. 按顺序点击四个角点：左上 → 右上 → 右下 → 左下")
    print("  2. 按 'r' 重置")
    print("  3. 按 's' 保存并退出")
    print("  4. 按 'q' 退出不保存")
    print("=" * 50)
    
    cv2.imshow('Perspective Correction', display_image)
    
    while True:
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('r'):
            # 重置
            points = []
            display_image = cv2.resize(image, (display_w, display_h))
            cv2.imshow('Perspective Correction', display_image)
            print("已重置，请重新选择角点")
        
        elif key == ord('s') and len(points) == 4:
            # 保存
            break
        
        elif key == ord('q'):
            cv2.destroyAllWindows()
            print("已取消")
            return
    
    cv2.destroyAllWindows()
    
    # 计算输出尺寸
    top_width = np.sqrt((points[1][0] - points[0][0])**2 + (points[1][1] - points[0][1])**2)
    bottom_width = np.sqrt((points[2][0] - points[3][0])**2 + (points[2][1] - points[3][1])**2)
    left_height = np.sqrt((points[3][0] - points[0][0])**2 + (points[3][1] - points[0][1])**2)
    right_height = np.sqrt((points[2][0] - points[1][0])**2 + (points[2][1] - points[1][1])**2)
    
    dst_width = int((top_width + bottom_width) / 2)
    dst_height = int((left_height + right_height) / 2)
    
    print(f"\n输出尺寸: {dst_width} x {dst_height}")
    
    # 透视变换
    src_points = np.array(points, dtype=np.float32)
    result = perspective_correction(image, src_points, dst_width, dst_height)
    
    # 保存结果
    cv2.imwrite(output_path, result, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f"已保存到: {output_path}")
    
    # 保存角点坐标
    points_file = Path(output_path).with_suffix('.json')
    with open(points_file, 'w') as f:
        json.dump({
            'source': image_path,
            'points': points,
            'output_size': [dst_width, dst_height]
        }, f, indent=2)
    print(f"角点坐标已保存到: {points_file}")


def command_line_correction(image_path: str, 
                            output_path: str,
                            tl: tuple,
                            tr: tuple,
                            br: tuple,
                            bl: tuple,
                            dst_width: int = None,
                            dst_height: int = None):
    """
    命令行模式透视校正
    """
    # 读取图片
    image = cv2.imread(image_path)
    if image is None:
        print(f"错误：无法读取图片 {image_path}")
        return
    
    h, w = image.shape[:2]
    
    # 源角点
    src_points = np.array([tl, tr, br, bl], dtype=np.float32)
    
    # 计算输出尺寸
    if dst_width is None or dst_height is None:
        top_width = np.sqrt((tr[0] - tl[0])**2 + (tr[1] - tl[1])**2)
        bottom_width = np.sqrt((br[0] - bl[0])**2 + (br[1] - bl[1])**2)
        left_height = np.sqrt((bl[0] - tl[0])**2 + (bl[1] - tl[1])**2)
        right_height = np.sqrt((br[0] - tr[0])**2 + (br[1] - tr[1])**2)
        
        dst_width = dst_width or int((top_width + bottom_width) / 2)
        dst_height = dst_height or int((left_height + right_height) / 2)
    
    print(f"原图尺寸: {w} x {h}")
    print(f"输出尺寸: {dst_width} x {dst_height}")
    
    # 透视变换
    result = perspective_correction(image, src_points, dst_width, dst_height)
    
    # 保存结果
    cv2.imwrite(output_path, result, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f"已保存到: {output_path}")


def parse_point(point_str: str) -> tuple:
    """解析坐标字符串，格式：x,y"""
    x, y = map(int, point_str.split(','))
    return (x, y)


def main():
    parser = argparse.ArgumentParser(
        description='透视校正脚本',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
    # 交互模式（自动输出到原文件夹）
    python perspective_correction.py -i image.jpg --interactive
    
    # 指定输出路径
    python perspective_correction.py -i image.jpg -o corrected.jpg --interactive
    
    # 指定角点
    python perspective_correction.py -i image.jpg \\
        --tl 100,50 --tr 1500,80 --br 1480,1580 --bl 120,1600
        """
    )
    parser.add_argument('-i', '--input', type=str, required=True, help='输入图片路径')
    parser.add_argument('-o', '--output', type=str, default=None, 
                        help='输出图片路径（默认自动命名：原文件名_corrected.jpg）')
    parser.add_argument('--interactive', action='store_true', help='交互模式')
    parser.add_argument('--tl', type=str, help='左上角坐标 (x,y)')
    parser.add_argument('--tr', type=str, help='右上角坐标 (x,y)')
    parser.add_argument('--br', type=str, help='右下角坐标 (x,y)')
    parser.add_argument('--bl', type=str, help='左下角坐标 (x,y)')
    parser.add_argument('--width', type=int, help='输出宽度')
    parser.add_argument('--height', type=int, help='输出高度')
    
    args = parser.parse_args()
    
    # 自动生成输出路径
    output_path = args.output or get_auto_output_path(args.input)
    
    if args.interactive:
        interactive_correction(args.input, output_path)
    elif args.tl and args.tr and args.br and args.bl:
        command_line_correction(
            args.input, output_path,
            parse_point(args.tl),
            parse_point(args.tr),
            parse_point(args.br),
            parse_point(args.bl),
            args.width,
            args.height
        )
    else:
        # 默认使用交互模式
        interactive_correction(args.input, output_path)


if __name__ == '__main__':
    main()
