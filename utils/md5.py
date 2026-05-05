#!/usr/bin/env python3
"""
MD5文件校验工具
用于判断两个文件是否完全相同

Usage:
    python md5.py <file1> <file2>
"""

import sys
import hashlib


def get_md5(file_path: str) -> str:
    """
    计算文件的MD5哈希值
    
    Args:
        file_path: 文件路径
        
    Returns:
        MD5哈希字符串
    """
    md5_hash = hashlib.md5()
    
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            md5_hash.update(chunk)
    
    return md5_hash.hexdigest()


def main():
    if len(sys.argv) != 3:
        print("Usage: python md5.py <file1> <file2>")
        print("Example: python md5.py output_vis/vis_00.jpg output_vis1/vis_00.jpg")
        sys.exit(1)
    
    file1 = sys.argv[1]
    file2 = sys.argv[2]
    
    try:
        md5_1 = get_md5(file1)
        md5_2 = get_md5(file2)
        
        print(f"文件1: {file1}")
        print(f"MD5:   {md5_1}")
        print()
        print(f"文件2: {file2}")
        print(f"MD5:   {md5_2}")
        print()
        
        if md5_1 == md5_2:
            print("✅ 两个文件完全相同！")
        else:
            print("❌ 两个文件不相同！")
            
    except FileNotFoundError as e:
        print(f"错误: 文件不存在 - {e.filename}")
        sys.exit(1)
    except Exception as e:
        print(f"错误: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
