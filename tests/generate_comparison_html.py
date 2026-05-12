import os
from pathlib import Path


def generate_html_comparison(diff_files_path, predict_dir1, predict_dir2, output_path):
    """生成HTML对比页面"""
    
    with open(diff_files_path, 'r', encoding='utf-8') as f:
        diff_files = [line.strip() for line in f if line.strip()]
    
    html_content = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>模型预测结果对比</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Microsoft YaHei', Arial, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            min-height: 100vh;
        }}
        
        .header {{
            text-align: center;
            color: white;
            margin-bottom: 30px;
            padding: 20px;
        }}
        
        .header h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }}
        
        .header p {{
            font-size: 1.1em;
            opacity: 0.9;
        }}
        
        .stats {{
            background: rgba(255,255,255,0.15);
            backdrop-filter: blur(10px);
            border-radius: 15px;
            padding: 20px;
            margin: 0 auto 30px;
            max-width: 800px;
            color: white;
            text-align: center;
        }}
        
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px;
            margin-top: 15px;
        }}
        
        .stat-item {{
            background: rgba(255,255,255,0.2);
            padding: 15px;
            border-radius: 10px;
        }}
        
        .stat-number {{
            font-size: 2em;
            font-weight: bold;
        }}
        
        .stat-label {{
            font-size: 0.9em;
            opacity: 0.9;
            margin-top: 5px;
        }}
        
        .comparison-container {{
            max-width: 1400px;
            margin: 0 auto;
        }}
        
        .image-pair {{
            background: white;
            border-radius: 15px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.2);
            margin-bottom: 30px;
            overflow: hidden;
        }}
        
        .image-header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 15px 20px;
            font-size: 1.2em;
            font-weight: bold;
        }}
        
        .image-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 2px;
            background: #e0e0e0;
        }}
        
        .image-wrapper {{
            background: #f5f5f5;
            padding: 20px;
        }}
        
        .image-label {{
            text-align: center;
            font-weight: bold;
            margin-bottom: 10px;
            color: #333;
            padding: 8px;
            border-radius: 5px;
        }}
        
        .image-label.model1 {{
            background: #e3f2fd;
            color: #1976d2;
        }}
        
        .image-label.model2 {{
            background: #f3e5f5;
            color: #7b1fa2;
        }}
        
        .image-wrapper img {{
            width: 100%;
            height: auto;
            border-radius: 8px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            cursor: pointer;
            transition: transform 0.3s ease;
        }}
        
        .image-wrapper img:hover {{
            transform: scale(1.02);
        }}
        
        .nav-buttons {{
            position: fixed;
            bottom: 30px;
            right: 30px;
            display: flex;
            gap: 10px;
            z-index: 1000;
        }}
        
        .nav-button {{
            background: white;
            border: none;
            width: 50px;
            height: 50px;
            border-radius: 50%;
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
            cursor: pointer;
            font-size: 20px;
            transition: all 0.3s ease;
        }}
        
        .nav-button:hover {{
            transform: scale(1.1);
            box-shadow: 0 6px 20px rgba(0,0,0,0.3);
        }}
        
        .lightbox {{
            display: none;
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0,0,0,0.95);
            z-index: 2000;
            justify-content: center;
            align-items: center;
        }}
        
        .lightbox.active {{
            display: flex;
        }}
        
        .lightbox img {{
            max-width: 90%;
            max-height: 90%;
            border-radius: 8px;
            box-shadow: 0 0 50px rgba(255,255,255,0.2);
        }}
        
        .lightbox-close {{
            position: absolute;
            top: 20px;
            right: 30px;
            color: white;
            font-size: 40px;
            cursor: pointer;
            transition: transform 0.3s ease;
        }}
        
        .lightbox-close:hover {{
            transform: rotate(90deg);
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🔍 模型预测结果对比</h1>
        <p>比较两个模型在同一数据集上的预测差异</p>
    </div>
    
    <div class="stats">
        <h2>📊 统计信息</h2>
        <div class="stats-grid">
            <div class="stat-item">
                <div class="stat-number">{len(diff_files)}</div>
                <div class="stat-label">有差异的图片</div>
            </div>
        </div>
    </div>
    
    <div class="comparison-container">
"""
    
    dir1_path = Path(predict_dir1)
    dir2_path = Path(predict_dir2)
    
    for filename in diff_files:
        img_path1 = dir1_path / filename
        img_path2 = dir2_path / filename
        
        rel_path1 = os.path.relpath(img_path1, start=Path(output_path).parent)
        rel_path2 = os.path.relpath(img_path2, start=Path(output_path).parent)
        
        html_content += f"""
        <div class="image-pair">
            <div class="image-header">📁 {filename}</div>
            <div class="image-grid">
                <div class="image-wrapper">
                    <div class="image-label model1">Model 1: lscd_predict_202605121912042</div>
                    <img src="{rel_path1}" alt="{filename} - Model 1" onclick="openLightbox('{rel_path1}')">
                </div>
                <div class="image-wrapper">
                    <div class="image-label model2">Model 2: lscd_predict_202605121913322</div>
                    <img src="{rel_path2}" alt="{filename} - Model 2" onclick="openLightbox('{rel_path2}')">
                </div>
            </div>
        </div>
"""
    
    html_content += """
    </div>
    
    <div class="nav-buttons">
        <button class="nav-button" onclick="scrollToTop()">↑</button>
        <button class="nav-button" onclick="scrollToBottom()">↓</button>
    </div>
    
    <div class="lightbox" id="lightbox" onclick="closeLightbox()">
        <span class="lightbox-close">&times;</span>
        <img id="lightbox-img" src="" alt="">
    </div>
    
    <script>
        function scrollToTop() {
            window.scrollTo({ top: 0, behavior: 'smooth' });
        }
        
        function scrollToBottom() {
            window.scrollTo({ top: document.body.scrollHeight, behavior: 'smooth' });
        }
        
        function openLightbox(src) {
            document.getElementById('lightbox-img').src = src;
            document.getElementById('lightbox').classList.add('active');
        }
        
        function closeLightbox() {
            document.getElementById('lightbox').classList.remove('active');
        }
        
        document.addEventListener('keydown', function(e) {
            if (e.key === 'Escape') {
                closeLightbox();
            }
        });
    </script>
</body>
</html>
"""
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"HTML对比页面已生成: {output_path}")


if __name__ == "__main__":
    diff_files_path = r"d:\A_pack\pack\tests\diff_result\diff_files.txt"
    predict_dir1 = r"d:\A_pack\pack\YOLO\runs\predict\lscd_predict_202605121912042"
    predict_dir2 = r"d:\A_pack\pack\YOLO\runs\predict\lscd_predict_202605121913322"
    output_path = r"d:\A_pack\pack\tests\comparison.html"
    
    generate_html_comparison(diff_files_path, predict_dir1, predict_dir2, output_path)
