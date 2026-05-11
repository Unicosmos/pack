"""
SKU 人工审核界面 — Gradio 4.x 原生实现
毕业设计展示用 · 100% 保留原始数据 · 无复杂依赖
使用方法: pip install gradio numpy pillow  →  python sku_review.py
"""

import gradio as gr
import json
import shutil
from pathlib import Path
from datetime import datetime
import numpy as np
from PIL import Image
import io
import base64

# ═══════════════════════════ 路径配置 ═══════════════════════════
CROPS_DIR = Path("crops")
SKU_DIR = Path("sku_output")
DB_PATH = SKU_DIR / "sku_database.json"
CANDIDATES_DIR = SKU_DIR / "new_candidates"
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ═══════════════════════════ 工具函数 ═══════════════════════════
def ensure_dirs():
    """初始化所需目录与空文件"""
    for d in [CROPS_DIR, SKU_DIR, CANDIDATES_DIR]:
        d.mkdir(parents=True, exist_ok=True)
    if not DB_PATH.exists():
        DB_PATH.write_text(json.dumps({}, ensure_ascii=False, indent=2), "utf-8")


def read_db():
    try:
        return json.loads(DB_PATH.read_text("utf-8"))
    except Exception:
        return {}


def write_db(db):
    DB_PATH.write_text(json.dumps(db, ensure_ascii=False, indent=2), "utf-8")


def get_folders():
    """获取 crops 下所有子文件夹名（有序）"""
    if not CROPS_DIR.exists():
        return []
    return sorted(d.name for d in CROPS_DIR.iterdir()
                  if d.is_dir() and not d.name.startswith("."))


def get_crop_images(folder_name):
    """获取指定文件夹下所有图片路径"""
    p = CROPS_DIR / folder_name
    if not p.exists():
        return []
    return sorted(str(f) for f in p.iterdir() if f.suffix.lower() in EXTS)


def get_sku_items(db, keyword=""):
    """
    获取 SKU 列表（用于库池展示）
    支持两种数据库结构：
    1. 新结构: {"skus": [{"sku_id": "xxx", "sku_name": "xxx", ...}, ...]}
    2. 旧结构: {"sku_id": {"name": "xxx", ...}, ...}
    返回: [{"id","name","cover","cnt"}, ...]
    """
    out = []
    
    # 兼容新旧两种数据库结构
    if "skus" in db:
        # 新结构: {"skus": [...]}
        sku_list = db["skus"]
        for sku in sku_list:
            sid = sku.get("sku_id", "")
            sname = sku.get("sku_name", sid)
            members = sku.get("members", [])
            # 优先使用数据库中已有的 member_count
            cnt = sku.get("member_count", 0)
            cover = None
            if members:
                # 从数据库中获取 cover 图片，避免每次都遍历目录
                for m in members:
                    if Path(m).exists():
                        cover = m
                        break
                # 如果数据库中没有找到可用的图片，才去读取目录
                if not cover:
                    sd = SKU_DIR / sid
                    if sd.exists():
                        fs = [f for f in sd.iterdir() if f.suffix.lower() in EXTS]
                        cnt = len(fs)
                        cover = str(sorted(fs)[0]) if fs else None
            out.append(dict(id=sid, name=sname, cover=cover, cnt=cnt))
    else:
        # 旧结构: {"sku_id": {...}}
        for sid, info in db.items():
            # 优先使用数据库中已有的统计信息
            cnt = info.get("image_count", 0)
            cover = None
            # 尝试从数据库中获取 cover 图片
            if "images" in info and info["images"]:
                first_img = info["images"][0]
                img_path = SKU_DIR / sid / first_img
                if img_path.exists():
                    cover = str(img_path)
            # 如果数据库中没有找到，才去读取目录
            if not cover:
                sd = SKU_DIR / sid
                if sd.exists():
                    fs = [f for f in sd.iterdir() if f.suffix.lower() in EXTS]
                    cnt = len(fs)
                    cover = str(sorted(fs)[0]) if fs else None
                else:
                    cnt, cover = 0, None
            out.append(dict(id=sid, name=info.get("name", sid), cover=cover, cnt=cnt))
    
    if keyword:
        k = keyword.lower()
        out = [o for o in out if k in o["id"].lower() or k in o["name"].lower()]
    return out[:200]


def get_sku_images(sid):
    """获取某 SKU 目录下所有图片路径"""
    sd = SKU_DIR / sid
    if not sd.exists():
        return []
    return sorted(str(f) for f in sd.iterdir() if f.suffix.lower() in EXTS)


def auto_sku_id(db):
    """自动生成不重复的 SKU 编号"""
    i = 1
    while f"{i:06d}" in db:
        i += 1
    return f"{i:06d}"


def add_log(logs, msg):
    """追加一条日志"""
    t = datetime.now().strftime("%H:%M:%S")
    return logs + [f"[{t}] {msg}"]


def render_logs(logs):
    """将日志列表转为显示文本"""
    return "\n".join(logs[-100:])


def rotate_image(image_path, angle):
    """旋转图片"""
    try:
        img = Image.open(image_path)
        rotated = img.rotate(angle, expand=True)
        # 生成临时文件路径（使用系统临时目录）
        import tempfile
        temp_dir = Path(tempfile.gettempdir()) / "sku_editor"
        temp_dir.mkdir(exist_ok=True)
        temp_path = temp_dir / f"rotated_{Path(image_path).name}"
        rotated.save(temp_path)
        return str(temp_path)
    except Exception as e:
        print(f"旋转图片失败: {e}")
        return None


def crop_image(image_path, x1, y1, x2, y2):
    """裁剪图片（使用百分比坐标）"""
    try:
        img = Image.open(image_path)
        width, height = img.size
        # 将百分比转换为像素坐标
        x1_px = int((x1 / 100) * width)
        y1_px = int((y1 / 100) * height)
        x2_px = int((x2 / 100) * width)
        y2_px = int((y2 / 100) * height)
        # 确保坐标在有效范围内
        x1_px = max(0, min(width, x1_px))
        y1_px = max(0, min(height, y1_px))
        x2_px = max(0, min(width, x2_px))
        y2_px = max(0, min(height, y2_px))
        # 确保 x1 < x2 且 y1 < y2
        if x1_px >= x2_px or y1_px >= y2_px:
            return None
        cropped = img.crop((x1_px, y1_px, x2_px, y2_px))
        # 生成临时文件路径（使用系统临时目录）
        import tempfile
        temp_dir = Path(tempfile.gettempdir()) / "sku_editor"
        temp_dir.mkdir(exist_ok=True)
        temp_path = temp_dir / f"cropped_{Path(image_path).name}"
        cropped.save(temp_path)
        return str(temp_path)
    except Exception as e:
        print(f"裁剪图片失败: {e}")
        return None


def save_edited_image(edited_path, original_path, sku_id):
    """保存编辑后的图片到 SKU 库，使用同名文件"""
    try:
        if not edited_path or not sku_id:
            return None
        
        # 确保 SKU 目录存在
        sku_dir = SKU_DIR / sku_id
        sku_dir.mkdir(exist_ok=True)
        
        # 生成目标文件路径（使用原始文件名）
        target_path = sku_dir / Path(original_path).name
        
        # 复制编辑后的图片到 SKU 库
        shutil.copy2(edited_path, target_path)
        
        return str(target_path)
    except Exception as e:
        print(f"保存编辑后的图片失败: {e}")
        return None


# ═══════════════════════════ 初始化 ═══════════════════════════
ensure_dirs()


# ═══════════════════════════ UI 构建 ═══════════════════════════
with gr.Blocks(
    title="SKU 人工审核工具"
) as demo:

    # ── 状态变量 ──
    st_folder_idx = gr.State(0)
    st_crop_sel = gr.State([])          # 左侧已选中的图片索引列表
    st_images = gr.State([])            # 当前文件夹的全部图片路径
    st_sku_id = gr.State(None)          # 右侧当前选中的 SKU ID
    st_logs = gr.State([])              # 操作日志
    st_sku_img_sel = gr.State([])       # SKU 详情区已选中的图片索引
    st_editing_image = gr.State(None)    # 当前正在编辑的图片路径
    st_edited_image = gr.State(None)     # 编辑后的图片数据

    # ── 标题 ──
    gr.HTML(
        '<div style="text-align:center;padding:4px 0">'
        '<h2 style="margin:0;color:#1a1a2e">📦 SKU 人工审核工具</h2>'
        '<p style="margin:2px 0 0;color:#888;font-size:13px">'
        '100% 保留原始数据 · 仅复制归类 · 安全可撤回</p></div>'
    )

    # ══════════════════ 顶部控制栏 ══════════════════
    with gr.Row(elem_id="top-bar"):
        with gr.Column(scale=3):
            with gr.Row():
                btn_prev = gr.Button("◀ 上一个", size="sm")
                html_folder = gr.HTML('<span style="font-size:14px;font-weight:600">📂 未加载</span>')
                btn_next = gr.Button("下一个 ▶", size="sm")
                # 文件夹选择下拉框
                ddl_folders = gr.Dropdown(
                    label="选择文件夹",
                    choices=[],
                    value=None,
                    interactive=True,
                    show_label=False,
                    scale=2
                )

        with gr.Column(scale=2):
            txt_search = gr.Textbox(
                label="SKU搜索", placeholder="输入编号或名称筛选…",
                scale=4, max_lines=1, show_label=True
            )

        with gr.Column(scale=4):
            with gr.Row():
                txt_newid = gr.Textbox(
                    label="新增SKU", placeholder="留空自动生成",
                    scale=4, max_lines=1, show_label=True
                )
                btn_add = gr.Button("➕ 新增", size="sm", variant="secondary")
                btn_save = gr.Button("💾 保存更新", size="sm", variant="primary")

    # ══════════════════ 中间双栏操作区 ══════════════════
    with gr.Row():

        # ─── 左侧：待审核图片区 ───
        with gr.Column(scale=1):
            gr.Markdown("### 📋 待审核图片")
            gr.HTML('<span style="font-size:12px;color:#999">点击图片选择/取消，支持多选 · 点击缩略图可预览大图</span>')
            gal_crop = gr.Gallery(
                columns=5, height=440, object_fit="contain",
                show_label=False,
                interactive=True,
                selected_index=None,
                allow_preview=True
            )
            with gr.Row():
                btn_clear = gr.Button("🧹 清空界面", size="sm")
                txt_sel_status = gr.Textbox(
                    "未选择图片", show_label=False, max_lines=1,
                    interactive=False, elem_id="hint-box"
                )
            
            # 已选中图片预览区
            gr.Markdown("### 📋 已选中图片")
            gr.HTML('<span style="font-size:12px;color:#999">显示当前已选择的图片，点击图片可取消选中</span>')
            gal_selected = gr.Gallery(
                columns=5, height="auto", object_fit="contain",
                show_label=False,
                interactive=True
            )
            
            # 图片编辑折叠面板
            acc_edit = gr.Accordion("🖼️ 图片编辑", open=False)
            with acc_edit:
                gr.Markdown("#### 选择要编辑的图片，然后进行裁剪或旋转")
                with gr.Row():
                    btn_edit_selected = gr.Button("编辑选中图片")
                with gr.Row():
                    gr.Markdown("##### 旋转图片")
                    angle = gr.Slider(minimum=-180, maximum=180, step=90, value=0, label="旋转角度")
                    btn_rotate = gr.Button("旋转")
                with gr.Row():
                    gr.Markdown("##### 裁剪图片")
                    with gr.Column(scale=1):
                        x1 = gr.Number(label="X1 (%)", value=0, minimum=0, maximum=100, step=1)
                        y1 = gr.Number(label="Y1 (%)", value=0, minimum=0, maximum=100, step=1)
                    with gr.Column(scale=1):
                        x2 = gr.Number(label="X2 (%)", value=100, minimum=0, maximum=100, step=1)
                        y2 = gr.Number(label="Y2 (%)", value=100, minimum=0, maximum=100, step=1)
                    btn_crop = gr.Button("裁剪")
                img_editor = gr.Image(label="编辑预览", type="filepath", interactive=True, elem_id="img-editor")
                with gr.Row():
                    btn_save_edit = gr.Button("保存编辑", variant="primary")
                    btn_cancel_edit = gr.Button("取消编辑")
        
        # 自定义 CSS 样式，使图片编辑预览高度自适应
        demo.css = """
        #img-editor img {
            max-height: none !important;
            height: auto !important;
        }
        .grid-wrap img {
            max-height: none !important;
            height: auto !important;
        }
        .grid-wrap.fixed-height {
            height: auto !important;
            max-height: none !important;
            min-height: 200px;
            overflow: visible !important; 
        }
        
        .gallery-container,
        .gallery {
            height: auto !important;
            max-height: none !important;
            overflow: visible !important;
        }

        .grid-wrap img {
            max-height: none !important;
            height: auto !important;
        }
        .gallery-item img {
            max-height: none !important;
            height: auto !important;
        }
        """

        # ─── 右侧：SKU 库池区 ───
        with gr.Column(scale=1):
            gr.Markdown("### 🗂️ SKU 库池")
            gr.HTML('<span style="font-size:12px;color:#999">点击 SKU 查看详情，点击图片选择/取消</span>')
            
            # 使用 Gallery 组件显示 SKU 封面图
            gal_sku = gr.Gallery(
                columns=6, height=300, object_fit="contain",
                show_label=False,
                interactive=True
            )
            
            txt_action = gr.Textbox(
                "操作提示：先在左侧选择图片，再点击右侧目标 SKU",
                show_label=False, max_lines=2, interactive=False, elem_id="hint-box"
            )
            with gr.Row():
                btn_assign = gr.Button("✅ 确认归类", size="sm", variant="primary")
                btn_recall = gr.Button("↩️ 撤回", size="sm")
                btn_del_sku = gr.Button("🗑️ 删除空SKU", size="sm", variant="stop")

            # SKU 详情折叠面板
            acc_detail = gr.Accordion("SKU 详情", open=False)
            with acc_detail:
                txt_sku_detail = gr.Textbox(show_label=False, interactive=False, lines=2)
                gal_sku_imgs = gr.Gallery(
                    columns=4, height="auto", object_fit="contain",
                    show_label=False,
                    interactive=True
                )

    # ══════════════════ 底部日志栏 ══════════════════
    with gr.Accordion("📜 操作日志", open=True):
        txt_log = gr.Textbox(
            show_label=False, interactive=False,
            lines=6, max_lines=10, autoscroll=True, elem_id="log-area"
        )
        sld_progress = gr.Slider(
            minimum=0, maximum=100, value=0,
            show_label=False, interactive=False
        )

    # ══════════════════════════════════════════════════════
    #                     事件绑定
    # ══════════════════════════════════════════════════════

    # ── 辅助：构建 SKU Gallery 更新值 ──
    def _sku_gallery_update(keyword, logs):
        db = read_db()
        items = get_sku_items(db, keyword)
        # 为没有图片的 SKU 使用默认图标
        covers = []
        for it in items:
            if it["cover"]:
                covers.append(it["cover"])
            else:
                # 使用包含 SKU 编号的默认图标
                sku_id = it["id"]
                covers.append(f"https://via.placeholder.com/200x200?text={sku_id}")
        return gr.update(value=covers), logs

    # ── 辅助：构建文件夹加载结果 ──
    def _load_folder(idx, logs):
        folders = get_folders()
        if not folders:
            return (
                '<span style="font-size:14px;font-weight:600;color:#e65100">'
                '⚠️ crops/ 目录为空，请放入切图文件夹</span>',
                gr.update(value=[], captions=[]),
                [], idx, "未选择图片", logs
            )
        idx = max(0, min(idx, len(folders) - 1))
        name = folders[idx]
        imgs = get_crop_images(name)
        html = (f'<span style="font-size:14px;font-weight:600">'
                f'📂 {name}（{idx+1}/{len(folders)}，{len(imgs)}张）</span>')
        logs = add_log(logs, f"切换文件夹 → {name}（{len(imgs)}张图片）")
        return html, gr.update(value=imgs), imgs, idx, "未选择图片", logs

    # ━━━━━━━━━━ 1. 页面初始化 ━━━━━━━━━━
    def on_init():
        logs = add_log([], "🚀 系统初始化完成")
        logs = add_log(logs, f"📁 crops 路径: {CROPS_DIR.resolve()}")
        logs = add_log(logs, f"📁 sku_output 路径: {SKU_DIR.resolve()}")

        # 加载第一个文件夹
        folders = get_folders()
        if not folders:
            sku_upd, _ = _sku_gallery_update("", logs)
            return (
                '<span style="font-size:14px;font-weight:600;color:#e65100">'
                '⚠️ crops/ 目录为空</span>',
                gr.update(value=[]),
                [], 0, sku_upd,
                "操作提示：先在左侧选择图片，再点击右侧目标 SKU",
                "未选择图片", render_logs(logs), gr.update(value=[]),
                gr.update(choices=[], value=None)
            )

        html, crop_upd, imgs, idx, sel_txt, logs = _load_folder(0, logs)
        sku_upd, _ = _sku_gallery_update("", logs)

        return (
            html, crop_upd, imgs, idx, sku_upd,
            "操作提示：先在左侧选择图片，再点击右侧目标 SKU",
            sel_txt, render_logs(logs), gr.update(value=[]),
            gr.update(choices=folders, value=folders[0] if folders else None)
        )

    demo.load(
        fn=on_init,
        outputs=[
            html_folder, gal_crop, st_images, st_folder_idx, gal_sku,
            txt_action, txt_sel_status, txt_log, gal_selected, ddl_folders
        ]
    )

    # ━━━━━━━━━━ 2. 文件夹切换 ━━━━━━━━━━
    def on_prev_folder(idx, lg):
        folders = get_folders()
        html, crop_upd, imgs, new_idx, sel_txt, new_logs = _load_folder(idx - 1, lg)
        sku_upd, new_logs = _sku_gallery_update(txt_search.value, add_log(new_logs, ""))
        folder_name = folders[new_idx] if 0 <= new_idx < len(folders) else None
        return html, crop_upd, imgs, new_idx, sel_txt, new_logs, gr.update(value=[]), sku_upd, folder_name

    btn_prev.click(
        fn=on_prev_folder,
        inputs=[st_folder_idx, st_logs],
        outputs=[html_folder, gal_crop, st_images, st_folder_idx, txt_sel_status, st_logs, gal_selected, gal_sku, ddl_folders]
    )

    def on_next_folder(idx, lg):
        folders = get_folders()
        html, crop_upd, imgs, new_idx, sel_txt, new_logs = _load_folder(idx + 1, lg)
        sku_upd, new_logs = _sku_gallery_update(txt_search.value, add_log(new_logs, ""))
        folder_name = folders[new_idx] if 0 <= new_idx < len(folders) else None
        return html, crop_upd, imgs, new_idx, sel_txt, new_logs, gr.update(value=[]), sku_upd, folder_name

    btn_next.click(
        fn=on_next_folder,
        inputs=[st_folder_idx, st_logs],
        outputs=[html_folder, gal_crop, st_images, st_folder_idx, txt_sel_status, st_logs, gal_selected, gal_sku, ddl_folders]
    )

    # ━━━━━━━━━━ 2.5 文件夹下拉框选择 ━━━━━━━━━━
    def on_folder_select(folder_name, logs):
        folders = get_folders()
        if folder_name in folders:
            idx = folders.index(folder_name)
            html, crop_upd, imgs, new_idx, sel_txt, new_logs = _load_folder(idx, logs)
            sku_upd, new_logs = _sku_gallery_update(txt_search.value, new_logs)
            return html, crop_upd, imgs, new_idx, sel_txt, gr.update(value=[]), sku_upd, new_logs, folder_name
        return html_folder.value, gr.update(), st_images.value, st_folder_idx.value, txt_sel_status.value, gr.update(), gr.update(), logs, folder_name

    ddl_folders.change(
        fn=on_folder_select,
        inputs=[ddl_folders, st_logs],
        outputs=[html_folder, gal_crop, st_images, st_folder_idx, txt_sel_status, gal_selected, gal_sku, st_logs, ddl_folders]
    )

    # ━━━━━━━━━━ 3. 左侧图片多选 ━━━━━━━━━━
    @gal_crop.select(
        inputs=[st_crop_sel, st_images, st_logs],
        outputs=[st_crop_sel, txt_sel_status, st_logs, gal_selected]
    )
    def on_crop_select(cur_sel, images, logs, evt: gr.SelectData):
        idx = evt.index
        new_sel = list(cur_sel) if cur_sel else []
        if idx in new_sel:
            new_sel.remove(idx)
        else:
            new_sel.append(idx)
        new_sel.sort()
        n = len(new_sel)
        if n > 0:
            # 获取已选中的图片和文件名
            selected_images = [images[i] for i in new_sel if i < len(images)]
            selected_names = [img.split('/')[-1] for img in selected_images]
            # 显示已选择的图片数量和名称
            txt = f"✅ 已选择 {n} 张图片: " + ", ".join(selected_names[:3])
            if n > 3:
                txt += f" 等{n}张"
        else:
            txt = "未选择图片"
        
        # 获取已选中的图片
        selected_images = [images[i] for i in new_sel if i < len(images)]
        return new_sel, txt, logs, gr.update(value=selected_images)

    # ━━━━━━━━━━ 4. 清空界面 ━━━━━━━━━━
    btn_clear.click(
        fn=lambda lg: (
            gr.update(value=[]),
            [], [], "未选择图片",
            gr.update(value=[]),
            add_log(lg, "🧹 已清空界面显示（原始文件未受影响）"),
            render_logs(add_log(lg, "🧹 已清空界面显示（原始文件未受影响）"))
        ),
        inputs=[st_logs],
        outputs=[gal_crop, st_images, st_crop_sel, txt_sel_status, gal_selected, st_logs, txt_log]
    )

    # ━━━━━━━━━━ 5. SKU 搜索 ━━━━━━━━━━
    txt_search.change(
    fn=lambda kw, lg: _sku_gallery_update(kw, lg if not kw else add_log(lg, f"🔍 搜索 SKU: '{kw}'")),
    inputs=[txt_search, st_logs],
    outputs=[gal_sku, st_logs]
)

    # ━━━━━━━━━━ 6. 右侧 SKU 点击 ━━━━━━━━━━
    @gal_sku.select(
        inputs=[st_crop_sel, st_logs],
        outputs=[st_sku_id, txt_action, acc_detail, txt_sku_detail,
                 gal_sku_imgs, st_sku_img_sel, st_logs]
    )
    def on_sku_click(crop_sel, logs, evt: gr.SelectData):
        db = read_db()
        items = get_sku_items(db, txt_search.value)
        idx = evt.index
        if idx < 0 or idx >= len(items):
            return None, "操作提示：请点击有效的 SKU", gr.update(open=False), "", \
                   gr.update(value=[], captions=[]), [], logs

        sku = items[idx]
        sid = sku["id"]
        n_crop = len(crop_sel) if crop_sel else 0

        # 操作提示
        if n_crop > 0:
            hint = f"即将把 {n_crop} 张图片归类至【{sid}】({sku['name']})，请点击「✅ 确认归类」"
        else:
            hint = f"已选中 SKU: {sid}（{sku['cnt']}张）— 请先在左侧勾选图片"

        # SKU 详情
        detail = f"编号: {sid}  |  名称: {sku['name']}  |  图片数: {sku['cnt']}"
        sku_imgs = get_sku_images(sid)

        logs = add_log(logs, f"选中 SKU: {sid}（{sku['cnt']}张图片）")

        return (
            sid, hint, gr.update(open=True), detail,
            gr.update(value=sku_imgs),
            [], logs
        )

    # ━━━━━━━━━━ 7. 确认归类 ━━━━━━━━━━
    def on_assign(crop_sel, sku_id, images, logs):
        # 校验
        if not sku_id:
            logs = add_log(logs, "⚠️ 请先选择目标 SKU")
            return (
                gr.update(), images, [], "未选择图片",
                "⚠️ 请先选择目标 SKU",
                *_sku_gallery_update(txt_search.value, logs),
                render_logs(logs)
            )
        if not crop_sel:
            logs = add_log(logs, "⚠️ 请先在左侧选择要归类的图片")
            return (
                gr.update(), images, [], "未选择图片",
                "⚠️ 请先在左侧选择要归类的图片",
                *_sku_gallery_update(txt_search.value, logs),
                render_logs(logs)
            )

        # 执行复制
        db = read_db()
        sku_subdir = SKU_DIR / sku_id
        sku_subdir.mkdir(exist_ok=True)

        copied = []
        for i in crop_sel:
            if i < len(images):
                src = Path(images[i])
                dst = sku_subdir / src.name
                if not dst.exists():
                    shutil.copy2(str(src), str(dst))
                    copied.append(src.name)

        # 更新数据库
        if sku_id not in db:
            db[sku_id] = {"name": sku_id, "images": [], "feature_mean": []}
        existing = db[sku_id].get("images", [])
        for name in copied:
            if name not in existing:
                existing.append(name)
        db[sku_id]["images"] = existing
        db[sku_id]["image_count"] = len(existing)
        write_db(db)

        logs = add_log(logs, f"✅ 归类 {len(copied)} 张图片至 {sku_id}: "
                             f"{', '.join(copied[:3])}{'...' if len(copied) > 3 else ''}")

        sku_upd, logs = _sku_gallery_update(txt_search.value, logs)
        names_preview = ", ".join(copied[:3]) + ("..." if len(copied) > 3 else "")

        return (
            gr.update(),                           # gal_crop 保持不变
            images,                                # st_images 不变
            [],                                    # st_crop_sel 清空
            "未选择图片",                           # txt_sel_status
            gr.update(value=[]),                   # gal_selected 清空
            f"✅ 成功归类 {len(copied)} 张图片至 {sku_id}\n{names_preview}",
            sku_upd,                               # gal_sku 刷新
            render_logs(logs)                      # txt_log
        )

    btn_assign.click(
        fn=on_assign,
        inputs=[st_crop_sel, st_sku_id, st_images, st_logs],
        outputs=[
            gal_crop, st_images, st_crop_sel, txt_sel_status,
            gal_selected, txt_action, gal_sku, txt_log
        ]
    )

    # ━━━━━━━━━━ 8. 撤回至待审核 ━━━━━━━━━━
    def on_recall(sku_id, sku_img_sel, logs):
        if not sku_id:
            logs = add_log(logs, "⚠️ 请先选择要撤回的 SKU")
            return gr.update(value=[], captions=[]), [], render_logs(logs)

        if not sku_img_sel:
            logs = add_log(logs, "⚠️ 请在 SKU 详情区勾选要撤回的图片")
            return gr.update(value=[], captions=[]), [], render_logs(logs)

        sku_imgs = get_sku_images(sku_id)
        removed = []
        for i in sku_img_sel:
            if i < len(sku_imgs):
                p = Path(sku_imgs[i])
                try:
                    p.unlink()       # 删除复制件（原文件在 crops/ 不受影响）
                    removed.append(p.name)
                except Exception as e:
                    logs = add_log(logs, f"⚠️ 删除失败 {p.name}: {e}")

        # 更新数据库
        db = read_db()
        if sku_id in db:
            existing = db[sku_id].get("images", [])
            for name in removed:
                if name in existing:
                    existing.remove(name)
            db[sku_id]["images"] = existing
            db[sku_id]["image_count"] = len(existing)
            write_db(db)

        logs = add_log(logs, f"↩️ 从 {sku_id} 撤回 {len(removed)} 张图片: "
                             f"{', '.join(removed[:3])}{'...' if len(removed) > 3 else ''}")

        # 刷新 SKU 详情
        new_imgs = get_sku_images(sku_id)
        return gr.update(value=new_imgs), [], render_logs(logs)

    btn_recall.click(
        fn=on_recall,
        inputs=[st_sku_id, st_sku_img_sel, st_logs],
        outputs=[gal_sku_imgs, st_sku_img_sel, txt_log]
    )

    # ━━━━━━━━━━ 9. 删除空 SKU ━━━━━━━━━━
    def on_delete_sku(sku_id, logs):
        if not sku_id:
            logs = add_log(logs, "⚠️ 请先选择要删除的 SKU")
            return *_sku_gallery_update(txt_search.value, logs), render_logs(logs)

        db = read_db()
        if sku_id not in db:
            logs = add_log(logs, f"⚠️ SKU {sku_id} 不存在于数据库中")
            return *_sku_gallery_update(txt_search.value, logs), render_logs(logs)

        sku_subdir = SKU_DIR / sku_id
        if sku_subdir.exists():
            imgs = [f for f in sku_subdir.iterdir() if f.suffix.lower() in EXTS]
            if imgs:
                # 非空 SKU，将图片移至 new_candidates
                CANDIDATES_DIR.mkdir(exist_ok=True)
                for f in imgs:
                    shutil.move(str(f), str(CANDIDATES_DIR / f.name))
                logs = add_log(logs, f"⚠️ SKU {sku_id} 非空（{len(imgs)}张），"
                                     f"图片已移至 new_candidates/")
                sku_subdir.rmdir()
            else:
                sku_subdir.rmdir()

        del db[sku_id]
        write_db(db)
        logs = add_log(logs, f"🗑️ 已删除 SKU: {sku_id}")

        sku_upd, logs = _sku_gallery_update(txt_search.value, logs)
        return sku_upd, render_logs(logs)

    btn_del_sku.click(
        fn=on_delete_sku,
        inputs=[st_sku_id, st_logs],
        outputs=[gal_sku, txt_log]
    )

    # ━━━━━━━━━━ 10. 新增/重命名 SKU ━━━━━━━━━━
    def on_add_sku(custom_id, logs):
        db = read_db()
        
        # 兼容新旧两种数据库结构
        if "skus" in db:
            # 新结构: {"skus": [...]}
            # 检查是否包含 | 分隔（格式：原编号|新名称）
            if "|" in custom_id:
                parts = custom_id.split("|", 1)
                old_sid = parts[0].strip()
                new_name = parts[1].strip()
                
                # 查找并更新 SKU
                found = False
                for sku in db["skus"]:
                    if sku.get("sku_id") == old_sid:
                        old_name = sku.get("sku_name", old_sid)
                        sku["sku_name"] = new_name
                        write_db(db)
                        logs = add_log(logs, f"✏️ 重命名 SKU: {old_sid} | {old_name} → {new_name}")
                        found = True
                        break
                
                if not found:
                    logs = add_log(logs, f"⚠️ 未找到 SKU '{old_sid}'")
                
                sku_upd, logs = _sku_gallery_update(txt_search.value, logs)
                return sku_upd, logs, f"✅ 重命名完成" if found else f"⚠️ 未找到 {old_sid}", render_logs(logs)
            else:
                # 只输入名称，编号自动递增
                name = custom_id.strip()
                sid = auto_sku_id(db)
                
                # 创建新 SKU
                new_sku = {
                    "sku_id": sid,
                    "sku_name": name,
                    "member_count": 0,
                    "members": [],
                    "feature_center": []
                }
                db["skus"].append(new_sku)
                write_db(db)
                (SKU_DIR / sid).mkdir(exist_ok=True)
                
                logs = add_log(logs, f"➕ 新增 SKU: {sid} | {name}")
                sku_upd, logs = _sku_gallery_update(txt_search.value, logs)
                return sku_upd, logs, f"✅ 已新增 SKU: {sid} | {name}", render_logs(logs)
        else:
            # 旧结构: {"sku_id": {...}}
            # 检查是否包含 | 分隔（格式：原编号|新名称）
            if "|" in custom_id:
                parts = custom_id.split("|", 1)
                old_sid = parts[0].strip()
                new_name = parts[1].strip()
                
                if old_sid in db:
                    old_name = db[old_sid].get("name", old_sid)
                    db[old_sid]["name"] = new_name
                    write_db(db)
                    logs = add_log(logs, f"✏️ 重命名 SKU: {old_sid} | {old_name} → {new_name}")
                else:
                    logs = add_log(logs, f"⚠️ 未找到 SKU '{old_sid}'")
                
                sku_upd, logs = _sku_gallery_update(txt_search.value, logs)
                return sku_upd, logs, f"✅ 重命名完成" if old_sid in db else f"⚠️ 未找到 {old_sid}", render_logs(logs)
            else:
                # 只输入名称，编号自动递增
                name = custom_id.strip()
                sid = auto_sku_id(db)
                
                # 校验唯一性
                if sid in db:
                    logs = add_log(logs, f"⚠️ SKU 编号 '{sid}' 已存在")
                    return gr.update(), logs, f"⚠️ 编号 '{sid}' 已存在", render_logs(logs)
                
                # 创建
                db[sid] = {"name": name, "images": [], "feature_mean": [], "image_count": 0}
                write_db(db)
                (SKU_DIR / sid).mkdir(exist_ok=True)
                
                logs = add_log(logs, f"➕ 新增 SKU: {sid} | {name}")
                sku_upd, logs = _sku_gallery_update(txt_search.value, logs)
                return sku_upd, logs, f"✅ 已新增 SKU: {sid} | {name}", render_logs(logs)

    btn_add.click(
        fn=on_add_sku,
        inputs=[txt_newid, st_logs],
        outputs=[gal_sku, st_logs, txt_action, txt_log]
    )

    # ━━━━━━━━━━ 11. 保存更新（含进度条）━━━━━━━━━━
    def on_save(logs, progress=gr.Progress()):
        progress(0, desc="开始保存…")
        logs = add_log(logs, "💾 开始保存 SKU 库更新…")

        # ① 同步数据库
        progress(0.2, desc="同步数据库…")
        db = read_db()
        
        # 兼容新旧两种数据库结构
        if "skus" in db:
            # 新结构: {"skus": [...]}
            for sku in db["skus"]:
                sid = sku.get("sku_id", "")
                if sid:
                    sd = SKU_DIR / sid
                    if sd.exists():
                        # 统计实际存在的图片
                        fs = [f for f in sd.iterdir() if f.suffix.lower() in EXTS]
                        sku["member_count"] = len(fs)
                        sku["members"] = [str(f) for f in sorted(fs)]
        else:
            # 旧结构: {"sku_id": {...}}
            for sid in list(db.keys()):
                sd = SKU_DIR / sid
                if sd.exists():
                    cnt = len([f for f in sd.iterdir() if f.suffix.lower() in EXTS])
                    db[sid]["image_count"] = cnt
        
        write_db(db)
        logs = add_log(logs, "  ✓ 数据库同步完成")

        # ② 特征矩阵更新已移除（由 build_library.py 单独处理）

        # ③ 最终确认写入
        progress(0.7, desc="最终写入…")
        write_db(db)

        progress(1.0, desc="保存完成")
        logs = add_log(logs, "✅ SKU 库保存成功！")

        return logs, render_logs(logs), 100

    btn_save.click(
        fn=on_save,
        inputs=[st_logs],
        outputs=[st_logs, txt_log, sld_progress]
    )

    # ━━━━━━━━━━ 12. SKU 详情区图片多选 ━━━━━━━━━━
    @gal_sku_imgs.select(
        inputs=[st_sku_img_sel, st_logs],
        outputs=[st_sku_img_sel, st_logs]
    )
    def on_sku_img_select(cur_sel, logs, evt: gr.SelectData):
        idx = evt.index
        new_sel = list(cur_sel) if cur_sel else []
        if idx in new_sel:
            new_sel.remove(idx)
        else:
            new_sel.append(idx)
        new_sel.sort()
        return new_sel, logs

    # ━━━━━━━━━━ 13. 已选中图片取消选择 ━━━━━━━━━━
    @gal_selected.select(
        inputs=[st_crop_sel, st_images, st_logs],
        outputs=[st_crop_sel, txt_sel_status, st_logs, gal_selected]
    )
    def on_selected_image_select(cur_sel, images, logs, evt: gr.SelectData):
        # 获取被点击的图片在预览区中的索引
        preview_idx = evt.index
        
        # 计算实际在原始图片列表中的索引
        if cur_sel and preview_idx < len(cur_sel):
            actual_idx = cur_sel[preview_idx]
            
            # 从选中列表中移除该索引
            new_sel = [idx for idx in cur_sel if idx != actual_idx]
            new_sel.sort()
            
            n = len(new_sel)
            txt = f"✅ 已选择 {n} 张图片" if n > 0 else "未选择图片"
            
            # 更新已选中图片预览区
            selected_images = [images[i] for i in new_sel if i < len(images)]
            return new_sel, txt, logs, gr.update(value=selected_images)
        
        return cur_sel, f"✅ 已选择 {len(cur_sel)} 张图片" if cur_sel else "未选择图片", logs, gr.update(value=[images[i] for i in cur_sel if i < len(images)])
    
    # ━━━━━━━━━━ 14. 图片编辑功能 ━━━━━━━━━━
    # 编辑选中图片
    def on_edit_selected(crop_sel, images, logs):
        if not crop_sel:
            logs = add_log(logs, "⚠️ 请先选择要编辑的图片")
            return None, None, logs, render_logs(logs)
        
        # 取第一张选中的图片进行编辑
        img_path = images[crop_sel[0]] if crop_sel[0] < len(images) else None
        if not img_path:
            logs = add_log(logs, "⚠️ 选中的图片不存在")
            return None, None, logs, render_logs(logs)
        
        logs = add_log(logs, f"开始编辑图片: {Path(img_path).name}")
        return img_path, img_path, logs, render_logs(logs)
    
    btn_edit_selected.click(
        fn=on_edit_selected,
        inputs=[st_crop_sel, st_images, st_logs],
        outputs=[st_editing_image, img_editor, st_logs, txt_log]
    )
    
    # 旋转图片
    def on_rotate(editing_image, angle, logs):
        if not editing_image:
            logs = add_log(logs, "⚠️ 请先选择要编辑的图片")
            return None, logs, render_logs(logs)
        
        rotated_path = rotate_image(editing_image, angle)
        if rotated_path:
            logs = add_log(logs, f"旋转图片 {angle} 度")
            return rotated_path, logs, render_logs(logs)
        else:
            logs = add_log(logs, "⚠️ 旋转图片失败")
            return None, logs, render_logs(logs)
    
    btn_rotate.click(
        fn=on_rotate,
        inputs=[st_editing_image, angle, st_logs],
        outputs=[img_editor, st_logs, txt_log]
    )
    
    # 裁剪图片
    def on_crop(editing_image, x1, y1, x2, y2, logs):
        if not editing_image:
            logs = add_log(logs, "⚠️ 请先选择要编辑的图片")
            return None, logs, render_logs(logs)
        
        cropped_path = crop_image(editing_image, x1, y1, x2, y2)
        if cropped_path:
            logs = add_log(logs, f"裁剪图片: ({x1}%,{y1}%)-({x2}%,{y2}%)")
            return cropped_path, logs, render_logs(logs)
        else:
            logs = add_log(logs, "⚠️ 裁剪图片失败，请检查坐标")
            return None, logs, render_logs(logs)
    
    btn_crop.click(
        fn=on_crop,
        inputs=[st_editing_image, x1, y1, x2, y2, st_logs],
        outputs=[img_editor, st_logs, txt_log]
    )
    
    # 保存编辑
    def on_save_edit(editing_image, edited_path, crop_sel, images, sku_id, logs):
        if not editing_image or not edited_path:
            logs = add_log(logs, "⚠️ 请先编辑图片")
            return None, crop_sel, f"✅ 已选择 {len(crop_sel)} 张图片" if crop_sel else "未选择图片", gr.update(value=[images[i] for i in crop_sel if i < len(images)]), logs, render_logs(logs)
        
        if not sku_id:
            logs = add_log(logs, "⚠️ 请先选择目标 SKU")
            return None, crop_sel, f"✅ 已选择 {len(crop_sel)} 张图片" if crop_sel else "未选择图片", gr.update(value=[images[i] for i in crop_sel if i < len(images)]), logs, render_logs(logs)
        
        # 保存编辑后的图片到 SKU 库
        target_path = save_edited_image(edited_path, editing_image, sku_id)
        if not target_path:
            logs = add_log(logs, "⚠️ 保存编辑后的图片失败")
            return None, crop_sel, f"✅ 已选择 {len(crop_sel)} 张图片" if crop_sel else "未选择图片", gr.update(value=[images[i] for i in crop_sel if i < len(images)]), logs, render_logs(logs)
        
        # 更新数据库
        db = read_db()
        if sku_id not in db:
            db[sku_id] = {"name": sku_id, "images": [], "feature_mean": []}
        existing = db[sku_id].get("images", [])
        img_name = Path(editing_image).name
        if img_name not in existing:
            existing.append(img_name)
        db[sku_id]["images"] = existing
        db[sku_id]["image_count"] = len(existing)
        write_db(db)
        
        logs = add_log(logs, f"✅ 保存编辑后的图片到 SKU {sku_id}: {img_name}")
        
        # 清空选中状态
        return None, [], "未选择图片", gr.update(value=[]), logs, render_logs(logs)
    
    btn_save_edit.click(
        fn=on_save_edit,
        inputs=[st_editing_image, img_editor, st_crop_sel, st_images, st_sku_id, st_logs],
        outputs=[st_editing_image, st_crop_sel, txt_sel_status, gal_selected, st_logs, txt_log]
    )
    
    # 取消编辑
    def on_cancel_edit(logs):
        # 清理临时文件
        try:
            import tempfile
            temp_dir = Path(tempfile.gettempdir()) / "sku_editor"
            if temp_dir.exists():
                for f in temp_dir.glob("cropped_*"):
                    f.unlink()
                for f in temp_dir.glob("rotated_*"):
                    f.unlink()
        except Exception as e:
            print(f"清理临时文件失败: {e}")
        logs = add_log(logs, "已取消编辑")
        return None, None, logs, render_logs(logs)
    
    btn_cancel_edit.click(
        fn=on_cancel_edit,
        inputs=[st_logs],
        outputs=[st_editing_image, img_editor, st_logs, txt_log]
    )


# ═══════════════════════════ 启动 ═══════════════════════════
if __name__ == "__main__":
    print("=" * 60)
    print("  SKU 人工审核工具 已启动")
    print(f"  crops 目录: {CROPS_DIR.resolve()}")
    print(f"  sku_output 目录: {SKU_DIR.resolve()}")
    print("=" * 60)
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        inbrowser=True,
        share=False
    )
