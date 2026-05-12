"""
SKU 审核 API
支持从 crops 文件夹导入图片到 sku_output 库

注意：
- 此模块使用独立的文件系统存储（sku_output目录）
- 与正式SKU库（pack.db + data/sku_library）完全独立
- 审核完成后可批量同步到正式库
"""
import json
import shutil
from pathlib import Path
from datetime import datetime
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File

from config import config

router = APIRouter(prefix="/api/sku-review", tags=["SKU审核"])

# 配置路径 - 从config获取基础路径，避免硬编码
BASE_DIR = config.paths.BASE_DIR
CROPS_DIR = BASE_DIR / "SKU" / "crops"
SKU_DIR = BASE_DIR / "SKU" / "sku_output"
DB_PATH = SKU_DIR / "sku_database.json"
CANDIDATES_DIR = SKU_DIR / "new_candidates"
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def ensure_dirs():
    """初始化所需目录与空文件"""
    for d in [CROPS_DIR, SKU_DIR, CANDIDATES_DIR]:
        d.mkdir(parents=True, exist_ok=True)
    if not DB_PATH.exists():
        DB_PATH.write_text(json.dumps({}, ensure_ascii=False, indent=2), "utf-8")

def read_db():
    """读取数据库"""
    try:
        data = json.loads(DB_PATH.read_text("utf-8"))
        if isinstance(data, dict):
            return data
        return {}
    except Exception as e:
        print(f"读取数据库失败: {e}")
        return {}

def write_db(db):
    """写入数据库，并添加时间戳"""
    now = datetime.now().isoformat()
    for sku_id, info in db.items():
        if isinstance(info, dict):
            if "created_at" not in info:
                info["created_at"] = now
            info["updated_at"] = now
    DB_PATH.write_text(json.dumps(db, ensure_ascii=False, indent=2), "utf-8")

def get_crop_folders():
    """获取 crops 下所有子文件夹名"""
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
    """获取 SKU 列表"""
    out = []
    
    if "skus" in db:
        # 新结构: {"skus": [...]}
        sku_list = db["skus"]
        for sku in sku_list:
            sid = sku.get("sku_id", "")
            sname = sku.get("sku_name", sid)
            members = sku.get("members", [])
            cnt = sku.get("member_count", 0)
            cover = None
            if members:
                for m in members:
                    if Path(m).exists():
                        cover = m
                        break
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
            cnt = info.get("image_count", 0)
            cover = None
            if "images" in info and info["images"]:
                first_img = info["images"][0]
                img_path = SKU_DIR / sid / first_img
                if img_path.exists():
                    cover = str(img_path)
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
    return out

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

# 初始化
ensure_dirs()


@router.get("/folders")
def get_folders():
    """获取所有待审核文件夹列表"""
    return {"success": True, "folders": get_crop_folders()}


@router.get("/folder-images/{folder_name}")
def get_folder_images(folder_name: str):
    """获取指定文件夹下的图片"""
    images = get_crop_images(folder_name)
    # 转换为相对于项目根目录的路径
    image_urls = []
    for img_path in images:
        path_obj = Path(img_path)
        if path_obj.is_absolute():
            rel_path = path_obj.relative_to("d:/A_pack/pack")
            url_path = str(rel_path).replace("\\", "/")
        else:
            url_path = str(img_path).replace("\\", "/")
        image_urls.append({
            "path": img_path,
            "url": f"/static/{url_path}",
            "name": path_obj.name
        })
    return {"success": True, "images": image_urls}


@router.get("/skus")
def get_skus(keyword: str = ""):
    """获取 SKU 列表"""
    db = read_db()
    skus = get_sku_items(db, keyword)
    # 转换封面图片为可访问的URL
    for sku in skus:
        if sku.get("cover"):
            path_obj = Path(sku["cover"])
            if path_obj.is_absolute():
                rel_path = str(path_obj.relative_to("d:/A_pack/pack"))
                url_path = rel_path.replace("\\", "/")
                sku["cover_url"] = f"/static/{url_path}"
            else:
                url_path = sku["cover"].replace("\\", "/")
                sku["cover_url"] = f"/static/{url_path}"
    return {"success": True, "skus": skus}


@router.get("/sku-images/{sku_id}")
def get_sku_images_api(sku_id: str):
    """获取指定SKU的所有图片"""
    images = get_sku_images(sku_id)
    image_urls = []
    for img_path in images:
        path_obj = Path(img_path)
        if path_obj.is_absolute():
            rel_path = path_obj.relative_to("d:/A_pack/pack")
            url_path = str(rel_path).replace("\\", "/")
        else:
            url_path = str(img_path).replace("\\", "/")
        image_urls.append({
            "path": img_path,
            "url": f"/static/{url_path}",
            "name": path_obj.name
        })
    return {"success": True, "images": image_urls}


@router.post("/assign-images")
def assign_images(sku_id: str, image_paths: List[str]):
    """将图片归类到指定SKU"""
    db = read_db()
    now = datetime.now().isoformat()
    
    sku_subdir = SKU_DIR / sku_id
    sku_subdir.mkdir(exist_ok=True)
    
    copied = []
    for img_path in image_paths:
        src = Path(img_path)
        if not src.exists():
            continue
        dst = sku_subdir / src.name
        if not dst.exists():
            shutil.copy2(str(src), str(dst))
            copied.append(src.name)
    
    # 更新数据库
    if sku_id not in db:
        db[sku_id] = {
            "name": sku_id,
            "images": [],
            "feature_mean": [],
            "image_count": 0,
            "created_at": now,
            "updated_at": now
        }
    existing = db[sku_id].get("images", [])
    for name in copied:
        if name not in existing:
            existing.append(name)
    db[sku_id]["images"] = existing
    db[sku_id]["image_count"] = len(existing)
    write_db(db)
    
    return {
        "success": True,
        "message": f"成功归类 {len(copied)} 张图片至 {sku_id}"
    }


@router.post("/recall-images")
def recall_images(sku_id: str, image_paths: List[str]):
    """从SKU中撤回图片"""
    if not sku_id or not image_paths:
        return {"success": False, "message": "参数不全"}
    
    db = read_db()
    sku_subdir = SKU_DIR / sku_id
    
    removed = []
    for img_path in image_paths:
        try:
            img_name = Path(img_path).name
            file_path = sku_subdir / img_name
            if file_path.exists():
                file_path.unlink()
                removed.append(img_name)
        except Exception as e:
            print(f"删除失败 {img_path}: {e}")
    
    # 更新数据库
    if sku_id in db:
        existing = db[sku_id].get("images", [])
        for name in removed:
            if name in existing:
                existing.remove(name)
        db[sku_id]["images"] = existing
        db[sku_id]["image_count"] = len(existing)
        write_db(db)
    
    return {
        "success": True,
        "message": f"从 {sku_id} 撤回 {len(removed)} 张图片"
    }


@router.post("/create-sku")
def create_sku(name: str = "", sku_id: str = ""):
    """创建新SKU"""
    db = read_db()
    
    if not sku_id:
        sku_id = auto_sku_id(db)
    
    if sku_id in db:
        return {"success": False, "message": f"SKU {sku_id} 已存在"}
    
    if not name:
        name = sku_id
    
    db[sku_id] = {
        "name": name,
        "images": [],
        "feature_mean": [],
        "image_count": 0,
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat()
    }
    
    (SKU_DIR / sku_id).mkdir(exist_ok=True)
    write_db(db)
    
    return {"success": True, "message": f"创建 SKU {sku_id} 成功", "sku_id": sku_id}


@router.put("/rename-sku")
def rename_sku(old_id: str, new_name: str):
    """重命名SKU"""
    db = read_db()
    
    if old_id not in db:
        return {"success": False, "message": f"SKU {old_id} 不存在"}
    
    db[old_id]["name"] = new_name
    write_db(db)
    
    return {"success": True, "message": f"SKU {old_id} 重命名为 {new_name}"}


@router.delete("/delete-sku/{sku_id}")
def delete_sku(sku_id: str):
    """删除SKU"""
    db = read_db()
    
    if sku_id not in db:
        return {"success": False, "message": f"SKU {sku_id} 不存在"}
    
    sku_subdir = SKU_DIR / sku_id
    if sku_subdir.exists():
        imgs = [f for f in sku_subdir.iterdir() if f.suffix.lower() in EXTS]
        if imgs:
            CANDIDATES_DIR.mkdir(exist_ok=True)
            for f in imgs:
                shutil.move(str(f), str(CANDIDATES_DIR / f.name))
    
    if sku_subdir.exists():
        shutil.rmtree(sku_subdir)
    
    del db[sku_id]
    write_db(db)
    
    return {"success": True, "message": f"已删除 SKU {sku_id}"}


@router.post("/save-database")
def save_database():
    """同步数据库与实际文件"""
    db = read_db()
    
    if "skus" in db:
        # 新结构
        for sku in db["skus"]:
            sid = sku.get("sku_id", "")
            if sid:
                sd = SKU_DIR / sid
                if sd.exists():
                    fs = [f for f in sd.iterdir() if f.suffix.lower() in EXTS]
                    sku["member_count"] = len(fs)
                    sku["members"] = [str(f) for f in sorted(fs)]
    else:
        # 旧结构
        for sid in list(db.keys()):
            sd = SKU_DIR / sid
            if sd.exists():
                cnt = len([f for f in sd.iterdir() if f.suffix.lower() in EXTS])
                db[sid]["image_count"] = cnt
    
    write_db(db)
    
    return {"success": True, "message": "数据库同步完成"}
