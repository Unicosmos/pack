"""
SKU管理API
支持完整的CRUD操作和图片管理
"""

import csv
import io
from typing import Optional, List
from pathlib import Path
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Query, UploadFile, File
from fastapi.responses import StreamingResponse, JSONResponse
from sqlalchemy.orm import Session
from sqlalchemy import or_, func
from pydantic import BaseModel, validator

from config import config
from database import get_db
from models.sku import SKU

router = APIRouter(prefix="/api/skus", tags=["SKU管理"])


class SKUCreate(BaseModel):
    sku_id: str
    sku_name: str
    description: Optional[str] = None
    category: Optional[str] = None
    tags: Optional[str] = None

    @validator('sku_id')
    def validate_sku_id(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError('SKU编号不能为空')
        if len(v) > 50:
            raise ValueError('SKU编号长度不能超过50')
        return v.strip()

    @validator('sku_name')
    def validate_sku_name(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError('SKU名称不能为空')
        if len(v) > 200:
            raise ValueError('SKU名称长度不能超过200')
        return v.strip()


class SKUUpdate(BaseModel):
    sku_name: Optional[str] = None
    description: Optional[str] = None
    category: Optional[str] = None
    tags: Optional[str] = None
    is_deleted: Optional[bool] = None
    status: Optional[str] = None


class SKUResponse(BaseModel):
    id: int
    sku_id: str
    sku_name: str
    description: Optional[str]
    category: Optional[str]
    is_deleted: bool
    status: str
    image_count: int
    tags: Optional[str]
    created_at: Optional[str]
    updated_at: Optional[str]


class SKUListResponse(BaseModel):
    success: bool
    skus: List[SKUResponse]
    total: int
    page: int
    page_size: int


class SKUStatsResponse(BaseModel):
    success: bool
    total_skus: int
    active_skus: int
    inactive_skus: int
    total_images: int


class MessageResponse(BaseModel):
    success: bool
    message: str


def _sku_to_response(sku: SKU) -> SKUResponse:
    return SKUResponse(
        id=sku.id,
        sku_id=sku.sku_id,
        sku_name=sku.sku_name,
        description=sku.description,
        category=sku.category,
        is_deleted=sku.is_deleted,
        status='inactive' if sku.is_deleted else 'active',
        image_count=sku.image_count,
        tags=sku.tags,
        created_at=sku.created_at.isoformat() if sku.created_at else None,
        updated_at=sku.updated_at.isoformat() if sku.updated_at else None,
    )


@router.get("", response_model=SKUListResponse)
async def list_skus(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=10000),
    search: Optional[str] = None,
    category: Optional[str] = None,
    status: Optional[str] = None,
    is_deleted: Optional[bool] = None,
    db: Session = Depends(get_db)
):
    """获取SKU列表（支持分页、搜索、筛选）"""
    query = db.query(SKU)

    if search:
        search_pattern = f"%{search}%"
        query = query.filter(
            or_(
                SKU.sku_id.ilike(search_pattern),
                SKU.sku_name.ilike(search_pattern),
                SKU.tags.ilike(search_pattern)
            )
        )

    if category:
        query = query.filter(SKU.category == category)

    if status is not None:
        # 兼容status参数：active=未删除，inactive=已删除
        if status == 'active':
            query = query.filter(SKU.is_deleted == False)
        elif status == 'inactive':
            query = query.filter(SKU.is_deleted == True)
    elif is_deleted is not None:
        query = query.filter(SKU.is_deleted == is_deleted)

    total = query.count()
    skus = query.order_by(SKU.created_at.desc()).offset((page - 1) * page_size).limit(page_size).all()

    return SKUListResponse(
        success=True,
        skus=[_sku_to_response(s) for s in skus],
        total=total,
        page=page,
        page_size=page_size
    )


@router.get("/categories")
async def list_categories(
    db: Session = Depends(get_db)
):
    """获取所有分类列表"""
    categories = db.query(SKU.category).filter(
        SKU.is_deleted == False,
        SKU.category.isnot(None),
        SKU.category != ""
    ).distinct().all()

    return {
        "success": True,
        "categories": [c[0] for c in categories if c[0]]
    }


@router.get("/stats", response_model=SKUStatsResponse)
async def get_sku_stats(
    db: Session = Depends(get_db)
):
    """获取SKU统计信息"""
    total = db.query(SKU).count()
    active = db.query(SKU).filter(SKU.is_deleted == False).count()
    inactive = db.query(SKU).filter(SKU.is_deleted == True).count()
    total_images = db.query(func.sum(SKU.image_count)).scalar() or 0

    return SKUStatsResponse(
        success=True,
        total_skus=total,
        active_skus=active,
        inactive_skus=inactive,
        total_images=total_images
    )


@router.get("/{sku_id}", response_model=SKUResponse)
async def get_sku(
    sku_id: str,
    db: Session = Depends(get_db)
):
    """获取单个SKU详情"""
    sku = db.query(SKU).filter(
        SKU.sku_id == sku_id
    ).first()

    if not sku:
        raise HTTPException(status_code=404, detail="SKU不存在")

    return _sku_to_response(sku)


@router.post("", response_model=SKUResponse)
async def create_sku(
    sku_data: SKUCreate,
    db: Session = Depends(get_db)
):
    """创建新SKU"""
    existing = db.query(SKU).filter(
        SKU.sku_id == sku_data.sku_id
    ).first()

    if existing:
        raise HTTPException(status_code=400, detail="SKU编号已存在")

    sku = SKU(
        sku_id=sku_data.sku_id,
        sku_name=sku_data.sku_name,
        description=sku_data.description,
        category=sku_data.category,
        tags=sku_data.tags,
        created_by=1,
        is_deleted=False
    )

    db.add(sku)
    db.commit()
    db.refresh(sku)

    return _sku_to_response(sku)


@router.put("/{sku_id}", response_model=SKUResponse)
async def update_sku(
    sku_id: str,
    sku_data: SKUUpdate,
    db: Session = Depends(get_db)
):
    """更新SKU信息"""
    sku = db.query(SKU).filter(
        SKU.sku_id == sku_id
    ).first()

    if not sku:
        raise HTTPException(status_code=404, detail="SKU不存在")

    update_data = sku_data.dict(exclude_unset=True)
    
    if 'status' in update_data:
        sku.is_deleted = update_data['status'] == 'inactive'
        del update_data['status']
    
    for field, value in update_data.items():
        setattr(sku, field, value)

    db.commit()
    db.refresh(sku)

    return _sku_to_response(sku)


@router.delete("/{sku_id}", response_model=MessageResponse)
async def delete_sku(
    sku_id: str,
    db: Session = Depends(get_db)
):
    """禁用SKU"""
    sku = db.query(SKU).filter(
        SKU.sku_id == sku_id
    ).first()

    if not sku:
        raise HTTPException(status_code=404, detail="SKU不存在")

    sku.is_deleted = True
    db.commit()

    return MessageResponse(success=True, message=f"SKU {sku_id} 已删除")


@router.post("/batch-delete", response_model=MessageResponse)
async def batch_delete_skus(
    sku_ids: List[str],
    db: Session = Depends(get_db)
):
    """批量禁用SKU"""
    if len(sku_ids) > 100:
        raise HTTPException(status_code=400, detail="单次最多禁用100个SKU")

    count = db.query(SKU).filter(
        SKU.sku_id.in_(sku_ids)
    ).update({SKU.is_deleted: True}, synchronize_session=False)

    db.commit()

    return MessageResponse(success=True, message=f"已禁用 {count} 个SKU")


@router.post("/import")
async def import_skus_csv(
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """从CSV文件导入SKU"""
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="只支持CSV文件")

    content = await file.read()
    try:
        text = content.decode('utf-8')
    except UnicodeDecodeError:
        text = content.decode('gbk')

    reader = csv.DictReader(io.StringIO(text))
    rows = list(reader)

    if not rows:
        raise HTTPException(status_code=400, detail="CSV文件为空")

    imported = 0
    updated = 0
    errors = []

    for i, row in enumerate(rows):
        try:
            sku_id = row.get('sku_id', '').strip()
            sku_name = row.get('sku_name', '').strip()

            if not sku_id or not sku_name:
                errors.append(f"第{i+2}行: sku_id或sku_name为空")
                continue

            existing = db.query(SKU).filter(
                SKU.sku_id == sku_id
            ).first()

            if existing:
                existing.sku_name = sku_name
                existing.description = row.get('description', existing.description)
                existing.category = row.get('category', existing.category)
                existing.tags = row.get('tags', existing.tags)
                existing.is_deleted = False
                updated += 1
            else:
                sku = SKU(
                    sku_id=sku_id,
                    sku_name=sku_name,
                    description=row.get('description'),
                    category=row.get('category'),
                    tags=row.get('tags'),
                    created_by=1,
                    is_deleted=False
                )
                db.add(sku)
                imported += 1

        except Exception as e:
            errors.append(f"第{i+2}行: {str(e)}")

    db.commit()

    return {
        "success": True,
        "message": f"导入完成: 新增 {imported} 个, 更新 {updated} 个",
        "errors": errors[:10] if errors else []
    }


@router.get("/export/download")
async def export_skus_csv(
    db: Session = Depends(get_db)
):
    """导出SKU为CSV"""
    skus = db.query(SKU).filter(SKU.is_deleted == False).order_by(SKU.sku_id).all()

    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=[
        'sku_id', 'sku_name', 'description', 'category', 'status', 'image_count', 'tags', 'created_at', 'updated_at'
    ])
    writer.writeheader()

    for sku in skus:
        writer.writerow({
            'sku_id': sku.sku_id,
            'sku_name': sku.sku_name,
            'description': sku.description or '',
            'category': sku.category or '',
            'status': sku.status,
            'image_count': sku.image_count,
            'tags': sku.tags or '',
            'created_at': sku.created_at.isoformat() if sku.created_at else '',
            'updated_at': sku.updated_at.isoformat() if sku.updated_at else '',
        })

    output.seek(0)
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=sku_export.csv"}
    )


@router.post("/sync-from-csv")
async def sync_from_sku_csv(
    db: Session = Depends(get_db)
):
    """从现有的sku_library.csv同步SKU到数据库"""
    csv_path = config.paths.SKU_INDEX

    if not csv_path.exists():
        raise HTTPException(status_code=404, detail="SKU索引文件不存在")

    sku_map = {}
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            sku_id = row.get('sku_id', '')
            if sku_id and sku_id not in sku_map:
                sku_map[sku_id] = {
                    'sku_name': row.get('sku_name', sku_id),
                    'category': row.get('category', ''),
                    'label': row.get('label', ''),
                }

    imported = 0
    for sku_id, info in sku_map.items():
        existing = db.query(SKU).filter(
            SKU.sku_id == sku_id
        ).first()

        if not existing:
            sku = SKU(
                sku_id=sku_id,
                sku_name=info['sku_name'],
                category=info.get('category'),
                created_by=1,
                is_deleted=False
            )
            db.add(sku)
            imported += 1
        else:
            existing.is_deleted = False

    db.commit()

    return {
        "success": True,
        "message": f"已从sku_library.csv同步 {imported} 个新SKU到数据库"
    }


@router.get("/{sku_id}/images")
async def get_sku_images(
    sku_id: str,
    db: Session = Depends(get_db)
):
    """获取SKU的所有图片"""
    sku = db.query(SKU).filter(
        SKU.sku_id == sku_id,
        SKU.is_deleted == False
    ).first()

    if not sku:
        raise HTTPException(status_code=404, detail="SKU不存在")

    sku_images_dir = config.paths.SKU_IMAGES_DIR / sku_id
    images = []
    
    if sku_images_dir.exists():
        exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
        for img_path in sorted(sku_images_dir.iterdir()):
            if img_path.suffix.lower() in exts:
                images.append({
                    'url': f"/static/sku_images/{sku_id}/{img_path.name}",
                    'filename': img_path.name
                })

    return {
        "success": True,
        "sku_id": sku_id,
        "images": images
    }


@router.post("/{sku_id}/images/upload")
async def upload_sku_images(
    sku_id: str,
    files: List[UploadFile] = File(...),
    db: Session = Depends(get_db)
):
    """上传SKU图片"""
    sku = db.query(SKU).filter(
        SKU.sku_id == sku_id,
        SKU.is_deleted == False
    ).first()

    if not sku:
        raise HTTPException(status_code=404, detail="SKU不存在")

    sku_images_dir = config.paths.SKU_IMAGES_DIR / sku_id
    sku_images_dir.mkdir(parents=True, exist_ok=True)

    uploaded_count = 0
    allowed_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    
    for file in files:
        ext = Path(file.filename).suffix.lower()
        if ext not in allowed_exts:
            continue
        
        file_path = sku_images_dir / file.filename
        contents = await file.read()
        file_path.write_bytes(contents)
        uploaded_count += 1

    sku.image_count = len(list(sku_images_dir.iterdir()))
    db.commit()

    return {
        "success": True,
        "message": f"成功上传 {uploaded_count} 张图片",
        "uploaded_count": uploaded_count
    }


@router.delete("/{sku_id}/images/{filename}")
async def delete_sku_image(
    sku_id: str,
    filename: str,
    db: Session = Depends(get_db)
):
    """删除SKU的单张图片"""
    sku = db.query(SKU).filter(
        SKU.sku_id == sku_id,
        SKU.is_deleted == False
    ).first()

    if not sku:
        raise HTTPException(status_code=404, detail="SKU不存在")

    file_path = config.paths.SKU_IMAGES_DIR / sku_id / filename
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="图片不存在")

    file_path.unlink()

    sku_images_dir = config.paths.SKU_IMAGES_DIR / sku_id
    sku.image_count = len(list(sku_images_dir.iterdir())) if sku_images_dir.exists() else 0
    db.commit()

    return {
        "success": True,
        "message": "图片已删除"
    }


@router.get("/{sku_id}/list-images")
async def list_sku_images(
    sku_id: str,
    db: Session = Depends(get_db)
):
    """获取SKU的图片列表（简化版，用于画廊显示）"""
    sku = db.query(SKU).filter(
        SKU.sku_id == sku_id,
        SKU.is_deleted == False
    ).first()

    if not sku:
        raise HTTPException(status_code=404, detail="SKU不存在")

    sku_images_dir = config.paths.SKU_IMAGES_DIR / sku_id
    images = []
    
    if sku_images_dir.exists():
        exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
        for img_path in sorted(sku_images_dir.iterdir())[:4]:
            if img_path.suffix.lower() in exts:
                images.append(f"/static/sku_images/{sku_id}/{img_path.name}")

    return {
        "success": True,
        "sku_id": sku_id,
        "images": images,
        "total": sku.image_count
    }
