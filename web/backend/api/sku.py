"""
SKU管理API
"""

import csv
from typing import Optional, List
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from pydantic import BaseModel

from auth import get_current_user_required
from config import config
from database import get_db
from models.user import User
from models.sku import SKU

router = APIRouter(prefix="/api/skus", tags=["SKU管理"])


class SKUCsvResponse(BaseModel):
    success: bool
    skus: List[dict]
    total: int
    page: int
    page_size: int


class SKUDetail(BaseModel):
    sku_id: str
    sku_name: str
    image_count: int
    category: Optional[str]
    description: Optional[str]


@router.get("", response_model=SKUCsvResponse)
async def list_skus_from_csv(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    search: Optional[str] = None,
    current_user: User = Depends(get_current_user_required),
    db: Session = Depends(get_db)
):
    """从CSV文件获取SKU列表"""
    csv_path = config.paths.SKU_INDEX

    if not csv_path.exists():
        return SKUCsvResponse(success=True, skus=[], total=0, page=page, page_size=page_size)

    skus = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if search:
                sku_id = row.get('sku_id', '')
                sku_name = row.get('sku_name', '')
                label = row.get('label', '')
                if search.lower() not in sku_id.lower() and \
                   search.lower() not in sku_name.lower() and \
                   search.lower() not in label.lower():
                    continue
            skus.append({
                'sku_id': row.get('sku_id', ''),
                'sku_name': row.get('sku_name', ''),
                'label': row.get('label', ''),
                'image_name': row.get('image_name', ''),
                'category': row.get('category', '')
            })

    total = len(skus)
    start = (page - 1) * page_size
    end = start + page_size
    paginated_skus = skus[start:end]

    return SKUCsvResponse(
        success=True,
        skus=paginated_skus,
        total=total,
        page=page,
        page_size=page_size
    )


@router.get("/export")
async def export_skus_csv(
    current_user: User = Depends(get_current_user_required)
):
    """导出SKU列表为CSV"""
    csv_path = config.paths.SKU_INDEX

    if not csv_path.exists():
        raise HTTPException(status_code=404, detail="SKU库文件不存在")

    def iterfile():
        with open(csv_path, 'r', encoding='utf-8') as f:
            yield f.read()

    return StreamingResponse(
        iterfile(),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=sku_library.csv"}
    )


@router.get("/stats")
async def get_sku_stats(
    current_user: User = Depends(get_current_user_required)
):
    """获取SKU统计信息"""
    csv_path = config.paths.SKU_INDEX

    if not csv_path.exists():
        return {"success": True, "total_skus": 0, "total_images": 0}

    sku_ids = set()
    total_images = 0

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            sku_ids.add(row.get('sku_id', ''))
            total_images += 1

    return {
        "success": True,
        "total_skus": len(sku_ids),
        "total_images": total_images
    }


@router.get("/{sku_id}")
async def get_sku_detail(
    sku_id: str,
    current_user: User = Depends(get_current_user_required)
):
    """获取SKU详情"""
    csv_path = config.paths.SKU_INDEX

    if not csv_path.exists():
        raise HTTPException(status_code=404, detail="SKU库文件不存在")

    sku_items = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get('sku_id') == sku_id:
                sku_items.append(row)

    if not sku_items:
        raise HTTPException(status_code=404, detail="SKU不存在")

    return {
        "success": True,
        "sku_id": sku_id,
        "sku_name": sku_items[0].get('sku_name', ''),
        "category": sku_items[0].get('category', ''),
        "images": [
            {
                "label": item.get('label', ''),
                "image_name": item.get('image_name', '')
            }
            for item in sku_items
        ]
    }
