import subprocess
import json
import uuid
import time
from typing import Dict, Any
from fastapi import APIRouter, BackgroundTasks

from config import config

router = APIRouter(prefix="/api/build", tags=["建库任务"])

_build_status: Dict[str, Any] = {
    "status": "idle",
    "message": "",
    "result": None
}

_feature_status: Dict[str, Any] = {
    "status": "idle",
    "message": "",
    "result": None
}

# 合并任务状态（支持持久化）
_build_and_extract_status: Dict[str, Any] = {
    "task_id": "",
    "status": "idle",
    "step": 0,
    "message": "",
    "output": "",
    "result": None
}


def _count_sku_output_images() -> int:
    """统计 sku_output 目录下的图片总数"""
    count = 0
    sku_output_dir = config.paths.SKU_OUTPUT_DIR
    if sku_output_dir.exists():
        images_dir = sku_output_dir / "images"
        if images_dir.exists():
            for sku_folder in images_dir.iterdir():
                if sku_folder.is_dir():
                    count += len([f for f in sku_folder.iterdir() if f.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}])
    return count


def _save_status():
    """保存任务状态到文件"""
    try:
        status_file = config.paths.STATUS_FILE
        status_file.parent.mkdir(parents=True, exist_ok=True)
        with open(status_file, "w", encoding="utf-8") as f:
            json.dump(_build_and_extract_status, f)
    except Exception as e:
        print(f"[ERROR] 保存状态失败: {e}")


def _load_status():
    """从文件加载任务状态"""
    global _build_and_extract_status
    status_file = config.paths.STATUS_FILE
    if status_file.exists():
        try:
            with open(status_file, "r", encoding="utf-8") as f:
                saved = json.load(f)
                _build_and_extract_status.update(saved)
            if _build_and_extract_status.get("status") == "running":
                start_time = _build_and_extract_status.get("started_at", 0)
                if time.time() - start_time > 3600:
                    _build_and_extract_status["status"] = "failed"
                    _build_and_extract_status["message"] = "任务超时"
                    _save_status()
        except Exception as e:
            print(f"[ERROR] 加载状态失败: {e}")


_load_status()


def run_build():
    global _build_status
    _build_status = {"status": "running", "message": "建库任务执行中...", "result": None}

    try:
        cmd = [
            "python",
            str(config.paths.BUILD_SCRIPT),
            "--input", str(config.paths.SKU_OUTPUT_DIR),
            "--output", str(config.paths.SKU_DIR),
            "--seed", "0"
        ]

        import os
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600, env=env)

        if result.returncode == 0:
            _build_status["status"] = "completed"
            _build_status["message"] = "建库完成"
            _build_status["result"] = {
                "output": result.stdout,
                "returncode": 0
            }

            meta_path = config.paths.SKU_DIR / "meta.json"
            if meta_path.exists():
                with open(meta_path, "r", encoding="utf-8") as f:
                    _build_status["result"]["meta"] = json.load(f)
        else:
            _build_status["status"] = "failed"
            _build_status["message"] = "建库失败"
            _build_status["result"] = {
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode
            }
    except subprocess.TimeoutExpired:
        _build_status = {"status": "failed", "message": "建库超时", "result": None}
    except Exception as e:
        _build_status = {"status": "failed", "message": str(e), "result": None}


def run_extract_feature():
    global _feature_status
    _feature_status = {"status": "running", "message": "特征提取执行中...", "result": None}

    try:
        cmd = [
            "python",
            str(config.paths.EXTRACT_SCRIPT),
            "--input", str(config.paths.SKU_DIR),
            "--csv", "sku_library.csv",
            "--weights", str(config.paths.SKU_MODEL_PATH),
            "--batch-size", "16",
            "--device", "cpu"
        ]

        if not (config.paths.SKU_DIR / "sku_library.csv").exists():
            raise FileNotFoundError(f"找不到 sku_library.csv，请先执行建库")

        import os
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"

        print(f"[DEBUG] 执行命令: {' '.join(cmd)}")

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800, env=env)

        print(f"[DEBUG] 返回码: {result.returncode}")
        print(f"[DEBUG] stdout: {result.stdout}")
        print(f"[DEBUG] stderr: {result.stderr}")

        if result.returncode == 0:
            _feature_status["status"] = "completed"
            _feature_status["message"] = "特征提取完成"
            _feature_status["result"] = {
                "output": result.stdout,
                "stderr": result.stderr,
                "returncode": 0
            }

            meta_path = config.paths.SKU_DIR / "feature_meta.json"
            if meta_path.exists():
                with open(meta_path, "r", encoding="utf-8") as f:
                    _feature_status["result"]["meta"] = json.load(f)
        else:
            _feature_status["status"] = "failed"
            _feature_status["message"] = f"特征提取失败: {result.stderr or '未知错误'}"
            _feature_status["result"] = {
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode
            }
    except subprocess.TimeoutExpired:
        _feature_status = {"status": "failed", "message": "特征提取超时", "result": None}
    except Exception as e:
        print(f"[ERROR] 特征提取异常: {str(e)}")
        _feature_status = {"status": "failed", "message": str(e), "result": None}


def _update_output(output: str):
    """更新输出缓冲"""
    _build_and_extract_status["output"] = output


def run_build_and_extract():
    global _build_and_extract_status

    _build_and_extract_status = {
        "task_id": str(uuid.uuid4()),
        "status": "running",
        "step": 1,
        "started_at": time.time(),
        "message": "正在执行建库...",
        "output": "",
        "result": None
    }
    _save_status()

    try:
        _build_and_extract_status["message"] = "正在执行建库（图片增强）..."

        cmd_build = [
            "python",
            str(config.paths.BUILD_SCRIPT),
            "--input", str(config.paths.SKU_OUTPUT_DIR),
            "--output", str(config.paths.SKU_DIR),
            "--seed", "0"
        ]

        import os
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"

        _update_output(f"[命令] {' '.join(cmd_build)}\n\n")

        process = subprocess.Popen(cmd_build, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   text=True, env=env)

        stdout, stderr = process.communicate()
        stdout = stdout or ""
        stderr = stderr or ""
        build_output = stdout + ("\n[ERROR]\n" + stderr if stderr else "")
        _update_output(build_output)

        if process.returncode != 0:
            _build_and_extract_status["status"] = "failed"
            _build_and_extract_status["message"] = f"建库失败: {stderr or '未知错误'}"
            _build_and_extract_status["result"] = {"step": 1, "returncode": process.returncode}
            _save_status()
            return

        _build_and_extract_status["step"] = 2
        _build_and_extract_status["message"] = "正在执行特征提取..."
        _save_status()

        cmd_extract = [
            "python",
            str(config.paths.EXTRACT_SCRIPT),
            "--input", str(config.paths.SKU_DIR),
            "--csv", "sku_library.csv",
            "--weights", str(config.paths.SKU_MODEL_PATH),
            "--batch-size", "16",
            "--device", "cpu"
        ]

        _update_output(build_output + f"\n\n[步骤 2/2] {' '.join(cmd_extract)}\n\n")

        process = subprocess.Popen(cmd_extract, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   text=True, env=env)

        stdout, stderr = process.communicate()
        stdout = stdout or ""
        stderr = stderr or ""
        extract_output = stdout + ("\n[ERROR]\n" + stderr if stderr else "")
        _update_output(build_output + extract_output)

        if process.returncode != 0:
            _build_and_extract_status["status"] = "failed"
            _build_and_extract_status["message"] = f"特征提取失败: {stderr or '未知错误'}"
            _build_and_extract_status["result"] = {"step": 2, "returncode": process.returncode}
            _save_status()
            return

        _build_and_extract_status["status"] = "completed"
        _build_and_extract_status["step"] = 3
        _build_and_extract_status["message"] = "全部完成！"
        _save_status()

        meta_data = {}
        meta_path = config.paths.SKU_DIR / "meta.json"
        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                meta_data["library"] = json.load(f)

        feature_meta_path = config.paths.SKU_DIR / "feature_meta.json"
        if feature_meta_path.exists():
            with open(feature_meta_path, "r", encoding="utf-8") as f:
                meta_data["features"] = json.load(f)

        _build_and_extract_status["result"] = {"step": 3, "meta": meta_data}

    except subprocess.TimeoutExpired:
        step_msg = "建库" if _build_and_extract_status["step"] == 1 else "特征提取"
        _build_and_extract_status["status"] = "failed"
        _build_and_extract_status["message"] = f"{step_msg}超时"
        _save_status()
    except Exception as e:
        print(f"[ERROR] 合并任务异常: {str(e)}")
        _build_and_extract_status["status"] = "failed"
        _build_and_extract_status["message"] = str(e)
        _save_status()


@router.get("/status")
async def get_status():
    return {"success": True, **_build_status}


@router.post("/library")
async def build_library(background_tasks: BackgroundTasks):
    if _build_status["status"] == "running":
        return {"success": False, "message": "建库任务正在进行中"}

    background_tasks.add_task(run_build)
    return {"success": True, "message": "建库任务已启动"}


@router.get("/feature/status")
async def get_feature_status():
    return {"success": True, **_feature_status}


@router.post("/feature/extract")
async def extract_feature(background_tasks: BackgroundTasks):
    if _feature_status["status"] == "running":
        return {"success": False, "message": "特征提取任务正在进行中"}

    background_tasks.add_task(run_extract_feature)
    return {"success": True, "message": "特征提取任务已启动"}


@router.get("/check-change")
async def check_change():
    return {
        "success": True,
        "image_count": _count_sku_output_images()
    }


@router.get("/combined/status")
async def get_combined_status():
    return {"success": True, **_build_and_extract_status}


@router.post("/combined/run")
async def run_combined(background_tasks: BackgroundTasks):
    if (_build_status["status"] == "running" or
        _feature_status["status"] == "running" or
        _build_and_extract_status["status"] == "running"):
        return {"success": False, "message": "有任务正在进行中，请稍候"}

    background_tasks.add_task(run_build_and_extract)
    return {"success": True, "message": "已启动完整建库流程"}


@router.get("/info")
async def get_info():
    sku_output_dir = config.paths.SKU_OUTPUT_DIR
    sku_dir = config.paths.SKU_DIR

    info = {
        "sku_output": {
            "exists": sku_output_dir.exists(),
            "path": str(sku_output_dir),
            "has_images": (sku_output_dir / "images").exists() if sku_output_dir.exists() else False,
            "has_database": (sku_output_dir / "sku_database.json").exists() if sku_output_dir.exists() else False
        },
        "sku_library": {
            "exists": sku_dir.exists(),
            "path": str(sku_dir),
            "has_images": (sku_dir / "images").exists() if sku_dir.exists() else False,
            "has_csv": (sku_dir / "sku_library.csv").exists() if sku_dir.exists() else False,
            "has_meta": (sku_dir / "meta.json").exists() if sku_dir.exists() else False,
            "has_features": (sku_dir / "sku_features.npy").exists() if sku_dir.exists() else False
        }
    }

    db_path = sku_output_dir / "sku_database.json"
    if db_path.exists():
        with open(db_path, "r", encoding="utf-8") as f:
            db = json.load(f)
            info["sku_output"]["sku_count"] = len(db)

    meta_path = sku_dir / "meta.json"
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            info["sku_library"]["meta"] = json.load(f)

    return {"success": True, "data": info}