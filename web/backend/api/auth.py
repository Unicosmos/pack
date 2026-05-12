"""
认证API
"""

from datetime import timedelta
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from pydantic import BaseModel

from auth import (
    Token,
    UserCreate,
    UserResponse,
    authenticate_user,
    create_access_token,
    get_current_user_required,
    get_user_by_username,
    get_password_hash
)
from database import get_db
from models.user import User
from core.utils.logger import logger

router = APIRouter(prefix="/api/auth", tags=["认证"])

ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24


class LoginRequest(BaseModel):
    username: str
    password: str


@router.post("/login", response_model=Token)
async def login(login_data: LoginRequest, db: Session = Depends(get_db)):
    """用户登录"""
    logger.info(f"========== 登录请求 ==========")
    logger.info(f"用户名: {login_data.username}")
    logger.info(f"密码长度: {len(login_data.password)} 位")
    
    try:
        user = authenticate_user(db, login_data.username, login_data.password)
        logger.info(f"用户验证结果: {'成功' if user else '失败'}")
        
        if not user:
            logger.warning(f"登录失败: 用户名或密码错误")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="用户名或密码错误",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        if not user.is_active:
            logger.warning(f"登录失败: 用户已被禁用")
            raise HTTPException(status_code=400, detail="用户已被禁用")

        logger.info(f"用户信息: id={user.id}, username={user.username}, is_admin={user.is_admin}")
        
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            data={"sub": user.username}, expires_delta=access_token_expires
        )
        
        logger.info(f"登录成功: access_token已生成，有效期 {ACCESS_TOKEN_EXPIRE_MINUTES} 分钟")
        logger.info("========== 登录请求结束 ==========")
        
        return {"access_token": access_token, "token_type": "bearer"}
        
    except Exception as e:
        logger.error(f"登录异常: {str(e)}")
        raise


@router.post("/register", response_model=UserResponse)
async def register(user_data: UserCreate, db: Session = Depends(get_db)):
    """用户注册"""
    logger.info(f"========== 注册请求 ==========")
    logger.info(f"用户名: {user_data.username}")
    logger.info(f"邮箱: {user_data.email}")
    
    try:
        existing_user = get_user_by_username(db, user_data.username)
        if existing_user:
            logger.warning(f"注册失败: 用户名已存在")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="用户名已存在"
            )

        user = User(
            username=user_data.username,
            password_hash=get_password_hash(user_data.password),
            email=user_data.email,
            is_active=True,
            is_admin=False
        )
        db.add(user)
        db.commit()
        db.refresh(user)
        
        logger.info(f"注册成功: 用户ID={user.id}")
        logger.info("========== 注册请求结束 ==========")
        
        return user
        
    except Exception as e:
        logger.error(f"注册异常: {str(e)}")
        raise


@router.get("/me", response_model=UserResponse)
async def get_me(current_user: User = Depends(get_current_user_required)):
    """获取当前用户信息"""
    logger.info(f"========== 获取用户信息 ==========")
    logger.info(f"当前用户: id={current_user.id}, username={current_user.username}")
    logger.info("========== 获取用户信息结束 ==========")
    return current_user


@router.get("/check")
async def check_auth(current_user: User = Depends(get_current_user_required)):
    """检查认证状态"""
    logger.info(f"认证检查: 用户 {current_user.username} 已认证")
    return {"authenticated": True, "username": current_user.username}
