# backend/routes/api_keys.py
from fastapi import APIRouter, HTTPException, Query, Depends
from fastapi.responses import JSONResponse
from typing import List, Optional
from pydantic import BaseModel, Field
from services.api_key_service import APIKeyService
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import logging

router = APIRouter()
logger = logging.getLogger("api_keys_router")
security = HTTPBearer()


def get_admin_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """관리자 인증"""
    if credentials.credentials != "admin_token":
        raise HTTPException(status_code=401, detail="Invalid admin credentials")
    return {"role": "admin"}


def get_api_key_auth(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """API 키 인증"""
    if credentials.credentials == "admin_token":
        return {"role": "admin", "api_key_info": None}
    
    # API 키 검증
    api_key_info = APIKeyService.validate_api_key(credentials.credentials)
    if not api_key_info:
        raise HTTPException(status_code=401, detail="Invalid or expired API key")
    
    return {"role": "user", "api_key_info": api_key_info}


class APIKeyCreateRequest(BaseModel):
    """API 키 생성 요청 모델"""
    name: str = Field(..., min_length=1, max_length=255, description="API 키 이름")
    description: str = Field("", max_length=1000, description="API 키 설명")
    expires_days: Optional[int] = Field(None, ge=1, le=3650, description="만료일 (일 단위)")

    class Config:
        json_schema_extra = {
            "example": {
                "name": "프론트엔드 앱",
                "description": "사용자 인터페이스용 API 키",
                "expires_days": 365
            }
        }


class APIKeyUpdateRequest(BaseModel):
    """API 키 업데이트 요청 모델"""
    name: Optional[str] = Field(None, min_length=1, max_length=255, description="새로운 이름")
    description: Optional[str] = Field(None, max_length=1000, description="새로운 설명")
    is_active: Optional[bool] = Field(None, description="활성화 상태")

    class Config:
        json_schema_extra = {
            "example": {
                "name": "업데이트된 이름",
                "description": "업데이트된 설명",
                "is_active": True
            }
        }


# =============================================================================
# 관리자용 API 키 관리 엔드포인트
# =============================================================================

@router.get("/admin/api-keys/stats", summary="API 키 통계 조회 (관리자)")
async def get_api_key_stats_admin(admin=Depends(get_admin_user)):
    """관리자용 API 키 통계 조회"""
    try:
        stats = APIKeyService.get_api_key_stats()
        return JSONResponse(content={
            "status": "success",
            **stats
        })
    except Exception as e:
        logger.error(f"API 키 통계 조회 실패: {e}")
        raise HTTPException(status_code=500, detail=f"API 키 통계 조회 실패: {str(e)}")

@router.get("/admin/api-keys", summary="API 키 목록 조회 (관리자)")
async def get_api_keys_admin(
    include_inactive: bool = Query(False, description="비활성화된 키 포함 여부"),
    admin=Depends(get_admin_user)
):
    """관리자용 API 키 목록 조회"""
    try:
        api_keys = APIKeyService.get_api_keys(include_inactive=include_inactive)
        return JSONResponse(content={
            "status": "success",
            "api_keys": api_keys,
            "total": len(api_keys)
        })
    except Exception as e:
        logger.error(f"API 키 목록 조회 실패: {e}")
        raise HTTPException(status_code=500, detail=f"API 키 목록 조회 실패: {str(e)}")


@router.post("/admin/api-keys", summary="API 키 생성 (관리자)")
async def create_api_key_admin(
    request: APIKeyCreateRequest,
    admin=Depends(get_admin_user)
):
    """관리자용 API 키 생성"""
    try:
        result = APIKeyService.create_api_key(
            name=request.name,
            description=request.description,
            expires_days=request.expires_days
        )
        
        logger.info(f"API 키 생성 완료: {request.name}")
        return JSONResponse(
            status_code=201,
            content={
                "status": "success",
                "message": "API 키가 성공적으로 생성되었습니다",
                **result
            }
        )
    except Exception as e:
        logger.error(f"API 키 생성 실패: {e}")
        raise HTTPException(status_code=500, detail=f"API 키 생성 실패: {str(e)}")


@router.get("/admin/api-keys/{key_id}", summary="특정 API 키 조회 (관리자)")
async def get_api_key_admin(
    key_id: int,
    admin=Depends(get_admin_user)
):
    """관리자용 특정 API 키 조회"""
    try:
        api_key = APIKeyService.get_api_key_by_id(key_id)
        if not api_key:
            raise HTTPException(status_code=404, detail=f"API 키 ID {key_id}를 찾을 수 없습니다")
        
        return JSONResponse(content={
            "status": "success",
            "api_key": api_key
        })
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"API 키 조회 실패: {e}")
        raise HTTPException(status_code=500, detail=f"API 키 조회 실패: {str(e)}")


@router.put("/admin/api-keys/{key_id}", summary="API 키 업데이트 (관리자)")
async def update_api_key_admin(
    key_id: int,
    request: APIKeyUpdateRequest,
    admin=Depends(get_admin_user)
):
    """관리자용 API 키 업데이트"""
    try:
        # 업데이트할 내용이 있는지 확인
        if not any([request.name, request.description is not None, request.is_active is not None]):
            raise HTTPException(status_code=400, detail="업데이트할 내용이 없습니다")
        
        result = APIKeyService.update_api_key(
            key_id=key_id,
            name=request.name,
            description=request.description,
            is_active=request.is_active
        )
        
        if not result:
            raise HTTPException(status_code=404, detail=f"API 키 ID {key_id}를 찾을 수 없습니다")
        
        logger.info(f"API 키 업데이트 완료: ID {key_id}")
        return JSONResponse(content={
            "status": "success",
            "message": "API 키가 성공적으로 업데이트되었습니다",
            "api_key": result
        })
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"API 키 업데이트 실패: {e}")
        raise HTTPException(status_code=500, detail=f"API 키 업데이트 실패: {str(e)}")


@router.delete("/admin/api-keys/{key_id}", summary="API 키 삭제 (관리자)")
async def delete_api_key_admin(
    key_id: int,
    soft_delete: bool = Query(True, description="소프트 삭제 여부"),
    admin=Depends(get_admin_user)
):
    """관리자용 API 키 삭제"""
    try:
        success = APIKeyService.delete_api_key(key_id, soft_delete=soft_delete)
        if not success:
            raise HTTPException(status_code=404, detail=f"API 키 ID {key_id}를 찾을 수 없습니다")
        
        delete_type = "비활성화" if soft_delete else "삭제"
        logger.info(f"API 키 {delete_type} 완료: ID {key_id}")
        
        return JSONResponse(content={
            "status": "success",
            "message": f"API 키가 성공적으로 {delete_type}되었습니다",
            "deleted_id": key_id,
            "soft_delete": soft_delete
        })
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"API 키 삭제 실패: {e}")
        raise HTTPException(status_code=500, detail=f"API 키 삭제 실패: {str(e)}")

# =============================================================================
# 사용자용 API 키 검증 엔드포인트
# =============================================================================

@router.get("/user/verify", summary="API 키 검증")
async def verify_api_key(auth=Depends(get_api_key_auth)):
    """사용자 API 키 검증"""
    try:
        if auth["role"] == "admin":
            return JSONResponse(content={
                "status": "success",
                "message": "관리자 토큰으로 인증됨",
                "role": "admin",
                "api_key_info": None
            })
        else:
            return JSONResponse(content={
                "status": "success",
                "message": "API 키 인증 성공",
                "role": "user",
                "api_key_info": {
                    "id": auth["api_key_info"]["id"],
                    "name": auth["api_key_info"]["name"],
                    "description": auth["api_key_info"]["description"]
                }
            })
    except Exception as e:
        logger.error(f"API 키 검증 실패: {e}")
        raise HTTPException(status_code=500, detail=f"API 키 검증 실패: {str(e)}")


@router.get("/user/info", summary="사용자 API 키 정보 조회")
async def get_user_api_key_info(auth=Depends(get_api_key_auth)):
    """사용자 API 키 정보 조회"""
    try:
        if auth["role"] == "admin":
            return JSONResponse(content={
                "status": "success",
                "message": "관리자 권한",
                "role": "admin"
            })
        else:
            api_key_info = auth["api_key_info"]
            return JSONResponse(content={
                "status": "success",
                "role": "user",
                "api_key": {
                    "id": api_key_info["id"],
                    "name": api_key_info["name"],
                    "description": api_key_info["description"],
                    "created_at": api_key_info["created_at"],
                    "expires_at": api_key_info["expires_at"]
                }
            })
    except Exception as e:
        logger.error(f"사용자 정보 조회 실패: {e}")
        raise HTTPException(status_code=500, detail=f"사용자 정보 조회 실패: {str(e)}")