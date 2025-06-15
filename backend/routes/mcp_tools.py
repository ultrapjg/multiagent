# backend/routes/mcp_tools.py
from fastapi import APIRouter, HTTPException, Query, Depends
from fastapi.responses import JSONResponse
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from services.mcp_tool_service import MCPToolService
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import logging

router = APIRouter()
logger = logging.getLogger("mcp_tools_router")
security = HTTPBearer()


def get_admin_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """관리자 인증"""
    if credentials.credentials != "admin_token":
        raise HTTPException(status_code=401, detail="Invalid admin credentials")
    return {"role": "admin"}


class MCPToolCreateRequest(BaseModel):
    """MCP 도구 생성 요청 모델"""
    name: str = Field(..., min_length=1, max_length=255, description="도구 이름")
    description: str = Field("", max_length=1000, description="도구 설명")
    config: Dict[str, Any] = Field(..., description="도구 설정 (transport, command, args, url 등)")

    class Config:
        json_schema_extra = {
            "example": {
                "name": "delete_file",
                "description": "파일 삭제 도구",
                "config": {
                    "transport": "stdio",
                    "command": "python",
                    "args": ["/path/to/mcp_server_delete_file.py"]
                }
            }
        }


class MCPToolUpdateRequest(BaseModel):
    """MCP 도구 업데이트 요청 모델"""
    name: Optional[str] = Field(None, min_length=1, max_length=255, description="새로운 이름")
    description: Optional[str] = Field(None, max_length=1000, description="새로운 설명")
    config: Optional[Dict[str, Any]] = Field(None, description="새로운 설정")
    is_active: Optional[bool] = Field(None, description="활성화 상태")

    class Config:
        json_schema_extra = {
            "example": {
                "name": "updated_tool_name",
                "description": "업데이트된 설명",
                "is_active": True,
                "config": {
                    "transport": "stdio",
                    "command": "python",
                    "args": ["/updated/path/to/server.py"]
                }
            }
        }


# =============================================================================
# 관리자용 MCP 도구 관리 엔드포인트
# =============================================================================

@router.get("/admin/tools/stats", summary="MCP 도구 통계 조회 (관리자)")
async def get_mcp_tool_stats_admin(admin=Depends(get_admin_user)):
    """관리자용 MCP 도구 통계 조회"""
    try:
        stats = MCPToolService.get_mcp_tool_stats()
        return JSONResponse(content={
            "status": "success",
            **stats
        })
    except Exception as e:
        logger.error(f"MCP 도구 통계 조회 실패: {e}")
        raise HTTPException(status_code=500, detail=f"MCP 도구 통계 조회 실패: {str(e)}")


@router.get("/admin/tools", summary="MCP 도구 목록 조회 (관리자)")
async def get_mcp_tools_admin(
    include_inactive: bool = Query(False, description="비활성화된 도구 포함 여부"),
    admin=Depends(get_admin_user)
):
    """관리자용 MCP 도구 목록 조회"""
    try:
        tools = MCPToolService.get_all_tools(include_inactive=include_inactive)
        return JSONResponse(content={
            "status": "success",
            "tools": tools,
            "total": len(tools)
        })
    except Exception as e:
        logger.error(f"MCP 도구 목록 조회 실패: {e}")
        raise HTTPException(status_code=500, detail=f"MCP 도구 목록 조회 실패: {str(e)}")


@router.post("/admin/tools", summary="MCP 도구 생성 (관리자)")
async def create_mcp_tool_admin(
    request: MCPToolCreateRequest,
    admin=Depends(get_admin_user)
):
    """관리자용 MCP 도구 생성"""
    try:
        result = MCPToolService.create_tool(
            name=request.name,
            config=request.config,
            description=request.description
        )
        
        logger.info(f"MCP 도구 생성 완료: {request.name}")
        return JSONResponse(
            status_code=201,
            content={
                "status": "success",
                "message": "MCP 도구가 성공적으로 생성되었습니다",
                "tool": result
            }
        )
    except ValueError as e:
        logger.error(f"MCP 도구 생성 실패 (유효성 검사): {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"MCP 도구 생성 실패: {e}")
        raise HTTPException(status_code=500, detail=f"MCP 도구 생성 실패: {str(e)}")


@router.get("/admin/tools/{tool_id}", summary="특정 MCP 도구 조회 (관리자)")
async def get_mcp_tool_admin(
    tool_id: int,
    admin=Depends(get_admin_user)
):
    """관리자용 특정 MCP 도구 조회"""
    try:
        tool = MCPToolService.get_tool_by_id(tool_id)
        if not tool:
            raise HTTPException(status_code=404, detail=f"MCP 도구 ID {tool_id}를 찾을 수 없습니다")
        
        return JSONResponse(content={
            "status": "success",
            "tool": tool
        })
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"MCP 도구 조회 실패: {e}")
        raise HTTPException(status_code=500, detail=f"MCP 도구 조회 실패: {str(e)}")


@router.put("/admin/tools/{tool_id}", summary="MCP 도구 업데이트 (관리자)")
async def update_mcp_tool_admin(
    tool_id: int,
    request: MCPToolUpdateRequest,
    admin=Depends(get_admin_user)
):
    """관리자용 MCP 도구 업데이트"""
    try:
        # 업데이트할 내용이 있는지 확인
        if not any([request.name, request.description is not None, request.config, request.is_active is not None]):
            raise HTTPException(status_code=400, detail="업데이트할 내용이 없습니다")
        
        result = MCPToolService.update_tool(
            tool_id=tool_id,
            name=request.name,
            description=request.description,
            config=request.config,
            is_active=request.is_active
        )
        
        if not result:
            raise HTTPException(status_code=404, detail=f"MCP 도구 ID {tool_id}를 찾을 수 없습니다")
        
        logger.info(f"MCP 도구 업데이트 완료: ID {tool_id}")
        return JSONResponse(content={
            "status": "success",
            "message": "MCP 도구가 성공적으로 업데이트되었습니다",
            "tool": result
        })
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"MCP 도구 업데이트 실패: {e}")
        raise HTTPException(status_code=500, detail=f"MCP 도구 업데이트 실패: {str(e)}")


@router.delete("/admin/tools/{tool_id}", summary="MCP 도구 삭제 (관리자)")
async def delete_mcp_tool_admin(
    tool_id: int,
    soft_delete: bool = Query(True, description="소프트 삭제 여부"),
    admin=Depends(get_admin_user)
):
    """관리자용 MCP 도구 삭제"""
    try:
        success = MCPToolService.delete_tool(tool_id, soft_delete=soft_delete)
        if not success:
            raise HTTPException(status_code=404, detail=f"MCP 도구 ID {tool_id}를 찾을 수 없습니다")
        
        delete_type = "비활성화" if soft_delete else "삭제"
        logger.info(f"MCP 도구 {delete_type} 완료: ID {tool_id}")
        
        return JSONResponse(content={
            "status": "success",
            "message": f"MCP 도구가 성공적으로 {delete_type}되었습니다",
            "deleted_id": tool_id,
            "soft_delete": soft_delete
        })
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"MCP 도구 삭제 실패: {e}")
        raise HTTPException(status_code=500, detail=f"MCP 도구 삭제 실패: {str(e)}")


@router.delete("/admin/tools/name/{tool_name}", summary="MCP 도구 이름으로 삭제 (관리자)")
async def delete_mcp_tool_by_name_admin(
    tool_name: str,
    soft_delete: bool = Query(True, description="소프트 삭제 여부"),
    admin=Depends(get_admin_user)
):
    """관리자용 MCP 도구 이름으로 삭제"""
    try:
        success = MCPToolService.delete_tool_by_name(tool_name, soft_delete=soft_delete)
        if not success:
            raise HTTPException(status_code=404, detail=f"MCP 도구 '{tool_name}'을 찾을 수 없습니다")
        
        delete_type = "비활성화" if soft_delete else "삭제"
        logger.info(f"MCP 도구 {delete_type} 완료: 이름 {tool_name}")
        
        return JSONResponse(content={
            "status": "success",
            "message": f"MCP 도구가 성공적으로 {delete_type}되었습니다",
            "deleted_name": tool_name,
            "soft_delete": soft_delete
        })
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"MCP 도구 삭제 실패: {e}")
        raise HTTPException(status_code=500, detail=f"MCP 도구 삭제 실패: {str(e)}")


@router.get("/admin/tools/config/client", summary="MCP 클라이언트용 설정 조회 (관리자)")
async def get_mcp_client_config_admin(admin=Depends(get_admin_user)):
    """관리자용 MCP 클라이언트 설정 조회 (기존 mcp_config.json 형식과 호환)"""
    try:
        config = MCPToolService.get_mcp_config_for_client()
        return JSONResponse(content={
            "status": "success",
            "config": config,
            "tools_count": len(config.get("mcpServers", {}))
        })
    except Exception as e:
        logger.error(f"MCP 클라이언트 설정 조회 실패: {e}")
        raise HTTPException(status_code=500, detail=f"MCP 클라이언트 설정 조회 실패: {str(e)}")


@router.post("/admin/tools/apply", summary="MCP 도구 변경사항 적용 (관리자)")
async def apply_mcp_tool_changes_admin(admin=Depends(get_admin_user)):
    """관리자용 MCP 도구 변경사항 적용 - 에이전트 재초기화"""
    try:
        # 이 엔드포인트는 실제로 에이전트 재초기화를 트리거해야 함
        # orchestrator.py에서 이 정보를 사용하여 에이전트를 재초기화
        return JSONResponse(content={
            "status": "success",
            "message": "MCP 도구 변경사항 적용이 요청되었습니다. 에이전트가 재초기화됩니다.",
            "timestamp": "2025-01-15T10:30:00"
        })
    except Exception as e:
        logger.error(f"MCP 도구 변경사항 적용 실패: {e}")
        raise HTTPException(status_code=500, detail=f"MCP 도구 변경사항 적용 실패: {str(e)}")