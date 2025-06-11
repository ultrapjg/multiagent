# backend/routes/messages.py
from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from typing import List, Optional
from models import MessageCreate, MessageUpdate, MessageResponse, MessageStats, DeleteResponse
from services.message import MessageService
import logging
import time
from datetime import datetime

router = APIRouter()

# 로거 설정
logger = logging.getLogger("messages_router")


def log_request(request: Request, endpoint_name: str, **kwargs):
    """요청 로깅 함수"""
    client_ip = request.client.host if request.client else "unknown"
    user_agent = request.headers.get("user-agent", "unknown")

    log_data = {
        "endpoint": endpoint_name,
        "method": request.method,
        "ip": client_ip,
        "user_agent": user_agent,
        "timestamp": datetime.now().isoformat(),
        **kwargs
    }

    logger.info(f"📍 [{endpoint_name}] {request.method} request from {client_ip}")


def log_response(endpoint_name: str, status_code: int, execution_time: float, **kwargs):
    """응답 로깅 함수"""
    status_emoji = "✅" if status_code < 400 else "❌"
    logger.info(f"{status_emoji} [{endpoint_name}] Response {status_code} | Time: {execution_time * 1000:.2f}ms")


# 통계 조회
@router.get("/stats/summary", response_model=MessageStats, summary="메시지 통계")
async def get_message_stats_endpoint(request: Request):
    """메시지 보드의 전체 통계를 조회합니다."""
    start_time = time.time()
    endpoint_name = "get_message_stats"

    log_request(request, endpoint_name)

    try:
        # 서비스 레이어만 호출 - 비즈니스 로직 없음
        stats_data = MessageService.get_message_stats()

        execution_time = time.time() - start_time
        log_response(endpoint_name, 200, execution_time)

        return MessageStats(**stats_data)

    except Exception as e:
        execution_time = time.time() - start_time
        log_response(endpoint_name, 500, execution_time)
        raise HTTPException(status_code=500, detail=f"통계 조회 실패: {str(e)}")


# 메시지 목록 조회
@router.get("/list", response_model=List[MessageResponse], summary="메시지 목록 조회")
async def get_messages_endpoint(
        request: Request,
        limit: Optional[int] = Query(100, ge=1, le=1000, description="조회할 메시지 수"),
        offset: Optional[int] = Query(0, ge=0, description="건너뛸 메시지 수"),
        author: Optional[str] = Query(None, description="특정 작성자의 메시지만 조회")
):
    """메시지 목록을 시간 역순으로 조회합니다."""
    start_time = time.time()
    endpoint_name = "get_messages"

    log_request(request, endpoint_name, limit=limit, offset=offset, author_filter=author)

    try:
        # 서비스 레이어만 호출 - 비즈니스 로직 없음
        messages_data = MessageService.get_messages(limit, offset, author)
        messages = [MessageResponse(**msg) for msg in messages_data]

        execution_time = time.time() - start_time
        log_response(endpoint_name, 200, execution_time)

        return messages

    except Exception as e:
        execution_time = time.time() - start_time
        log_response(endpoint_name, 500, execution_time)
        raise HTTPException(status_code=500, detail=f"메시지 조회 실패: {str(e)}")


# 메시지 생성
@router.post("/create", response_model=MessageResponse, summary="메시지 생성")
async def create_message_endpoint(request: Request, message: MessageCreate):
    """새로운 메시지를 생성합니다."""
    start_time = time.time()
    endpoint_name = "create_message"

    log_request(request, endpoint_name, content_length=len(message.content), author=message.author)

    try:
        # 서비스 레이어만 호출 - 데이터베이스 로직 없음
        message_data = MessageService.create_message(message.content, message.author)

        execution_time = time.time() - start_time
        log_response(endpoint_name, 201, execution_time)

        return MessageResponse(**message_data)

    except Exception as e:
        execution_time = time.time() - start_time
        log_response(endpoint_name, 500, execution_time)
        raise HTTPException(status_code=500, detail=f"메시지 생성 실패: {str(e)}")


# 특정 메시지 조회
@router.get("/{message_id}", response_model=MessageResponse, summary="특정 메시지 조회")
async def get_message_endpoint(request: Request, message_id: int):
    """특정 ID의 메시지를 조회합니다."""
    start_time = time.time()
    endpoint_name = "get_message"

    log_request(request, endpoint_name, message_id=message_id)

    try:
        # 서비스 레이어만 호출 - 데이터베이스 로직 없음
        message_data = MessageService.get_message_by_id(message_id)

        if not message_data:
            execution_time = time.time() - start_time
            log_response(endpoint_name, 404, execution_time)
            raise HTTPException(status_code=404, detail=f"메시지 ID {message_id}를 찾을 수 없습니다")

        execution_time = time.time() - start_time
        log_response(endpoint_name, 200, execution_time)

        return MessageResponse(**message_data)

    except HTTPException:
        raise
    except Exception as e:
        execution_time = time.time() - start_time
        log_response(endpoint_name, 500, execution_time)
        raise HTTPException(status_code=500, detail=f"메시지 조회 실패: {str(e)}")


# 메시지 수정
@router.put("/{message_id}", response_model=MessageResponse, summary="메시지 수정")
async def update_message_endpoint(request: Request, message_id: int, message_update: MessageUpdate):
    """특정 ID의 메시지를 수정합니다."""
    start_time = time.time()
    endpoint_name = "update_message"

    log_request(request, endpoint_name, message_id=message_id)

    try:
        # HTTP 레벨 입력 검증만 처리
        if not message_update.content and not message_update.author:
            raise HTTPException(status_code=400, detail="수정할 내용이 없습니다")

        # 서비스 레이어만 호출 - 비즈니스 로직 없음
        updated_message = MessageService.update_message(
            message_id,
            message_update.content,
            message_update.author
        )

        if not updated_message:
            execution_time = time.time() - start_time
            log_response(endpoint_name, 404, execution_time)
            raise HTTPException(status_code=404, detail=f"메시지 ID {message_id}를 찾을 수 없습니다")

        execution_time = time.time() - start_time
        log_response(endpoint_name, 200, execution_time)

        return MessageResponse(**updated_message)

    except HTTPException:
        raise
    except Exception as e:
        execution_time = time.time() - start_time
        log_response(endpoint_name, 500, execution_time)
        raise HTTPException(status_code=500, detail=f"메시지 수정 실패: {str(e)}")


# 메시지 삭제
@router.delete("/{message_id}", response_model=DeleteResponse, summary="메시지 삭제")
async def delete_message_endpoint(request: Request, message_id: int):
    """특정 ID의 메시지를 삭제합니다."""
    start_time = time.time()
    endpoint_name = "delete_message"

    log_request(request, endpoint_name, message_id=message_id)

    try:
        # 서비스 레이어만 호출 - 데이터베이스 로직 없음
        deleted = MessageService.delete_message(message_id)

        if not deleted:
            execution_time = time.time() - start_time
            log_response(endpoint_name, 404, execution_time)
            raise HTTPException(status_code=404, detail=f"메시지 ID {message_id}를 찾을 수 없습니다")

        execution_time = time.time() - start_time
        log_response(endpoint_name, 200, execution_time)

        return DeleteResponse(
            message="메시지가 성공적으로 삭제되었습니다",
            deleted_id=message_id
        )

    except HTTPException:
        raise
    except Exception as e:
        execution_time = time.time() - start_time
        log_response(endpoint_name, 500, execution_time)
        raise HTTPException(status_code=500, detail=f"메시지 삭제 실패: {str(e)}")


# 모든 메시지 삭제
@router.delete("/deleteAll", summary="모든 메시지 삭제 (주의!)")
async def delete_all_messages_endpoint(request: Request):
    """모든 메시지를 삭제합니다. 이 작업은 되돌릴 수 없습니다."""
    start_time = time.time()
    endpoint_name = "delete_all_messages"

    log_request(request, endpoint_name)

    try:
        # 서비스 레이어만 호출 - 데이터베이스 로직 없음
        deleted_count = MessageService.delete_all_messages()

        execution_time = time.time() - start_time
        log_response(endpoint_name, 200, execution_time)

        return JSONResponse(
            content={
                "message": f"모든 메시지가 삭제되었습니다",
                "deleted_count": deleted_count
            }
        )

    except Exception as e:
        execution_time = time.time() - start_time
        log_response(endpoint_name, 500, execution_time)
        raise HTTPException(status_code=500, detail=f"전체 삭제 실패: {str(e)}")