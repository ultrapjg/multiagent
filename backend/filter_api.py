from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from core.input_filter import InputFilter, init_filter_db, test_filter_connection
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from psycopg2.extras import RealDictCursor
from database import get_db_connection
import logging
import time
from datetime import datetime

router = APIRouter()
logger = logging.getLogger("filter_api")


class FilterRule(BaseModel):
    """필터 규칙 모델"""
    id: Optional[int] = None
    name: str = Field(..., min_length=1, max_length=255, description="규칙 이름")
    pattern: str = Field(..., min_length=1, description="정규식 패턴")
    created_at: Optional[str] = None
    updated_at: Optional[str] = None

    class Config:
        json_schema_extra = {
            "example": {
                "name": "이메일 패턴",
                "pattern": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
            }
        }


class FilterTestRequest(BaseModel):
    """필터 테스트 요청 모델"""
    text: str = Field(..., min_length=1, description="테스트할 텍스트")

    class Config:
        json_schema_extra = {
            "example": {
                "text": "제 이메일은 test@example.com 입니다."
            }
        }


class FilterTestResponse(BaseModel):
    """필터 테스트 응답 모델"""
    is_sensitive: bool = Field(..., description="민감한 내용 포함 여부")
    matched_rules: List[Dict[str, Any]] = Field(..., description="매칭된 규칙들")
    message: str = Field(..., description="결과 메시지")
    test_text: str = Field(..., description="테스트한 텍스트")


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


@router.get("/filters", response_model=List[FilterRule], summary="필터 규칙 목록 조회")
async def get_filters(request: Request):
    """모든 필터 규칙을 조회합니다."""
    start_time = time.time()
    endpoint_name = "get_filters"
    
    log_request(request, endpoint_name)
    
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            cursor.execute("""
                SELECT id, name, pattern, created_at, updated_at 
                FROM filter_rules 
                ORDER BY id
            """)
            rows = cursor.fetchall()
            
            rules = []
            for row in rows:
                rules.append(FilterRule(
                    id=row['id'],
                    name=row['name'],
                    pattern=row['pattern'],
                    created_at=row['created_at'].isoformat() if row['created_at'] else None,
                    updated_at=row['updated_at'].isoformat() if row['updated_at'] else None
                ))
        
        execution_time = time.time() - start_time
        log_response(endpoint_name, 200, execution_time, count=len(rules))
        
        return rules
        
    except Exception as e:
        execution_time = time.time() - start_time
        log_response(endpoint_name, 500, execution_time)
        logger.error(f"💥 필터 규칙 조회 실패: {e}")
        raise HTTPException(status_code=500, detail=f"필터 규칙 조회 실패: {str(e)}")


@router.post("/filters", response_model=FilterRule, summary="필터 규칙 생성")
async def create_filter(request: Request, rule: FilterRule):
    """새로운 필터 규칙을 생성합니다."""
    start_time = time.time()
    endpoint_name = "create_filter"
    
    log_request(request, endpoint_name, rule_name=rule.name)
    
    try:
        # 정규식 패턴 유효성 검사
        import re
        try:
            re.compile(rule.pattern)
        except re.error as e:
            raise HTTPException(status_code=400, detail=f"잘못된 정규식 패턴: {str(e)}")
        
        with get_db_connection() as conn:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            # 중복 이름 검사
            cursor.execute("SELECT id FROM filter_rules WHERE name = %s", (rule.name,))
            if cursor.fetchone():
                raise HTTPException(status_code=400, detail=f"규칙 이름 '{rule.name}'이 이미 존재합니다.")
            
            # 규칙 생성
            cursor.execute("""
                INSERT INTO filter_rules (name, pattern) 
                VALUES (%s, %s) 
                RETURNING id, name, pattern, created_at, updated_at
            """, (rule.name, rule.pattern))
            
            row = cursor.fetchone()
            conn.commit()
            
            created_rule = FilterRule(
                id=row['id'],
                name=row['name'],
                pattern=row['pattern'],
                created_at=row['created_at'].isoformat(),
                updated_at=row['updated_at'].isoformat()
            )
        
        # 필터 규칙 다시 로드
        InputFilter.reload_rules()
        
        execution_time = time.time() - start_time
        log_response(endpoint_name, 201, execution_time)
        
        return created_rule
        
    except HTTPException:
        raise
    except Exception as e:
        execution_time = time.time() - start_time
        log_response(endpoint_name, 500, execution_time)
        logger.error(f"💥 필터 규칙 생성 실패: {e}")
        raise HTTPException(status_code=500, detail=f"필터 규칙 생성 실패: {str(e)}")


@router.put("/filters", summary="필터 규칙 전체 교체")
async def replace_filters(request: Request, rules: List[FilterRule]):
    """모든 필터 규칙을 새로운 규칙들로 교체합니다."""
    start_time = time.time()
    endpoint_name = "replace_filters"
    
    log_request(request, endpoint_name, rules_count=len(rules))
    
    try:
        # 모든 정규식 패턴 유효성 검사
        import re
        for rule in rules:
            try:
                re.compile(rule.pattern)
            except re.error as e:
                raise HTTPException(status_code=400, detail=f"규칙 '{rule.name}'의 잘못된 정규식 패턴: {str(e)}")
        
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            # 기존 규칙 모두 삭제
            cursor.execute("DELETE FROM filter_rules")
            
            # 새 규칙들 삽입
            for rule in rules:
                cursor.execute("""
                    INSERT INTO filter_rules (name, pattern) 
                    VALUES (%s, %s)
                """, (rule.name, rule.pattern))
            
            conn.commit()
        
        # 필터 규칙 다시 로드
        InputFilter.reload_rules()
        
        execution_time = time.time() - start_time
        log_response(endpoint_name, 200, execution_time)
        
        return JSONResponse(content={
            "status": "success",
            "message": f"{len(rules)}개의 규칙이 성공적으로 교체되었습니다.",
            "rules_count": len(rules)
        })
        
    except HTTPException:
        raise
    except Exception as e:
        execution_time = time.time() - start_time
        log_response(endpoint_name, 500, execution_time)
        logger.error(f"💥 필터 규칙 교체 실패: {e}")
        raise HTTPException(status_code=500, detail=f"필터 규칙 교체 실패: {str(e)}")


@router.delete("/filters/{rule_id}", summary="필터 규칙 삭제")
async def delete_filter(request: Request, rule_id: int):
    """특정 ID의 필터 규칙을 삭제합니다."""
    start_time = time.time()
    endpoint_name = "delete_filter"
    
    log_request(request, endpoint_name, rule_id=rule_id)
    
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("DELETE FROM filter_rules WHERE id = %s", (rule_id,))
            
            if cursor.rowcount == 0:
                execution_time = time.time() - start_time
                log_response(endpoint_name, 404, execution_time)
                raise HTTPException(status_code=404, detail=f"필터 규칙 ID {rule_id}를 찾을 수 없습니다.")
            
            conn.commit()
        
        # 필터 규칙 다시 로드
        InputFilter.reload_rules()
        
        execution_time = time.time() - start_time
        log_response(endpoint_name, 200, execution_time)
        
        return JSONResponse(content={
            "status": "success",
            "message": f"필터 규칙 ID {rule_id}가 성공적으로 삭제되었습니다.",
            "deleted_id": rule_id
        })
        
    except HTTPException:
        raise
    except Exception as e:
        execution_time = time.time() - start_time
        log_response(endpoint_name, 500, execution_time)
        logger.error(f"💥 필터 규칙 삭제 실패: {e}")
        raise HTTPException(status_code=500, detail=f"필터 규칙 삭제 실패: {str(e)}")


@router.post("/filters/test", response_model=FilterTestResponse, summary="필터 테스트")
async def test_filter(request: Request, test_request: FilterTestRequest):
    """주어진 텍스트에 대해 필터 규칙을 테스트합니다."""
    start_time = time.time()
    endpoint_name = "test_filter"
    
    log_request(request, endpoint_name, text_length=len(test_request.text))
    
    try:
        result = InputFilter.contains_sensitive(test_request.text)
        
        response = FilterTestResponse(
            is_sensitive=result["is_sensitive"],
            matched_rules=result["matched_rules"],
            message=result["message"],
            test_text=test_request.text
        )
        
        execution_time = time.time() - start_time
        log_response(endpoint_name, 200, execution_time, 
                    is_sensitive=result["is_sensitive"], 
                    matched_count=len(result["matched_rules"]))
        
        return response
        
    except Exception as e:
        execution_time = time.time() - start_time
        log_response(endpoint_name, 500, execution_time)
        logger.error(f"💥 필터 테스트 실패: {e}")
        raise HTTPException(status_code=500, detail=f"필터 테스트 실패: {str(e)}")


@router.get("/filters/stats", summary="필터 통계")
async def get_filter_stats(request: Request):
    """필터 규칙 통계를 조회합니다."""
    start_time = time.time()
    endpoint_name = "get_filter_stats"
    
    log_request(request, endpoint_name)
    
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            # 전체 규칙 수
            cursor.execute("SELECT COUNT(*) as total FROM filter_rules")
            total_rules = cursor.fetchone()['total']
            
            # 최근 생성된 규칙
            cursor.execute("""
                SELECT name, created_at 
                FROM filter_rules 
                ORDER BY created_at DESC 
                LIMIT 5
            """)
            recent_rules = cursor.fetchall()
        
        stats = {
            "total_rules": total_rules,
            "recent_rules": [
                {
                    "name": rule['name'],
                    "created_at": rule['created_at'].isoformat() if rule['created_at'] else None
                }
                for rule in recent_rules
            ],
            "filter_status": "active" if total_rules > 0 else "inactive"
        }
        
        execution_time = time.time() - start_time
        log_response(endpoint_name, 200, execution_time)
        
        return JSONResponse(content=stats)
        
    except Exception as e:
        execution_time = time.time() - start_time
        log_response(endpoint_name, 500, execution_time)
        logger.error(f"💥 필터 통계 조회 실패: {e}")
        raise HTTPException(status_code=500, detail=f"필터 통계 조회 실패: {str(e)}")


@router.post("/filters/reload", summary="필터 규칙 리로드")
async def reload_filters(request: Request):
    """메모리에 로드된 필터 규칙을 데이터베이스에서 다시 로드합니다."""
    start_time = time.time()
    endpoint_name = "reload_filters"
    
    log_request(request, endpoint_name)
    
    try:
        success = InputFilter.reload_rules()
        rules_count = InputFilter.get_rules_count()
        
        if success:
            execution_time = time.time() - start_time
            log_response(endpoint_name, 200, execution_time)
            
            return JSONResponse(content={
                "status": "success",
                "message": f"필터 규칙이 성공적으로 리로드되었습니다.",
                "rules_count": rules_count
            })
        else:
            execution_time = time.time() - start_time
            log_response(endpoint_name, 500, execution_time)
            
            return JSONResponse(
                status_code=500,
                content={
                    "status": "error",
                    "message": "필터 규칙 리로드에 실패했습니다.",
                    "rules_count": 0
                }
            )
        
    except Exception as e:
        execution_time = time.time() - start_time
        log_response(endpoint_name, 500, execution_time)
        logger.error(f"💥 필터 규칙 리로드 실패: {e}")
        raise HTTPException(status_code=500, detail=f"필터 규칙 리로드 실패: {str(e)}")


@router.get("/filters/health", summary="필터 서비스 상태 확인")
async def filter_health_check(request: Request):
    """필터 서비스의 상태를 확인합니다."""
    start_time = time.time()
    endpoint_name = "filter_health"
    
    log_request(request, endpoint_name)
    
    try:
        # 데이터베이스 연결 테스트
        db_status = test_filter_connection()
        
        # 필터 규칙 로드 상태 확인
        rules_count = InputFilter.get_rules_count()
        
        health_status = {
            "status": "healthy" if db_status else "unhealthy",
            "database_connection": "ok" if db_status else "failed",
            "rules_loaded": rules_count,
            "filter_active": rules_count > 0,
            "timestamp": datetime.now().isoformat()
        }
        
        execution_time = time.time() - start_time
        log_response(endpoint_name, 200 if db_status else 503, execution_time)
        
        return JSONResponse(
            status_code=200 if db_status else 503,
            content=health_status
        )
        
    except Exception as e:
        execution_time = time.time() - start_time
        log_response(endpoint_name, 500, execution_time)
        logger.error(f"💥 필터 상태 확인 실패: {e}")
        raise HTTPException(status_code=500, detail=f"필터 상태 확인 실패: {str(e)}")