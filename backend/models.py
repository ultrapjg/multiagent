from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional


class MessageCreate(BaseModel):
    """메시지 생성 요청 모델"""
    content: str = Field(..., min_length=1, max_length=2000, description="메시지 내용")
    author: str = Field(..., min_length=1, max_length=100, description="작성자 이름")

    class Config:
        json_schema_extra = {
            "example": {
                "content": "안녕하세요! 첫 번째 메시지입니다.",
                "author": "홍길동"
            }
        }


class MessageUpdate(BaseModel):
    """메시지 수정 요청 모델"""
    content: Optional[str] = Field(None, min_length=1, max_length=2000, description="수정할 메시지 내용")
    author: Optional[str] = Field(None, min_length=1, max_length=100, description="수정할 작성자 이름")


class MessageResponse(BaseModel):
    """메시지 응답 모델"""
    id: int = Field(..., description="메시지 ID")
    content: str = Field(..., description="메시지 내용")
    author: str = Field(..., description="작성자 이름")
    created_at: str = Field(..., description="생성 시간 (ISO 형식)")

    class Config:
        json_schema_extra = {
            "example": {
                "id": 1,
                "content": "안녕하세요! 첫 번째 메시지입니다.",
                "author": "홍길동",
                "created_at": "2024-01-15T10:30:00"
            }
        }


class MessageStats(BaseModel):
    """메시지 통계 모델"""
    total_messages: int = Field(..., description="전체 메시지 수")
    total_authors: int = Field(..., description="전체 작성자 수")
    recent_message_count: int = Field(..., description="최근 24시간 메시지 수")
    most_active_author: Optional[str] = Field(None, description="가장 활발한 작성자")


class DeleteResponse(BaseModel):
    """삭제 응답 모델"""
    message: str = Field(..., description="결과 메시지")
    deleted_id: int = Field(..., description="삭제된 메시지 ID")

    class Config:
        json_schema_extra = {
            "example": {
                "message": "Message deleted successfully",
                "deleted_id": 1
            }
        }