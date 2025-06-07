import uvicorn
from fastapi import FastAPI, HTTPException, Depends, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import Optional, Dict, Any
import json
import asyncio
import logging

from services.supervisor_service import SupervisorService
from services.message_service import MessageService
from database import init_db
from core.agent_service import MCPAgentService
from core.tool_service import MCPToolService
from routes.messages import router as messages_router

app = FastAPI(title="LangGraph MCP Agents API", version="2.0.0")

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://localhost:8502"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 라우터 등록
app.include_router(messages_router, prefix="/messages", tags=["messages"])

# 서비스 인스턴스
agent_service = MCPAgentService()
tool_service = MCPToolService()
supervisor_service = SupervisorService()

# 보안
security = HTTPBearer()


def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.credentials != "user_token":
        raise HTTPException(status_code=401, detail="Invalid user credentials")
    return {"role": "user"}


def get_admin_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.credentials != "admin_token":
        raise HTTPException(status_code=401, detail="Invalid admin credentials")
    return {"role": "admin"}


# 데이터 모델
class ChatMessage(BaseModel):
    message: str
    thread_id: Optional[str] = "default"


class ToolConfig(BaseModel):
    name: str
    config: Dict[str, Any]
    description: Optional[str] = ""


class AgentConfig(BaseModel):
    model_name: str = "claude-3-5-sonnet-latest"
    system_prompt: Optional[str] = None


@app.on_event("startup")
async def startup_event():
    print("🚀 서버 시작 중...")
    try:
        init_db()
        print("✅ 데이터베이스 초기화 완료")

        """서버 시작 시 에이전트 초기화"""
        await agent_service.initialize_agent()
        await supervisor_service.initialize_agent()
    except Exception as e:
        print(f"❌ 데이터베이스 초기화 실패: {e}")
        raise e
    print("🎉 서버 시작 완료!")


@app.websocket("/api/user/chat")
async def websocket_chat(websocket: WebSocket):
    """실시간 채팅 웹소켓"""
    await websocket.accept()

    try:
        while True:
            # 사용자 메시지 수신
            data = await websocket.receive_text()
            message_data = json.loads(data)
            message = message_data.get("message", "")
            thread_id = message_data.get("thread_id", "default")

            result = MessageService.create_message(message, "admin")

            # 에이전트가 초기화되지 않은 경우 자동 초기화
            if not agent_service.agent:
                await agent_service.initialize_agent()

            # 스트리밍 응답 전송
            async for chunk in agent_service.chat_stream(message, thread_id):
                await websocket.send_text(json.dumps({
                    "type": "response_chunk",
                    "data": chunk,
                    "thread_id": thread_id
                }))

            # 완료 신호
            await websocket.send_text(json.dumps({
                "type": "response_complete",
                "thread_id": thread_id
            }))

    except WebSocketDisconnect:
        pass
    except Exception as e:
        await websocket.send_text(json.dumps({
            "type": "error",
            "data": f"오류 발생: {str(e)}"
        }))


@app.get("/api/user/status")
async def get_user_status(user=Depends(get_current_user)):
    """사용자용 상태 정보"""
    status = await agent_service.get_agent_status()
    return {
        "agent_ready": status["is_initialized"],
        "tools_available": status["tools_count"]
    }


@app.get("/api/admin/tools")
async def get_tools(admin=Depends(get_admin_user)):
    """모든 도구 조회"""
    tools = tool_service.get_all_tools()
    return {"tools": tools, "count": len(tools)}


@app.post("/api/admin/tools")
async def create_tool(tool: ToolConfig, admin=Depends(get_admin_user)):
    """새 도구 추가"""
    result = tool_service.add_tool(tool.name, tool.config)
    if result["success"]:
        return result
    else:
        raise HTTPException(status_code=400, detail=result["message"])


@app.delete("/api/admin/tools/{tool_name}")
async def delete_tool(tool_name: str, admin=Depends(get_admin_user)):
    """도구 삭제"""
    result = tool_service.remove_tool(tool_name)
    if result["success"]:
        return result
    else:
        raise HTTPException(status_code=404, detail=result["message"])


@app.post("/api/admin/tools/apply")
async def apply_tool_changes(admin=Depends(get_admin_user)):
    """도구 변경사항 적용 (에이전트 재초기화)"""
    try:
        # MCP 설정 다시 로드하여 에이전트 재초기화
        success = await agent_service.initialize_agent()
        if success:
            return {"message": "도구 변경사항이 성공적으로 적용되었습니다."}
        else:
            raise HTTPException(status_code=500, detail="에이전트 재초기화 실패")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"적용 실패: {str(e)}")


@app.get("/api/admin/agent/status")
async def get_agent_status(admin=Depends(get_admin_user)):
    """에이전트 상태 정보"""
    return await agent_service.get_agent_status()


@app.post("/api/admin/agent/reinitialize")
async def reinitialize_agent(config: AgentConfig, admin=Depends(get_admin_user)):
    """에이전트 재초기화"""
    try:
        success = await agent_service.initialize_agent(
            model_name=config.model_name,
            system_prompt=config.system_prompt
        )
        if success:
            return {"message": "에이전트가 성공적으로 재초기화되었습니다."}
        else:
            raise HTTPException(status_code=500, detail="에이전트 초기화 실패")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"초기화 실패: {str(e)}")


@app.get("/api/admin/stats")
async def get_admin_stats(admin=Depends(get_admin_user)):
    """운영자 통계"""
    tools = tool_service.get_all_tools()
    agent_status = await agent_service.get_agent_status()

    return {
        "active_tools": len(tools),
        "agent_initialized": agent_status["is_initialized"],
        "model_name": agent_status.get("model_name", "None"),
        "total_conversations": 0,  # TODO: 실제 대화 수 계산
        "daily_users": 1  # TODO: 실제 사용자 수 계산
    }


# 헬스체크
@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "LangGraph MCP Agents"}


@app.websocket("/user/chat")
async def websocket_chat(websocket: WebSocket):
    """실시간 채팅 웹소켓"""
    await websocket.accept()

    try:
        while True:
            # 사용자 메시지 수신
            data = await websocket.receive_text()
            message_data = json.loads(data)
            message = message_data.get("message", "")
            thread_id = message_data.get("thread_id", "default")

            result = MessageService.create_message(message, "admin")

            # 에이전트가 초기화되지 않은 경우 자동 초기화
            if not agent_service.agent:
                await supervisor_service.initialize_agent()

            # 스트리밍 응답 전송
            async for chunk in supervisor_service.chat_stream(message, thread_id):
                await websocket.send_text(json.dumps({
                    "type": "response_chunk",
                    "data": chunk,
                    "thread_id": thread_id
                }))

            # 완료 신호
            await websocket.send_text(json.dumps({
                "type": "response_complete",
                "thread_id": thread_id
            }))

    except WebSocketDisconnect:
        pass
    except Exception as e:
        await websocket.send_text(json.dumps({
            "type": "error",
            "data": f"오류 발생: {str(e)}"
        }))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)