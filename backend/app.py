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

# HITL을 위한 전역 변수
supervisor_instances: Dict[str, SupervisorService] = {}
active_websockets: Dict[str, WebSocket] = {}
pending_hitl_messages: Dict[str, asyncio.Queue] = {}

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


# HITL 콜백 함수 생성
def create_hitl_callback(thread_id: str):
    """각 스레드별 HITL 콜백 생성"""

    def hitl_callback(message: str, context: Dict) -> str:
        """WebSocket을 통해 프론트엔드로 승인 요청을 보내는 콜백"""

        # WebSocket 연결 확인
        ws = active_websockets.get(thread_id)
        if not ws:
            print(f"WebSocket 연결 없음: {thread_id}")
            return "rejected"  # 안전을 위해 거부

        try:
            # 현재 실행 중인 이벤트 루프 가져오기
            loop = asyncio.get_event_loop()

            # 비동기 작업을 Future로 만들어 실행
            future = asyncio.ensure_future(ws.send_json({
                "type": "response_chunk",
                "data": message  # 승인 메시지 전체를 전송
            }))

            # Future를 스케줄링만 하고 바로 반환
            # 실제 전송은 이벤트 루프가 처리

            print(f"HITL 메시지 전송 스케줄링 완료")

            # 특별한 플래그 반환하여 비동기 대기 시작
            return "__WAIT_FOR_ASYNC_INPUT__"

        except Exception as e:
            print(f"HITL 메시지 전송 실패: {e}")
            import traceback
            traceback.print_exc()
            return "rejected"

    return hitl_callback


@app.on_event("startup")
async def startup_event():
    print("🚀 서버 시작 중...")
    try:
        init_db()
        print("✅ 데이터베이스 초기화 완료")

        # 서버 시작 시 에이전트 초기화
        await agent_service.initialize_agent()
        print("✅ Agent Service 초기화 완료")
    except Exception as e:
        print(f"❌ 초기화 실패: {e}")
        raise e
    print("🎉 서버 시작 완료!")


@app.on_event("shutdown")
async def shutdown_event():
    """서버 종료 시 정리"""
    for supervisor in supervisor_instances.values():
        await supervisor.cleanup_mcp_client()


@app.websocket("/user/chat")
async def websocket_endpoint_user(websocket: WebSocket):
    await websocket.accept()
    thread_id = "default"

    try:
        active_websockets[thread_id] = websocket

        # Supervisor 초기화
        if thread_id not in supervisor_instances:
            supervisor = SupervisorService()
            supervisor.set_human_input_callback(create_hitl_callback(thread_id))
            await supervisor.initialize_agent(
                model_name="qwen2.5:32b",
                hitl_enabled=True
            )
            supervisor_instances[thread_id] = supervisor
        else:
            supervisor = supervisor_instances[thread_id]

        # 🚨 핵심: 메시지 처리를 논블로킹으로 변경
        chat_task = None

        while True:
            try:
                # 타임아웃을 짧게 설정하여 블로킹 방지
                data = await asyncio.wait_for(
                    websocket.receive_json(),
                    timeout=0.1  # 100ms 타임아웃
                )

                print(f"📥 메시지 수신: {data}")
                message = data.get("message", "")

                # HITL 승인 응답 처리 (최우선)
                if message.startswith("[HITL_APPROVAL]"):
                    approval = message.replace("[HITL_APPROVAL]", "").strip()
                    print(f"🎯 HITL 승인 수신: {approval}")

                    # 즉시 응답
                    await websocket.send_json({
                        "type": "approval_received",
                        "data": f"승인 '{approval}' 처리 중"
                    })

                    # Supervisor에 승인 전달
                    if hasattr(supervisor, 'human_input_queue'):
                        try:
                            await supervisor.human_input_queue.put(approval)
                            supervisor.waiting_for_human_input = False
                            print(f"✅ 승인 처리 완료: {approval}")

                            await websocket.send_json({
                                "type": "approval_processed",
                                "data": "워크플로우 재개됨"
                            })
                        except Exception as e:
                            print(f"❌ 승인 처리 실패: {e}")

                # 일반 메시지 처리
                elif message and not message.startswith("["):
                    print(f"💬 일반 메시지: {message}")

                    # 기존 채팅이 있으면 취소
                    if chat_task and not chat_task.done():
                        chat_task.cancel()

                    # 새 채팅 시작 (백그라운드)
                    chat_task = asyncio.create_task(
                        process_chat_message(websocket, supervisor, message, thread_id)
                    )

            except asyncio.TimeoutError:
                # 타임아웃은 정상 - 계속 진행
                continue

            except Exception as e:
                print(f"❌ 메시지 처리 오류: {e}")
                break

    except WebSocketDisconnect:
        print("WebSocket 연결 종료")
    except Exception as e:
        print(f"WebSocket 오류: {e}")


async def process_chat_message(websocket, supervisor, message, thread_id):
    """채팅 메시지 처리 (백그라운드 태스크)"""
    try:
        print(f"🚀 채팅 처리 시작: {message}")

        async for chunk in supervisor.chat_stream(message, thread_id):
            if not chunk.startswith("\n🤚") and not chunk.startswith("\n💭"):
                await websocket.send_json({
                    "type": "response_chunk",
                    "data": chunk
                })
                await asyncio.sleep(0.01)

        await websocket.send_json({"type": "response_complete"})
        print("✅ 채팅 처리 완료")

    except Exception as e:
        print(f"❌ 채팅 처리 오류: {e}")


@app.post("/api/user/hitl/approve")
async def handle_hitl_approval(request: Dict, user=Depends(get_current_user)):
    """REST API를 통한 HITL 승인 처리"""
    approval = request.get("approval")
    thread_id = request.get("thread_id", "default")

    # Supervisor 인스턴스 확인
    supervisor = supervisor_instances.get(thread_id)

    if supervisor and supervisor.waiting_for_human_input:
        success = await supervisor.set_human_input_async(approval)
        if success:
            return {"status": "success", "approval": approval}
        else:
            return {"status": "error", "message": "No pending approval"}
    else:
        return {"status": "error", "message": "Supervisor not found or not waiting"}


@app.get("/api/user/status")
async def get_user_status(user=Depends(get_current_user)):
    """사용자용 상태 정보"""
    # 기본 thread_id 사용
    thread_id = "default"
    supervisor = supervisor_instances.get(thread_id)

    if supervisor:
        status = await supervisor.get_agent_status()
        return {
            "agent_ready": status.get("is_initialized", False),
            "tools_available": status.get("tools_count", 0),
            "hitl_config": status.get("hitl_config", {})
        }
    else:
        # Supervisor가 없으면 Agent Service 상태 반환
        status = await agent_service.get_agent_status()
        return {
            "agent_ready": status["is_initialized"],
            "tools_available": status["tools_count"],
            "hitl_config": {}
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


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)