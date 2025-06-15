import uvicorn
from fastapi import FastAPI, HTTPException, Depends, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import Optional, Dict, Any
import json
import asyncio
import logging

from services.supervisor import SupervisorService
from services.message import MessageService
from services.api_key_service import APIKeyService  # 추가
from database import init_db
from core.agent_service import MCPAgentService
from core.tool_service import MCPToolService
from core.input_filter import InputFilter, init_filter_db  # 추가
from routes.messages import router as messages_router
from routes.api_keys import router as api_keys_router  # 추가
from routes.mcp_tools import router as mcp_tools_router
from filter_api import router as filter_router  # 추가

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
app.include_router(filter_router, prefix="/filters", tags=["filters"])  # 추가
app.include_router(api_keys_router, prefix="/api", tags=["api_keys"])  # 추가
app.include_router(mcp_tools_router, prefix="/api", tags=["mcp_tools"])  # 추가

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
    """사용자 인증 - API 키 또는 사용자 토큰"""
    if credentials.credentials == "user_token":
        return {"role": "user", "auth_type": "token"}
    
    # API 키 검증
    api_key_info = APIKeyService.validate_api_key(credentials.credentials)
    if api_key_info:
        return {"role": "user", "auth_type": "api_key", "api_key_info": api_key_info}
    
    raise HTTPException(status_code=401, detail="Invalid user credentials or API key")


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

# WebSocket 인증 함수 추가
async def authenticate_websocket(websocket: WebSocket) -> Optional[Dict]:
    """WebSocket 연결 인증"""
    try:
        # Authorization 헤더에서 토큰/API 키 추출
        auth_header = websocket.headers.get("authorization")
        if not auth_header:
            # Query parameter에서 API 키 확인
            api_key = websocket.query_params.get("api_key")
            if api_key:
                auth_header = f"Bearer {api_key}"
        
        if not auth_header or not auth_header.startswith("Bearer "):
            return None
        
        token = auth_header.split(" ")[1]
        
        # 관리자 토큰 확인
        if token == "admin_token":
            return {"role": "admin", "auth_type": "token"}
        
        # 사용자 토큰 확인
        if token == "user_token":
            return {"role": "user", "auth_type": "token"}
        
        # API 키 검증
        api_key_info = APIKeyService.validate_api_key(token)
        if api_key_info:
            return {"role": "user", "auth_type": "api_key", "api_key_info": api_key_info}
        
        return None
    except Exception as e:
        logging.error(f"WebSocket 인증 실패: {e}")
        return None

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


async def get_default_model_name() -> str:
    """기본 모델 이름을 결정하는 함수"""
    # 1. agent_service가 초기화되어 있고 model이 있으면 해당 모델명 사용
    if (hasattr(agent_service, 'model') and 
        agent_service.model is not None and 
        hasattr(agent_service.model, 'model_name')):
        model_name = agent_service.model.model_name
        print(f"📋 Agent Service에서 모델명 가져옴: {model_name}")
        return model_name
    else:
        # 로컬 모델 사용
        default_model = "qwen2.5:32b"
        print(f"📋 외부 API 키 없음 - 로컬 모델 사용: {default_model}")
        return default_model


# 입력 필터링 미들웨어 함수
async def filter_user_input(message: str) -> Dict[str, Any]:
    """사용자 입력에 대한 필터링 검사"""
    try:
        filter_result = InputFilter.contains_sensitive(message)
        return filter_result
    except Exception as e:
        print(f"❌ 입력 필터링 중 오류: {e}")
        return {
            "is_sensitive": False,
            "matched_rules": [],
            "message": f"필터링 검사 중 오류 발생: {str(e)}"
        }


@app.on_event("startup")
async def startup_event():
    print("🚀 서버 시작 중...")
    try:
        # 1. 데이터베이스 초기화
        init_db()
        print("✅ 기본 데이터베이스 초기화 완료")
        
        # 2. 필터 데이터베이스 초기화
        init_filter_db()
        print("✅ 필터 데이터베이스 초기화 완료")
        
        # 3. 필터 규칙 로드
        InputFilter.load_rules()
        rules_count = InputFilter.get_rules_count()
        print(f"✅ 필터 규칙 로드 완료: {rules_count}개 규칙")

        # 4. API 키 서비스 초기화 확인
        try:
            api_key_stats = APIKeyService.get_api_key_stats()
            print(f"✅ API 키 서비스 초기화 완료: {api_key_stats['total_keys']}개 키 등록됨")
        except Exception as e:
            print(f"⚠️ API 키 서비스 초기화 중 오류: {e}")

        # 🔄 개선된 에이전트 초기화 프로세스
        
        # 4. 먼저 기본 모델명 결정
        default_model = await get_default_model_name()
        
        # 5. Agent Service 초기화
        print(f"🤖 Agent Service 초기화 중... (모델: {default_model})")
        agent_init_success = await agent_service.initialize_agent(model_name=default_model)
        
        if agent_init_success:
            print("✅ Agent Service 초기화 완료")
            
            # 6. Agent Service에서 실제 사용된 모델명 가져오기
            actual_model_name = default_model
            if (hasattr(agent_service, 'model') and 
                agent_service.model is not None and 
                hasattr(agent_service.model, 'model_name')):
                actual_model_name = agent_service.model.model_name
                print(f"📋 Agent Service 실제 사용 모델: {actual_model_name}")
            
            # 7. Supervisor Service를 같은 모델로 초기화
            print(f"👥 Supervisor Service 초기화 중... (모델: {actual_model_name})")
            
            # Supervisor 서비스를 글로벌로 하나만 생성하지 말고,
            # 필요시 동적으로 생성하도록 변경
            print(f"✅ Supervisor Service는 요청시 동적으로 생성됩니다 (기본 모델: {actual_model_name})")
            
        else:
            print("❌ Agent Service 초기화 실패")
            
    except Exception as e:
        print(f"❌ 초기화 실패: {e}")
        import traceback
        traceback.print_exc()
        raise e
    print("🎉 서버 시작 완료!")


@app.on_event("shutdown")
async def shutdown_event():
    """서버 종료 시 정리"""
    print("🔄 서버 종료 중...")
    
    # Supervisor 인스턴스들 정리
    for thread_id, supervisor in supervisor_instances.items():
        try:
            await supervisor.cleanup_mcp_client()
            print(f"✅ Supervisor 정리 완료: {thread_id}")
        except Exception as e:
            print(f"⚠️ Supervisor 정리 중 오류 ({thread_id}): {e}")
    
    # Agent Service 정리
    if hasattr(agent_service, 'cleanup_mcp_client'):
        try:
            await agent_service.cleanup_mcp_client()
            print("✅ Agent Service 정리 완료")
        except Exception as e:
            print(f"⚠️ Agent Service 정리 중 오류: {e}")
    
    print("✅ 서버 종료 완료")


async def get_or_create_supervisor(thread_id: str) -> SupervisorService:
    """스레드별 Supervisor 인스턴스 생성 또는 반환"""
    if thread_id not in supervisor_instances:
        print(f"🔄 새로운 Supervisor 생성 중... (thread: {thread_id})")
        
        # Agent Service에서 사용 중인 모델명 가져오기
        model_name = "claude-3-5-sonnet-latest"  # 기본값
        
        if (hasattr(agent_service, 'model') and 
            agent_service.model is not None and 
            hasattr(agent_service.model, 'model_name')):
            model_name = agent_service.model.model_name
            print(f"📋 Agent Service 모델명 사용: {model_name}")
        else:
            # Agent Service가 초기화되지 않았다면 기본 모델명 다시 결정
            model_name = await get_default_model_name()
            print(f"📋 기본 모델명 사용: {model_name}")
        
        # 새 Supervisor 생성
        supervisor = SupervisorService()
        supervisor.set_human_input_callback(create_hitl_callback(thread_id))
        
        # Agent Service와 같은 모델로 초기화
        await supervisor.initialize_agent(
            model_name=model_name,
            hitl_enabled=True
        )
        
        supervisor_instances[thread_id] = supervisor
        print(f"✅ Supervisor 생성 완료 (thread: {thread_id}, model: {model_name})")
    
    return supervisor_instances[thread_id]


@app.websocket("/api/user/chat")
async def websocket_endpoint_user(websocket: WebSocket):
    # 인증 확인
    auth_info = await authenticate_websocket(websocket)
    if not auth_info:
        await websocket.close(code=1008, reason="Authentication required")
        return

    await websocket.accept()
    thread_id = "default"

    try:
        active_websockets[thread_id] = websocket

        # 인증 정보 로깅
        if auth_info["auth_type"] == "api_key":
            api_key_name = auth_info["api_key_info"]["name"]
            logging.info(f"WebSocket 연결: API 키 '{api_key_name}' 사용")
        else:
            logging.info(f"WebSocket 연결: {auth_info['auth_type']} 사용")

        # 🔄 개선된 Supervisor 초기화
        supervisor = await get_or_create_supervisor(thread_id)

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
                            error_str = str(e)
                            if "(1000," in error_str:
                                print(f"✅ [{thread_id if 'thread_id' in locals() else 'unknown'}] WebSocket 정상 종료")
                            elif "connection closed" in error_str.lower():
                                print(f"ℹ️ [{thread_id if 'thread_id' in locals() else 'unknown'}] WebSocket 연결 종료")
                            else:
                                print(f"❌ [{thread_id if 'thread_id' in locals() else 'unknown'}] 실제 처리 오류: {e}")
                            break

                # 일반 메시지 처리
                elif message and not message.startswith("["):
                    print(f"💬 일반 메시지: {message}")

                    # 🔒 입력 필터링 검사
                    filter_result = await filter_user_input(message)
                    
                    if filter_result["is_sensitive"]:
                        # 민감한 내용 감지 시 사용자에게 알림
                        warning_message = f"""🚨 민감한 내용이 감지되었습니다.

{filter_result['message']}

매칭된 규칙:
{chr(10).join([f"- {rule['name']}" for rule in filter_result['matched_rules']])}

메시지 처리가 차단되었습니다."""
                        
                        await websocket.send_json({
                            "type": "response_chunk",
                            "data": warning_message
                        })
                        
                        await websocket.send_json({
                            "type": "response_complete"
                        })
                        
                        print(f"🚨 민감한 내용으로 인해 메시지 차단: {len(filter_result['matched_rules'])}개 규칙 매칭")
                        continue

                    # 메시지 저장 (필터 통과한 경우만) - 인증 정보에 따라 작성자 설정
                    if auth_info["auth_type"] == "api_key":
                        author = f"API:{auth_info['api_key_info']['name']}"
                    else:
                        author = "admin"

                    # 메시지 저장
                    MessageService.create_message(message, "admin")
                    
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
                error_str = str(e)
                if "(1000," in error_str:
                    print(f"✅ [{thread_id if 'thread_id' in locals() else 'unknown'}] WebSocket 정상 종료")
                elif "connection closed" in error_str.lower():
                    print(f"ℹ️ [{thread_id if 'thread_id' in locals() else 'unknown'}] WebSocket 연결 종료")
                else:
                    print(f"❌ [{thread_id if 'thread_id' in locals() else 'unknown'}] 실제 처리 오류: {e}")
                break

    except WebSocketDisconnect:
        print("WebSocket 연결 종료")
    except Exception as e:
        print(f"WebSocket 오류: {e}")
    finally:
        # 연결 정리
        if thread_id in active_websockets:
            del active_websockets[thread_id]


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


# 추후 삭제
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

    # 필터 상태 정보 추가
    filter_status = {
        "rules_count": InputFilter.get_rules_count(),
        "active": InputFilter.get_rules_count() > 0
    }

    # API 키 정보 추가
    api_key_info = None
    if user.get("auth_type") == "api_key":
        api_key_data = user.get("api_key_info", {})
        api_key_info = {
            "name": api_key_data.get("name"),
            "description": api_key_data.get("description"),
            "created_at": api_key_data.get("created_at")
        }

    if supervisor:
        status = await supervisor.get_agent_status()
        return {
            "agent_ready": status.get("is_initialized", False),
            "tools_available": status.get("tools_count", 0),
            "hitl_config": status.get("hitl_config", {}),
            "model_name": status.get("model_name", "Unknown"),
            "filter_status": filter_status,
            "auth_type": user.get("auth_type"),
            "api_key_info": api_key_info
        }
    else:
        # Supervisor가 없으면 Agent Service 상태 반환
        status = await agent_service.get_agent_status()
        return {
            "agent_ready": status["is_initialized"],
            "tools_available": status["tools_count"],
            "hitl_config": {},
            "model_name": status.get("model_name", "Unknown"),
            "filter_status": filter_status,
            "auth_type": user.get("auth_type"),
            "api_key_info": api_key_info            
        }

# API 키 검증 전용 엔드포인트 추가
@app.post("/api/user/verify-key")
async def verify_api_key_endpoint(request: dict):
    """API 키 검증 엔드포인트"""
    try:
        api_key = request.get("api_key")
        if not api_key:
            raise HTTPException(status_code=400, detail="API 키가 필요합니다")
        
        api_key_info = APIKeyService.validate_api_key(api_key)
        if not api_key_info:
            raise HTTPException(status_code=401, detail="유효하지 않거나 만료된 API 키입니다")
        
        return {
            "status": "valid",
            "message": "API 키가 유효합니다",
            "api_key_info": {
                "name": api_key_info["name"],
                "description": api_key_info["description"],
                "created_at": api_key_info["created_at"]
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"API 키 검증 실패: {str(e)}")   

@app.get("/api/admin/tools")
async def get_tools(admin=Depends(get_admin_user)):
    """모든 도구 조회 - 데이터베이스에서"""
    try:
        from services.mcp_tool_service import MCPToolService
        tools = MCPToolService.get_all_tools(include_inactive=False)
        
        # 기존 형식과 호환되도록 변환
        formatted_tools = []
        for tool in tools:
            formatted_tools.append({
                "id": tool['name'],
                "name": tool['name'],
                "description": tool.get('description', ''),
                "transport": tool.get('transport', 'stdio'),
                "command": tool.get('command', ''),
                "args": tool.get('args', []),
                "url": tool.get('url', ''),
                "active": tool.get('is_active', True),
                "config": tool.get('config', {})
            })
        
        return {"tools": formatted_tools, "count": len(formatted_tools)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"도구 조회 실패: {str(e)}")


@app.post("/api/admin/tools")
async def create_tool(tool: ToolConfig, admin=Depends(get_admin_user)):
    """새 도구 추가 - 데이터베이스에"""
    try:
        from services.mcp_tool_service import MCPToolService
        
        result = MCPToolService.create_tool(
            name=tool.name,
            config=tool.config,
            description=tool.description
        )
        
        return {
            "success": True,
            "message": f"도구 '{tool.name}'이 성공적으로 추가되었습니다.",
            "tool": result
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"도구 추가 실패: {str(e)}")


@app.delete("/api/admin/tools/{tool_name}")
async def delete_tool(tool_name: str, admin=Depends(get_admin_user)):
    """도구 삭제 - 데이터베이스에서"""
    try:
        from services.mcp_tool_service import MCPToolService
        
        success = MCPToolService.delete_tool_by_name(tool_name, soft_delete=True)
        
        if success:
            return {
                "success": True,
                "message": f"도구 '{tool_name}'이 성공적으로 삭제되었습니다."
            }
        else:
            raise HTTPException(status_code=404, detail=f"도구 '{tool_name}'을 찾을 수 없습니다.")
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"도구 삭제 실패: {str(e)}")



@app.post("/api/admin/tools/apply")
async def apply_tool_changes(admin=Depends(get_admin_user)):
    """도구 변경사항 적용 (에이전트 재초기화) - 데이터베이스 기반"""
    try:
        # Agent Service 재초기화
        success = await agent_service.initialize_agent()
        if not success:
            raise HTTPException(status_code=500, detail="Agent Service 초기화 실패")
        
        # 모든 Supervisor 인스턴스도 재초기화
        updated_supervisors = []
        failed_supervisors = []
        
        for thread_id, supervisor in list(supervisor_instances.items()):
            try:
                await supervisor.initialize_agent(
                    model_name=getattr(agent_service.model, 'model_name', 'claude-3-5-sonnet-latest'),
                    hitl_enabled=True
                )
                updated_supervisors.append(thread_id)
                print(f"✅ Supervisor 재초기화 완료: {thread_id}")
            except Exception as e:
                print(f"❌ Supervisor 재초기화 실패: {thread_id} -> {e}")
                failed_supervisors.append({"thread_id": thread_id, "error": str(e)})
                del supervisor_instances[thread_id]
        
        return {
            "message": "도구 변경사항이 성공적으로 적용되었습니다 (데이터베이스 기반)",
            "agent_service": "재초기화 완료",
            "updated_supervisors": updated_supervisors,
            "failed_supervisors": failed_supervisors
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"적용 실패: {str(e)}")


@app.get("/api/admin/agent/status")
async def get_agent_status(admin=Depends(get_admin_user)):
    """에이전트 상태 정보"""
    agent_status = await agent_service.get_agent_status()
    
    # Supervisor 인스턴스들의 상태도 포함
    supervisor_statuses = {}
    for thread_id, supervisor in supervisor_instances.items():
        try:
            supervisor_status = await supervisor.get_agent_status()
            supervisor_statuses[thread_id] = {
                "is_initialized": supervisor_status.get("is_initialized", False),
                "model_name": supervisor_status.get("model_name", "Unknown"),
                "tools_count": supervisor_status.get("tools_count", 0),
                "hitl_enabled": supervisor_status.get("hitl_config", {}).get("enabled", False)
            }
        except Exception as e:
            supervisor_statuses[thread_id] = {"error": str(e)}
    
    # 필터 상태 정보 추가
    filter_status = {
        "rules_count": InputFilter.get_rules_count(),
        "active": InputFilter.get_rules_count() > 0,
        "all_rules": InputFilter.get_all_rules()
    }
    
    return {
        "agent_service": agent_status,
        "supervisor_instances": supervisor_statuses,
        "total_supervisor_instances": len(supervisor_instances),
        "filter_status": filter_status  # 추가
    }


@app.post("/api/admin/agent/reinitialize")
async def reinitialize_agent(config: AgentConfig, admin=Depends(get_admin_user)):
    """에이전트 재초기화"""
    try:
        # 1. Agent Service 재초기화
        success = await agent_service.initialize_agent(
            model_name=config.model_name,
            system_prompt=config.system_prompt
        )
        
        if not success:
            raise HTTPException(status_code=500, detail="Agent Service 초기화 실패")
        
        # 2. 모든 Supervisor 인스턴스도 새 모델로 재초기화
        updated_supervisors = []
        failed_supervisors = []
        
        for thread_id, supervisor in list(supervisor_instances.items()):
            try:
                await supervisor.initialize_agent(
                    model_name=config.model_name,
                    hitl_enabled=True
                )
                updated_supervisors.append(thread_id)
                print(f"✅ Supervisor 재초기화 완료: {thread_id} -> {config.model_name}")
            except Exception as e:
                print(f"❌ Supervisor 재초기화 실패: {thread_id} -> {e}")
                failed_supervisors.append({"thread_id": thread_id, "error": str(e)})
                # 실패한 인스턴스는 제거
                del supervisor_instances[thread_id]
        
        return {
            "message": f"에이전트가 성공적으로 재초기화되었습니다 (모델: {config.model_name})",
            "agent_service": "재초기화 완료",
            "updated_supervisors": updated_supervisors,
            "failed_supervisors": failed_supervisors
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"초기화 실패: {str(e)}")


@app.get("/api/admin/stats")
async def get_admin_stats(admin=Depends(get_admin_user)):
    """운영자 통계"""
    # MCP 도구 통계
    from services.mcp_tool_service import MCPToolService
    mcp_stats = MCPToolService.get_mcp_tool_stats()
    agent_status = await agent_service.get_agent_status()
    
    # 필터 통계 추가
    filter_stats = {
        "total_rules": InputFilter.get_rules_count(),
        "active": InputFilter.get_rules_count() > 0
    }

    # API 키 통계 추가
    api_key_stats = APIKeyService.get_api_key_stats()

    return {
        "active_tools": mcp_stats.get("active_tools", 0),
        "total_tools": mcp_stats.get("total_tools", 0),
        "agent_initialized": agent_status["is_initialized"],
        "model_name": agent_status.get("model_name", "None"),
        "supervisor_instances": len(supervisor_instances),
        "active_websockets": len(active_websockets),
        "filter_stats": filter_stats,  # 추가
        "api_key_stats": api_key_stats,  # 추가
        "mcp_tool_stats": mcp_stats,  # 추가
        "total_conversations": 0,  # TODO: 실제 대화 수 계산
        "daily_users": 1  # TODO: 실제 사용자 수 계산
    }


# 헬스체크
@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "LangGraph MCP Agents"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)