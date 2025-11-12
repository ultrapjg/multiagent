import streamlit as st
import websocket
import json
import threading
import time
import queue
from typing import Optional
import re
import requests
import os

BACKEND_URL=os.getenv("BACKEND_URL", "http://localhost:8000")
BACKEND_WEBSOCKET=os.getenv("BACKEND_WEBSOCKET", "ws://localhost:8000")

st.set_page_config(
    page_title="🤖 AI Assistant with HITL",
    page_icon="🤖",
    layout="wide"
)

def verify_api_key(api_key: str) -> Optional[dict]:
    """API 키 검증"""
    try:
        response = requests.post(
            f"{BACKEND_URL}/api/user/verify-key",
            json={"api_key": api_key},
            timeout=10
        )
        
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 401:
            return None
        else:
            st.error(f"서버 오류: {response.status_code}")
            return None
    except requests.exceptions.RequestException as e:
        st.error(f"연결 오류: {e}")
        return None


def check_api_key_authentication():
    """API 키 인증 확인"""
    if "api_key_verified" not in st.session_state:
        st.session_state.api_key_verified = False
    if "api_key" not in st.session_state:
        st.session_state.api_key = ""
    if "api_key_info" not in st.session_state:
        st.session_state.api_key_info = None

    if not st.session_state.api_key_verified:
        st.title("🔐 API 키 인증")
        st.markdown("AI Assistant를 사용하려면 유효한 API 키가 필요합니다.")
        
        # API 키 입력 섹션
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            with st.container():
                st.markdown("### API 키 입력")
                
                # API 키 입력 폼
                with st.form("api_key_form"):
                    api_key_input = st.text_input(
                        "API 키",
                        type="password",
                        placeholder="ak_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
                        help="관리자로부터 받은 API 키를 입력하세요"
                    )
                    
                    col_submit, col_demo = st.columns(2)
                    
                    with col_submit:
                        submit_button = st.form_submit_button(
                            "🔑 인증하기",
                            use_container_width=True,
                            type="primary"
                        )
                    
                    with col_demo:
                        demo_button = st.form_submit_button(
                            "🎮 데모 모드",
                            use_container_width=True,
                            help="데모용 토큰 사용 (개발/테스트용)"
                        )
                    
                    if submit_button and api_key_input:
                        with st.spinner("API 키 검증 중..."):
                            verification_result = verify_api_key(api_key_input)
                            
                            if verification_result:
                                st.session_state.api_key = api_key_input
                                st.session_state.api_key_verified = True
                                st.session_state.api_key_info = verification_result.get("api_key_info", {})
                                
                                st.success("✅ API 키 인증 성공!")
                                st.info(f"**API 키 이름:** {st.session_state.api_key_info.get('name', 'Unknown')}")
                                if st.session_state.api_key_info.get('description'):
                                    st.info(f"**설명:** {st.session_state.api_key_info['description']}")
                                
                                time.sleep(1)
                                st.rerun()
                            else:
                                st.error("❌ 유효하지 않거나 만료된 API 키입니다.")
                    
                    elif demo_button:
                        # 데모 모드 (개발/테스트용)
                        st.session_state.api_key = "user_token"  # 기존 토큰 사용
                        st.session_state.api_key_verified = True
                        st.session_state.api_key_info = {
                            "name": "데모 모드",
                            "description": "개발/테스트용 데모 계정"
                        }
                        
                        st.success("✅ 데모 모드로 접속합니다!")
                        time.sleep(1)
                        st.rerun()
                    
                    elif submit_button and not api_key_input:
                        st.error("API 키를 입력해주세요.")
                
                # 도움말 섹션
                st.markdown("---")
                st.markdown("### 📋 도움말")
                with st.expander("API 키 관련 안내"):
                    st.markdown("""
                    **API 키란?**
                    - AI Assistant 서비스에 접근하기 위한 인증 키입니다
                    - 각 사용자/애플리케이션마다 고유한 키가 발급됩니다
                    
                    **API 키 형식:**
                    - `ak_` 로 시작하는 44자리 문자열입니다
                    - 예: `ak_abc123def456ghi789jkl012mno345pqr678stu901`
                    
                    **API 키 발급:**
                    - 시스템 관리자에게 요청하세요
                    - 관리자 대시보드에서 새 API 키를 생성할 수 있습니다
                    
                    **보안 주의사항:**
                    - API 키는 비밀번호와 같으니 안전하게 보관하세요
                    - 다른 사람과 공유하지 마세요
                    - 키가 노출되었다면 즉시 관리자에게 알리세요
                    """)
                
                with st.expander("문제 해결"):
                    st.markdown("""
                    **API 키 인증이 실패하는 경우:**
                    1. API 키를 정확히 입력했는지 확인하세요
                    2. 키가 만료되지 않았는지 확인하세요
                    3. 네트워크 연결을 확인하세요
                    4. 관리자에게 키 상태를 문의하세요
                    
                    **데모 모드:**
                    - 개발/테스트 목적으로만 사용하세요
                    - 실제 운영 환경에서는 유효한 API 키를 사용하세요
                    """)
        
        return False
    return True

class HITLWebSocketClient:
    """Human-in-the-Loop 지원 웹소켓 클라이언트"""

    def __init__(self, url: str, api_key: str = None):
        self.url = url
        self.api_key = api_key        
        self.ws = None
        self.is_connected = False
        self.response_buffer = ""
        self.approval_queue = queue.Queue()
        self.waiting_for_approval = False
        self.current_approval_context = None

    def connect(self):
        """웹소켓 연결"""
        try:
            # API 키를 URL 파라미터로 추가
            if self.api_key:
                connection_url = f"{self.url}?api_key={self.api_key}"
            else:
                connection_url = self.url            
            self.ws = websocket.create_connection(connection_url)
            self.is_connected = True
            return True
        except Exception as e:
            st.error(f"연결 실패: {e}")
            return False

    def send_message(self, message: str, thread_id: str = "default"):
        """메시지 전송"""
        if not self.is_connected:
            return False

        try:
            data = {
                "message": message,
                "thread_id": thread_id
            }
            self.ws.send(json.dumps(data))
            return True
        except Exception as e:
            st.error(f"메시지 전송 실패: {e}")
            return False

    def send_approval_response(self, approval_response: str, thread_id: str = "default"):
        """승인 응답 전송"""
        if not self.is_connected:
            return False

        try:
            data = {
                "message": f"[HITL_APPROVAL]{approval_response}",
                "thread_id": thread_id
            }

            self.ws.send(json.dumps(data))
            st.info(f"승인 응답 전송: {approval_response}")

            # 🚨 서버 응답 확인 (수정된 부분)
            try:
                self.ws.settimeout(5.0)  # 5초 대기
                response = self.ws.recv()
                response_data = json.loads(response)

                st.success(f"✅ 서버 응답: {response_data.get('data', '성공')}")
                return True

            except Exception as recv_e:
                st.warning(f"⚠️ 서버 응답 대기 시간 초과: {recv_e}")
                return True  # 메시지는 전송되었으므로 True

        except Exception as e:
            st.error(f"승인 응답 전송 실패: {e}")
            return False

    def receive_response(self) -> Optional[dict]:
        """응답 수신 (HITL 지원)"""
        if not self.is_connected:
            return None

        try:
            response = self.ws.recv()
            data = json.loads(response)

            if data.get("type") == "response_chunk":
                chunk_data = data.get("data", "")

                # 다양한 HITL 패턴 감지 (더 광범위하게)
                hitl_patterns = [
                    "Human Approval 필요",
                    "Human Input 필요",
                    "승인 요청",
                    "고위험 작업 승인",
                    "고위험 도구 실행",
                    "이 작업을 진행하시겠습니까",
                    "승인하시겠습니까",
                    "(approved/rejected)"
                ]

                # HITL 패턴이 감지되면 특별 처리
                if any(pattern in chunk_data for pattern in hitl_patterns):
                    return {
                        "type": "hitl_request",
                        "data": chunk_data,
                        "raw_response": data
                    }

                return {
                    "type": "response_chunk",
                    "data": chunk_data
                }

            elif data.get("type") == "response_complete":
                return {"type": "response_complete"}

            elif data.get("type") == "error":
                return {
                    "type": "error",
                    "data": data.get("data", "알 수 없는 오류")
                }

        except Exception as e:
            st.error(f"응답 수신 실패: {e}")
            return None

    def close(self):
        """연결 종료"""
        if self.ws:
            self.ws.close()
        self.is_connected = False


def send_hitl_approval_to_backend(approval_response: str, thread_id: str = "default", api_key: str = None):
    """백엔드로 HITL 승인 응답 전송 (REST API 사용)"""
    try:
        headers = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        else:
            headers["Authorization"] = "Bearer user_token"  # 폴백
        
        response = requests.post(
            f"{BACKEND_URL}/api/user/hitl/approve",
            json={
                "approval": approval_response,
                "thread_id": thread_id
            },
            headers=headers,
            timeout=5
        )

        if response.status_code == 200:
            st.success(f"✅ 승인 응답 전송 완료: {approval_response}")
            return True
        else:
            st.error(f"❌ 승인 응답 전송 실패: {response.status_code}")
            return False
    except Exception as e:
        st.error(f"❌ 승인 응답 전송 오류: {e}")
        return False


def extract_approval_details(hitl_message: str) -> dict:
    """HITL 메시지에서 승인 상세 정보 추출"""
    details = {
        "type": "unknown",
        "tool_name": None,
        "confidence": None,
        "reason": None,
        "keywords": [],
        "options": ["approved", "rejected"]
    }

    # 키워드 추출
    keyword_match = re.search(r'감지된 키워드[:\s]*([^\n]+)', hitl_message)
    if keyword_match:
        keywords_str = keyword_match.group(1)
        details["keywords"] = [k.strip() for k in keywords_str.split(',')]

    # 도구명 추출
    tool_patterns = [
        r'실행 예정 도구[:\s]*([^\n]+)',
        r'실행 도구[:\s]*([^\n]+)',
        r"'([^']+)'\s*도구"
    ]
    for pattern in tool_patterns:
        match = re.search(pattern, hitl_message)
        if match:
            details["tool_name"] = match.group(1).strip()
            break

    # 메시지 타입 판단
    message_lower = hitl_message.lower()

    # 고위험 작업/도구 실행 승인
    if any(keyword in message_lower for keyword in ["고위험 작업", "고위험 도구", "시스템에 영향을 줄 수 있습니다"]):
        details["type"] = "high_risk_tool"
        details["options"] = ["approved", "rejected", "modified"]

    # 도구 실행 승인
    elif any(keyword in message_lower for keyword in ["도구 실행", "tool", "파일", "삭제", "시스템"]):
        details["type"] = "tool_execution"
        details["options"] = ["approved", "rejected", "modified"]

    # 최종 답변 승인
    elif any(keyword in message_lower for keyword in ["최종 답변", "final answer"]):
        details["type"] = "final_answer"
        details["options"] = ["approved", "rejected", "modified"]

    # Human Input 요청
    elif any(keyword in message_lower for keyword in ["human input", "추가 정보", "더 구체적인"]):
        details["type"] = "input_request"
        details["options"] = ["provide_input"]

    return details


def render_hitl_approval_ui(hitl_message: str, approval_details: dict,
                            ws_client: Optional[HITLWebSocketClient] = None) -> Optional[str]:
    """HITL 승인 UI 렌더링"""
    st.markdown("---")
    st.markdown("### 🤚 Human Approval 필요")

    # 현재 사용 중인 API 키 정보 표시
    api_key_info = st.session_state.get("api_key_info", {})
    if api_key_info and api_key_info.get("name") != "데모 모드":
        st.info(f"🔑 API 키: {api_key_info.get('name', 'Unknown')}")

    # 메시지 표시
    with st.expander("📋 승인 요청 상세", expanded=True):
        st.markdown(hitl_message)

    # 승인 타입에 따른 UI
    approval_response = None

    if approval_details["type"] in ["high_risk_tool", "tool_execution"]:
        keywords = approval_details.get("keywords", [])
        tool_name = approval_details.get("tool_name", "unknown")

        if keywords:
            st.error(f"⚠️ 고위험 키워드 감지: {', '.join(keywords)}")

        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("✅ 승인", key="approve_tool", type="primary"):
                approval_response = "approved"
        with col2:
            if st.button("❌ 거부", key="reject_tool"):
                approval_response = "rejected"
        with col3:
            if st.button("✏️ 수정", key="modify_tool"):
                approval_response = "modified"

    elif approval_details["type"] == "final_answer":
        st.info("✅ 최종 답변 승인이 필요합니다.")

        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("✅ 승인", key="approve_final", type="primary"):
                approval_response = "approved"
        with col2:
            if st.button("❌ 거부", key="reject_final"):
                approval_response = "rejected"
        with col3:
            if st.button("✏️ 수정 요청", key="modify_final"):
                approval_response = "modified"

    elif approval_details["type"] == "input_request":
        st.info("💭 추가 정보가 필요합니다.")

        additional_input = st.text_area(
            "추가 정보를 입력하세요:",
            key="additional_input",
            placeholder="더 구체적인 정보나 지시사항을 입력해주세요..."
        )

        if st.button("📤 정보 제공", key="provide_input", type="primary"):
            if additional_input.strip():
                approval_response = additional_input.strip()
            else:
                st.warning("추가 정보를 입력해주세요.")

    else:
        # 알 수 없는 타입의 승인 요청
        st.error("❓ 알 수 없는 승인 요청입니다.")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("✅ 승인", key="approve_unknown", type="primary"):
                approval_response = "approved"
        with col2:
            if st.button("❌ 거부", key="reject_unknown"):
                approval_response = "rejected"

    # 디버깅 정보 (개발용)
    with st.expander("🔍 디버깅 정보", expanded=False):
        st.json({
            "detected_type": approval_details["type"],
            "confidence": approval_details.get("confidence"),
            "tool_name": approval_details.get("tool_name"),
            "keywords": approval_details.get("keywords"),
            "options": approval_details["options"],
            "api_key_name": api_key_info.get("name", "Unknown")
        })

    st.markdown("---")
    return approval_response


def main():
    # API 키 인증 확인
    if not check_api_key_authentication():
        return
        
    st.title("🤖 AI Assistant with Human-in-the-Loop")
    st.markdown("Human-in-the-Loop 기능이 지원되는 LangGraph MCP 에이전트와 대화해보세요!")

    # API 키 정보 표시 및 로그아웃
    with st.sidebar:
        st.header("👤 사용자 정보")
        
        # API 키 정보 표시
        api_key_info = st.session_state.get("api_key_info", {})
        if api_key_info:
            st.success("✅ 인증됨")
            st.info(f"**키 이름:** {api_key_info.get('name', 'Unknown')}")
            if api_key_info.get('description'):
                st.info(f"**설명:** {api_key_info['description']}")
        
        # 로그아웃 버튼
        if st.button("🚪 로그아웃", use_container_width=True):
            # 세션 상태 초기화
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
        
        st.markdown("---")

    # HITL 설정 사이드바
    with st.sidebar:
        st.header("⚙️ HITL 설정")

        hitl_enabled = st.checkbox("🤚 Human-in-the-Loop 활성화", value=True)

        if hitl_enabled:
            st.subheader("승인 옵션")
            require_tool_approval = st.checkbox("🔧 도구 실행 전 승인", value=True)
            require_final_approval = st.checkbox("✅ 최종 답변 승인", value=False)

        st.markdown("---")
        st.header("📊 서버 상태")

        # 서버 상태 확인
        try:
            headers = {}
            if st.session_state.api_key:
                headers["Authorization"] = f"Bearer {st.session_state.api_key}"

            response = requests.get(
                f"{BACKEND_URL}/api/user/status",
                headers=headers,
                timeout=5
            )
            if response.status_code == 200:
                status = response.json()
                st.success("✅ 서버 연결됨")
                st.info(f"🤖 에이전트: {'준비됨' if status['agent_ready'] else '초기화 중'}")
                st.info(f"🛠️ 도구: {status['tools_available']}개")
                st.info(f"🔑 인증: {status.get('auth_type', 'unknown')}")

                # HITL 상태 표시
                if 'hitl_config' in status:
                    st.info(f"🤚 HITL: {'활성화됨' if status['hitl_config'].get('enabled') else '비활성화됨'}")
            else:
                st.error("❌ 서버 오류")
        except:
            st.error("❌ 서버 연결 실패")

        if st.button("🔄 새로고침"):
            st.rerun()

    # 세션 상태 초기화
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = "default"
    if "waiting_for_approval" not in st.session_state:
        st.session_state.waiting_for_approval = False
    if "current_hitl_data" not in st.session_state:
        st.session_state.current_hitl_data = None
    if "current_ws_client" not in st.session_state:
        st.session_state.current_ws_client = None

    # 채팅 기록 표시
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant" and message.get("type") == "hitl":
                # HITL 메시지는 특별한 스타일로 표시
                st.markdown("🤚 **Human Approval 요청**")
                st.markdown(message["content"])
            else:
                st.markdown(message["content"])

    # HITL 승인 대기 중인 경우
    if st.session_state.waiting_for_approval and st.session_state.current_hitl_data:
        hitl_message = st.session_state.current_hitl_data.get("message", "")
        approval_details = extract_approval_details(hitl_message)

        approval_response = render_hitl_approval_ui(
            hitl_message,
            approval_details,
            st.session_state.current_ws_client
        )

        if approval_response:
            st.write(f"🎯 승인 응답: {approval_response}")
            st.write(f"🔗 WebSocket 클라이언트 존재: {st.session_state.current_ws_client is not None}")

            # WebSocket을 통해 승인 응답 전송
            if st.session_state.current_ws_client:
                st.write(f"🔗 연결 상태: {st.session_state.current_ws_client.is_connected}")

                # **중요**: 연결 상태 재확인
                try:
                    # ping/pong 테스트
                    st.session_state.current_ws_client.ws.ping()
                    st.write("✅ WebSocket ping 성공")
                except Exception as ping_e:
                    st.error(f"❌ WebSocket ping 실패: {ping_e}")
                    st.session_state.current_ws_client.is_connected = False

                if st.session_state.current_ws_client.send_approval_response(approval_response,
                                                                             st.session_state.thread_id):
                    # 승인 응답 메시지 추가
                    st.session_state.messages.append({
                        "role": "user",
                        "content": f"[Human Approval] {approval_response}",
                        "type": "approval"
                    })

                    # 상태 초기화
                    st.session_state.waiting_for_approval = False
                    st.session_state.current_hitl_data = None

                    st.success(f"✅ 승인 응답 완료: {approval_response}")

                    # WebSocket 클라이언트가 있으면 계속 응답 받기
                    with st.chat_message("assistant"):
                        response_placeholder = st.empty()
                        full_response = ""

                        # 초기 대기 메시지
                        response_placeholder.markdown("🔄 승인 처리 중... 서버 응답 대기")

                        # 🚨 중요: 서버 워크플로우 재개 대기
                        st.write("⏳ 서버에서 워크플로우를 재개하고 있습니다...")
                        time.sleep(2)  # 2초 초기 대기

                        # 응답 대기 설정
                        max_wait_time = 120  # 2분 대기
                        start_time = time.time()
                        consecutive_none_count = 0
                        max_consecutive_none = 10  # 연속 None 10회까지 허용
                        response_started = False

                        st.write("🔄 AI 응답 대기 중...")

                        while time.time() - start_time < max_wait_time:
                            try:
                                response_data = st.session_state.current_ws_client.receive_response()

                                if response_data is None:
                                    consecutive_none_count += 1

                                    # 응답이 시작되기 전에는 더 관대하게 대기
                                    if not response_started and consecutive_none_count < max_consecutive_none:
                                        elapsed = time.time() - start_time
                                        response_placeholder.markdown(f"🔄 서버 응답 대기 중... ({elapsed:.1f}초)")
                                        time.sleep(0.5)  # 0.5초 대기 후 재시도
                                        continue
                                    # 응답이 시작된 후에는 빠르게 종료
                                    elif response_started and consecutive_none_count >= 3:
                                        st.write("ℹ️ 응답 완료로 판단됨")
                                        break
                                    # 너무 오래 대기했으면 종료
                                    elif consecutive_none_count >= max_consecutive_none:
                                        st.warning("⚠️ 서버 응답 대기 시간 초과")
                                        break
                                    else:
                                        time.sleep(0.2)
                                        continue

                                # 응답 데이터 처리
                                consecutive_none_count = 0  # None 카운트 리셋

                                if response_data["type"] == "response_chunk":
                                    chunk = response_data["data"]
                                    full_response += chunk
                                    response_placeholder.markdown(full_response + "▌")
                                    response_started = True
                                    st.write(f"📥 응답 수신 중... ({len(full_response)}자)")
                                    time.sleep(0.01)

                                elif response_data["type"] == "response_complete":
                                    st.write("✅ 응답 완료 신호 수신")
                                    break

                                elif response_data["type"] == "error":
                                    st.error(f"서버 오류: {response_data['data']}")
                                    break

                                elif response_data["type"] in ["approval_received", "approval_processed"]:
                                    st.info(f"📋 승인 상태: {response_data['data']}")
                                    # 승인 처리 메시지는 무시하고 계속 대기
                                    continue

                                elif response_data["type"] == "heartbeat":
                                    st.write("💓 서버 연결 확인")
                                    continue

                                else:
                                    st.write(f"🔍 기타 응답: {response_data['type']}")
                                    continue

                            except Exception as e:
                                st.error(f"응답 수신 중 오류: {e}")
                                time.sleep(0.5)
                                continue

                        # 최종 응답 처리
                        if full_response:
                            response_placeholder.markdown(full_response)
                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": full_response
                            })
                            st.success(f"✅ 최종 답변 수신 완료 ({len(full_response)}자)")
                        else:
                            response_placeholder.markdown("⚠️ 서버로부터 응답을 받지 못했습니다.")
                            st.warning("서버 처리가 지연되고 있을 수 있습니다.")

                        # 🚀 모든 처리 완료 후에만 연결 종료
                        st.write("🔌 WebSocket 연결 종료")
                        st.session_state.current_ws_client.close()
                        st.session_state.current_ws_client = None

                    time.sleep(0.5)
                    st.rerun()
                else:
                    st.error("❌ 승인 응답 전송 실패")
            else:
                # REST API로 전송 (폴백)
                if send_hitl_approval_to_backend(approval_response, st.session_state.thread_id, st.session_state.api_key):
                    st.success(f"✅ 승인 응답 완료: {approval_response}")
                    # 상태 초기화
                    st.session_state.waiting_for_approval = False
                    st.session_state.current_hitl_data = None
                    time.sleep(0.5)
                    st.rerun()
                else:
                    st.error("❌ 승인 응답 전송 실패")

    # 사용자 입력 (승인 대기 중이 아닐 때만)
    if not st.session_state.waiting_for_approval:
        if prompt := st.chat_input("메시지를 입력하세요..."):
            # 사용자 메시지 표시
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # AI 응답
            with st.chat_message("assistant"):
                response_placeholder = st.empty()
                full_response = ""

                # 웹소켓 클라이언트 생성 및 연결
                client = HITLWebSocketClient(
                    f"{BACKEND_WEBSOCKET}/api/user/chat",
                    api_key=st.session_state.api_key
                )

                if client.connect():
                    # HITL 설정을 포함하여 메시지 전송
                    enhanced_prompt = prompt
                    if hitl_enabled:
                        enhanced_prompt += f" [HITL_CONFIG: enabled={hitl_enabled}]"

                    if client.send_message(enhanced_prompt, st.session_state.thread_id):
                        # 스트리밍 응답 수신
                        while True:
                            response_data = client.receive_response()
                            if response_data is None:
                                break

                            if response_data["type"] == "response_chunk":
                                chunk = response_data["data"]
                                full_response += chunk
                                response_placeholder.markdown(full_response + "▌")
                                time.sleep(0.01)

                            elif response_data["type"] == "hitl_request":
                                # HITL 요청 처리
                                hitl_data = response_data["data"]

                                # HITL 메시지 저장
                                st.session_state.messages.append({
                                    "role": "assistant",
                                    "content": hitl_data,
                                    "type": "hitl"
                                })

                                # HITL 대기 상태 설정
                                st.session_state.waiting_for_approval = True
                                st.session_state.current_hitl_data = {"message": hitl_data}
                                st.session_state.current_ws_client = client  # 클라이언트 저장

                                response_placeholder.markdown("🤚 **Human Approval이 필요합니다. 아래에서 승인해주세요.**")
                                break

                            elif response_data["type"] == "response_complete":
                                break

                            elif response_data["type"] == "error":
                                st.error(response_data["data"])
                                break

                        if not st.session_state.waiting_for_approval:
                            response_placeholder.markdown(full_response)
                            client.close()

                    else:
                        client.close()
                else:
                    response_placeholder.markdown("❌ 연결 실패. 서버가 실행 중인지 확인하세요.")
                    full_response = "연결 실패"

            # AI 응답 저장 (HITL 요청이 아닌 경우만)
            if full_response and not st.session_state.waiting_for_approval:
                st.session_state.messages.append({"role": "assistant", "content": full_response})

            # HITL 요청이 있으면 페이지 새로고침
            if st.session_state.waiting_for_approval:
                st.rerun()


if __name__ == "__main__":
    main()