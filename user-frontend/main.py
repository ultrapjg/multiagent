import streamlit as st
import websocket
import json
import threading
import time
import queue
from typing import Optional
import re
import requests

st.set_page_config(
    page_title="🤖 AI Assistant with HITL",
    page_icon="🤖",
    layout="wide"
)


class HITLWebSocketClient:
    """Human-in-the-Loop 지원 웹소켓 클라이언트"""

    def __init__(self, url: str):
        self.url = url
        self.ws = None
        self.is_connected = False
        self.response_buffer = ""
        self.approval_queue = queue.Queue()
        self.waiting_for_approval = False
        self.current_approval_context = None

    def connect(self):
        """웹소켓 연결"""
        try:
            self.ws = websocket.create_connection(self.url)
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
            # HITL 승인 응답임을 나타내는 특별한 형식으로 전송
            data = {
                "message": f"[HITL_APPROVAL]{approval_response}",
                "thread_id": thread_id,
                "type": "hitl_approval"
            }
            self.ws.send(json.dumps(data))
            st.info(f"승인 응답 전송: {approval_response}")
            return True
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


def send_hitl_approval_to_backend(approval_response: str, thread_id: str = "default"):
    """백엔드로 HITL 승인 응답 전송 (REST API 사용)"""
    try:
        response = requests.post(
            "http://localhost:8000/api/user/hitl/approve",
            json={
                "approval": approval_response,
                "thread_id": thread_id
            },
            headers={"Authorization": "Bearer user_token"},
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

    # 낮은 신뢰도 승인
    elif any(keyword in message_lower for keyword in ["낮은 신뢰도", "신뢰도는 낮습니다", "low confidence"]):
        details["type"] = "low_confidence"
        details["options"] = ["approved", "rejected", "need_input"]

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
        st.warning(f"🔧 도구 '{tool_name}' 실행 승인이 필요합니다.")

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

    elif approval_details["type"] == "low_confidence":
        confidence = approval_details.get("confidence", 0.0)
        st.warning(f"⚠️ 낮은 신뢰도 ({confidence:.2f}) 결과입니다. 계속 진행하시겠습니까?")

        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("✅ 진행", key="approve_confidence", type="primary"):
                approval_response = "approved"
        with col2:
            if st.button("❌ 중단", key="reject_confidence"):
                approval_response = "rejected"
        with col3:
            if st.button("💭 추가 정보 제공", key="need_input"):
                approval_response = "need_input"

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
            "options": approval_details["options"]
        })

    st.markdown("---")
    return approval_response


def main():
    st.title("🤖 AI Assistant with Human-in-the-Loop")
    st.markdown("Human-in-the-Loop 기능이 지원되는 LangGraph MCP 에이전트와 대화해보세요!")

    # HITL 설정 사이드바
    with st.sidebar:
        st.header("⚙️ HITL 설정")

        hitl_enabled = st.checkbox("🤚 Human-in-the-Loop 활성화", value=True)

        if hitl_enabled:
            st.subheader("승인 옵션")
            require_tool_approval = st.checkbox("🔧 도구 실행 전 승인", value=True)
            require_low_confidence_approval = st.checkbox("⚠️ 낮은 신뢰도 시 승인", value=True)
            require_final_approval = st.checkbox("✅ 최종 답변 승인", value=False)

            confidence_threshold = st.slider(
                "신뢰도 임계값",
                min_value=0.0,
                max_value=1.0,
                value=0.7,
                step=0.1,
                help="이 값 이하의 신뢰도에서 승인 요청"
            )

        st.markdown("---")
        st.header("📊 서버 상태")

        # 서버 상태 확인
        try:
            response = requests.get(
                "http://localhost:8000/api/user/status",
                headers={"Authorization": "Bearer user_token"},
                timeout=5
            )
            if response.status_code == 200:
                status = response.json()
                st.success("✅ 서버 연결됨")
                st.info(f"🤖 에이전트: {'준비됨' if status['agent_ready'] else '초기화 중'}")
                st.info(f"🛠️ 도구: {status['tools_available']}개")

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
            # WebSocket을 통해 승인 응답 전송
            if st.session_state.current_ws_client:
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

                        # 승인 후 계속되는 응답 수신
                        while True:
                            response_data = st.session_state.current_ws_client.receive_response()
                            if response_data is None:
                                break

                            if response_data["type"] == "response_chunk":
                                chunk = response_data["data"]
                                full_response += chunk
                                response_placeholder.markdown(full_response + "▌")
                                time.sleep(0.01)
                            elif response_data["type"] == "response_complete":
                                break
                            elif response_data["type"] == "error":
                                st.error(response_data["data"])
                                break

                        if full_response:
                            response_placeholder.markdown(full_response)
                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": full_response
                            })

                        # 클라이언트 정리
                        st.session_state.current_ws_client.close()
                        st.session_state.current_ws_client = None

                    time.sleep(0.5)
                    st.rerun()
                else:
                    st.error("❌ 승인 응답 전송 실패")
            else:
                # REST API로 전송 (폴백)
                if send_hitl_approval_to_backend(approval_response, st.session_state.thread_id):
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
                client = HITLWebSocketClient("ws://localhost:8000/user/chat")

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

    # 도움말
    with st.expander("💡 HITL 사용법", expanded=False):
        st.markdown("""
        ### Human-in-the-Loop 기능 사용법

        1. **고위험 작업 승인**: AI가 삭제, 수정 등 위험한 작업을 수행하려 할 때 승인 요청
        2. **낮은 신뢰도 승인**: 결과의 신뢰도가 낮을 때 계속 진행할지 확인
        3. **최종 답변 승인**: 답변을 사용자에게 제공하기 전 최종 검토
        4. **추가 정보 요청**: AI가 더 정확한 답변을 위해 추가 정보 요청

        ### 테스트 쿼리 예시
        - "중요한 파일을 삭제해줘" (고위험 작업 승인 테스트)
        - "시스템 설정을 변경해줘" (도구 승인 테스트)
        - "복잡한 분석을 해줘" (낮은 신뢰도 테스트)
        - "안전한 작업을 해줘" (일반 작업)
        """)


if __name__ == "__main__":
    main()