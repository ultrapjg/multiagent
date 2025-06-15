import streamlit as st
import requests
import json
from typing import Dict, List, Any
import re
from datetime import datetime
import pandas as pd
import os
import time

BACKEND_URL=os.getenv("BACKEND_URL", "http://backend:8000")

st.set_page_config(
    page_title="⚙️ 운영자 대시보드",
    page_icon="⚙️",
    layout="wide"
)

class AdminAPIClient:
    """운영자용 API 클라이언트"""
    
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.headers = {"Authorization": "Bearer admin_token"}
    
    def get_tools(self) -> List[Dict]:
        """도구 목록 조회"""
        try:
            response = requests.get(f"{self.base_url}/api/admin/tools", headers=self.headers)
            response.raise_for_status()
            return response.json().get("tools", [])
        except Exception as e:
            st.error(f"도구 조회 실패: {e}")
            return []
    
    def add_tool(self, name: str, config: Dict) -> bool:
        """도구 추가"""
        try:
            data = {"name": name, "config": config}
            response = requests.post(f"{self.base_url}/api/admin/tools", headers=self.headers, json=data)
            response.raise_for_status()
            return True
        except Exception as e:
            st.error(f"도구 추가 실패: {e}")
            return False
    
    def delete_tool(self, name: str) -> bool:
        """도구 삭제"""
        try:
            response = requests.delete(f"{self.base_url}/api/admin/tools/{name}", headers=self.headers)
            response.raise_for_status()
            return True
        except Exception as e:
            st.error(f"도구 삭제 실패: {e}")
            return False
    
    def apply_changes(self) -> bool:
        """변경사항 적용"""
        try:
            response = requests.post(f"{self.base_url}/api/admin/tools/apply", headers=self.headers)
            response.raise_for_status()
            return True
        except Exception as e:
            st.error(f"변경사항 적용 실패: {e}")
            return False
    
    def get_agent_status(self) -> Dict:
        """에이전트 상태 조회"""
        try:
            response = requests.get(f"{self.base_url}/api/admin/agent/status", headers=self.headers)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            st.error(f"상태 조회 실패: {e}")
            return {}
    
    def get_stats(self) -> Dict:
        """통계 조회"""
        try:
            response = requests.get(f"{self.base_url}/api/admin/stats", headers=self.headers)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            st.error(f"통계 조회 실패: {e}")
            return {}
    
    def reinitialize_agent(self, model_name: str) -> bool:
        """에이전트 재초기화"""
        try:
            data = {"model_name": model_name}
            response = requests.post(f"{self.base_url}/api/admin/agent/reinitialize", headers=self.headers, json=data)
            response.raise_for_status()
            return True
        except Exception as e:
            st.error(f"에이전트 재초기화 실패: {e}")
            return False

    # 필터 관련 메서드 추가
    def get_filters(self) -> List[Dict]:
        """필터 규칙 목록 조회"""
        try:
            response = requests.get(f"{self.base_url}/filters/filters")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            st.error(f"필터 규칙 조회 실패: {e}")
            return []
    
    def add_filter(self, name: str, pattern: str) -> bool:
        """필터 규칙 추가"""
        try:
            data = {"name": name, "pattern": pattern}
            response = requests.post(f"{self.base_url}/filters/filters", json=data)
            response.raise_for_status()
            return True
        except Exception as e:
            st.error(f"필터 규칙 추가 실패: {e}")
            return False
    
    def delete_filter(self, rule_id: int) -> bool:
        """필터 규칙 삭제"""
        try:
            response = requests.delete(f"{self.base_url}/filters/filters/{rule_id}")
            response.raise_for_status()
            return True
        except Exception as e:
            st.error(f"필터 규칙 삭제 실패: {e}")
            return False
    
    def test_filter(self, text: str) -> Dict:
        """필터 테스트"""
        try:
            data = {"text": text}
            response = requests.post(f"{self.base_url}/filters/filters/test", json=data)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            st.error(f"필터 테스트 실패: {e}")
            return {}
    
    def get_filter_stats(self) -> Dict:
        """필터 통계 조회"""
        try:
            response = requests.get(f"{self.base_url}/filters/filters/stats")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            st.error(f"필터 통계 조회 실패: {e}")
            return {}
    
    def reload_filters(self) -> bool:
        """필터 규칙 리로드"""
        try:
            response = requests.post(f"{self.base_url}/filters/filters/reload")
            response.raise_for_status()
            return True
        except Exception as e:
            st.error(f"필터 규칙 리로드 실패: {e}")
            return False

    # API 키 관련 메서드 추가
    def get_api_keys(self, include_inactive: bool = False) -> List[Dict]:
        """API 키 목록 조회"""
        try:
            response = requests.get(
                f"{self.base_url}/api/admin/api-keys",
                headers=self.headers,
                params={"include_inactive": include_inactive}
            )
            response.raise_for_status()
            return response.json().get("api_keys", [])
        except Exception as e:
            st.error(f"API 키 조회 실패: {e}")
            return []
    
    def create_api_key(self, name: str, description: str = "", expires_days: int = None) -> Dict:
        """API 키 생성"""
        try:
            data = {
                "name": name,
                "description": description,
                "expires_days": expires_days
            }
            response = requests.post(
                f"{self.base_url}/api/admin/api-keys",
                headers=self.headers,
                json=data
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            st.error(f"API 키 생성 실패: {e}")
            return {}
    
    def update_api_key(self, key_id: int, name: str = None, description: str = None, is_active: bool = None) -> bool:
        """API 키 업데이트"""
        try:
            data = {}
            if name is not None:
                data["name"] = name
            if description is not None:
                data["description"] = description
            if is_active is not None:
                data["is_active"] = is_active
            
            response = requests.put(
                f"{self.base_url}/api/admin/api-keys/{key_id}",
                headers=self.headers,
                json=data
            )
            response.raise_for_status()
            return True
        except Exception as e:
            st.error(f"API 키 업데이트 실패: {e}")
            return False
    
    def delete_api_key(self, key_id: int, soft_delete: bool = True) -> bool:
        """API 키 삭제"""
        try:
            response = requests.delete(
                f"{self.base_url}/api/admin/api-keys/{key_id}",
                headers=self.headers,
                params={"soft_delete": soft_delete}
            )
            response.raise_for_status()
            return True
        except Exception as e:
            st.error(f"API 키 삭제 실패: {e}")
            return False
    
    def get_api_key_stats(self) -> Dict:
        """API 키 통계 조회"""
        try:
            response = requests.get(f"{self.base_url}/api/admin/api-keys/stats", headers=self.headers)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            st.error(f"API 키 통계 조회 실패: {e}")
            return {}


def render_api_key_management_tab(api_client: AdminAPIClient):
    """API 키 관리 탭 렌더링"""
    st.subheader("🔑 API 키 관리")
    
    # API 키 통계 표시
    col1, col2, col3, col4 = st.columns(4)
    
    api_key_stats = api_client.get_api_key_stats()
    
    with col1:
        st.metric(
            label="🔑 전체 API 키",
            value=api_key_stats.get("total_keys", 0)
        )
    
    with col2:
        st.metric(
            label="✅ 활성 API 키",
            value=api_key_stats.get("active_keys", 0)
        )
    
    with col3:
        st.metric(
            label="⏰ 만료된 키",
            value=api_key_stats.get("expired_keys", 0)
        )
    
    with col4:
        st.metric(
            label="📊 최근 사용",
            value=api_key_stats.get("recent_used_keys", 0)
        )
    
    st.markdown("---")
    
    # API 키 목록과 생성 섹션
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.write("**현재 API 키 목록:**")
        
        # 필터 옵션
        col_filter1, col_filter2 = st.columns(2)
        with col_filter1:
            include_inactive = st.checkbox("비활성화된 키 포함", value=False)
        with col_filter2:
            if st.button("🔄 새로고침", key="refresh_api_keys"):
                st.rerun()
        
        api_keys = api_client.get_api_keys(include_inactive=include_inactive)
        
        if api_keys:
            for api_key in api_keys:
                # 상태에 따른 색상 결정
                if not api_key['is_active']:
                    status_color = "🔴"
                    status_text = "비활성화"
                elif api_key.get('is_expired', False):
                    status_color = "🟠"
                    status_text = "만료됨"
                else:
                    status_color = "🟢"
                    status_text = "활성"
                
                with st.expander(f"{status_color} {api_key['name']} ({status_text})"):
                    col_info, col_action = st.columns([3, 1])
                    
                    with col_info:
                        st.write(f"**ID:** {api_key['id']}")
                        st.write(f"**키 미리보기:** {api_key['key_preview']}")
                        st.write(f"**설명:** {api_key['description'] or '설명 없음'}")
                        st.write(f"**생성일:** {api_key['created_at']}")
                        
                        if api_key['expires_at']:
                            st.write(f"**만료일:** {api_key['expires_at']}")
                        else:
                            st.write("**만료일:** 만료되지 않음")
                        
                        if api_key['last_used_at']:
                            st.write(f"**마지막 사용:** {api_key['last_used_at']}")
                        else:
                            st.write("**마지막 사용:** 사용 안됨")
                    
                    with col_action:
                        # 활성화/비활성화 토글
                        if api_key['is_active']:
                            if st.button("⏸️ 비활성화", key=f"deactivate_{api_key['id']}"):
                                if api_client.update_api_key(api_key['id'], is_active=False):
                                    st.success("API 키가 비활성화되었습니다!")
                                    st.rerun()
                        else:
                            if st.button("▶️ 활성화", key=f"activate_{api_key['id']}"):
                                if api_client.update_api_key(api_key['id'], is_active=True):
                                    st.success("API 키가 활성화되었습니다!")
                                    st.rerun()
                        
                        # 삭제 버튼
                        if st.button("🗑️ 삭제", key=f"delete_api_key_{api_key['id']}"):
                            if api_client.delete_api_key(api_key['id'], soft_delete=True):
                                st.success("API 키가 삭제되었습니다!")
                                st.rerun()
                        
                        # 수정 버튼
                        if st.button("✏️ 수정", key=f"edit_api_key_{api_key['id']}"):
                            st.session_state[f"editing_key_{api_key['id']}"] = True
                        
                        # 수정 폼 (세션 상태에 따라 표시)
                        if st.session_state.get(f"editing_key_{api_key['id']}", False):
                            st.write("**수정:**")
                            new_name = st.text_input(
                                "이름",
                                value=api_key['name'],
                                key=f"edit_name_{api_key['id']}"
                            )
                            new_desc = st.text_area(
                                "설명",
                                value=api_key['description'],
                                key=f"edit_desc_{api_key['id']}"
                            )
                            
                            col_save, col_cancel = st.columns(2)
                            with col_save:
                                if st.button("💾 저장", key=f"save_{api_key['id']}"):
                                    if api_client.update_api_key(
                                        api_key['id'], 
                                        name=new_name, 
                                        description=new_desc
                                    ):
                                        st.success("API 키가 수정되었습니다!")
                                        del st.session_state[f"editing_key_{api_key['id']}"]
                                        st.rerun()
                            
                            with col_cancel:
                                if st.button("❌ 취소", key=f"cancel_{api_key['id']}"):
                                    del st.session_state[f"editing_key_{api_key['id']}"]
                                    st.rerun()
        else:
            st.info("등록된 API 키가 없습니다.")
    
    with col2:
        st.write("**새 API 키 생성:**")
        
        with st.form("create_api_key_form"):
            api_key_name = st.text_input(
                "API 키 이름 *",
                placeholder="예: 프론트엔드 앱"
            )
            
            api_key_description = st.text_area(
                "설명",
                placeholder="이 API 키의 용도를 설명하세요...",
                height=100
            )
            
            # 만료 설정
            expires_option = st.selectbox(
                "만료 설정",
                ["만료되지 않음", "30일", "90일", "180일", "365일", "사용자 정의"]
            )
            
            expires_days = None
            if expires_option == "30일":
                expires_days = 30
            elif expires_option == "90일":
                expires_days = 90
            elif expires_option == "180일":
                expires_days = 180
            elif expires_option == "365일":
                expires_days = 365
            elif expires_option == "사용자 정의":
                expires_days = st.number_input(
                    "만료일 (일 단위)",
                    min_value=1,
                    max_value=3650,
                    value=30
                )
            
            # 보안 경고
            st.warning("⚠️ **보안 주의사항**\n- API 키는 생성 시에만 표시됩니다\n- 안전한 곳에 보관하세요\n- 정기적으로 키를 교체하세요")
            
            submitted = st.form_submit_button("🔑 API 키 생성", use_container_width=True)
            
            if submitted and api_key_name:
                result = api_client.create_api_key(
                    name=api_key_name,
                    description=api_key_description,
                    expires_days=expires_days
                )
                
                if result and 'api_key' in result:
                    st.success("✅ API 키가 성공적으로 생성되었습니다!")
                    
                    # 생성된 API 키 표시 (중요!)
                    st.markdown("### 🔑 생성된 API 키")
                    st.code(result['api_key'], language=None)
                    st.error("⚠️ **중요**: 이 키는 다시 표시되지 않습니다. 지금 복사하여 안전한 곳에 보관하세요!")
                    
                    # 키 정보 표시
                    st.json({
                        "id": result['id'],
                        "name": result['name'],
                        "description": result['description'],
                        "created_at": result['created_at'],
                        "expires_at": result['expires_at']
                    })
                    
                    time.sleep(3)  # 3초 후 새로고침
                    st.rerun()
            elif submitted:
                st.error("API 키 이름을 입력해주세요.")
    
    # 빠른 액션 섹션
    st.markdown("---")
    st.subheader("🚀 빠른 액션")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔄 모든 키 새로고침", use_container_width=True):
            st.rerun()
    
    with col2:
        if st.button("⚠️ 만료된 키 정리", use_container_width=True):
            # 만료된 키들을 비활성화
            api_keys = api_client.get_api_keys(include_inactive=True)
            expired_count = 0
            for key in api_keys:
                if key.get('is_expired', False) and key['is_active']:
                    if api_client.update_api_key(key['id'], is_active=False):
                        expired_count += 1
            
            if expired_count > 0:
                st.success(f"✅ {expired_count}개의 만료된 키를 비활성화했습니다!")
                st.rerun()
            else:
                st.info("정리할 만료된 키가 없습니다.")
    
    with col3:
        if st.button("📊 사용 통계 보기", use_container_width=True):
            stats = api_client.get_api_key_stats()
            if stats:
                st.json(stats)


def check_admin_login():
    """관리자 로그인 확인"""
    if "admin_logged_in" not in st.session_state:
        st.session_state.admin_logged_in = False
    
    if not st.session_state.admin_logged_in:
        st.title("🔐 관리자 로그인")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            with st.form("login_form"):
                username = st.text_input("사용자명")
                password = st.text_input("비밀번호", type="password")
                
                if st.form_submit_button("로그인", use_container_width=True):
                    if username == "admin" and password == "admin123":
                        st.session_state.admin_logged_in = True
                        st.success("로그인 성공!")
                        st.rerun()
                    else:
                        st.error("잘못된 인증 정보입니다.")
        return False
    return True


def render_filter_management_tab(api_client: AdminAPIClient):
    """필터 관리 탭 렌더링"""
    st.subheader("🛡️ 입력 필터 관리")
    
    # 필터 통계 표시
    col1, col2, col3, col4 = st.columns(4)
    
    filter_stats = api_client.get_filter_stats()
    
    with col1:
        st.metric(
            label="🛡️ 활성 필터 규칙",
            value=filter_stats.get("total_rules", 0)
        )
    
    with col2:
        st.metric(
            label="📊 필터 상태",
            value="활성화" if filter_stats.get("filter_status") == "active" else "비활성화"
        )
    
    with col3:
        if st.button("🔄 필터 리로드", use_container_width=True):
            if api_client.reload_filters():
                st.success("필터 규칙이 리로드되었습니다!")
                st.rerun()
    
    with col4:
        if st.button("📊 통계 새로고침", use_container_width=True):
            st.rerun()
    
    st.markdown("---")
    
    # 필터 규칙 목록과 추가 섹션
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.write("**현재 필터 규칙:**")
        filters = api_client.get_filters()
        
        if filters:
            for filter_rule in filters:
                with st.expander(f"🛡️ {filter_rule['name']}"):
                    col_info, col_action = st.columns([3, 1])
                    
                    with col_info:
                        st.write(f"**규칙 ID:** {filter_rule.get('id', 'N/A')}")
                        st.write(f"**패턴:** `{filter_rule['pattern']}`")
                        
                        if filter_rule.get('created_at'):
                            created_at = datetime.fromisoformat(filter_rule['created_at'].replace('Z', '+00:00'))
                            st.write(f"**생성일:** {created_at.strftime('%Y-%m-%d %H:%M:%S')}")
                        
                        # 패턴 유효성 검사
                        try:
                            re.compile(filter_rule['pattern'])
                            st.success("✅ 유효한 정규식 패턴")
                        except re.error as e:
                            st.error(f"❌ 잘못된 정규식 패턴: {e}")
                    
                    with col_action:
                        if st.button("🗑️ 삭제", key=f"delete_filter_{filter_rule.get('id')}"):
                            if api_client.delete_filter(filter_rule.get('id')):
                                st.success(f"필터 규칙 '{filter_rule['name']}'이 삭제되었습니다!")
                                st.rerun()
        else:
            st.info("등록된 필터 규칙이 없습니다.")
    
    with col2:
        st.write("**새 필터 규칙 추가:**")
        
        with st.form("add_filter_form"):
            filter_name = st.text_input(
                "규칙 이름",
                placeholder="예: 이메일 패턴"
            )
            
            filter_pattern = st.text_area(
                "정규식 패턴",
                placeholder="예: [a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}",
                height=100
            )
            
            # 일반적인 패턴 예시
            st.write("**일반적인 패턴 예시:**")
            pattern_examples = {
                "이메일": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
                "전화번호": r"01[0-9]-\d{3,4}-\d{4}",
                "주민등록번호": r"\d{6}-[1-4]\d{6}",
                "신용카드": r"\d{4}-\d{4}-\d{4}-\d{4}",
                "욕설": r"(바보|멍청이|쓰레기)",
                "URL": r"https?://[^\s]+",
                "IP 주소": r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b"
            }
            
            selected_example = st.selectbox(
                "패턴 예시 선택:",
                ["직접 입력"] + list(pattern_examples.keys())
            )
            
            if selected_example != "직접 입력":
                st.code(pattern_examples[selected_example])
                if st.button("패턴 적용", key="apply_pattern"):
                    st.session_state.temp_pattern = pattern_examples[selected_example]
                    st.rerun()
            
            # 세션 상태에서 임시 패턴 적용
            if hasattr(st.session_state, 'temp_pattern'):
                filter_pattern = st.session_state.temp_pattern
                del st.session_state.temp_pattern
            
            # 패턴 유효성 실시간 검사
            if filter_pattern:
                try:
                    re.compile(filter_pattern)
                    st.success("✅ 유효한 정규식 패턴")
                except re.error as e:
                    st.error(f"❌ 잘못된 정규식: {e}")
            
            submitted = st.form_submit_button("필터 추가", use_container_width=True)
            
            if submitted and filter_name and filter_pattern:
                try:
                    # 패턴 유효성 최종 검사
                    re.compile(filter_pattern)
                    
                    if api_client.add_filter(filter_name, filter_pattern):
                        st.success("필터 규칙이 성공적으로 추가되었습니다!")
                        st.rerun()
                except re.error as e:
                    st.error(f"정규식 패턴 오류: {e}")
                except Exception as e:
                    st.error(f"필터 추가 실패: {e}")
    
    # 필터 테스트 섹션
    st.markdown("---")
    st.subheader("🧪 필터 테스트")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        test_text = st.text_area(
            "테스트할 텍스트를 입력하세요:",
            placeholder="예: 제 이메일은 test@example.com 입니다.",
            height=100
        )
        
        if st.button("🧪 필터 테스트 실행", use_container_width=True):
            if test_text:
                test_result = api_client.test_filter(test_text)
                
                if test_result:
                    if test_result.get("is_sensitive"):
                        st.error("🚨 민감한 내용이 감지되었습니다!")
                        st.write(f"**메시지:** {test_result.get('message', '')}")
                        
                        matched_rules = test_result.get("matched_rules", [])
                        if matched_rules:
                            st.write("**매칭된 규칙들:**")
                            for rule in matched_rules:
                                st.write(f"- **{rule['name']}** (ID: {rule['id']})")
                                st.code(rule['pattern'])
                    else:
                        st.success("✅ 민감한 내용이 감지되지 않았습니다.")
                        st.write(f"**메시지:** {test_result.get('message', '')}")
            else:
                st.warning("테스트할 텍스트를 입력해주세요.")
    
    with col2:
        st.write("**테스트 예시:**")
        test_examples = [
            "안녕하세요!",
            "제 이메일은 test@example.com 입니다.",
            "전화번호는 010-1234-5678 입니다.",
            "주민등록번호는 123456-1234567 입니다.",
            "이 바보야!",
            "https://example.com 에 방문해보세요."
        ]
        
        for i, example in enumerate(test_examples):
            if st.button(f"예시 {i+1}", key=f"test_example_{i}", use_container_width=True):
                st.session_state.test_text = example
                st.rerun()


def main():
    if not check_admin_login():
        return
    
    st.title("⚙️ LangGraph MCP 에이전트 운영자 대시보드")
    
    # 로그아웃 버튼
    col1, col2, col3 = st.columns([6, 1, 1])
    with col3:
        if st.button("로그아웃"):
            st.session_state.admin_logged_in = False
            st.rerun()
    
    st.markdown("---")
    
    # API 클라이언트 초기화
    api_client = AdminAPIClient(BACKEND_URL)
    
    # 탭 생성 - 필터 관리 탭 추가
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "📊 대시보드", 
        "🔧 도구 관리", 
        "🛡️ 필터 관리",  # 새로 추가
        "🔑 API 키 관리",  # 새로 추가
        "🤖 에이전트 관리", 
        "📈 모니터링", 
        "📋 사용자 요청 조회"
    ])
    
    # =============================================================================
    # 대시보드 탭
    # =============================================================================
    with tab1:
        st.subheader("📊 시스템 현황")
        
        # 통계 정보
        stats = api_client.get_stats()
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric(
                label="🛠️ 활성 도구",
                value=stats.get("active_tools", 0)
            )
        
        with col2:
            st.metric(
                label="🤖 에이전트 상태",
                value="초기화됨" if stats.get("agent_initialized") else "초기화 안됨"
            )
        
        with col3:
            st.metric(
                label="🛡️ 필터 규칙",
                value=stats.get("filter_stats", {}).get("total_rules", 0)
            )
        
        with col4:
            st.metric(
                label="💬 총 대화",
                value=stats.get("total_conversations", 0)
            )
        
        with col5:
            st.metric(
                label="👥 일일 사용자",
                value=stats.get("daily_users", 0)
            )
        
        st.markdown("---")
        
        # 에이전트 상태 상세 정보
        agent_status = api_client.get_agent_status()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🤖 에이전트 상태")
            if agent_status:
                agent_service_status = agent_status.get("agent_service", {})
                st.write(f"**초기화 여부:** {'✅' if agent_service_status.get('is_initialized') else '❌'}")
                st.write(f"**모델:** {agent_service_status.get('model_name', 'Unknown')}")
                st.write(f"**도구 수:** {agent_service_status.get('tools_count', 0)}개")
                st.write(f"**MCP 클라이언트:** {'✅' if agent_service_status.get('mcp_client_active') else '❌'}")
                
                # 필터 상태 정보 추가
                filter_status = agent_status.get("filter_status", {})
                st.write(f"**필터 규칙:** {filter_status.get('rules_count', 0)}개")
                st.write(f"**필터 활성화:** {'✅' if filter_status.get('active') else '❌'}")
            else:
                st.error("에이전트 상태를 가져올 수 없습니다.")
        
        with col2:
            st.subheader("🔄 빠른 액션")
            if st.button("🔄 에이전트 재시작", use_container_width=True):
                if api_client.apply_changes():
                    st.success("에이전트가 재시작되었습니다!")
                    st.rerun()
            
            if st.button("🛡️ 필터 리로드", use_container_width=True):
                if api_client.reload_filters():
                    st.success("필터 규칙이 리로드되었습니다!")
                    st.rerun()
            
            if st.button("📊 상태 새로고침", use_container_width=True):
                st.rerun()
    
    # =============================================================================
    # 도구 관리 탭
    # =============================================================================
    with tab2:
        st.subheader("🔧 MCP 도구 관리")
        
        # 현재 도구 목록
        tools = api_client.get_tools()
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.write("**등록된 도구:**")
            if tools:
                for tool in tools:
                    with st.expander(f"🛠️ {tool['name']}"):
                        col_info, col_action = st.columns([3, 1])
                        
                        with col_info:
                            st.write(f"**Transport:** {tool.get('transport', 'stdio')}")
                            st.write(f"**Command:** {tool.get('command', 'N/A')}")
                            if tool.get('args'):
                                st.write(f"**Args:** {', '.join(tool['args'])}")
                            if tool.get('url'):
                                st.write(f"**URL:** {tool['url']}")
                        
                        with col_action:
                            if st.button("❌ 삭제", key=f"delete_{tool['name']}"):
                                if api_client.delete_tool(tool['name']):
                                    st.success(f"도구 '{tool['name']}'이 삭제되었습니다!")
                                    st.rerun()
            else:
                st.info("등록된 도구가 없습니다.")
        
        with col2:
            st.write("**새 도구 추가:**")
            
            # Smithery 링크
            st.markdown("**[Smithery](https://smithery.ai/)에서 도구 찾기**")
            
            with st.form("add_tool_form"):
                tool_name = st.text_input("도구 이름")
                
                # 도구 타입 선택
                transport_type = st.selectbox(
                    "Transport 타입",
                    ["stdio", "streamable_http"]
                )
                
                if transport_type == "stdio":
                    command = st.text_input("Command", value="python")
                    args_text = st.text_area(
                        "Arguments (한 줄에 하나씩)",
                        placeholder="예:\n/path/to/server.py\n--option\nvalue"
                    )
                    
                    # JSON 직접 입력 옵션
                    use_json = st.checkbox("JSON 직접 입력")
                    if use_json:
                        tool_json = st.text_area(
                            "전체 JSON 설정",
                            height=200,
                            placeholder='{"command": "python", "args": ["/path/to/server.py"], "transport": "stdio"}'
                        )
                    
                else:  # streamable_http
                    url = st.text_input("서버 URL", placeholder="http://localhost:3000/mcp")
                    tool_json = None
                
                submitted = st.form_submit_button("도구 추가", use_container_width=True)
                
                if submitted and tool_name:
                    try:
                        if transport_type == "stdio":
                            if use_json and tool_json:
                                config = json.loads(tool_json)
                            else:
                                args = [arg.strip() for arg in args_text.split('\n') if arg.strip()]
                                config = {
                                    "command": command,
                                    "args": args,
                                    "transport": "stdio"
                                }
                        else:  # streamable_http
                            config = {
                                "url": url,
                                "transport": "streamable_http"
                            }
                        
                        if api_client.add_tool(tool_name, config):
                            st.success("도구가 성공적으로 추가되었습니다!")
                            st.rerun()
                            
                    except json.JSONDecodeError:
                        st.error("올바른 JSON 형식이 아닙니다.")
                    except Exception as e:
                        st.error(f"도구 추가 실패: {e}")
        
        # 변경사항 적용
        st.markdown("---")
        col1, col2, col3 = st.columns([2, 1, 2])
        
        with col2:
            if st.button("🔄 변경사항 적용", use_container_width=True, type="primary"):
                with st.spinner("에이전트 재초기화 중..."):
                    if api_client.apply_changes():
                        st.success("변경사항이 에이전트에 적용되었습니다!")
                        st.rerun()
    
    # =============================================================================
    # 필터 관리 탭 (새로 추가)
    # =============================================================================
    with tab3:
        render_filter_management_tab(api_client)

    # =============================================================================
    # API 키 관리 탭 (새로 추가)
    # =============================================================================
    with tab4:
        render_api_key_management_tab(api_client)    
    
    # =============================================================================
    # 에이전트 관리 탭
    # =============================================================================
    with tab5:
        st.subheader("🤖 에이전트 설정")
        
        # 현재 에이전트 상태
        agent_status = api_client.get_agent_status()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**현재 에이전트 상태:**")
            if agent_status:
                st.json(agent_status)
            else:
                st.error("에이전트 상태를 가져올 수 없습니다.")
        
        with col2:
            st.write("**에이전트 재설정:**")
            
            with st.form("agent_config_form"):
                model_options = [
                    "claude-3-5-sonnet-latest",
                    "claude-3-5-haiku-latest", 
                    "claude-3-7-sonnet-latest",
                    "gpt-4o",
                    "gpt-4o-mini",
                    "qwen2.5:32b",
                ]
                
                selected_model = st.selectbox(
                    "모델 선택",
                    model_options,
                    index=0
                )
                
                custom_prompt = st.text_area(
                    "시스템 프롬프트 (선택사항)",
                    placeholder="사용자 지정 시스템 프롬프트를 입력하세요...",
                    height=100
                )
                
                if st.form_submit_button("에이전트 재초기화", use_container_width=True):
                    with st.spinner("에이전트 재초기화 중..."):
                        if api_client.reinitialize_agent(selected_model):
                            st.success("에이전트가 성공적으로 재초기화되었습니다!")
                            st.rerun()
        
        st.markdown("---")
        
        # 시스템 프롬프트 파일 편집 안내
        st.subheader("📝 시스템 프롬프트 파일 편집")
        st.info("""
        **시스템 프롬프트를 영구적으로 변경하려면:**
        1. `prompts/system_prompt.yaml` 파일을 편집하세요
        2. 변경 후 "변경사항 적용" 버튼을 클릭하세요
        3. 변경사항이 자동으로 반영됩니다
        """)
    
    # =============================================================================
    # 모니터링 탭
    # =============================================================================
    with tab6:
        st.subheader("📈 실시간 모니터링")
        
        # 자동 새로고침 설정
        auto_refresh = st.checkbox("자동 새로고침 (10초마다)")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**서버 상태:**")
            try:
                health_response = requests.get(BACKEND_URL+"/health", timeout=5)
                if health_response.status_code == 200:
                    st.success("✅ 백엔드 서버 정상")
                    health_data = health_response.json()
                    st.json(health_data)
                else:
                    st.error("❌ 백엔드 서버 오류")
            except:
                st.error("❌ 백엔드 서버 연결 실패")
        
        with col2:
            st.write("**에이전트 메트릭:**")
            agent_status = api_client.get_agent_status()
            if agent_status:
                agent_service_status = agent_status.get("agent_service", {})
                # 간단한 메트릭 표시
                metrics = {
                    "초기화 상태": "✅ 완료" if agent_service_status.get('is_initialized') else "❌ 실패",
                    "사용 가능한 도구": f"{agent_service_status.get('tools_count', 0)}개",
                    "모델": agent_service_status.get('model_name', 'Unknown'),
                    "MCP 연결": "✅ 활성" if agent_service_status.get('mcp_client_active') else "❌ 비활성",
                    "필터 규칙": f"{agent_status.get('filter_status', {}).get('rules_count', 0)}개"
                }
                
                for key, value in metrics.items():
                    st.write(f"**{key}:** {value}")
        
        # 필터 상태 모니터링 추가
        st.markdown("---")
        st.subheader("🛡️ 필터 시스템 모니터링")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**필터 통계:**")
            filter_stats = api_client.get_filter_stats()
            if filter_stats:
                st.json(filter_stats)
        
        with col2:
            st.write("**최근 필터 활동:**")
            st.info("필터 활동 로그는 향후 버전에서 구현될 예정입니다.")
        
        # 로그 섹션 (향후 구현)
        st.markdown("---")
        st.subheader("📋 최근 활동 로그")
        st.info("로그 기능은 향후 버전에서 구현될 예정입니다.")
        
        # 자동 새로고침
        if auto_refresh:
            time.sleep(10)
            st.rerun()

    # =============================================================================
    # 사용자 요청 조회
    # =============================================================================
    with tab7:
        st.subheader("📋 요청 목록")

        if st.button("🔄 새로고침", key="main_refresh"):
            st.rerun()

        # 메시지 조회
        success, messages = get_messages()

        if not success:
            st.error(f"메시지 로딩 실패: {messages}")
            st.info(f"FastAPI 백엔드 서버가 실행 중인지 확인해주세요. ({BACKEND_URL})")
            return

        if not messages:
            st.info("아직 메시지가 없습니다. 첫 번째 메시지를 작성해보세요!")
            return

        # 메시지 표시 옵션
        col1, col2 = st.columns([2, 1])
        with col1:
            st.write(f"총 {len(messages)}개의 메시지")
        with col2:
            view_mode = st.selectbox("보기 모드", ["카드뷰", "테이블뷰"])

        st.markdown("---")

        if view_mode == "카드뷰":
            # 카드 형태로 메시지 표시
            for i, message in enumerate(messages):
                with st.container():
                    col1, col2 = st.columns([10, 1])

                    with col1:
                        st.markdown(f"""
                        <div class="message-card">
                            <div class="message-author">👤 {message['author']}</div>
                            <div class="message-time">🕒 {format_datetime(message['created_at'])}</div>
                            <div class="message-content">{message['content']}</div>
                        </div>
                        """, unsafe_allow_html=True)

        else:
            # 테이블 형태로 메시지 표시
            df_data = []
            for message in messages:
                df_data.append({
                    "ID": message['id'],
                    "작성자": message['author'],
                    "메시지": message['content'][:100] + ("..." if len(message['content']) > 100 else ""),
                    "작성시간": format_datetime(message['created_at'])
                })

            df = pd.DataFrame(df_data)

            # 선택 가능한 데이터프레임 (최신 Streamlit 방식)
            selected_rows = st.dataframe(
                df,
                use_container_width=True,
                hide_index=True,
                on_select="rerun",
                selection_mode="single-row"
            )

            # 선택된 행 처리 (새로운 방식)
            if hasattr(selected_rows, 'selection') and selected_rows.selection:
                # selection이 존재하고 rows 속성이 있는 경우
                if hasattr(selected_rows.selection, 'rows') and len(selected_rows.selection.rows) > 0:
                    selected_idx = selected_rows.selection.rows[0]
                    selected_message = messages[selected_idx]

                    st.markdown("### 📄 선택된 메시지 상세")
                    col1, col2 = st.columns([3, 1])

                    with col1:
                        st.markdown(f"""
                                **작성자:** {selected_message['author']}  
                                **작성시간:** {format_datetime(selected_message['created_at'])}  
                                **메시지:**  
                                {selected_message['content']}
                                """)


# API 엔드포인트 설정
API_BASE_URL = BACKEND_URL

def get_messages(limit: int = 100):
    """메시지 조회 API 호출"""
    try:
        response = requests.get(f"{API_BASE_URL}/messages/list?limit={limit}")
        if response.status_code == 200:
            return True, response.json()
        else:
            return False, f"Error: {response.status_code} - {response.text}"
    except requests.exceptions.RequestException as e:
        return False, f"Connection error: {str(e)}"

def format_datetime(datetime_str: str):
    """날짜시간 포맷팅"""
    try:
        dt = datetime.fromisoformat(datetime_str.replace('Z', '+00:00'))
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except:
        return datetime_str

if __name__ == "__main__":
    import time
    main()