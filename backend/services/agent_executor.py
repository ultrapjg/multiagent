import logging
from datetime import datetime
from typing import Dict, List, Any, Optional

from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.prebuilt import ToolNode

import inspect
from typing import Dict, Any, Optional
import json
import asyncio
import re

class AgentExecutorService:
    """MCP 도구 실행을 담당하는 서비스"""

    def __init__(self, model, tools):
        self.model = model
        self.tools = tools
        self.logger = logging.getLogger(__name__)

    async def generate_tool_input(self, tool, user_query: str, state: Dict[str, Any]) -> Dict[str, Any]:
        """도구별 맞춤형 입력 생성"""
        tool_name = getattr(tool, 'name', str(tool))

        try:
            # 1. 도구의 스키마/시그니처 분석
            tool_schema = await self.analyze_tool_schema(tool)

            # 2. LLM을 활용한 도구별 입력 생성
            tool_input = await self.generate_smart_tool_input(
                tool_name=tool_name,
                tool_schema=tool_schema,
                user_query=user_query,
                previous_results=state.get("tool_results", []),
                context=state.get("reasoning_trace", [])
            )

            self.logger.info(f"도구 '{tool_name}' 입력 생성: {tool_input}")
            return tool_input

        except Exception as e:
            self.logger.error(f"도구 '{tool_name}' 입력 생성 실패: {e}")
            # 폴백: 기본 입력
            return self.get_fallback_input(tool_name, user_query)

    async def analyze_tool_schema(self, tool) -> Dict[str, Any]:
        """도구의 스키마/파라미터 분석"""
        tool_name = getattr(tool, 'name', str(tool))

        schema_info = {
            "name": tool_name,
            "parameters": {},
            "required_params": [],
            "optional_params": [],
            "description": ""
        }

        try:
            # MCP 도구의 스키마 정보 추출
            if hasattr(tool, 'args_schema') and tool.args_schema:
                # Pydantic 모델에서 스키마 추출
                if hasattr(tool.args_schema, 'model_json_schema'):
                    json_schema = tool.args_schema.model_json_schema()

                    properties = json_schema.get('properties', {})
                    required = json_schema.get('required', [])

                    for param_name, param_info in properties.items():
                        schema_info["parameters"][param_name] = {
                            "type": param_info.get('type', 'string'),
                            "description": param_info.get('description', ''),
                            "required": param_name in required
                        }

                        if param_name in required:
                            schema_info["required_params"].append(param_name)
                        else:
                            schema_info["optional_params"].append(param_name)

            # 도구의 설명 정보 추출
            if hasattr(tool, 'description'):
                schema_info["description"] = tool.description
            elif hasattr(tool, '__doc__') and tool.__doc__:
                schema_info["description"] = tool.__doc__

            self.logger.info(f"도구 '{tool_name}' 스키마 분석 완료: {schema_info}")

        except Exception as e:
            self.logger.warning(f"도구 '{tool_name}' 스키마 분석 실패: {e}")

        return schema_info

    async def generate_smart_tool_input(self,
                                        tool_name: str,
                                        tool_schema: Dict[str, Any],
                                        user_query: str,
                                        previous_results: List[Dict],
                                        context: List[str]) -> Dict[str, Any]:
        """LLM을 활용한 지능적 도구 입력 생성"""

        # 이전 결과에서 유용한 정보 추출
        previous_info = self.extract_relevant_info(previous_results, tool_name)

        input_generation_prompt = ChatPromptTemplate.from_messages([
            ("system", """당신은 도구 실행을 위한 최적의 입력을 생성하는 전문가입니다.

    도구 정보:
    - 이름: {tool_name}
    - 설명: {tool_description}
    - 필수 파라미터: {required_params}
    - 선택 파라미터: {optional_params}
    - 파라미터 상세: {parameters_detail}

    컨텍스트:
    - 사용자 쿼리: {user_query}
    - 이전 도구 결과: {previous_results}
    - 추론 과정: {reasoning_context}

    **중요 규칙**:
    1. 도구의 스키마에 맞는 파라미터만 생성하세요
    2. 필수 파라미터는 반드시 포함하세요
    3. 사용자 쿼리에서 구체적인 값을 추출하세요
    4. 이전 결과를 활용하여 더 정확한 입력을 만드세요
    5. 파일 경로, 시간, 수량 등은 정확하게 추출하세요

    JSON 형식으로 도구 입력을 생성하세요:
    {{
      "parameter1": "value1",
      "parameter2": "value2"
    }}"""),
            ("human", "위 도구를 실행하기 위한 최적의 입력을 생성해주세요.")
        ])

        try:
            response = await self.model.ainvoke(
                input_generation_prompt.format_messages(
                    tool_name=tool_name,
                    tool_description=tool_schema.get("description", "설명 없음"),
                    required_params=", ".join(tool_schema.get("required_params", [])),
                    optional_params=", ".join(tool_schema.get("optional_params", [])),
                    parameters_detail=json.dumps(tool_schema.get("parameters", {}), ensure_ascii=False, indent=2),
                    user_query=user_query,
                    previous_results=previous_info,
                    reasoning_context=context[-3:] if context else []
                )
            )

            # JSON 파싱 시도
            try:
                tool_input = json.loads(response.content)

                # 필수 파라미터 검증
                required_params = tool_schema.get("required_params", [])
                for param in required_params:
                    if param not in tool_input:
                        self.logger.warning(f"필수 파라미터 '{param}' 누락 - 기본값 추가")
                        tool_input[param] = self.get_default_value_for_param(param, user_query)

                return tool_input

            except json.JSONDecodeError:
                self.logger.warning(f"도구 입력 JSON 파싱 실패 - 폴백 로직 사용")
                return self.get_fallback_input(tool_name, user_query)

        except Exception as e:
            self.logger.error(f"스마트 도구 입력 생성 실패: {e}")
            return self.get_fallback_input(tool_name, user_query)

    def extract_relevant_info(self, previous_results: List[Dict], current_tool: str) -> str:
        """이전 결과에서 현재 도구에 관련된 정보 추출"""
        if not previous_results:
            return "이전 결과 없음"

        relevant_info = []
        for result in previous_results[-3:]:  # 최근 3개만
            tool_name = result.get("tool_name", "")
            output = result.get("output", "")
            success = result.get("success", False)

            if success and output:
                # 파일 경로, 시간, ID 등 유용한 정보 추출
                relevant_info.append(f"[{tool_name}] {output[:200]}...")

        return "\n".join(relevant_info) if relevant_info else "관련 정보 없음"

    def get_fallback_input(self, tool_name: str, user_query: str) -> Dict[str, Any]:
        """도구별 폴백 입력 생성"""
        tool_name_lower = tool_name.lower()

        # 도구 타입별 기본 입력 패턴
        if any(keyword in tool_name_lower for keyword in ['file', 'delete', 'remove']):
            # 파일 관련 도구
            return self.extract_file_params(user_query)
        elif any(keyword in tool_name_lower for keyword in ['time', 'clock', 'date']):
            # 시간 관련 도구
            return self.extract_time_params(user_query)
        elif any(keyword in tool_name_lower for keyword in ['search', 'find', 'query']):
            # 검색 관련 도구
            return self.extract_search_params(user_query)
        elif any(keyword in tool_name_lower for keyword in ['create', 'make', 'generate']):
            # 생성 관련 도구
            return self.extract_creation_params(user_query)
        else:
            # 일반적인 경우
            return {"query": user_query}

    def extract_file_params(self, user_query: str) -> Dict[str, Any]:
        """파일 관련 파라미터 추출"""
        import re

        # 파일 경로 패턴 매칭
        file_patterns = [
            r'["\']([^"\']+\.[a-zA-Z0-9]+)["\']',  # "file.txt" 형태
            r'(\S+\.[a-zA-Z0-9]+)',  # file.txt 형태
            r'([/\\][\w\s/\\.-]+)',  # 경로 형태
        ]

        file_paths = []
        for pattern in file_patterns:
            matches = re.findall(pattern, user_query)
            file_paths.extend(matches)

        if file_paths:
            return {
                "path": file_paths[0],
                "filename": file_paths[0].split('/')[-1].split('\\')[-1],
                "query": user_query
            }
        else:
            return {
                "path": "",
                "query": user_query
            }

    def extract_time_params(self, user_query: str) -> Dict[str, Any]:
        """시간 관련 파라미터 추출"""
        query_lower = user_query.lower()

        if any(keyword in query_lower for keyword in ['현재', 'current', 'now', '지금']):
            return {
                "timezone": "Asia/Seoul",
                "format": "YYYY-MM-DD HH:mm:ss",
                "query": user_query
            }
        else:
            return {"query": user_query}

    def extract_search_params(self, user_query: str) -> Dict[str, Any]:
        """검색 관련 파라미터 추출"""
        # 검색어에서 따옴표나 특수 문자 제거
        clean_query = user_query.strip('"\'')

        return {
            "query": clean_query,
            "limit": 10,
            "type": "all"
        }

    def extract_creation_params(self, user_query: str) -> Dict[str, Any]:
        """생성 관련 파라미터 추출"""
        return {
            "content": user_query,
            "format": "text",
            "query": user_query
        }

    def get_default_value_for_param(self, param_name: str, user_query: str) -> Any:
        """파라미터별 기본값 생성"""
        param_lower = param_name.lower()

        if param_lower in ['path', 'file', 'filename']:
            return ""
        elif param_lower in ['query', 'text', 'content', 'message']:
            return user_query
        elif param_lower in ['limit', 'count', 'max']:
            return 10
        elif param_lower in ['format', 'type']:
            return "text"
        elif param_lower in ['timezone']:
            return "Asia/Seoul"
        else:
            return user_query

    async def execute_tools(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """실제 MCP 도구 실행 - 다중 도구 지원"""
        self.logger.info("MCP 도구 실행 단계...")

        # 계획이 생략되었거나 추가 도구가 불필요한 경우 건너뛰기
        if state["current_step"] in ["plan_skipped", "plan_failed"]:
            self.logger.info("계획 단계에서 도구 실행이 불필요하다고 판단됨 - 도구 실행 생략")
            state["current_step"] = "tools_skipped"
            return state

        # 이미 충분한 결과가 있는지 재확인
        if state["evaluation_results"]:
            latest_evaluation = state["evaluation_results"][-1]
            confidence = latest_evaluation.get("confidence", 0.0)
            if confidence >= 1.0:
                self.logger.info(f"완벽한 신뢰도({confidence:.2f}) 달성 - 도구 실행 생략")
                state["current_step"] = "tools_skipped"
                return state

        try:
            if not self.tools:
                self.logger.info("사용 가능한 도구가 없음. 모델 지식만으로 답변 생성")
                state["current_step"] = "tools_executed"
                return state

            # ⭐ 개선된 도구 선택 로직: 계획된 도구 목록 활용
            selected_tools = await self.select_tools_from_plan(state)

            if not selected_tools:
                self.logger.info("실행할 도구가 없음")
                state["current_step"] = "tools_skipped"
                return state

            # ⭐ 다중 도구 실행
            execution_strategy = state.get("tool_execution_strategy", "sequential")

            if execution_strategy == "parallel":
                # 병렬 실행 (위험할 수 있으므로 신중하게)
                await self.execute_tools_parallel(state, selected_tools)
            else:
                # 순차 실행 (기본값)
                await self.execute_tools_sequential(state, selected_tools)

            state["current_step"] = "tools_executed"

        except Exception as e:
            self.logger.error(f"MCP 도구 실행 실패: {e}")
            tool_result = {
                "tool_name": "fallback_knowledge",
                "input": state["user_query"],
                "output": f"도구 실행 실패로 모델 지식 기반 답변 시도: {str(e)}",
                "timestamp": datetime.now().isoformat(),
                "success": False
            }
            state["tool_results"].append(tool_result)
            state["current_step"] = "tools_executed"

        return state

    async def select_tools_from_plan(self, state: Dict[str, Any]) -> List[Any]:
        """계획된 도구 목록에서 실행할 도구들 선택"""

        # 이미 성공적으로 실행된 도구들 확인
        executed_tools = set()
        for tool_result in state["tool_results"]:
            if tool_result.get("success", False):
                executed_tools.add(tool_result.get("tool_name", ""))

        # 계획된 도구 목록이 있는지 확인
        planned_tools = state.get("planned_tools", [])
        current_priority_tool = state.get("current_priority_tool")
        needs_multiple_tools = state.get("needs_multiple_tools", False)

        if planned_tools:
            self.logger.info(f"계획된 도구 목록 활용: {planned_tools}")

            # 아직 실행되지 않은 도구들만 필터링
            pending_tools = [tool_name for tool_name in planned_tools
                             if tool_name not in executed_tools]

            if not pending_tools:
                self.logger.info("계획된 모든 도구가 이미 실행됨")
                return []

            # 실제 도구 객체 찾기
            selected_tool_objects = []
            available_tool_names = {getattr(tool, 'name', str(tool)): tool for tool in self.tools}

            if needs_multiple_tools:
                # 다중 도구 실행: 최대 2-3개까지
                max_tools_per_iteration = min(3, len(pending_tools))
                tools_to_execute = pending_tools[:max_tools_per_iteration]

                for tool_name in tools_to_execute:
                    if tool_name in available_tool_names:
                        selected_tool_objects.append(available_tool_names[tool_name])
                        self.logger.info(f"다중 실행 도구 선택: {tool_name}")
            else:
                # 단일 도구 실행: 우선순위가 높은 하나만
                priority_tool = current_priority_tool if current_priority_tool in pending_tools else pending_tools[0]

                if priority_tool in available_tool_names:
                    selected_tool_objects.append(available_tool_names[priority_tool])
                    self.logger.info(f"우선순위 도구 선택: {priority_tool}")

            return selected_tool_objects

        else:
            # 기존 로직: 사용 가능한 도구 중에서 최적 선택
            self.logger.info("계획된 도구 목록이 없음 - 기존 도구 선택 로직 사용")
            available_tools = [tool for tool in self.tools
                               if getattr(tool, 'name', str(tool)) not in executed_tools]

            if not available_tools:
                return []

            selected_tool = await self.select_best_tool(state["user_query"], available_tools)
            return [selected_tool] if selected_tool else []

    async def execute_tools_sequential(self, state: Dict[str, Any], selected_tools: List[Any]):
        """도구들을 순차적으로 실행"""
        self.logger.info(f"순차 도구 실행 시작: {len(selected_tools)}개 도구")

        for i, tool in enumerate(selected_tools, 1):
            tool_name = getattr(tool, 'name', f'tool_{i}')

            self.logger.info(f"도구 {i}/{len(selected_tools)} 실행 중: {tool_name}")

            try:
                # 개별 도구 실행
                tool_result = await self.execute_single_tool(tool, state["user_query"], state)

                if tool_result:
                    state["tool_results"].append(tool_result)
                    state["reasoning_trace"].append(f"도구 {i}/{len(selected_tools)} 완료: {tool_name}")

                    # 도구 실행 결과가 충분히 좋으면 조기 종료
                    if tool_result.get("success") and len(state["tool_results"]) >= 2:
                        # 간단한 품질 체크
                        output_length = len(str(tool_result.get("output", "")))
                        if output_length > 100:  # 충분한 정보가 있다고 판단
                            self.logger.info(f"충분한 결과 획득 - 남은 도구 실행 생략 ({output_length} chars)")
                            break

            except Exception as e:
                self.logger.error(f"도구 {tool_name} 실행 실패: {e}")
                # 실패해도 계속 진행
                continue

    async def execute_tools_parallel(self, state: Dict[str, Any], selected_tools: List[Any]):
        """도구들을 병렬로 실행 (주의: 리소스 경합 가능)"""
        self.logger.info(f"병렬 도구 실행 시작: {len(selected_tools)}개 도구")

        # 병렬 실행 태스크 생성
        tasks = []
        for tool in selected_tools:
            task = self.execute_single_tool(tool, state["user_query"], state)
            tasks.append(task)

        # 모든 도구 병렬 실행
        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for i, result in enumerate(results):
                tool_name = getattr(selected_tools[i], 'name', f'parallel_tool_{i}')

                if isinstance(result, Exception):
                    self.logger.error(f"병렬 도구 {tool_name} 실행 실패: {result}")
                    continue

                if result:
                    state["tool_results"].append(result)
                    state["reasoning_trace"].append(f"병렬 도구 완료: {tool_name}")

        except Exception as e:
            self.logger.error(f"병렬 도구 실행 중 오류: {e}")

    async def execute_single_tool(self, tool, query: str, state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """단일 도구 실행 - 개선된 입력 생성"""
        tool_name = getattr(tool, 'name', 'unknown_tool')

        try:
            # ⭐ 도구별 맞춤형 입력 생성
            tool_input = await self.generate_tool_input(tool, query, state)

            self.logger.info(f"도구 '{tool_name}' 실행 시작 - 입력: {tool_input}")

            # 도구 실행을 위한 메시지 구성
            tool_node = ToolNode([tool])

            # 도구 호출 ID 생성 (고유하게)
            call_id = f"call_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_{tool_name}"

            tool_message = AIMessage(
                content="",
                tool_calls=[{
                    "name": tool_name,
                    "args": tool_input,  # ⭐ 개선된 입력 사용
                    "id": call_id
                }]
            )

            # 실제 도구 실행
            self.logger.info(f"도구 '{tool_name}' 호출 중...")
            tool_response = await tool_node.ainvoke({"messages": [tool_message]})

            if tool_response and "messages" in tool_response:
                for msg in tool_response["messages"]:
                    if hasattr(msg, 'content') and msg.content:
                        # 성공적인 결과 처리
                        tool_result = {
                            "tool_name": tool_name,
                            "input": tool_input,  # 원본 입력 저장
                            "input_query": query,  # 사용자 쿼리도 별도 저장
                            "output": msg.content,
                            "timestamp": datetime.now().isoformat(),
                            "success": True,
                            "call_id": call_id,
                            "execution_time": self.get_execution_time(call_id)
                        }

                        # 결과 품질 체크
                        output_quality = self.assess_output_quality(msg.content, tool_name)
                        tool_result["quality_score"] = output_quality

                        self.logger.info(f"도구 '{tool_name}' 실행 성공 (품질: {output_quality:.2f})")
                        return tool_result
                    elif hasattr(msg, 'content'):
                        # 빈 결과 처리
                        self.logger.warning(f"도구 '{tool_name}' 실행 결과가 비어있음")
                        return {
                            "tool_name": tool_name,
                            "input": tool_input,
                            "input_query": query,
                            "output": "도구 실행 완료되었지만 결과가 비어있습니다.",
                            "timestamp": datetime.now().isoformat(),
                            "success": False,
                            "call_id": call_id,
                            "error_reason": "empty_result"
                        }

            # 응답이 없는 경우
            self.logger.warning(f"도구 '{tool_name}' 응답이 없음")
            return {
                "tool_name": tool_name,
                "input": tool_input,
                "input_query": query,
                "output": "도구에서 응답을 받지 못했습니다.",
                "timestamp": datetime.now().isoformat(),
                "success": False,
                "call_id": call_id,
                "error_reason": "no_response"
            }

        except Exception as e:
            error_msg = str(e)
            self.logger.error(f"도구 '{tool_name}' 실행 실패: {error_msg}")

            # 오류 유형별 분류
            error_type = self.classify_error(error_msg)

            return {
                "tool_name": tool_name,
                "input": tool_input if 'tool_input' in locals() else {"query": query},
                "input_query": query,
                "output": f"도구 실행 실패: {error_msg}",
                "timestamp": datetime.now().isoformat(),
                "success": False,
                "error_reason": error_type,
                "error_details": error_msg
            }

    def assess_output_quality(self, output: str, tool_name: str) -> float:
        """도구 출력의 품질 평가"""
        if not output or output.strip() == "":
            return 0.0

        # 기본 품질 점수
        quality_score = 0.5

        # 길이 기반 평가
        if len(output) > 50:
            quality_score += 0.2
        if len(output) > 200:
            quality_score += 0.1

        # 내용 기반 평가
        output_lower = output.lower()

        # 성공 지표
        success_indicators = ['성공', 'success', 'complete', '완료', 'done']
        if any(indicator in output_lower for indicator in success_indicators):
            quality_score += 0.2

        # 오류 지표 (감점)
        error_indicators = ['error', 'fail', '실패', '오류', 'exception']
        if any(indicator in output_lower for indicator in error_indicators):
            quality_score -= 0.3

        # 도구 타입별 특별 평가
        tool_lower = tool_name.lower()
        if 'time' in tool_lower and any(time_format in output for time_format in [':', '-', '/']):
            quality_score += 0.1
        elif 'file' in tool_lower and ('파일' in output_lower or 'file' in output_lower):
            quality_score += 0.1

        return max(0.0, min(1.0, quality_score))

    def classify_error(self, error_msg: str) -> str:
        """오류 유형 분류"""
        error_lower = error_msg.lower()

        if 'permission' in error_lower or '권한' in error_lower:
            return "permission_error"
        elif 'not found' in error_lower or '찾을 수 없' in error_lower:
            return "not_found_error"
        elif 'timeout' in error_lower or '시간초과' in error_lower:
            return "timeout_error"
        elif 'network' in error_lower or '네트워크' in error_lower:
            return "network_error"
        elif 'invalid' in error_lower or '유효하지' in error_lower:
            return "validation_error"
        else:
            return "unknown_error"

    def get_execution_time(self, call_id: str) -> float:
        """실행 시간 측정 (간단한 구현)"""
        # 실제로는 시작 시간을 기록하고 계산해야 함
        # 여기서는 call_id에서 타임스탬프를 추출하여 대략적으로 계산
        try:
            timestamp_part = call_id.split('_')[1] + call_id.split('_')[2]
            # 간단한 더미 실행 시간 (실제로는 start_time을 별도로 관리해야 함)
            return 0.5  # 기본값
        except:
            return 0.0


async def select_best_tool(self, user_query: str, available_tools: List) -> Optional[Any]:
    """사용자 요청에 가장 적합한 도구 선택 - 향상된 버전"""
    if not available_tools:
        return None

    if len(available_tools) == 1:
        return available_tools[0]

    # 도구명과 설명 수집 (더 자세한 정보 포함)
    tool_descriptions = []
    tool_schemas = {}

    for tool in available_tools:
        tool_name = getattr(tool, 'name', str(tool))

        # 도구 스키마 분석
        try:
            schema = await self.analyze_tool_schema(tool)
            tool_schemas[tool_name] = schema

            # 더 자세한 설명 생성
            description = f"- {tool_name}"
            if schema.get("description"):
                description += f": {schema['description']}"
            if schema.get("required_params"):
                description += f" (필수: {', '.join(schema['required_params'])})"

            tool_descriptions.append(description)
        except Exception as e:
            self.logger.warning(f"도구 '{tool_name}' 스키마 분석 실패: {e}")
            tool_descriptions.append(f"- {tool_name}")

    # ⭐ 개선된 LLM 기반 도구 선택
    selection_prompt = ChatPromptTemplate.from_messages([
        ("system", """당신은 사용자 요청에 가장 적합한 도구를 선택하는 전문가입니다.

사용 가능한 도구들과 상세 정보:
{tools}

**도구 선택 기준:**
1. **기능 적합성**: 사용자 요청과 도구의 기능이 얼마나 일치하는가?
2. **파라미터 호환성**: 사용자 요청에서 필요한 파라미터를 추출할 수 있는가?
3. **우선순위 규칙**:
   - 파일 삭제 요청 → delete_file, remove_file 등 선택
   - 시간 조회 요청 → get_current_time, get_time 등 선택
   - 파일 검색 요청 → search_files, find_files 등 선택
   - 파일 생성 요청 → create_file, write_file 등 선택

**중요**: 
- 사용자 요청과 정확히 일치하는 도구만 선택하세요
- 관련 없는 도구는 절대 선택하지 마세요
- 애매한 경우 가장 안전한 도구를 선택하세요

다음 형식으로 응답하세요:
{{
  "selected_tool": "도구명",
  "confidence": 0.0-1.0,
  "reason": "선택 이유",
  "alternative": "대안 도구명 (있다면)"
}}"""),
        ("human", "사용자 요청: {query}\n\n가장 적합한 도구를 선택해주세요.")
    ])

    try:
        response = await self.model.ainvoke(
            selection_prompt.format_messages(
                query=user_query,
                tools="\n".join(tool_descriptions)
            )
        )

        # JSON 파싱 시도
        try:
            selection_result = json.loads(response.content)
            selected_tool_name = selection_result.get("selected_tool", "").strip()
            confidence = selection_result.get("confidence", 0.5)
            reason = selection_result.get("reason", "")

            self.logger.info(f"LLM 도구 선택: '{selected_tool_name}' (신뢰도: {confidence:.2f})")
            self.logger.info(f"선택 이유: {reason}")

            # 선택된 도구 찾기
            for tool in available_tools:
                tool_name = getattr(tool, 'name', str(tool))
                if tool_name == selected_tool_name:

                    # 선택된 도구의 입력 호환성 검증
                    if await self.validate_tool_compatibility(tool, user_query):
                        self.logger.info(f"✅ 적절한 도구 선택됨: {selected_tool_name}")
                        return tool
                    else:
                        self.logger.warning(f"⚠️ 도구 호환성 검증 실패: {selected_tool_name}")
                        # 대안 도구 시도
                        alternative = selection_result.get("alternative")
                        if alternative:
                            for alt_tool in available_tools:
                                if getattr(alt_tool, 'name', str(alt_tool)) == alternative:
                                    self.logger.info(f"대안 도구 사용: {alternative}")
                                    return alt_tool

            # 매칭 실패 시 유사한 이름의 도구 찾기
            self.logger.warning(f"정확한 매칭 실패. 유사한 도구 검색: '{selected_tool_name}'")
            similar_tool = self.find_similar_tool(selected_tool_name, available_tools)
            if similar_tool:
                return similar_tool

        except json.JSONDecodeError:
            self.logger.warning("도구 선택 결과 JSON 파싱 실패")
            # 폴백: 키워드 기반 선택
            return self.select_tool_by_keywords(user_query, available_tools)

        # 모든 매칭이 실패한 경우
        self.logger.warning("도구 매칭 완전 실패. 첫 번째 도구 사용")
        return available_tools[0]

    except Exception as e:
        self.logger.error(f"도구 선택 실패: {e}")
        return available_tools[0]


async def validate_tool_compatibility(self, tool, user_query: str) -> bool:
    """도구와 사용자 요청의 호환성 검증"""
    try:
        tool_name = getattr(tool, 'name', str(tool))
        schema = await self.analyze_tool_schema(tool)

        # 필수 파라미터를 사용자 쿼리에서 추출할 수 있는지 검증
        required_params = schema.get("required_params", [])

        if not required_params:
            return True  # 필수 파라미터가 없으면 호환성 OK

        # 간단한 호환성 체크
        query_lower = user_query.lower()

        for param in required_params:
            param_lower = param.lower()

            # 파라미터 타입별 검증
            if param_lower in ['path', 'file', 'filename']:
                # 파일 관련 파라미터
                if not any(indicator in query_lower for indicator in ['.', '/', '\\', '파일', 'file']):
                    self.logger.warning(f"파일 관련 파라미터 '{param}' 요구되지만 쿼리에 파일 정보 없음")
                    return False
            elif param_lower in ['query', 'text', 'content']:
                # 텍스트 파라미터는 항상 사용자 쿼리로 채울 수 있음
                continue

        return True

    except Exception as e:
        self.logger.warning(f"도구 호환성 검증 실패: {e}")
        return True  # 검증 실패 시 일단 호환성 있다고 가정


def find_similar_tool(self, target_name: str, available_tools: List) -> Optional[Any]:
    """유사한 이름의 도구 찾기"""
    target_lower = target_name.lower()

    # 정확한 매칭 재시도
    for tool in available_tools:
        tool_name = getattr(tool, 'name', str(tool)).lower()
        if tool_name == target_lower:
            return tool

    # 부분 매칭
    for tool in available_tools:
        tool_name = getattr(tool, 'name', str(tool)).lower()
        if target_lower in tool_name or tool_name in target_lower:
            self.logger.info(f"유사한 도구 발견: {tool_name}")
            return tool

    # 키워드 기반 매칭
    keywords = target_lower.split('_')
    for tool in available_tools:
        tool_name = getattr(tool, 'name', str(tool)).lower()
        if any(keyword in tool_name for keyword in keywords if len(keyword) > 2):
            self.logger.info(f"키워드 기반 유사 도구: {tool_name}")
            return tool

    return None


def select_tool_by_keywords(self, user_query: str, available_tools: List) -> Optional[Any]:
    """키워드 기반 도구 선택 (폴백 방법)"""
    query_lower = user_query.lower()

    # 키워드별 우선순위 매핑
    keyword_mappings = [
        (['삭제', 'delete', 'remove', '지워', '제거'], ['delete', 'remove', 'del']),
        (['시간', 'time', '현재시간', 'clock'], ['time', 'clock', 'current']),
        (['검색', 'search', 'find', '찾기'], ['search', 'find', 'query']),
        (['생성', 'create', 'make', '만들기'], ['create', 'make', 'generate']),
        (['파일', 'file'], ['file', 'document']),
        (['폴더', 'folder', 'directory'], ['folder', 'dir', 'directory']),
    ]

    # 사용자 쿼리에서 키워드 감지
    for query_keywords, tool_keywords in keyword_mappings:
        if any(keyword in query_lower for keyword in query_keywords):
            # 해당하는 도구 찾기
            for tool in available_tools:
                tool_name = getattr(tool, 'name', str(tool)).lower()
                if any(tk in tool_name for tk in tool_keywords):
                    self.logger.info(f"키워드 기반 도구 선택: {tool_name}")
                    return tool

    # 매칭되는 것이 없으면 첫 번째 도구
    return available_tools[0] if available_tools else None
