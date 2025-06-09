import logging
from datetime import datetime
from typing import Dict, List, Any, Optional

from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.prebuilt import ToolNode


class AgentExecutorService:
    """MCP 도구 실행을 담당하는 서비스"""

    def __init__(self, model, tools):
        self.model = model
        self.tools = tools
        self.logger = logging.getLogger(__name__)

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
        """단일 도구 실행"""
        tool_name = getattr(tool, 'name', 'unknown_tool')

        try:
            # 도구 실행을 위한 메시지 구성
            tool_node = ToolNode([tool])

            tool_message = AIMessage(
                content="",
                tool_calls=[{
                    "name": tool_name,
                    "args": {"query": query},
                    "id": f"call_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{tool_name}"
                }]
            )

            # 실제 도구 실행
            tool_response = await tool_node.ainvoke({"messages": [tool_message]})

            if tool_response and "messages" in tool_response:
                for msg in tool_response["messages"]:
                    if hasattr(msg, 'content'):
                        tool_result = {
                            "tool_name": tool_name,
                            "input": query,
                            "output": msg.content,
                            "timestamp": datetime.now().isoformat(),
                            "success": True
                        }
                        self.logger.info(f"도구 '{tool_name}' 실행 성공")
                        return tool_result

            # 결과가 없는 경우
            self.logger.warning(f"도구 '{tool_name}' 실행 결과가 비어있음")
            return None

        except Exception as e:
            self.logger.error(f"도구 '{tool_name}' 실행 실패: {e}")
            return {
                "tool_name": tool_name,
                "input": query,
                "output": f"도구 실행 실패: {str(e)}",
                "timestamp": datetime.now().isoformat(),
                "success": False
            }

    async def select_best_tool(self, user_query: str, available_tools: List) -> Optional[Any]:
        """사용자 요청에 가장 적합한 도구 선택"""
        if not available_tools:
            return None

        if len(available_tools) == 1:
            return available_tools[0]

        # 도구명과 설명 수집
        tool_descriptions = []
        for tool in available_tools:
            tool_name = getattr(tool, 'name', str(tool))
            tool_descriptions.append(f"- {tool_name}")

        # LLM 기반 도구 선택
        selection_prompt = ChatPromptTemplate.from_messages([
            ("system", """사용자 요청에 가장 적합한 도구를 선택하세요.

사용 가능한 도구들:
{tools}

**선택 기준:**
- 파일 삭제 요청 → delete_file 선택
- 시간 조회 요청 → get_current_time 선택
- 관련 없는 도구는 절대 선택하지 마세요

정확한 도구명만 반환하세요."""),
            ("human", "사용자 요청: {query}")
        ])

        try:
            response = await self.model.ainvoke(
                selection_prompt.format_messages(
                    query=user_query,
                    tools="\n".join(tool_descriptions)
                )
            )

            selected_tool_name = response.content.strip()
            self.logger.info(f"LLM이 선택한 도구: '{selected_tool_name}'")

            # 선택된 도구 찾기
            for tool in available_tools:
                if getattr(tool, 'name', str(tool)) == selected_tool_name:
                    self.logger.info(f"✅ 적절한 도구 선택됨: {selected_tool_name}")
                    return tool

            # 매칭 실패 시 첫 번째 도구 (폴백)
            self.logger.warning(f"도구 매칭 실패. 첫 번째 도구 사용: {getattr(available_tools[0], 'name', 'unknown')}")
            return available_tools[0]

        except Exception as e:
            self.logger.error(f"도구 선택 실패: {e}")
            return available_tools[0]
