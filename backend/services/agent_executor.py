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
        """실제 MCP 도구 실행"""
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

            # 이미 성공적으로 실행된 도구들 확인
            executed_tools = set()
            for tool_result in state["tool_results"]:
                if tool_result.get("success", False):
                    executed_tools.add(tool_result.get("tool_name", ""))

            # 아직 실행되지 않은 도구가 있는지 확인
            available_tools = [tool for tool in self.tools
                               if getattr(tool, 'name', str(tool)) not in executed_tools]

            if not available_tools:
                self.logger.info("모든 관련 도구가 이미 실행됨 - 추가 실행 생략")
                state["current_step"] = "tools_skipped"
                return state

            # 🚀 개선된 도구 선택 로직
            selected_tool = await self.select_best_tool(state["user_query"], available_tools)

            if not selected_tool:
                self.logger.info("사용자 요청에 적합한 도구가 없음")
                state["current_step"] = "tools_skipped"
                return state

            tool_name = getattr(selected_tool, 'name', 'mcp_tool')

            self.logger.info(f"새로운 도구 '{tool_name}' 실행 중...")

            # 실행 계획에서 추출한 도구 정보를 바탕으로 실제 도구 실행
            tool_node = ToolNode([selected_tool])

            # 도구 실행을 위한 메시지 구성
            tool_message = AIMessage(
                content="",
                tool_calls=[{
                    "name": tool_name,
                    "args": {"query": state["user_query"]},
                    "id": f"call_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                }]
            )

            # 실제 도구 실행
            tool_response = await tool_node.ainvoke({"messages": [tool_message]})

            if tool_response and "messages" in tool_response:
                for msg in tool_response["messages"]:
                    if hasattr(msg, 'content'):
                        tool_result = {
                            "tool_name": tool_name,
                            "input": state["user_query"],
                            "output": msg.content,
                            "timestamp": datetime.now().isoformat(),
                            "success": True
                        }

                        state["tool_results"].append(tool_result)
                        state["reasoning_trace"].append(f"새로운 MCP 도구 실행 완료: {tool_result['tool_name']}")
                        self.logger.info(f"도구 '{tool_result['tool_name']}' 실행 성공")
            else:
                self.logger.warning("도구 실행 결과가 비어있음")

            state["current_step"] = "tools_executed"

        except Exception as e:
            self.logger.error(f"MCP 도구 실행 실패: {e}")
            # 도구 실행 실패 시에도 모델 지식으로 답변 시도
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
