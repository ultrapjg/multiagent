import asyncio
import json
import os
from typing import Dict, List, Any, AsyncGenerator, Optional, Literal, TypedDict, Callable
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_mcp_adapters.client import MultiServerMCPClient, load_mcp_tools
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import ToolNode
import yaml
import logging
from datetime import datetime
from enum import Enum


class WorkflowState(TypedDict):
    """워크플로우 상태 정의"""
    messages: List[Any]
    user_query: str
    current_step: str
    tool_results: List[Dict[str, Any]]
    evaluation_results: List[Dict[str, Any]]
    confidence_score: float
    next_action: str
    iteration_count: int
    max_iterations: int
    final_answer: str
    reasoning_trace: List[str]
    # Human-in-the-loop 관련 필드
    human_approval_needed: bool
    human_input_requested: bool
    human_response: Optional[str]
    pending_decision: Optional[Dict[str, Any]]
    hitl_enabled: bool


class ToolEvaluationResult(Enum):
    """도구 평가 결과"""
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILURE = "failure"
    NEEDS_MORE_INFO = "needs_more_info"


class HumanApprovalType(Enum):
    """Human approval 타입"""
    TOOL_EXECUTION = "tool_execution"
    HIGH_IMPACT_DECISION = "high_impact_decision"
    LOW_CONFIDENCE = "low_confidence"
    FINAL_ANSWER = "final_answer"


class SupervisorService:
    """Human-in-the-loop 기능을 포함한 동적 워크플로우 기반 LangGraph MCP 에이전트 서비스"""

    def __init__(self):
        self.mcp_client = None
        self.model = None
        self.evaluator_model = None
        self.tools = []
        self.workflow = None
        self.checkpointer = InMemorySaver()
        self.timeout_seconds = 120

        # Human-in-the-loop 설정
        self.hitl_config = {
            "enabled": True,
            "require_approval_for_tools": False,  # 도구 실행 전 승인 필요
            "require_approval_for_low_confidence": True,  # 낮은 신뢰도 시 승인 필요
            "require_approval_for_final_answer": False,  # 최종 답변 전 승인 필요
            "confidence_threshold": 0.7,  # 이 값 이하면 human approval 요청
            "high_impact_tools": ["file_operations", "external_api_calls", "system_commands"]  # 고위험 도구
        }

        # Human input callback
        self.human_input_callback: Optional[Callable] = None

        # 로깅 설정
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def set_human_input_callback(self, callback: Callable[[str, Dict], str]):
        """Human input callback 설정"""
        self.human_input_callback = callback

    def configure_hitl(self, **config):
        """Human-in-the-loop 설정 업데이트"""
        self.hitl_config.update(config)
        self.logger.info(f"HITL 설정 업데이트: {self.hitl_config}")

    async def initialize_agent(self,
                               model_name: str = "qwen2.5:32b",
                               evaluator_model_name: Optional[str] = None,
                               mcp_config: Optional[Dict] = None,
                               system_prompt: Optional[str] = None,
                               hitl_enabled: bool = True):
        """동적 워크플로우 에이전트 초기화"""
        try:
            self.logger.info(f"HITL 지원 동적 워크플로우 에이전트 초기화 시작: {model_name}")

            # HITL 설정
            self.hitl_config["enabled"] = hitl_enabled

            # 기존 클라이언트 정리
            await self.cleanup_mcp_client()

            # MCP 설정 로드 및 도구 초기화
            if mcp_config is None:
                mcp_config = self.load_mcp_config()

            if mcp_config and mcp_config.get("mcpServers"):
                self.mcp_client = MultiServerMCPClient(mcp_config["mcpServers"])
                self.tools = await self.mcp_client.get_tools()
            else:
                self.tools = []

            # 메인 모델과 평가 모델 초기화
            self.model = self.create_model(model_name)
            self.evaluator_model = self.create_model(evaluator_model_name or model_name)

            # 동적 워크플로우 생성
            self.workflow = self._create_dynamic_workflow()

            self.logger.info(f"HITL 지원 동적 워크플로우 에이전트 초기화 완료. 도구 {len(self.tools)}개 로드됨")
            return True

        except Exception as e:
            self.logger.error(f"에이전트 초기화 실패: {e}")
            return False

    def _create_dynamic_workflow(self) -> StateGraph:
        """Human-in-the-loop 기능을 포함한 동적 워크플로우 생성"""
        workflow = StateGraph(WorkflowState)

        # 노드 추가
        workflow.add_node("analyze_query", self._analyze_query)
        workflow.add_node("plan_execution", self._plan_execution)
        workflow.add_node("human_approval", self._human_approval)  # Human approval 노드
        workflow.add_node("execute_tools", self._execute_tools)
        workflow.add_node("evaluate_results", self._evaluate_results)
        workflow.add_node("synthesize_answer", self._synthesize_answer)
        workflow.add_node("quality_check", self._quality_check)
        workflow.add_node("simple_answer", self._simple_answer)
        workflow.add_node("human_input", self._human_input)  # Human input 노드

        # 워크플로우 연결
        workflow.add_edge(START, "analyze_query")

        # 분석 단계 후 조건부 분기
        workflow.add_conditional_edges(
            "analyze_query",
            self._decide_after_analysis,
            {
                "simple": "simple_answer",
                "complex": "plan_execution"
            }
        )

        workflow.add_edge("simple_answer", END)

        # 계획 후 Human approval 체크
        workflow.add_conditional_edges(
            "plan_execution",
            self._decide_after_planning,
            {
                "execute": "execute_tools",
                "need_approval": "human_approval",  # Human approval 필요
                "skip_to_synthesize": "synthesize_answer",
                "end": END
            }
        )

        # Human approval 후 분기
        workflow.add_conditional_edges(
            "human_approval",
            self._decide_after_approval,
            {
                "approved": "execute_tools",
                "rejected": "plan_execution",  # 다시 계획
                "modified": "plan_execution",  # 수정된 계획으로
                "need_input": "human_input"  # 추가 입력 필요
            }
        )

        # Human input 후 계획으로 돌아가기
        workflow.add_edge("human_input", "plan_execution")

        workflow.add_edge("execute_tools", "evaluate_results")

        # 평가 후 낮은 신뢰도 시 Human approval
        workflow.add_conditional_edges(
            "evaluate_results",
            self._decide_next_step,
            {
                "continue": "plan_execution",
                "synthesize": "synthesize_answer",
                "need_approval": "human_approval",  # 낮은 신뢰도로 인한 approval
                "end": END
            }
        )

        workflow.add_edge("synthesize_answer", "quality_check")
        workflow.add_conditional_edges(
            "quality_check",
            self._decide_final_step,
            {
                "approved": END,
                "retry": "plan_execution",
                "need_approval": "human_approval"  # 최종 답변 승인
            }
        )

        return workflow.compile(checkpointer=self.checkpointer)

    async def _analyze_query(self, state: WorkflowState) -> WorkflowState:
        """사용자 쿼리 분석"""
        self.logger.info("사용자 쿼리 분석 중...")

        analysis_prompt = ChatPromptTemplate.from_messages([
            ("system", """당신은 사용자 쿼리를 분석하는 전문가입니다.
다음 관점에서 쿼리를 분석하세요:
1. 쿼리의 의도와 목적
2. 필요한 정보의 종류
3. 적합한 도구들
4. 예상되는 복잡도 (1-10)
5. 단계별 해결 계획

**중요**: 다음과 같은 경우 "도구 불필요"로 분류하세요:
- 단순한 인사 (안녕, 안녕하세요, 반가워요 등)
- 일반적인 질문이나 상식 문의
- 창의적 글쓰기 요청
- 설명이나 정의 요청
- 도구 없이 모델 지식으로 충분히 답변 가능한 경우

Available tools: {tools}

JSON 형식으로 응답하세요:
{{
  "analysis": {{
    "intention_and_purpose": "쿼리의 의도와 목적",
    "type_of_information_required": "필요한 정보 종류",
    "suitable_tools": "적합한 도구들 또는 '도구 불필요'",
    "expected_complexity": 1-10,
    "step_by_step_solution_plan": ["해결 계획"],
    "requires_tools": true/false
  }}
}}"""),
            ("human", "쿼리: {query}")
        ])

        tool_names = [tool.name if hasattr(tool, 'name') else str(tool) for tool in self.tools]

        try:
            response = await self.model.ainvoke(
                analysis_prompt.format_messages(
                    query=state["user_query"],
                    tools=", ".join(tool_names) if tool_names else "없음"
                )
            )

            # JSON 파싱 시도
            try:
                analysis_result = json.loads(response.content)
                requires_tools = analysis_result.get("analysis", {}).get("requires_tools", True)
                suitable_tools = analysis_result.get("analysis", {}).get("suitable_tools", "")

                # 도구가 필요 없는 경우 체크
                if (not requires_tools or
                        "도구 불필요" in suitable_tools or
                        "특정 도구가 필요하지 않습니다" in suitable_tools or
                        "단순한 텍스트 응답으로 충분합니다" in suitable_tools):
                    self.logger.info("도구가 필요 없는 쿼리로 판단 - 직접 답변 생성")
                    state["current_step"] = "simple_query"
                    state["reasoning_trace"].append("도구가 필요 없는 간단한 쿼리로 판단되어 직접 답변을 생성합니다.")
                    return state

            except json.JSONDecodeError:
                self.logger.warning("분석 결과 JSON 파싱 실패 - 기본 워크플로우 진행")

            # 분석 결과를 상태에 저장
            state["reasoning_trace"].append(f"쿼리 분석: {response.content}")
            state["current_step"] = "query_analyzed"

        except Exception as e:
            self.logger.error(f"쿼리 분석 실패: {e}")
            state["reasoning_trace"].append(f"쿼리 분석 실패: {str(e)}")
            state["current_step"] = "query_analyzed"  # 실패해도 계속 진행

        return state

    async def _plan_execution(self, state: WorkflowState) -> WorkflowState:
        """실행 계획 수립"""
        self.logger.info("실행 계획 수립 중...")

        # 첫 번째 반복이면 무조건 도구 실행 계획을 세워야 함
        if state["iteration_count"] == 0:
            self.logger.info("첫 번째 반복 - 도구 실행 계획 수립")
        else:
            # 이전 평가 결과 확인 - 이미 충분한 결과가 있는지 체크
            if state["evaluation_results"]:
                latest_evaluation = state["evaluation_results"][-1]
                confidence = latest_evaluation.get("confidence", 0.0)
                evaluation_type = latest_evaluation.get("evaluation", "")

                # 이미 높은 신뢰도를 달성했다면 추가 도구 실행 생략
                if confidence >= 0.95 or evaluation_type == ToolEvaluationResult.SUCCESS.value:
                    self.logger.info(f"이미 충분한 결과 달성 (신뢰도: {confidence:.2f}) - 도구 실행 생략")
                    state["reasoning_trace"].append("이미 충분한 결과를 얻었으므로 추가 도구 실행을 생략합니다.")
                    state["current_step"] = "plan_skipped"
                    return state

        # 이전 도구 결과 중복 실행 방지
        executed_tools = set()
        for tool_result in state["tool_results"]:
            if tool_result.get("success", False):
                executed_tools.add(tool_result.get("tool_name", ""))

        planning_prompt = ChatPromptTemplate.from_messages([
            ("system", """당신은 실행 계획을 수립하는 전문가입니다.
사용자 쿼리를 분석하고 사용 가능한 도구 목록을 확인하여 실행 계획을 세우세요.

**중요 규칙**:
1. 사용 가능한 도구 목록에서만 도구를 선택할 수 있습니다.
2. 사용자 요청에 적합한 도구가 없다면 "적합한 도구 없음"이라고 명시하세요.
3. 이미 성공적으로 실행된 도구는 다시 실행하지 마세요.
4. 첫 번째 실행이라면 사용자 쿼리에 가장 적합한 도구를 선택하세요.

현재 상황:
- 사용자 쿼리: {query}
- 현재 반복: {iteration}/{max_iterations}
- 이전 도구 결과: {tool_results}
- 이전 평가 결과: {evaluation_results}
- 이미 실행된 도구: {executed_tools}

**실제 사용 가능한 도구**: {tools}

다음 형식으로 계획을 제시하세요:
- 다음에 사용할 도구들: [실제 존재하는 도구만 나열]
- 각 도구를 사용하는 이유: [구체적 이유]
- 예상되는 결과: [예상 결과]
- 적합한 도구가 없다면: "적합한 도구 없음 - 사용자에게 직접 설명 필요"
- 추가 도구가 필요하지 않다면: "추가 도구 불필요"

**주의**: 존재하지 않는 도구(예: get_weather, search_web 등)를 제안하지 마세요."""),
            ("human", "다음 실행 계획을 수립해주세요.")
        ])

        tool_names = [tool.name if hasattr(tool, 'name') else str(tool) for tool in self.tools]

        try:
            response = await self.model.ainvoke(
                planning_prompt.format_messages(
                    query=state["user_query"],
                    iteration=state["iteration_count"],
                    max_iterations=state["max_iterations"],
                    tool_results=state["tool_results"][-3:] if state["tool_results"] else "없음",
                    evaluation_results=state["evaluation_results"][-3:] if state["evaluation_results"] else "없음",
                    executed_tools=", ".join(executed_tools) if executed_tools else "없음",
                    tools=", ".join(tool_names) if tool_names else "없음"
                )
            )

            response_content = response.content.lower()

            # 적합한 도구가 없는 경우 체크
            if any(phrase in response_content for phrase in ["적합한 도구 없음", "사용자에게 직접 설명", "존재하지 않는다면"]):
                self.logger.info("사용자 요청에 적합한 도구가 없음 - 도구 실행 없이 직접 답변")
                state["reasoning_trace"].append("사용 가능한 도구 중 사용자 요청에 적합한 도구가 없어 직접 답변을 제공합니다.")
                state["current_step"] = "no_suitable_tools"
                return state

            # 첫 번째 반복이 아닌 경우에만 "추가 도구 불필요" 체크
            if state["iteration_count"] > 0:
                if any(phrase in response_content for phrase in ["추가 도구 불필요", "추가적인 도구", "더 이상", "필요하지 않"]):
                    self.logger.info("계획 단계에서 추가 도구 실행이 불필요하다고 판단됨")
                    state["reasoning_trace"].append("추가 도구 실행이 불필요하다고 판단되어 계획을 생략합니다.")
                    state["current_step"] = "plan_skipped"
                    return state

            state["reasoning_trace"].append(f"실행 계획: {response.content}")
            state["current_step"] = "plan_ready"

        except Exception as e:
            self.logger.error(f"실행 계획 수립 실패: {e}")
            state["reasoning_trace"].append(f"실행 계획 수립 실패: {str(e)}")
            state["current_step"] = "plan_failed"

        return state

    async def _human_approval(self, state: WorkflowState) -> WorkflowState:
        """Human approval 요청 처리"""
        self.logger.info("Human approval 요청 중...")

        if not self.hitl_config.get("enabled", False):
            self.logger.info("HITL이 비활성화됨 - 자동 승인")
            state["human_response"] = "approved"
            return state

        pending_decision = state.get("pending_decision", {})
        approval_type = pending_decision.get("type", "unknown")

        # 승인 요청 메시지 구성
        approval_message = self._create_approval_message(state, approval_type, pending_decision)

        try:
            if self.human_input_callback:
                # Callback을 통해 human input 요청
                human_response = self.human_input_callback(approval_message, pending_decision)
                state["human_response"] = human_response
                state["reasoning_trace"].append(f"Human approval 응답: {human_response}")
            else:
                # Callback이 없으면 자동 승인 (개발/테스트 용)
                self.logger.warning("Human input callback이 설정되지 않음 - 자동 승인")
                state["human_response"] = "approved"

        except Exception as e:
            self.logger.error(f"Human approval 처리 실패: {e}")
            state["human_response"] = "approved"  # 실패 시 자동 승인

        state["human_approval_needed"] = False
        return state

    async def _human_input(self, state: WorkflowState) -> WorkflowState:
        """Human input 요청 처리"""
        self.logger.info("Human input 요청 중...")

        if not self.hitl_config.get("enabled", False):
            state["human_response"] = "continue"
            return state

        input_message = "추가 정보나 지시사항을 제공해주세요:"

        try:
            if self.human_input_callback:
                human_input = self.human_input_callback(input_message, {"type": "input_request"})
                state["human_response"] = human_input
                state["reasoning_trace"].append(f"Human input 수신: {human_input}")
                # 받은 입력을 사용자 쿼리에 추가
                state["user_query"] += f"\n[추가 정보: {human_input}]"
            else:
                self.logger.warning("Human input callback이 설정되지 않음")
                state["human_response"] = "continue"

        except Exception as e:
            self.logger.error(f"Human input 처리 실패: {e}")
            state["human_response"] = "continue"

        state["human_input_requested"] = False
        return state

    async def _execute_tools(self, state: WorkflowState) -> WorkflowState:
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

            # 첫 번째 사용 가능한 도구 실행
            selected_tool = available_tools[0]
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

    async def _evaluate_results(self, state: WorkflowState) -> WorkflowState:
        """도구 결과 평가"""
        self.logger.info("도구 결과 평가 중...")

        if not state["tool_results"]:
            state["evaluation_results"].append({
                "evaluation": ToolEvaluationResult.FAILURE.value,
                "confidence": 0.0,
                "reason": "실행된 도구가 없음"
            })
            return state

        latest_result = state["tool_results"][-1]

        evaluation_prompt = ChatPromptTemplate.from_messages([
            ("system", """당신은 도구 실행 결과를 평가하는 전문가입니다.
다음 기준으로 결과를 평가하세요:

1. 정확성: 결과가 쿼리에 정확히 답하는가?
2. 완성도: 답변이 완전한가, 아니면 추가 정보가 필요한가?
3. 신뢰성: 결과를 신뢰할 수 있는가?
4. 관련성: 사용자 쿼리와 관련이 있는가?

평가 결과를 JSON 형식으로:
{{
  "evaluation": "success|partial|failure|needs_more_info",
  "confidence": 0.0-1.0,
  "reason": "평가 이유",
  "missing_info": ["부족한 정보 목록"],
  "next_steps": ["제안하는 다음 단계"]
}}"""),
            ("human", """
사용자 쿼리: {query}
도구 결과: {tool_result}
이전 컨텍스트: {context}

평가해주세요.""")
        ])

        try:
            response = await self.evaluator_model.ainvoke(
                evaluation_prompt.format_messages(
                    query=state["user_query"],
                    tool_result=latest_result,
                    context=state["reasoning_trace"][-3:]
                )
            )

            # JSON 파싱 시도
            try:
                evaluation = json.loads(response.content)
            except json.JSONDecodeError:
                # JSON 파싱 실패 시 기본 평가
                evaluation = {
                    "evaluation": ToolEvaluationResult.PARTIAL.value,
                    "confidence": 0.5,
                    "reason": "평가 결과 파싱 실패",
                    "missing_info": [],
                    "next_steps": []
                }

            state["evaluation_results"].append(evaluation)
            state["confidence_score"] = evaluation.get("confidence", 0.5)
            state["reasoning_trace"].append(f"평가 결과: {evaluation['evaluation']} (신뢰도: {evaluation['confidence']})")

        except Exception as e:
            self.logger.error(f"결과 평가 실패: {e}")
            state["evaluation_results"].append({
                "evaluation": ToolEvaluationResult.FAILURE.value,
                "confidence": 0.0,
                "reason": f"평가 실패: {str(e)}"
            })

        state["current_step"] = "results_evaluated"
        state["iteration_count"] += 1

        return state

    async def _synthesize_answer(self, state: WorkflowState) -> WorkflowState:
        """최종 답변 합성"""
        self.logger.info("최종 답변 합성 중...")

        # 적합한 도구가 없는 경우 특별 처리
        if state.get("current_step") == "no_suitable_tools":
            synthesis_prompt = ChatPromptTemplate.from_messages([
                ("system", """당신은 사용자에게 정중하고 도움이 되는 답변을 제공하는 AI 어시스턴트입니다.
현재 상황에서는 사용자의 요청에 적합한 도구가 없어서 실시간 정보를 제공할 수 없습니다.

다음과 같이 답변하세요:
1. 사용자의 요청을 이해했음을 표현
2. 현재 해당 정보를 제공할 수 있는 도구가 없음을 정중하게 설명
3. 일반적인 정보나 대안을 제공 (가능한 경우)
4. 다른 방법이나 추천 사항 제시

사용자 요청: {query}
사용 가능한 도구들: {available_tools}
추론 과정: {reasoning_trace}"""),
                ("human", "적절한 답변을 제공해주세요.")
            ])

            tool_names = [tool.name if hasattr(tool, 'name') else str(tool) for tool in self.tools]

            try:
                response = await self.model.ainvoke(
                    synthesis_prompt.format_messages(
                        query=state["user_query"],
                        available_tools=", ".join(tool_names) if tool_names else "없음",
                        reasoning_trace=state["reasoning_trace"]
                    )
                )

                state["final_answer"] = response.content
                state["current_step"] = "answer_synthesized"

            except Exception as e:
                self.logger.error(f"답변 합성 실패: {e}")
                state[
                    "final_answer"] = f"죄송합니다. 현재 '{state['user_query']}'에 대한 실시간 정보를 제공할 수 있는 도구가 없어서 정확한 답변을 드리기 어렵습니다. 다른 질문이 있으시면 언제든지 말씀해 주세요."

        else:
            # 기존 로직: 도구 결과를 바탕으로 답변 합성
            synthesis_prompt = ChatPromptTemplate.from_messages([
                ("system", """당신은 수집된 정보를 바탕으로 최종 답변을 합성하는 전문가입니다.
다음 정보들을 종합하여 사용자 쿼리에 대한 정확하고 완전한 답변을 제공하세요:

- 사용자 쿼리: {query}
- 도구 실행 결과들: {tool_results}
- 평가 결과들: {evaluation_results}
- 추론 과정: {reasoning_trace}

답변은 다음과 같이 구성하세요:
1. 직접적인 답변
2. 근거가 되는 정보
3. 신뢰도 및 한계점 (필요시)"""),
                ("human", "최종 답변을 합성해주세요.")
            ])

            try:
                response = await self.model.ainvoke(
                    synthesis_prompt.format_messages(
                        query=state["user_query"],
                        tool_results=state["tool_results"],
                        evaluation_results=state["evaluation_results"],
                        reasoning_trace=state["reasoning_trace"]
                    )
                )

                state["final_answer"] = response.content
                state["current_step"] = "answer_synthesized"

            except Exception as e:
                self.logger.error(f"답변 합성 실패: {e}")
                state["final_answer"] = f"답변 합성 중 오류가 발생했습니다: {str(e)}"

        return state

    async def _quality_check(self, state: WorkflowState) -> WorkflowState:
        """품질 검사"""
        self.logger.info("답변 품질 검사 중...")

        quality_prompt = ChatPromptTemplate.from_messages([
            ("system", """당신은 답변 품질을 검사하는 전문가입니다.
다음 기준으로 답변을 평가하세요:

1. 완성도: 쿼리에 완전히 답했는가?
2. 정확성: 제공된 정보가 정확한가?
3. 명확성: 이해하기 쉬운가?
4. 관련성: 사용자 요청과 관련이 있는가?

'approved' 또는 'retry' 중 하나로 응답하세요."""),
            ("human", """
사용자 쿼리: {query}
생성된 답변: {answer}
품질을 평가해주세요.""")
        ])

        try:
            response = await self.evaluator_model.ainvoke(
                quality_prompt.format_messages(
                    query=state["user_query"],
                    answer=state["final_answer"]
                )
            )

            state["next_action"] = "approved" if "approved" in response.content.lower() else "retry"
            state["reasoning_trace"].append(f"품질 검사: {state['next_action']}")

        except Exception as e:
            self.logger.error(f"품질 검사 실패: {e}")
            state["next_action"] = "approved"  # 실패 시 승인으로 처리

        return state

    async def _simple_answer(self, state: WorkflowState) -> WorkflowState:
        """도구 없이 간단한 직접 답변"""
        self.logger.info("간단한 직접 답변 생성 중...")

        simple_prompt = ChatPromptTemplate.from_messages([
            ("system", """당신은 친근하고 도움이 되는 AI 어시스턴트입니다.
사용자의 간단한 쿼리에 대해 자연스럽고 적절한 답변을 제공하세요.

특별한 도구나 실시간 정보가 필요하지 않은 간단한 질문이므로, 
당신의 기본 지식과 대화 능력을 활용하여 답변하세요.

답변은 다음과 같이 작성하세요:
- 자연스럽고 친근한 톤
- 적절한 길이 (너무 길지 않게)
- 필요시 추가 도움 제안"""),
            ("human", "사용자 쿼리: {query}")
        ])

        try:
            response = await self.model.ainvoke(
                simple_prompt.format_messages(query=state["user_query"])
            )

            state["final_answer"] = response.content
            state["current_step"] = "simple_answer_generated"
            state["reasoning_trace"].append("간단한 직접 답변 생성 완료")

        except Exception as e:
            self.logger.error(f"간단한 답변 생성 실패: {e}")
            # 폴백 답변
            if any(greeting in state["user_query"].lower() for greeting in ["안녕", "반가", "hi", "hello"]):
                state["final_answer"] = "안녕하세요! 무엇을 도와드릴까요?"
            else:
                state["final_answer"] = "네, 무엇을 도와드릴까요?"
            state["current_step"] = "simple_answer_generated"

        return state

    def _create_approval_message(self, state: WorkflowState, approval_type: str, pending_decision: Dict) -> str:
        """승인 요청 메시지 생성"""
        if approval_type == HumanApprovalType.TOOL_EXECUTION.value:
            tool_name = pending_decision.get("tool_name", "unknown")
            tool_args = pending_decision.get("tool_args", {})
            return f"""
🤖 도구 실행 승인 요청

도구명: {tool_name}
인수: {json.dumps(tool_args, ensure_ascii=False, indent=2)}
이유: {pending_decision.get('reason', '사용자 요청 처리')}

승인하시겠습니까? (approved/rejected/modified)
"""

        elif approval_type == HumanApprovalType.LOW_CONFIDENCE.value:
            confidence = pending_decision.get("confidence", 0.0)
            return f"""
⚠️ 낮은 신뢰도 결과 승인 요청

현재 신뢰도: {confidence:.2f}
결과: {pending_decision.get('result', '')}
이유: 신뢰도가 임계값({self.hitl_config['confidence_threshold']}) 이하입니다.

계속 진행하시겠습니까? (approved/rejected/need_input)
"""

        elif approval_type == HumanApprovalType.FINAL_ANSWER.value:
            answer = pending_decision.get("answer", "")
            return f"""
✅ 최종 답변 승인 요청

답변:
{answer}

이 답변을 사용자에게 제공하시겠습니까? (approved/rejected/modified)
"""

        else:
            return f"""
❓ 승인 요청

내용: {pending_decision.get('content', '알 수 없는 요청')}
승인하시겠습니까? (approved/rejected)
"""

    def _decide_after_analysis(self, state: WorkflowState) -> Literal["simple", "complex"]:
        """분석 단계 후 단순/복잡 쿼리 결정"""
        current_step = state.get("current_step", "")

        if current_step == "simple_query":
            self.logger.info("간단한 쿼리로 판단 - 직접 답변 생성")
            return "simple"
        else:
            self.logger.info("복잡한 쿼리로 판단 - 전체 워크플로우 실행")
            return "complex"

    def _decide_after_planning(self, state: WorkflowState) -> Literal[
        "execute", "need_approval", "skip_to_synthesize", "end"]:
        """계획 단계 후 다음 동작 결정 (HITL 포함)"""
        if state["iteration_count"] >= state["max_iterations"]:
            return "end"

        current_step = state.get("current_step", "")

        if current_step == "plan_skipped":
            return "skip_to_synthesize"
        elif current_step == "no_suitable_tools":
            return "skip_to_synthesize"
        elif current_step == "plan_ready":
            # 도구 실행 전 승인이 필요한지 체크
            if self._needs_approval_for_tools(state):
                return "need_approval"
            else:
                return "execute"
        elif current_step == "plan_failed":
            return "end"
        else:
            return "execute"

    def _needs_approval_for_tools(self, state: WorkflowState) -> bool:
        """도구 실행 전 승인이 필요한지 판단"""
        if not self.hitl_config.get("enabled", False):
            return False

        if self.hitl_config.get("require_approval_for_tools", False):
            # 고위험 도구인지 체크
            high_impact_tools = self.hitl_config.get("high_impact_tools", [])
            for tool in self.tools:
                tool_name = getattr(tool, 'name', str(tool))
                if any(risk_tool in tool_name.lower() for risk_tool in high_impact_tools):
                    state["pending_decision"] = {
                        "type": HumanApprovalType.TOOL_EXECUTION.value,
                        "tool_name": tool_name,
                        "tool_args": {"query": state["user_query"]},
                        "reason": "고위험 도구 실행"
                    }
                    return True

        return False

    def _decide_after_approval(self, state: WorkflowState) -> Literal["approved", "rejected", "modified", "need_input"]:
        """Human approval 후 결정"""
        human_response = state.get("human_response", "approved").lower()

        if "approved" in human_response or "승인" in human_response:
            return "approved"
        elif "rejected" in human_response or "거부" in human_response:
            return "rejected"
        elif "modified" in human_response or "수정" in human_response:
            return "modified"
        elif "input" in human_response or "입력" in human_response:
            return "need_input"
        else:
            return "approved"  # 기본값

    def _decide_next_step(self, state: WorkflowState) -> Literal["continue", "synthesize", "need_approval", "end"]:
        """다음 단계 결정 (HITL 포함)"""
        # 먼저 최대 반복 횟수 체크
        if state["iteration_count"] >= state["max_iterations"]:
            self.logger.info(f"최대 반복 횟수({state['max_iterations']}) 도달 - 종료")
            return "end"

        if not state["evaluation_results"]:
            self.logger.info("평가 결과가 없음 - 계속 진행")
            return "continue"

        latest_evaluation = state["evaluation_results"][-1]
        evaluation_type = latest_evaluation.get("evaluation", "")
        confidence = latest_evaluation.get("confidence", 0.0)

        self.logger.info(
            f"평가 결과 확인: type={evaluation_type}, confidence={confidence:.2f}, iteration={state['iteration_count']}")

        # 낮은 신뢰도에 대한 Human approval 체크
        if (self.hitl_config.get("enabled", False) and
                self.hitl_config.get("require_approval_for_low_confidence", False) and
                confidence < self.hitl_config.get("confidence_threshold", 0.7)):
            state["pending_decision"] = {
                "type": HumanApprovalType.LOW_CONFIDENCE.value,
                "confidence": confidence,
                "result": latest_evaluation.get("reason", ""),
                "evaluation_type": evaluation_type
            }
            return "need_approval"

        # 기존 로직
        if confidence >= 1.0:
            self.logger.info(f"완벽한 신뢰도({confidence:.2f}) 달성 - 즉시 답변 합성")
            return "synthesize"
        elif confidence >= 0.95 and evaluation_type == ToolEvaluationResult.SUCCESS.value:
            self.logger.info(f"매우 높은 신뢰도({confidence:.2f})와 성공 결과 - 답변 합성")
            return "synthesize"
        elif evaluation_type == ToolEvaluationResult.SUCCESS.value and confidence >= 0.8:
            self.logger.info(f"높은 신뢰도({confidence:.2f})와 성공 결과 - 답변 합성")
            return "synthesize"
        elif evaluation_type == ToolEvaluationResult.NEEDS_MORE_INFO.value and confidence < 0.9:
            self.logger.info(f"추가 정보 필요({confidence:.2f}) - 계속 진행")
            return "continue"
        elif confidence >= 0.7:
            self.logger.info(f"적절한 신뢰도({confidence:.2f}) - 답변 합성")
            return "synthesize"
        else:
            self.logger.info(f"낮은 신뢰도({confidence:.2f}) - 계속 진행")
            return "continue"

    def _decide_final_step(self, state: WorkflowState) -> Literal["approved", "retry", "need_approval"]:
        """최종 단계 결정 (HITL 포함)"""
        if state["iteration_count"] >= state["max_iterations"]:
            return "approved"

        # 최종 답변 승인이 필요한지 체크
        if (self.hitl_config.get("enabled", False) and
                self.hitl_config.get("require_approval_for_final_answer", False)):
            state["pending_decision"] = {
                "type": HumanApprovalType.FINAL_ANSWER.value,
                "answer": state.get("final_answer", "")
            }
            return "need_approval"

        return state.get("next_action", "approved")

    async def chat_stream(self, message: str, thread_id: str = "default", hitl_enabled: bool = None) -> AsyncGenerator[
        str, None]:
        """Human-in-the-loop 지원 스트리밍 채팅"""
        if not self.workflow:
            yield "워크플로우가 초기화되지 않았습니다."
            return

        # HITL 설정 오버라이드
        if hitl_enabled is not None:
            original_hitl = self.hitl_config["enabled"]
            self.hitl_config["enabled"] = hitl_enabled

        # 초기 상태 설정
        initial_state = WorkflowState(
            messages=[HumanMessage(content=message)],
            user_query=message,
            current_step="start",
            tool_results=[],
            evaluation_results=[],
            confidence_score=0.0,
            next_action="",
            iteration_count=0,
            max_iterations=3,
            final_answer="",
            reasoning_trace=[],
            # HITL 필드 초기화
            human_approval_needed=False,
            human_input_requested=False,
            human_response=None,
            pending_decision=None,
            hitl_enabled=self.hitl_config.get("enabled", False)
        )

        config = {
            "configurable": {"thread_id": thread_id},
            "recursion_limit": 50
        }

        try:
            final_state = None
            step_count = 0
            max_steps = 15  # 스텝 수를 더 줄임 (25 -> 15)
            workflow_completed = False
            answer_provided = False

            async for event in self.workflow.astream(initial_state, config=config):
                step_count += 1
                if step_count > max_steps:
                    self.logger.error(f"최대 스텝 수({max_steps}) 초과 - 강제 종료")
                    break

                for node_name, node_state in event.items():
                    # Human approval이나 input이 필요한 경우 사용자에게 알림
                    if node_name == "human_approval":
                        pending_decision = node_state.get("pending_decision", {})
                        approval_message = self._create_approval_message(node_state,
                                                                         pending_decision.get("type", ""),
                                                                         pending_decision)
                        yield f"\n🤚 **Human Approval 필요**\n{approval_message}\n"

                    elif node_name == "human_input":
                        yield f"\n💭 **Human Input 필요**\n추가 정보나 지시사항을 제공해주세요.\n"

                    # 시스템 로그
                    if node_state.get("current_step"):
                        self.logger.info(f"워크플로우 단계: {node_state['current_step']} (노드: {node_name}) - 스텝: {step_count}")

                    final_state = node_state

                    # 답변이 이미 제공되었는지 체크
                    if node_state.get("final_answer") and not answer_provided:
                        # 첫 번째 final_answer가 생성되면 사용자에게 제공
                        if node_name in ["synthesize_answer", "simple_answer"]:
                            confidence = node_state.get("confidence_score", 0.0)
                            iterations = node_state.get("iteration_count", 0)
                            self.logger.info(f"최종 답변 생성 - 신뢰도: {confidence:.2f}, 반복: {iterations}, 스텝: {step_count}")

                            yield node_state["final_answer"]
                            answer_provided = True

                            # quality_check가 비활성화되어 있거나 간단한 답변이면 즉시 종료
                            if (node_name == "simple_answer" or
                                    not self.hitl_config.get("require_approval_for_final_answer", False)):
                                self.logger.info("답변 제공 완료 - 워크플로우 조기 종료")
                                workflow_completed = True
                                break

                    # quality_check에서 approved가 나오면 종료
                    elif (node_name == "quality_check" and
                          node_state.get("next_action") == "approved" and
                          answer_provided):
                        self.logger.info("품질 검사 통과 - 워크플로우 종료")
                        workflow_completed = True
                        break

                # 워크플로우가 완료되면 외부 루프도 종료
                if workflow_completed:
                    break

            # 답변이 제공되지 않았다면 마지막으로 시도
            if not answer_provided and final_state and final_state.get("final_answer"):
                self.logger.info("마지막 시도로 답변 제공")
                yield final_state["final_answer"]
            elif not answer_provided:
                yield "죄송합니다. 요청을 처리하는 중에 문제가 발생했습니다."

        except Exception as e:
            self.logger.error(f"HITL 워크플로우 실행 실패: {e}")
            yield f"요청을 처리하는 중에 오류가 발생했습니다: {str(e)}"

        finally:
            # HITL 설정 복원
            if hitl_enabled is not None:
                self.hitl_config["enabled"] = original_hitl

    def create_model(self, model_name: str):
        """모델 생성"""
        output_tokens = {
            "claude-3-5-sonnet-latest": 8192,
            "claude-3-5-haiku-latest": 8192,
            "claude-3-7-sonnet-latest": 64000,
            "gpt-4o": 16000,
            "gpt-4o-mini": 16000,
        }

        if model_name.startswith("claude"):
            return ChatAnthropic(
                model_name=model_name,
                temperature=0.1
            )
        elif model_name.startswith("gpt"):
            return ChatOpenAI(
                model=model_name,
                temperature=0.1,
                max_tokens=output_tokens.get(model_name, 16000)
            )
        elif model_name.startswith("qwen2.5:32b"):
            return ChatOllama(
                model="qwen2.5:32b",
                temperature=0.1
            )
        else:
            raise ValueError(f"지원하지 않는 모델: {model_name}")

    def load_mcp_config(self) -> Dict:
        """MCP 설정 파일 로드"""
        config_path = "mcp-config/mcp_config.json"
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            self.logger.warning(f"MCP 설정 파일을 찾을 수 없습니다: {config_path}")
            return {"mcpServers": {}}
        except Exception as e:
            self.logger.error(f"MCP 설정 로드 실패: {e}")
            return {"mcpServers": {}}

    async def get_agent_status(self) -> Dict[str, Any]:
        """에이전트 상태 정보 (HITL 정보 포함)"""
        tools_count = len(self.tools) if self.tools else 0

        return {
            "is_initialized": self.workflow is not None,
            "model_name": getattr(self.model, 'model_name', 'Unknown') if self.model else None,
            "evaluator_model": getattr(self.evaluator_model, 'model_name', 'Unknown') if self.evaluator_model else None,
            "tools_count": tools_count,
            "mcp_client_active": self.mcp_client is not None,
            "workflow_active": self.workflow is not None,
            # HITL 상태 정보
            "hitl_config": self.hitl_config,
            "human_input_callback_set": self.human_input_callback is not None
        }

    async def cleanup_mcp_client(self):
        """MCP 클라이언트 정리"""
        if self.mcp_client:
            try:
                await self.mcp_client.__aexit__(None, None, None)
            except Exception as e:
                self.logger.warning(f"MCP 클라이언트 정리 중 오류: {e}")
            finally:
                self.mcp_client = None