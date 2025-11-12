import asyncio
import json
import logging
from enum import Enum
from typing import Dict, List, Any, AsyncGenerator, Optional, Literal, TypedDict, Callable
import os

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_mcp_adapters.client import MultiServerMCPClient
# New, correct way
from langchain_community.chat_models import ChatOllama
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, END, START

from .agent_executor import AgentExecutorService
from .result_aggregator import ResultAggregatorService
from .risk_assessment import LLMRiskAssessmentService

class WorkflowState(TypedDict):
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
    approval_type: Optional[str]
    approval_message: Optional[str]

    # ⭐ 다중 도구 실행 관련 필드 추가
    planned_tools: List[str]  # 계획된 도구 목록
    current_priority_tool: Optional[str]  # 현재 우선순위 도구
    tool_execution_strategy: str  # "sequential" | "parallel"
    needs_multiple_tools: bool  # 다중 도구 필요 여부
    remaining_tools: List[str]  # 아직 실행되지 않은 도구들
    tool_execution_progress: Dict[str, Any]  # 도구 실행 진행 상황


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
            "require_approval_for_low_confidence": False,  # 낮은 신뢰도 시 승인 필요
            "require_approval_for_final_answer": False,  # 최종 답변 전 승인 필요
            "confidence_threshold": 0.7,  # 이 값 이하면 human approval 요청
            "high_impact_tools": ["file_operations", "external_api_calls", "system_commands"]  # 고위험 도구
        }

        # Human input callback - 기본 callback 설정
        self.human_input_callback: Optional[Callable] = self._default_human_input_callback

        # Human input queue for async communication
        self.human_input_queue = asyncio.Queue()
        self.waiting_for_human_input = False
        self.pending_approval_message = None
        self.pending_approval_context = None

        # 로깅 설정
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        self.agent_executor = None
        self.result_aggregator = None
        self.risk_assessor = None

    def _default_human_input_callback(self, message: str, context: Dict) -> str:
        """기본 human input callback - 비동기 대기를 위한 플래그 설정"""
        self.logger.info("기본 human input callback 호출됨 - 비동기 입력 대기 모드")

        # 사용자에게 보여질 메시지를 저장
        self.pending_approval_message = message
        self.pending_approval_context = context

        # 비동기 입력 대기를 위한 특별한 플래그 반환
        return "__WAIT_FOR_ASYNC_INPUT__"

    def set_human_input_callback(self, callback: Callable[[str, Dict], str]):
        """Human input callback 설정"""
        if callback is None:
            self.logger.warning("None callback 전달됨 - 기본 callback 유지")
            return

        self.human_input_callback = callback
        self.logger.info(f"Human input callback 설정 완료: {callback}")

    async def set_human_input_async(self, human_input: str) -> bool:
        """비동기 human input 설정"""
        try:
            self.logger.info(f"Human input 설정: {human_input}")

            if hasattr(self, 'human_input_queue'):
                await self.human_input_queue.put(human_input)
                self.waiting_for_human_input = False
                self.logger.info("Human input 큐에 추가 완료")
                return True
            else:
                self.logger.warning("human_input_queue가 없음")
                return False

        except Exception as e:
            self.logger.error(f"Human input 설정 실패: {e}")
            return False

    def get_pending_approval_info(self) -> Dict[str, Any]:
        """현재 대기 중인 승인 정보 반환"""
        if self.waiting_for_human_input:
            return {
                "waiting": True,
                "message": self.pending_approval_message,
                "context": self.pending_approval_context
            }
        return {"waiting": False}

    def configure_hitl(self, **config):
        """Human-in-the-loop 설정 업데이트"""
        self.hitl_config.update(config)
        self.logger.info(f"HITL 설정 업데이트: {self.hitl_config}")

    async def initialize_agent(self,
                               model_name: Optional[str] = None,
                               evaluator_model_name: Optional[str] = None,
                               mcp_config: Optional[Dict] = None,
                               system_prompt: Optional[str] = None,
                               hitl_enabled: bool = True,
                               human_input_callback: Optional[Callable] = None):
        """동적 워크플로우 에이전트 초기화"""
        try:
            if model_name is None:
                model_name = os.getenv("LOCAL_MODEL_NAME", "qwen:7b")
            self.logger.info(f"HITL 지원 동적 워크플로우 에이전트 초기화 시작: {model_name}")

            # Human input callback 설정
            if human_input_callback:
                self.set_human_input_callback(human_input_callback)
                self.logger.info("초기화 시 human input callback 설정됨")

            # HITL 설정
            self.hitl_config["enabled"] = hitl_enabled
            if hitl_enabled:
                self.hitl_config.update({
                    "require_approval_for_tools": True,  # 도구 승인 활성화
                    "require_approval_for_low_confidence": False,  # 낮은 신뢰도 승인 활성화
                    "confidence_threshold": 0.8,  # 임계값을 높여서 더 자주 트리거
                    # 고위험 키워드
                    "high_impact_tools": [
                        "삭제", "제거", "지우", "delete", "remove", "rm",
                        "파괴", "destroy", "kill", "terminate",
                        "포맷", "format", "초기화", "reset",
                        "수정", "변경", "modify", "edit", "change",
                        "시스템", "system", "관리자", "admin", "root"
                    ]
                })

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

            # 분리된 서비스 초기화
            self.agent_executor = AgentExecutorService(self.model, self.tools)
            self.result_aggregator = ResultAggregatorService(self.model, self.evaluator_model, self.tools)

            # 🆕 LLM 위험도 평가 서비스 초기화
            if self.model:
                self.risk_assessor = LLMRiskAssessmentService(
                    model=self.model,
                    evaluator_model=self.evaluator_model
                )
                self.logger.info("✅ LLM 기반 위험도 평가 서비스 초기화 완료")

            # 동적 워크플로우 생성
            self.workflow = self._create_dynamic_workflow()

            self.logger.info(f"HITL 지원 동적 워크플로우 에이전트 초기화 완료. 도구 {len(self.tools)}개 로드됨")
            return True

        except Exception as e:
            self.logger.error(f"에이전트 초기화 실패: {e}")
            return False

    # 🆕 LLM 기반 위험도 평가 메서드 추가
    async def _needs_approval_for_tools_async(self, state: WorkflowState) -> bool:
        """LLM 기반 비동기 위험도 평가로 승인 필요 여부 판단"""
        if not self.hitl_config.get("enabled", False):
            return False

        if not self.risk_assessor:
            self.logger.warning("위험도 평가 서비스가 초기화되지 않음")
            return True  # 안전을 위해 승인 필요

        try:
            # 계획된 도구 정보 수집
            planned_tools = state.get("planned_tools", [])
            if not planned_tools:
                return False

            # 도구 설명 수집
            tool_descriptions = {}
            for tool in self.tools:
                tool_name = getattr(tool, 'name', str(tool))
                if hasattr(tool, 'description'):
                    tool_descriptions[tool_name] = tool.description

            # 컨텍스트 정보 준비
            context = {
                "previous_actions": state.get("reasoning_trace", [])[-3:],
                "iteration_count": state.get("iteration_count", 0),
                "tool_results": len(state.get("tool_results", []))
            }

            # LLM 위험도 평가 실행
            risk_assessment = await self.risk_assessor.assess_risk(
                user_query=state["user_query"],
                planned_tools=planned_tools,
                tool_descriptions=tool_descriptions,
                context=context
            )

            # 평가 결과를 상태에 저장
            if risk_assessment["approval_required"]:
                # LLM이 생성한 승인 메시지 사용
                approval_message = await self.risk_assessor.generate_approval_message(
                    risk_assessment, state["user_query"]
                )

                pending_decision = {
                    "type": HumanApprovalType.TOOL_EXECUTION.value,
                    "risk_assessment": risk_assessment,
                    "planned_tools": planned_tools,
                    "llm_generated": True
                }

                state["pending_decision"] = pending_decision
                state["approval_type"] = HumanApprovalType.TOOL_EXECUTION.value
                state["approval_message"] = approval_message
                state["human_approval_needed"] = True

                self.logger.info(f"LLM 위험도 평가 결과: {risk_assessment['risk_level']} - 승인 필요")
                return True
            else:
                self.logger.info(f"LLM 위험도 평가 결과: {risk_assessment['risk_level']} - 승인 불필요")
                return False

        except Exception as e:
            self.logger.error(f"LLM 위험도 평가 실패: {e}")
            # 평가 실패 시 안전을 위해 승인 필요
            return True

    def _create_dynamic_workflow(self) -> StateGraph:
        """Human-in-the-loop 기능을 포함한 동적 워크플로우 생성"""
        workflow = StateGraph(WorkflowState)

        # 노드 추가
        workflow.add_node("analyze_query", self._analyze_query)
        workflow.add_node("plan_execution", self._plan_execution)
        workflow.add_node("human_approval", self._human_approval)  # Human approval 노드
        workflow.add_node("execute_tools", self._execute_tools_wrapper)
        workflow.add_node("evaluate_results", self._evaluate_results_wrapper)
        workflow.add_node("synthesize_answer", self._synthesize_answer_wrapper)
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
        """실행 계획 수립 - 다중 도구 실행 지원"""
        self.logger.info("실행 계획 수립 중...")

        # 고위험 키워드 체크 및 HITL 설정을 여기서 먼저 수행
        if self.hitl_config.get("enabled", False) and self.hitl_config.get("require_approval_for_tools", False):
            query_lower = state["user_query"].lower()
            high_impact_keywords = self.hitl_config.get("high_impact_tools", [])

            detected_keywords = []
            for keyword in high_impact_keywords:
                if keyword.lower() in query_lower:
                    detected_keywords.append(keyword)

            if detected_keywords:
                # ... HITL 로직 유지 ...
                pass

        # 첫 번째 반복이면 무조건 도구 실행 계획을 세워야 함
        if state["iteration_count"] == 0:
            self.logger.info("첫 번째 반복 - 도구 실행 계획 수립")
        else:
            # 이전 평가 결과 확인
            if state["evaluation_results"]:
                latest_evaluation = state["evaluation_results"][-1]
                confidence = latest_evaluation.get("confidence", 0.0)
                evaluation_type = latest_evaluation.get("evaluation", "")

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

        # ⭐ 개선된 계획 수립: 다중 도구 실행 계획
        planning_prompt = ChatPromptTemplate.from_messages([
            ("system", """당신은 실행 계획을 수립하는 전문가입니다.
    사용자 쿼리를 분석하고 사용 가능한 도구 목록을 확인하여 **순차적으로 실행할 도구들의 계획**을 세우세요.

    **중요 규칙**:
    1. 사용 가능한 도구 목록에서만 도구를 선택할 수 있습니다.
    2. 사용자 요청에 적합한 도구가 없다면 "적합한 도구 없음"이라고 명시하세요.
    3. 이미 성공적으로 실행된 도구는 다시 실행하지 마세요.
    4. **여러 도구가 필요한 경우 우선순위와 실행 순서를 명시하세요.**
    5. 각 도구가 필요한 이유와 예상 결과를 설명하세요.

    현재 상황:
    - 사용자 쿼리: {query}
    - 현재 반복: {iteration}/{max_iterations}
    - 이전 도구 결과: {tool_results}
    - 이전 평가 결과: {evaluation_results}
    - 이미 실행된 도구: {executed_tools}

    **실제 사용 가능한 도구**: {tools}

    다음 JSON 형식으로 계획을 제시하세요:
    {{
      "execution_plan": {{
        "next_tools": ["도구1", "도구2", "도구3"],  // 실행할 도구들 (우선순위 순)
        "current_priority_tool": "도구1",  // 이번에 실행할 도구
        "tool_reasons": {{
          "도구1": "사용 이유",
          "도구2": "사용 이유"
        }},
        "expected_results": {{
          "도구1": "예상 결과",
          "도구2": "예상 결과"
        }},
        "execution_strategy": "sequential|parallel",  // 실행 전략
        "estimated_iterations": 2,  // 예상 반복 횟수
        "needs_multiple_tools": true,  // 다중 도구 필요 여부
        "status": "tools_planned|no_suitable_tools|additional_tools_not_needed"
      }}
    }}

    **주의**: 존재하지 않는 도구를 제안하지 마세요."""),
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

            # JSON 파싱 시도
            try:
                plan_data = json.loads(response.content)
                execution_plan = plan_data.get("execution_plan", {})

                # 계획된 도구들을 상태에 저장
                state["planned_tools"] = execution_plan.get("next_tools", [])
                state["current_priority_tool"] = execution_plan.get("current_priority_tool")
                state["tool_execution_strategy"] = execution_plan.get("execution_strategy", "sequential")
                state["needs_multiple_tools"] = execution_plan.get("needs_multiple_tools", False)

                plan_status = execution_plan.get("status", "tools_planned")

                if plan_status == "no_suitable_tools":
                    self.logger.info("사용자 요청에 적합한 도구가 없음")
                    state["current_step"] = "no_suitable_tools"
                    return state
                elif plan_status == "additional_tools_not_needed":
                    self.logger.info("추가 도구 실행이 불필요함")
                    state["current_step"] = "plan_skipped"
                    return state
                else:
                    state["current_step"] = "plan_ready"
                    self.logger.info(f"도구 실행 계획 완료: {state['planned_tools']}")

            except json.JSONDecodeError:
                # JSON 파싱 실패 시 기존 로직으로 폴백
                self.logger.warning("계획 JSON 파싱 실패 - 기존 로직 사용")
                response_content = response.content.lower()

                if any(phrase in response_content for phrase in ["적합한 도구 없음", "사용자에게 직접 설명"]):
                    state["current_step"] = "no_suitable_tools"
                    return state
                elif state["iteration_count"] > 0 and any(
                        phrase in response_content for phrase in ["추가 도구 불필요", "더 이상"]):
                    state["current_step"] = "plan_skipped"
                    return state
                else:
                    state["current_step"] = "plan_ready"

            state["reasoning_trace"].append(f"실행 계획: {response.content}")

        except Exception as e:
            self.logger.error(f"실행 계획 수립 실패: {e}")
            state["reasoning_trace"].append(f"실행 계획 수립 실패: {str(e)}")
            state["current_step"] = "plan_failed"

        return state

    async def _human_approval(self, state: WorkflowState) -> WorkflowState:
        """Human approval 요청 처리 - LLM 위험도 평가 결과 포함"""
        self.logger.info("=== Human approval 요청 시작 (LLM 위험도 평가 기반) ===")

        pending_decision = state.get("pending_decision")
        approval_message = state.get("approval_message")

        # LLM 생성 승인 메시지가 있는지 확인
        if pending_decision and pending_decision.get("llm_generated"):
            self.logger.info("LLM 기반 위험도 평가 결과 사용")

            # 위험도 평가 상세 정보 로깅
            risk_assessment = pending_decision.get("risk_assessment", {})
            if risk_assessment:
                risk_level = risk_assessment.get("risk_level", "unknown")
                risk_score = risk_assessment.get("overall_risk_score", 0.0)
                self.logger.info(f"위험도: {risk_level} ({risk_score:.2f})")

                risk_categories = risk_assessment.get("risk_categories", [])
                if risk_categories:
                    self.logger.info(f"위험 카테고리: {', '.join(risk_categories)}")

        if not self.hitl_config.get("enabled", False):
            self.logger.info("HITL이 비활성화됨 - 자동 승인")
            state["human_response"] = "approved"
            return state

        # 승인 메시지 준비 - LLM 생성 메시지 우선 사용
        if approval_message:
            final_approval_message = approval_message
            self.logger.info("LLM 생성 승인 메시지 사용")
        elif pending_decision:
            # 폴백: 기본 메시지 생성
            final_approval_message = self._create_approval_message(
                state,
                pending_decision.get("type", "unknown"),
                pending_decision
            )
            self.logger.info("기본 승인 메시지 생성")
        else:
            # 최종 폴백
            final_approval_message = f"""⚠️ 작업 승인이 필요합니다.

요청: {state['user_query']}

이 작업을 진행하시겠습니까? (approved/rejected)"""
            self.logger.warning("승인 정보 없음 - 기본 메시지 사용")

        self.logger.info(f"최종 승인 메시지:\n{final_approval_message}")

        try:
            if self.human_input_callback:
                self.logger.info("Human input callback 호출 중...")
                self.waiting_for_human_input = True

                # Callback 호출
                human_response = self.human_input_callback(final_approval_message, pending_decision or {})

                # 비동기 입력 대기 처리
                if human_response == "__WAIT_FOR_ASYNC_INPUT__":
                    self.logger.info("비동기 입력 대기 모드 진입")

                    try:
                        human_response = await asyncio.wait_for(
                            self.human_input_queue.get(),
                            timeout=300.0  # 5분 timeout
                        )
                        state["human_response"] = human_response
                        self.logger.info(f"Human 비동기 응답 수신: {human_response}")

                        # 🆕 LLM 위험도 평가 결과와 함께 응답 로깅
                        if pending_decision and pending_decision.get("risk_assessment"):
                            risk_info = pending_decision["risk_assessment"]
                            self.logger.info(f"위험도 {risk_info.get('risk_level')} 작업에 대한 사용자 응답: {human_response}")

                    except asyncio.TimeoutError:
                        self.logger.error("Human input timeout - 안전을 위해 자동 거부")
                        state["human_response"] = "rejected"
                        state["reasoning_trace"].append("Human input timeout으로 인한 자동 거부")
                elif isinstance(human_response, str):
                    state["human_response"] = human_response
                    self.logger.info(f"Human callback 동기 응답 수신: {human_response}")
                else:
                    self.logger.warning(f"예상치 못한 callback 응답 타입: {type(human_response)}")
                    state["human_response"] = "rejected"

            else:
                self.logger.error("⚠️ Human input callback이 설정되지 않음!")
                state["human_response"] = "rejected"
                state["reasoning_trace"].append("Human input callback 부재로 인한 자동 거부")

        except Exception as e:
            self.logger.error(f"Human approval 처리 실패: {e}")
            state["human_response"] = "rejected"
            state["reasoning_trace"].append(f"Human approval 처리 실패로 인한 자동 거부: {str(e)}")

        finally:
            self.waiting_for_human_input = False

        # 상태 초기화
        state["human_approval_needed"] = False

        self.logger.info(f"=== Human approval 완료: {state.get('human_response')} ===")
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

    async def _execute_tools_wrapper(self, state: WorkflowState) -> WorkflowState:
        """도구 실행 래퍼 함수"""
        if self.agent_executor:
            return await self.agent_executor.execute_tools(state)
        else:
            self.logger.error("AgentExecutorService가 초기화되지 않음")
            state["current_step"] = "tools_skipped"
            return state

    async def _evaluate_results_wrapper(self, state: WorkflowState) -> WorkflowState:
        """결과 평가 래퍼 함수"""
        if self.result_aggregator:
            return await self.result_aggregator.evaluate_results(state)
        else:
            self.logger.error("ResultAggregatorService가 초기화되지 않음")
            state["current_step"] = "evaluation_failed"
            return state

    async def _synthesize_answer_wrapper(self, state: WorkflowState) -> WorkflowState:
        """답변 합성 래퍼 함수"""
        if self.result_aggregator:
            return await self.result_aggregator.synthesize_answer(state)
        else:
            self.logger.error("ResultAggregatorService가 초기화되지 않음")
            state["final_answer"] = "답변 합성 서비스를 사용할 수 없습니다."
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

    def _create_fallback_approval_message(self, state: WorkflowState, pending_decision: Dict) -> str:
        """폴백 승인 메시지 생성"""
        keywords = pending_decision.get("keywords", [])
        tool_name = pending_decision.get("tool_name", "알 수 없는 도구")
        risk_level = pending_decision.get("risk_level", "보통")

        risk_emoji = "🔴" if risk_level == "high" else "🟡"

        return f"""{risk_emoji} 고위험 작업 승인 요청

감지된 키워드: {', '.join(keywords) if keywords else '없음'}
요청 내용: {state['user_query']}
위험도: {risk_level}

이 작업을 진행하시겠습니까? (approved/rejected/modified)"""

    def _create_approval_message(self, state: WorkflowState, approval_type: str, pending_decision: Dict) -> str:
        """승인 요청 메시지 생성 - 개선된 버전"""
        if not pending_decision:
            return "승인이 필요합니다. (approved/rejected)"

        if approval_type == HumanApprovalType.TOOL_EXECUTION.value:
            tool_name = pending_decision.get("tool_name", "unknown")
            tool_args = pending_decision.get("tool_args", {})
            reason = pending_decision.get("reason", "사용자 요청 처리")
            keywords = pending_decision.get("keywords", [])
            available_tools = pending_decision.get("available_tools", [])
            risk_level = pending_decision.get("risk_level", "보통")

            # 위험도에 따른 이모지 선택
            risk_emoji = "🔴" if risk_level == "high" else "🟡"

            # 더 자세한 승인 메시지
            message = f"""{risk_emoji} 고위험 도구 실행 승인 요청

감지된 키워드: {', '.join(keywords) if keywords else '없음'}
요청 내용: {state['user_query']}
위험도: {risk_level}

⚠️ 이 작업은 시스템에 영향을 줄 수 있습니다.
정말로 진행하시겠습니까? (approved/rejected/modified)"""

            return message

        elif approval_type == HumanApprovalType.FINAL_ANSWER.value:
            answer = pending_decision.get("answer", "")
            return f"""✅ 최종 답변 승인 요청

답변:
{answer}

이 답변을 사용자에게 제공하시겠습니까? (approved/rejected/modified)"""

        else:
            content = pending_decision.get("content", "알 수 없는 요청")
            return f"""❓ 승인 요청

내용: {content}
승인하시겠습니까? (approved/rejected)"""

    def _decide_after_analysis(self, state: WorkflowState) -> Literal["simple", "complex"]:
        """분석 단계 후 단순/복잡 쿼리 결정"""
        current_step = state.get("current_step", "")

        if current_step == "simple_query":
            self.logger.info("간단한 쿼리로 판단 - 직접 답변 생성")
            return "simple"
        else:
            self.logger.info("복잡한 쿼리로 판단 - 전체 워크플로우 실행")
            return "complex"

    async def _decide_after_planning(self, state: WorkflowState) -> Literal[
        "execute", "need_approval", "skip_to_synthesize", "end"]:
        """계획 단계 후 다음 동작 결정 (LLM 기반 위험도 평가 포함)"""
        if state["iteration_count"] >= state["max_iterations"]:
            return "end"

        current_step = state.get("current_step", "")

        if current_step == "plan_skipped":
            return "skip_to_synthesize"
        elif current_step == "no_suitable_tools":
            return "skip_to_synthesize"
        elif current_step == "plan_ready":
            # 🆕 LLM 기반 위험도 평가로 승인 필요 여부 결정
            needs_approval = await self._needs_approval_for_tools_async(state)

            if needs_approval:
                self.logger.info("LLM 위험도 평가 - human_approval로 분기")
                # pending_decision 확인
                pending_decision = state.get("pending_decision")
                if pending_decision:
                    self.logger.info(f"LLM 생성 pending_decision 확인됨")
                else:
                    self.logger.error("LLM 위험도 평가 후 pending_decision이 설정되지 않음!")
                    # 긴급 복구
                    state["pending_decision"] = {
                        "type": HumanApprovalType.TOOL_EXECUTION.value,
                        "reason": "LLM 위험도 평가 결과 승인 필요",
                        "llm_generated": True
                    }
                    state[
                        "approval_message"] = f"LLM이 다음 작업을 위험하다고 판단했습니다:\n\n{state['user_query']}\n\n진행하시겠습니까? (approved/rejected)"
                return "need_approval"
            else:
                return "execute"
        elif current_step == "plan_failed":
            return "end"
        else:
            return "execute"

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
        """다음 단계 결정 (다중 도구 실행 지원)"""

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

        # ⭐ 다중 도구 실행 상황 고려
        planned_tools = state.get("planned_tools", [])
        needs_multiple_tools = state.get("needs_multiple_tools", False)

        # 실행된 도구와 남은 도구 계산
        executed_tools = set()
        for tool_result in state["tool_results"]:
            if tool_result.get("success", False):
                executed_tools.add(tool_result.get("tool_name", ""))

        remaining_tools = [tool for tool in planned_tools if tool not in executed_tools]

        self.logger.info(f"도구 실행 상황: 계획된={len(planned_tools)}, 실행됨={len(executed_tools)}, 남은것={len(remaining_tools)}")

        # 높은 신뢰도면 즉시 답변 합성
        if confidence >= 1.0:
            self.logger.info(f"완벽한 신뢰도({confidence:.2f}) 달성 - 즉시 답변 합성")
            return "synthesize"
        elif confidence >= 0.95 and evaluation_type == ToolEvaluationResult.SUCCESS.value:
            self.logger.info(f"매우 높은 신뢰도({confidence:.2f})와 성공 결과 - 답변 합성")
            return "synthesize"

        # 다중 도구가 필요한 상황에서의 판단
        if needs_multiple_tools and remaining_tools:
            # 아직 실행할 도구가 남아있는 경우
            if confidence < 0.8:
                self.logger.info(f"다중 도구 필요: 신뢰도 부족({confidence:.2f}) & 남은 도구 있음 - 계속 진행")
                return "continue"
            elif confidence >= 0.8 and len(executed_tools) >= 2:
                # 적당한 신뢰도이고 이미 2개 이상 도구를 실행했으면 충분
                self.logger.info(f"다중 도구: 적절한 신뢰도({confidence:.2f}) & 충분한 도구 실행 - 답변 합성")
                return "synthesize"
            else:
                self.logger.info(f"다중 도구: 추가 도구 실행 필요 - 계속 진행")
                return "continue"

        # 단일 도구이거나 모든 계획된 도구가 실행된 경우
        if evaluation_type == ToolEvaluationResult.SUCCESS.value and confidence >= 0.8:
            self.logger.info(f"성공 결과 & 높은 신뢰도({confidence:.2f}) - 답변 합성")
            return "synthesize"
        elif evaluation_type == ToolEvaluationResult.NEEDS_MORE_INFO.value and confidence < 0.9:
            if remaining_tools:
                self.logger.info(f"추가 정보 필요 & 남은 도구 있음 - 계속 진행")
                return "continue"
            else:
                self.logger.info(f"추가 정보 필요하지만 남은 도구 없음 - 답변 합성")
                return "synthesize"
        elif confidence >= 0.7:
            self.logger.info(f"적절한 신뢰도({confidence:.2f}) - 답변 합성")
            return "synthesize"
        else:
            if remaining_tools and state["iteration_count"] < state["max_iterations"] - 1:
                self.logger.info(f"낮은 신뢰도({confidence:.2f}) & 남은 도구/반복 있음 - 계속 진행")
                return "continue"
            else:
                self.logger.info(f"낮은 신뢰도({confidence:.2f}) 하지만 옵션 소진 - 답변 합성")
                return "synthesize"

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

        # 초기 상태 설정 - 모든 필드 명시적으로 설정
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
            hitl_enabled=self.hitl_config.get("enabled", False),
            # 추가 HITL 상태
            approval_type=None,
            approval_message=None
        )

        config = {
            "configurable": {"thread_id": thread_id},
            "recursion_limit": 50
        }

        try:
            final_state = None
            step_count = 0
            max_steps = 15
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
                        try:
                            # 안전한 승인 메시지 추출
                            approval_message = None

                            if node_state.get("approval_message"):
                                approval_message = str(node_state["approval_message"])
                            elif node_state.get("pending_decision"):
                                pending_decision = node_state["pending_decision"]
                                if isinstance(pending_decision, dict):
                                    approval_message = f"""🤖 고위험 작업 승인 요청

요청: {node_state.get('user_query', 'unknown')}

승인하시겠습니까? (approved/rejected)"""
                            else:
                                approval_message = f"""🤖 작업 승인 요청

요청: {node_state.get('user_query', 'unknown')}
승인하시겠습니까? (approved/rejected)"""

                            if approval_message:
                                yield f"\n🤚 **Human Approval 필요**\n{approval_message}\n"
                                self.logger.info("Human Approval 메시지 전송됨")
                            else:
                                self.logger.warning("Human Approval 메시지를 생성할 수 없음")

                        except Exception as e:
                            self.logger.error(f"Human Approval 메시지 처리 실패: {e}")
                            # 안전한 폴백 메시지
                            fallback_message = f"🤖 작업 승인이 필요합니다.\n요청: {message}\n승인하시겠습니까? (approved/rejected)"
                            yield f"\n🤚 **Human Approval 필요**\n{fallback_message}\n"

                    elif node_name == "human_input":
                        yield f"\n💭 **Human Input 필요**\n추가 정보나 지시사항을 제공해주세요.\n"
                        self.logger.info("Human Input 메시지 전송됨")

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
            import traceback
            self.logger.error(f"상세 오류: {traceback.format_exc()}")
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
        elif model_name.startswith("qwen"):
            return ChatOllama(
                model=os.getenv("LOCAL_MODEL_NAME", "qwen:7b"),
                temperature=0.1
            )
        else:
            raise ValueError(f"지원하지 않는 모델: {model_name}")

    def load_mcp_config(self) -> Dict:
        """MCP 설정 파일 로드 - 데이터베이스에서"""
        try:
            from services.mcp_tool_service import MCPToolService
            config = MCPToolService.get_mcp_config_for_client()
            
            if config and config.get("mcpServers"):
                self.logger.info(f"데이터베이스에서 MCP 설정 로드 완료: {len(config['mcpServers'])}개 도구")
            else:
                self.logger.warning("데이터베이스에 활성화된 MCP 도구가 없습니다")
                
            return config
            
        except Exception as e:
            self.logger.error(f"데이터베이스에서 MCP 설정 로드 실패: {e}")
            # 폴백: 기존 JSON 파일에서 로드 시도
            try:
                config_path = "mcp-config/mcp_config.json"
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    self.logger.info(f"폴백: JSON 파일에서 MCP 설정 로드 완료")
                    return config
            except FileNotFoundError:
                self.logger.warning(f"MCP 설정 파일을 찾을 수 없습니다: {config_path}")
                return {"mcpServers": {}}
            except Exception as file_e:
                self.logger.error(f"JSON 파일에서도 MCP 설정 로드 실패: {file_e}")
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
            "human_input_callback_set": self.human_input_callback is not None,
            # 분리된 서비스 상태 추가
            "agent_executor_initialized": self.agent_executor is not None,
            "result_aggregator_initialized": self.result_aggregator is not None,
            # 🆕 LLM 위험도 평가 서비스 상태
            "risk_assessor_initialized": self.risk_assessor is not None

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
