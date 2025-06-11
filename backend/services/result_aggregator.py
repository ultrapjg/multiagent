import json
import logging
from enum import Enum
from typing import Dict, Any

from langchain_core.prompts import ChatPromptTemplate


class ToolEvaluationResult(Enum):
    """도구 평가 결과"""
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILURE = "failure"
    NEEDS_MORE_INFO = "needs_more_info"


class ResultAggregatorService:
    """결과 평가 및 답변 합성을 담당하는 서비스"""

    def __init__(self, model, evaluator_model, tools):
        self.model = model
        self.evaluator_model = evaluator_model
        self.tools = tools
        self.logger = logging.getLogger(__name__)

    async def evaluate_results(self, state: Dict[str, Any]) -> Dict[str, Any]:
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

    async def synthesize_answer(self, state: Dict[str, Any]) -> Dict[str, Any]:
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
