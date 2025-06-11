import json
import logging
from enum import Enum
from typing import Dict, List, Any, Optional, Tuple
from langchain_core.prompts import ChatPromptTemplate


class RiskLevel(Enum):
    """위험도 레벨"""
    SAFE = "safe"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RiskCategory(Enum):
    """위험 카테고리"""
    DATA_DESTRUCTION = "data_destruction"
    SYSTEM_MODIFICATION = "system_modification"
    SECURITY_BREACH = "security_breach"
    FINANCIAL_IMPACT = "financial_impact"
    PRIVACY_VIOLATION = "privacy_violation"
    SERVICE_DISRUPTION = "service_disruption"
    IRREVERSIBLE_ACTION = "irreversible_action"


class LLMRiskAssessmentService:
    """LLM 기반 위험도 평가 서비스"""

    def __init__(self, model, evaluator_model=None):
        self.model = model
        self.evaluator_model = evaluator_model or model
        self.logger = logging.getLogger(__name__)

        # 위험도 임계값 설정
        self.risk_thresholds = {
            RiskLevel.SAFE: 0.0,
            RiskLevel.LOW: 0.2,
            RiskLevel.MEDIUM: 0.4,
            RiskLevel.HIGH: 0.6,
            RiskLevel.CRITICAL: 0.8
        }

    async def assess_risk(self,
                          user_query: str,
                          planned_tools: List[str],
                          tool_descriptions: Dict[str, str] = None,
                          context: Dict[str, Any] = None) -> Dict[str, Any]:
        """사용자 쿼리와 계획된 도구들의 위험도를 LLM으로 평가"""

        self.logger.info(f"LLM 위험도 평가 시작: {user_query}")

        # 도구 설명 정보 준비
        tools_info = self._prepare_tools_info(planned_tools, tool_descriptions)

        # 컨텍스트 정보 준비
        context_info = self._prepare_context_info(context or {})

        # 위험도 평가 프롬프트
        risk_assessment_prompt = ChatPromptTemplate.from_messages([
            ("system", """당신은 사이버 보안 및 시스템 안전성 전문가입니다. 
사용자의 요청과 실행 예정인 도구들을 분석하여 위험도를 평가해주세요.

**평가 기준:**

🔴 **CRITICAL (0.8-1.0)**: 즉각적이고 돌이킬 수 없는 심각한 피해 가능
- 시스템 전체 파괴, 중요 데이터 완전 삭제
- 보안 시스템 완전 무력화
- 재정적으로 심각한 손실 (수백만원 이상)

🟠 **HIGH (0.6-0.8)**: 상당한 피해나 복구 비용 발생 가능
- 중요 파일/폴더 삭제, 시스템 설정 변경
- 민감한 정보 노출 위험
- 서비스 중단 가능성

🟡 **MEDIUM (0.4-0.6)**: 제한적 피해나 복구 가능한 문제
- 일반 파일 수정/삭제
- 설정 변경 (복구 가능)
- 임시적 서비스 영향

🟢 **LOW (0.2-0.4)**: 경미한 영향, 쉽게 복구 가능
- 로그 파일 조회, 임시 파일 생성
- 읽기 전용 작업

⚪ **SAFE (0.0-0.2)**: 위험 없음
- 정보 조회, 시간 확인, 계산

**특별 고려사항:**
- 사용자의 의도와 맥락을 고려하세요
- 도구의 실제 기능과 권한을 분석하세요  
- 복구 가능성과 영향 범위를 평가하세요
- 연쇄 반응이나 부작용을 고려하세요

**현재 상황:**
사용자 요청: {user_query}
실행 예정 도구들: {tools_info}
컨텍스트: {context_info}

다음 JSON 형식으로 평가 결과를 제공하세요:
{{
  "overall_risk_score": 0.0-1.0,
  "risk_level": "safe|low|medium|high|critical",
  "risk_categories": ["category1", "category2"],
  "risk_analysis": {{
    "potential_damages": ["피해 목록"],
    "affected_resources": ["영향받을 자원들"],
    "reversibility": "쉽게복구가능|어려움|불가능",
    "impact_scope": "개인|팀|조직|전체시스템"
  }},
  "approval_required": true/false,
  "approval_reason": "승인이 필요한 구체적 이유",
  "mitigation_suggestions": ["위험 완화 방안들"],
  "alternative_approaches": ["더 안전한 대안들"],
  "confidence": 0.0-1.0
}}"""),
            ("human", "위 요청의 위험도를 평가해주세요.")
        ])

        try:
            # LLM 위험도 평가 실행
            response = await self.model.ainvoke(
                risk_assessment_prompt.format_messages(
                    user_query=user_query,
                    tools_info=tools_info,
                    context_info=context_info
                )
            )

            # JSON 파싱 및 검증
            risk_assessment = await self._parse_and_validate_assessment(response.content)

            # 위험도 임계값 기반 승인 필요 여부 결정
            needs_approval = self._determine_approval_requirement(risk_assessment)
            risk_assessment["approval_required"] = needs_approval

            self.logger.info(
                f"위험도 평가 완료: {risk_assessment['risk_level']} ({risk_assessment['overall_risk_score']:.2f})")

            return risk_assessment

        except Exception as e:
            self.logger.error(f"LLM 위험도 평가 실패: {e}")
            # 안전을 위한 폴백: 알 수 없는 경우 고위험으로 분류
            return self._create_fallback_assessment(user_query, planned_tools, str(e))

    async def generate_approval_message(self,
                                        risk_assessment: Dict[str, Any],
                                        user_query: str) -> str:
        """위험도 평가 결과를 바탕으로 승인 메시지 생성"""

        risk_level = risk_assessment.get("risk_level", "high")
        risk_score = risk_assessment.get("overall_risk_score", 0.8)
        potential_damages = risk_assessment.get("risk_analysis", {}).get("potential_damages", [])
        mitigation_suggestions = risk_assessment.get("mitigation_suggestions", [])

        # 위험도별 이모지 및 색상
        risk_indicators = {
            "critical": "🚨",
            "high": "🔴",
            "medium": "🟡",
            "low": "🟢",
            "safe": "⚪"
        }

        indicator = risk_indicators.get(risk_level, "⚠️")

        message = f"""{indicator} **위험도 평가 결과**

**요청 내용:** {user_query}
**위험도:** {risk_level.upper()} ({risk_score:.1f}/1.0)
**평가 이유:** {risk_assessment.get('approval_reason', '위험 요소 감지')}

"""

        if potential_damages:
            message += f"**예상 피해:**\n"
            for damage in potential_damages[:3]:  # 상위 3개만 표시
                message += f"• {damage}\n"
            message += "\n"

        if mitigation_suggestions:
            message += f"**위험 완화 방안:**\n"
            for suggestion in mitigation_suggestions[:2]:  # 상위 2개만 표시
                message += f"• {suggestion}\n"
            message += "\n"

        message += """**승인 옵션:**
• `approved` - 위험을 감수하고 진행
• `rejected` - 작업 중단
• `modified` - 더 안전한 방법 제안 요청

진행하시겠습니까?"""

        return message

    def _prepare_tools_info(self, planned_tools: List[str], tool_descriptions: Dict[str, str] = None) -> str:
        """도구 정보 준비"""
        if not planned_tools:
            return "실행 예정 도구 없음"

        tools_info = []
        for tool_name in planned_tools:
            if tool_descriptions and tool_name in tool_descriptions:
                tools_info.append(f"- {tool_name}: {tool_descriptions[tool_name]}")
            else:
                tools_info.append(f"- {tool_name}")

        return "\n".join(tools_info)

    def _prepare_context_info(self, context: Dict[str, Any]) -> str:
        """컨텍스트 정보 준비"""
        context_parts = []

        if context.get("user_role"):
            context_parts.append(f"사용자 역할: {context['user_role']}")

        if context.get("system_state"):
            context_parts.append(f"시스템 상태: {context['system_state']}")

        if context.get("previous_actions"):
            context_parts.append(f"이전 작업: {', '.join(context['previous_actions'])}")

        if context.get("sensitive_data_present"):
            context_parts.append("민감한 데이터 존재: 예")

        return " | ".join(context_parts) if context_parts else "추가 컨텍스트 없음"

    async def _parse_and_validate_assessment(self, response_content: str) -> Dict[str, Any]:
        """LLM 응답 파싱 및 검증"""
        try:
            assessment = json.loads(response_content)

            # 필수 필드 검증
            required_fields = ["overall_risk_score", "risk_level", "approval_reason"]
            for field in required_fields:
                if field not in assessment:
                    raise ValueError(f"필수 필드 누락: {field}")

            # 위험도 점수 검증
            risk_score = assessment["overall_risk_score"]
            if not (0.0 <= risk_score <= 1.0):
                raise ValueError(f"잘못된 위험도 점수: {risk_score}")

            # 신뢰도 검증
            confidence = assessment.get("confidence", 0.5)
            if not (0.0 <= confidence <= 1.0):
                assessment["confidence"] = 0.5

            return assessment

        except json.JSONDecodeError as e:
            self.logger.error(f"JSON 파싱 실패: {e}")
            raise ValueError(f"LLM 응답 형식 오류: {e}")
        except Exception as e:
            self.logger.error(f"평가 결과 검증 실패: {e}")
            raise

    def _determine_approval_requirement(self, risk_assessment: Dict[str, Any]) -> bool:
        """위험도 평가 결과를 바탕으로 승인 필요 여부 결정"""
        risk_score = risk_assessment.get("overall_risk_score", 0.8)
        confidence = risk_assessment.get("confidence", 0.5)

        # 고위험 (0.6 이상) 또는 신뢰도가 낮은 경우 (0.7 미만) 승인 필요
        if risk_score >= 0.6:
            return True

        if confidence < 0.7 and risk_score >= 0.4:
            return True

        # 특정 위험 카테고리가 있는 경우
        risk_categories = risk_assessment.get("risk_categories", [])
        critical_categories = [
            RiskCategory.DATA_DESTRUCTION.value,
            RiskCategory.SECURITY_BREACH.value,
            RiskCategory.IRREVERSIBLE_ACTION.value
        ]

        if any(cat in critical_categories for cat in risk_categories):
            return True

        return False

    def _create_fallback_assessment(self, user_query: str, planned_tools: List[str], error: str) -> Dict[str, Any]:
        """평가 실패 시 안전한 폴백 평가"""
        self.logger.warning(f"폴백 위험도 평가 사용: {error}")

        return {
            "overall_risk_score": 0.8,  # 안전을 위해 높은 위험도
            "risk_level": RiskLevel.HIGH.value,
            "risk_categories": [RiskCategory.SYSTEM_MODIFICATION.value],
            "risk_analysis": {
                "potential_damages": ["알 수 없는 위험으로 인한 잠재적 피해"],
                "affected_resources": ["시스템 전반"],
                "reversibility": "불확실",
                "impact_scope": "알 수 없음"
            },
            "approval_required": True,
            "approval_reason": f"위험도 평가 실패로 인한 안전 조치 (오류: {error})",
            "mitigation_suggestions": [
                "작업을 더 작은 단위로 분할",
                "백업 생성 후 진행",
                "테스트 환경에서 먼저 시도"
            ],
            "alternative_approaches": ["수동 확인 후 진행"],
            "confidence": 0.3,
            "fallback_used": True,
            "original_error": error
        }