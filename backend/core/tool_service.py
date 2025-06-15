from typing import Dict, List, Any
from services.mcp_tool_service import MCPToolService
import logging

logger = logging.getLogger("tool_service")


class MCPToolService:
    """MCP 도구 관리 서비스 - 데이터베이스 기반"""
    
    def __init__(self):
        # 더 이상 JSON 파일 경로가 필요하지 않음
        pass
    
    def get_all_tools(self) -> List[Dict]:
        """모든 도구 조회 - 데이터베이스에서"""
        try:
            from services.mcp_tool_service import MCPToolService as DBMCPToolService
            tools = DBMCPToolService.get_all_tools(include_inactive=False)
            
            # 기존 형식과 호환되도록 변환
            formatted_tools = []
            for tool in tools:
                formatted_tools.append({
                    "id": tool['name'],  # 기존 코드와의 호환성을 위해
                    "name": tool['name'],
                    "description": tool.get('description', ''),
                    "transport": tool.get('transport', 'stdio'),
                    "command": tool.get('command', ''),
                    "args": tool.get('args', []),
                    "url": tool.get('url', ''),
                    "active": tool.get('is_active', True),
                    "config": tool.get('config', {})
                })
            
            return formatted_tools
            
        except Exception as e:
            logger.error(f"데이터베이스에서 도구 조회 실패: {e}")
            return []
    
    def add_tool(self, tool_name: str, tool_config: Dict) -> Dict:
        """도구 추가 - 데이터베이스에"""
        try:
            from services.mcp_tool_service import MCPToolService as DBMCPToolService
            
            # 기본 설정 검증
            if "transport" not in tool_config:
                tool_config["transport"] = "stdio"
            
            result = DBMCPToolService.create_tool(
                name=tool_name,
                config=tool_config,
                description=tool_config.get('description', '')
            )
            
            return {
                "success": True,
                "message": f"도구 '{tool_name}'이 추가되었습니다.",
                "tool": {
                    "id": result['name'],
                    "name": result['name'],
                    "config": result['config']
                }
            }
            
        except Exception as e:
            logger.error(f"도구 추가 실패: {e}")
            return {
                "success": False,
                "message": f"도구 추가 실패: {str(e)}"
            }
    
    def remove_tool(self, tool_name: str) -> Dict:
        """도구 제거 - 데이터베이스에서"""
        try:
            from services.mcp_tool_service import MCPToolService as DBMCPToolService
            
            success = DBMCPToolService.delete_tool_by_name(tool_name, soft_delete=True)
            
            if success:
                return {
                    "success": True,
                    "message": f"도구 '{tool_name}'이 제거되었습니다."
                }
            else:
                return {
                    "success": False,
                    "message": f"도구 '{tool_name}'을 찾을 수 없습니다."
                }
                
        except Exception as e:
            logger.error(f"도구 제거 실패: {e}")
            return {
                "success": False,
                "message": f"도구 제거 실패: {str(e)}"
            }
    
    def update_tool(self, tool_name: str, tool_config: Dict) -> Dict:
        """도구 업데이트 - 데이터베이스에서"""
        try:
            from services.mcp_tool_service import MCPToolService as DBMCPToolService
            
            # 먼저 도구 찾기
            existing_tool = DBMCPToolService.get_tool_by_name(tool_name)
            if not existing_tool:
                return {
                    "success": False,
                    "message": f"도구 '{tool_name}'을 찾을 수 없습니다."
                }
            
            # 업데이트
            result = DBMCPToolService.update_tool(
                tool_id=existing_tool['id'],
                config=tool_config,
                description=tool_config.get('description', existing_tool['description'])
            )
            
            if result:
                return {
                    "success": True,
                    "message": f"도구 '{tool_name}'이 업데이트되었습니다."
                }
            else:
                return {
                    "success": False,
                    "message": f"도구 '{tool_name}' 업데이트에 실패했습니다."
                }
                
        except Exception as e:
            logger.error(f"도구 업데이트 실패: {e}")
            return {
                "success": False,
                "message": f"도구 업데이트 실패: {str(e)}"
            }
    
    def load_mcp_config(self) -> Dict:
        """MCP 설정 로드 - 데이터베이스에서"""
        try:
            from services.mcp_tool_service import MCPToolService as DBMCPToolService
            return DBMCPToolService.get_mcp_config_for_client()
            
        except Exception as e:
            logger.error(f"MCP 설정 로드 실패: {e}")
            return {"mcpServers": {}}