import json
import os
from typing import Dict, List, Any

class MCPToolService:
    """MCP 도구 관리 서비스"""
    
    def __init__(self):
        self.config_path = "mcp-config/mcp_config.json"
        self.ensure_config_dir()
    
    def ensure_config_dir(self):
        """설정 디렉토리 확인 및 생성"""
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
        if not os.path.exists(self.config_path):
            self.save_config({"mcpServers": {}})
    
    def load_config(self) -> Dict:
        """MCP 설정 로드"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"설정 로드 실패: {e}")
            return {"mcpServers": {}}
    
    def save_config(self, config: Dict):
        """MCP 설정 저장"""
        try:
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"설정 저장 실패: {e}")
    
    def get_all_tools(self) -> List[Dict]:
        """모든 도구 조회"""
        config = self.load_config()
        tools = []
        
        for name, server_config in config.get("mcpServers", {}).items():
            tools.append({
                "id": name,
                "name": name,
                "description": server_config.get("description", ""),
                "transport": server_config.get("transport", "stdio"),
                "command": server_config.get("command", ""),
                "args": server_config.get("args", []),
                "url": server_config.get("url", ""),
                "active": True,
                "config": server_config
            })
        
        return tools
    
    def add_tool(self, tool_name: str, tool_config: Dict) -> Dict:
        """도구 추가"""
        config = self.load_config()
        
        # 기본 설정 검증
        if "transport" not in tool_config:
            tool_config["transport"] = "stdio"
        
        config["mcpServers"][tool_name] = tool_config
        self.save_config(config)
        
        return {
            "success": True,
            "message": f"도구 '{tool_name}'이 추가되었습니다.",
            "tool": {
                "id": tool_name,
                "name": tool_name,
                "config": tool_config
            }
        }
    
    def remove_tool(self, tool_name: str) -> Dict:
        """도구 제거"""
        config = self.load_config()
        
        if tool_name in config.get("mcpServers", {}):
            del config["mcpServers"][tool_name]
            self.save_config(config)
            return {
                "success": True,
                "message": f"도구 '{tool_name}'이 제거되었습니다."
            }
        else:
            return {
                "success": False,
                "message": f"도구 '{tool_name}'을 찾을 수 없습니다."
            }
    
    def update_tool(self, tool_name: str, tool_config: Dict) -> Dict:
        """도구 업데이트"""
        config = self.load_config()
        
        if tool_name in config.get("mcpServers", {}):
            config["mcpServers"][tool_name] = tool_config
            self.save_config(config)
            return {
                "success": True,
                "message": f"도구 '{tool_name}'이 업데이트되었습니다."
            }
        else:
            return {
                "success": False,
                "message": f"도구 '{tool_name}'을 찾을 수 없습니다."
            }
