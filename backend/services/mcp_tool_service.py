# backend/services/mcp_tool_service.py
from typing import List, Optional, Dict, Any
from psycopg2.extras import RealDictCursor
from database import get_db_connection
import json
import logging
from datetime import datetime

logger = logging.getLogger("mcp_tool_service")


class MCPToolService:
    """데이터베이스 기반 MCP 도구 관리 서비스"""

    @staticmethod
    def get_all_tools(include_inactive: bool = False) -> List[Dict[str, Any]]:
        """
        모든 MCP 도구 조회

        Args:
            include_inactive: 비활성화된 도구도 포함할지 여부

        Returns:
            List[Dict]: 도구 목록
        """
        try:
            logger.info(f"📋 MCP 도구 목록 조회 | 비활성화 포함: {include_inactive}")

            with get_db_connection() as conn:
                cursor = conn.cursor(cursor_factory=RealDictCursor)

                if include_inactive:
                    query = """
                        SELECT id, name, description, transport, command, args, url, config, is_active, created_at, updated_at
                        FROM mcp_tools 
                        ORDER BY created_at DESC
                    """
                    cursor.execute(query)
                else:
                    query = """
                        SELECT id, name, description, transport, command, args, url, config, is_active, created_at, updated_at
                        FROM mcp_tools 
                        WHERE is_active = true
                        ORDER BY created_at DESC
                    """
                    cursor.execute(query)

                rows = cursor.fetchall()

                tools = []
                for row in rows:
                    tools.append({
                        'id': row['id'],
                        'name': row['name'],
                        'description': row['description'],
                        'transport': row['transport'],
                        'command': row['command'],
                        'args': row['args'] if row['args'] else [],
                        'url': row['url'],
                        'config': row['config'],
                        'is_active': row['is_active'],
                        'created_at': row['created_at'].isoformat(),
                        'updated_at': row['updated_at'].isoformat()
                    })

                logger.info(f"✅ MCP 도구 목록 조회 완료 | 반환: {len(tools)}개")
                return tools

        except Exception as e:
            logger.error(f"💥 MCP 도구 목록 조회 실패 | 오류: {str(e)}")
            raise e

    @staticmethod
    def get_tool_by_id(tool_id: int) -> Optional[Dict[str, Any]]:
        """
        특정 MCP 도구 조회

        Args:
            tool_id: 도구 ID

        Returns:
            Optional[Dict]: 도구 정보 또는 None
        """
        try:
            logger.info(f"🔍 MCP 도구 조회 | ID: {tool_id}")

            with get_db_connection() as conn:
                cursor = conn.cursor(cursor_factory=RealDictCursor)

                cursor.execute(
                    """
                    SELECT id, name, description, transport, command, args, url, config, is_active, created_at, updated_at
                    FROM mcp_tools 
                    WHERE id = %s
                    """,
                    (tool_id,)
                )

                row = cursor.fetchone()

                if not row:
                    logger.warning(f"🔍 MCP 도구를 찾을 수 없음 | ID: {tool_id}")
                    return None

                result = {
                    'id': row['id'],
                    'name': row['name'],
                    'description': row['description'],
                    'transport': row['transport'],
                    'command': row['command'],
                    'args': row['args'] if row['args'] else [],
                    'url': row['url'],
                    'config': row['config'],
                    'is_active': row['is_active'],
                    'created_at': row['created_at'].isoformat(),
                    'updated_at': row['updated_at'].isoformat()
                }

                logger.info(f"✅ MCP 도구 조회 완료 | ID: {tool_id}")
                return result

        except Exception as e:
            logger.error(f"💥 MCP 도구 조회 실패 | ID: {tool_id} | 오류: {str(e)}")
            raise e

    @staticmethod
    def get_tool_by_name(tool_name: str) -> Optional[Dict[str, Any]]:
        """
        이름으로 MCP 도구 조회

        Args:
            tool_name: 도구 이름

        Returns:
            Optional[Dict]: 도구 정보 또는 None
        """
        try:
            with get_db_connection() as conn:
                cursor = conn.cursor(cursor_factory=RealDictCursor)

                cursor.execute(
                    """
                    SELECT id, name, description, transport, command, args, url, config, is_active, created_at, updated_at
                    FROM mcp_tools 
                    WHERE name = %s
                    """,
                    (tool_name,)
                )

                row = cursor.fetchone()

                if not row:
                    return None

                return {
                    'id': row['id'],
                    'name': row['name'],
                    'description': row['description'],
                    'transport': row['transport'],
                    'command': row['command'],
                    'args': row['args'] if row['args'] else [],
                    'url': row['url'],
                    'config': row['config'],
                    'is_active': row['is_active'],
                    'created_at': row['created_at'].isoformat(),
                    'updated_at': row['updated_at'].isoformat()
                }

        except Exception as e:
            logger.error(f"💥 이름으로 MCP 도구 조회 실패 | 이름: {tool_name} | 오류: {str(e)}")
            raise e

    @staticmethod
    def create_tool(name: str, config: Dict[str, Any], description: str = "") -> Dict[str, Any]:
        """
        새로운 MCP 도구 생성

        Args:
            name: 도구 이름
            config: 도구 설정
            description: 설명

        Returns:
            Dict: 생성된 도구 정보
        """
        try:
            logger.info(f"📝 MCP 도구 생성 시작 | 이름: {name}")

            # 설정에서 필드 추출
            transport = config.get("transport", "stdio")
            command = config.get("command")
            args = config.get("args", [])
            url = config.get("url")

            with get_db_connection() as conn:
                cursor = conn.cursor(cursor_factory=RealDictCursor)

                # 중복 이름 검사
                cursor.execute("SELECT id FROM mcp_tools WHERE name = %s", (name,))
                if cursor.fetchone():
                    raise ValueError(f"도구 이름 '{name}'이 이미 존재합니다.")

                cursor.execute(
                    """
                    INSERT INTO mcp_tools (name, description, transport, command, args, url, config) 
                    VALUES (%s, %s, %s, %s, %s, %s, %s) 
                    RETURNING id, name, description, transport, command, args, url, config, is_active, created_at, updated_at
                    """,
                    (name, description, transport, command, json.dumps(args) if args else None, url, json.dumps(config))
                )

                row = cursor.fetchone()
                conn.commit()

                result = {
                    'id': row['id'],
                    'name': row['name'],
                    'description': row['description'],
                    'transport': row['transport'],
                    'command': row['command'],
                    'args': row['args'] if row['args'] else [],
                    'url': row['url'],
                    'config': row['config'],
                    'is_active': row['is_active'],
                    'created_at': row['created_at'].isoformat(),
                    'updated_at': row['updated_at'].isoformat()
                }

                logger.info(f"✅ MCP 도구 생성 완료 | ID: {row['id']}, 이름: {name}")
                return result

        except Exception as e:
            logger.error(f"💥 MCP 도구 생성 실패 | 이름: {name} | 오류: {str(e)}")
            raise e

    @staticmethod
    def update_tool(tool_id: int, name: Optional[str] = None, 
                   description: Optional[str] = None, 
                   config: Optional[Dict[str, Any]] = None,
                   is_active: Optional[bool] = None) -> Optional[Dict[str, Any]]:
        """
        MCP 도구 정보 업데이트

        Args:
            tool_id: 도구 ID
            name: 새로운 이름
            description: 새로운 설명
            config: 새로운 설정
            is_active: 활성화 상태

        Returns:
            Optional[Dict]: 업데이트된 도구 정보 또는 None
        """
        try:
            if not any([name, description, config, is_active is not None]):
                raise ValueError("업데이트할 내용이 없습니다")

            logger.info(f"✏️ MCP 도구 수정 시작 | ID: {tool_id}")

            with get_db_connection() as conn:
                cursor = conn.cursor(cursor_factory=RealDictCursor)

                # 도구 존재 여부 확인
                cursor.execute("SELECT id FROM mcp_tools WHERE id = %s", (tool_id,))
                if not cursor.fetchone():
                    logger.warning(f"🔍 수정할 MCP 도구를 찾을 수 없음 | ID: {tool_id}")
                    return None

                # 동적 쿼리 생성
                update_fields = []
                values = []

                if name:
                    update_fields.append("name = %s")
                    values.append(name)

                if description is not None:
                    update_fields.append("description = %s")
                    values.append(description)

                if config is not None:
                    # 설정 업데이트 시 관련 필드도 함께 업데이트
                    transport = config.get("transport", "stdio")
                    command = config.get("command")
                    args = config.get("args", [])
                    url = config.get("url")

                    update_fields.extend([
                        "config = %s",
                        "transport = %s",
                        "command = %s",
                        "args = %s",
                        "url = %s"
                    ])
                    values.extend([
                        json.dumps(config),
                        transport,
                        command,
                        json.dumps(args) if args else None,
                        url
                    ])

                if is_active is not None:
                    update_fields.append("is_active = %s")
                    values.append(is_active)

                values.append(tool_id)

                query = f"""
                    UPDATE mcp_tools 
                    SET {', '.join(update_fields)}
                    WHERE id = %s
                    RETURNING id, name, description, transport, command, args, url, config, is_active, created_at, updated_at
                """

                cursor.execute(query, values)
                row = cursor.fetchone()
                conn.commit()

                result = {
                    'id': row['id'],
                    'name': row['name'],
                    'description': row['description'],
                    'transport': row['transport'],
                    'command': row['command'],
                    'args': row['args'] if row['args'] else [],
                    'url': row['url'],
                    'config': row['config'],
                    'is_active': row['is_active'],
                    'created_at': row['created_at'].isoformat(),
                    'updated_at': row['updated_at'].isoformat()
                }

                logger.info(f"✅ MCP 도구 수정 완료 | ID: {tool_id}")
                return result

        except Exception as e:
            logger.error(f"💥 MCP 도구 수정 실패 | ID: {tool_id} | 오류: {str(e)}")
            raise e

    @staticmethod
    def delete_tool(tool_id: int, soft_delete: bool = True) -> bool:
        """
        MCP 도구 삭제

        Args:
            tool_id: 도구 ID
            soft_delete: 소프트 삭제 여부 (True: 비활성화, False: 완전 삭제)

        Returns:
            bool: 삭제 성공 여부
        """
        try:
            logger.info(f"🗑️ MCP 도구 삭제 시작 | ID: {tool_id}, 소프트삭제: {soft_delete}")

            with get_db_connection() as conn:
                cursor = conn.cursor()

                if soft_delete:
                    # 소프트 삭제: 비활성화
                    cursor.execute(
                        "UPDATE mcp_tools SET is_active = false WHERE id = %s", 
                        (tool_id,)
                    )
                else:
                    # 하드 삭제: 완전 제거
                    cursor.execute("DELETE FROM mcp_tools WHERE id = %s", (tool_id,))

                if cursor.rowcount == 0:
                    logger.warning(f"🗑️ 삭제할 MCP 도구를 찾을 수 없음 | ID: {tool_id}")
                    return False

                conn.commit()
                delete_type = "비활성화" if soft_delete else "삭제"
                logger.info(f"✅ MCP 도구 {delete_type} 완료 | ID: {tool_id}")
                return True

        except Exception as e:
            logger.error(f"💥 MCP 도구 삭제 실패 | ID: {tool_id} | 오류: {str(e)}")
            raise e

    @staticmethod
    def delete_tool_by_name(tool_name: str, soft_delete: bool = True) -> bool:
        """
        이름으로 MCP 도구 삭제

        Args:
            tool_name: 도구 이름
            soft_delete: 소프트 삭제 여부

        Returns:
            bool: 삭제 성공 여부
        """
        try:
            logger.info(f"🗑️ MCP 도구 삭제 시작 | 이름: {tool_name}")

            with get_db_connection() as conn:
                cursor = conn.cursor()

                if soft_delete:
                    cursor.execute(
                        "UPDATE mcp_tools SET is_active = false WHERE name = %s", 
                        (tool_name,)
                    )
                else:
                    cursor.execute("DELETE FROM mcp_tools WHERE name = %s", (tool_name,))

                if cursor.rowcount == 0:
                    logger.warning(f"🗑️ 삭제할 MCP 도구를 찾을 수 없음 | 이름: {tool_name}")
                    return False

                conn.commit()
                delete_type = "비활성화" if soft_delete else "삭제"
                logger.info(f"✅ MCP 도구 {delete_type} 완료 | 이름: {tool_name}")
                return True

        except Exception as e:
            logger.error(f"💥 MCP 도구 삭제 실패 | 이름: {tool_name} | 오류: {str(e)}")
            raise e

    @staticmethod
    def get_mcp_config_for_client() -> Dict[str, Any]:
        """
        MCP 클라이언트용 설정 반환 (기존 mcp_config.json 형식과 호환)

        Returns:
            Dict: MCP 클라이언트 설정
        """
        try:
            logger.info("🔧 MCP 클라이언트용 설정 생성")

            tools = MCPToolService.get_all_tools(include_inactive=False)
            
            mcp_servers = {}
            for tool in tools:
                mcp_servers[tool['name']] = tool['config']

            config = {"mcpServers": mcp_servers}
            
            logger.info(f"✅ MCP 클라이언트 설정 생성 완료 | 활성 도구: {len(tools)}개")
            return config

        except Exception as e:
            logger.error(f"💥 MCP 클라이언트 설정 생성 실패 | 오류: {str(e)}")
            return {"mcpServers": {}}

    @staticmethod
    def get_mcp_tool_stats() -> Dict[str, Any]:
        """
        MCP 도구 통계 조회

        Returns:
            Dict: MCP 도구 사용 통계
        """
        try:
            logger.info("📊 MCP 도구 통계 조회 시작")

            with get_db_connection() as conn:
                cursor = conn.cursor(cursor_factory=RealDictCursor)

                # 전체 도구 수
                cursor.execute("SELECT COUNT(*) as total FROM mcp_tools")
                total_tools = cursor.fetchone()['total']

                # 활성 도구 수
                cursor.execute("SELECT COUNT(*) as active FROM mcp_tools WHERE is_active = true")
                active_tools = cursor.fetchone()['active']

                # Transport 별 통계
                cursor.execute("""
                    SELECT transport, COUNT(*) as count 
                    FROM mcp_tools 
                    WHERE is_active = true
                    GROUP BY transport
                """)
                transport_stats = {row['transport']: row['count'] for row in cursor.fetchall()}

                # 가장 최근에 생성된 도구
                cursor.execute("""
                    SELECT name, created_at 
                    FROM mcp_tools 
                    ORDER BY created_at DESC 
                    LIMIT 1
                """)
                latest_tool = cursor.fetchone()

                result = {
                    'total_tools': total_tools,
                    'active_tools': active_tools,
                    'inactive_tools': total_tools - active_tools,
                    'transport_stats': transport_stats,
                    'latest_tool': {
                        'name': latest_tool['name'] if latest_tool else None,
                        'created_at': latest_tool['created_at'].isoformat() if latest_tool else None
                    } if latest_tool else None
                }

                logger.info(f"✅ MCP 도구 통계 조회 완료 | {result}")
                return result

        except Exception as e:
            logger.error(f"💥 MCP 도구 통계 조회 실패 | 오류: {str(e)}")
            raise e