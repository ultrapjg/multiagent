# backend/services/message.py
from typing import List, Optional, Dict, Any
from psycopg2.extras import RealDictCursor
from database import get_db_connection
from models import MessageCreate, MessageUpdate, MessageResponse, MessageStats
import logging
from datetime import datetime

logger = logging.getLogger("message_service")


class MessageService:
    """메시지 관련 비즈니스 로직을 처리하는 서비스 클래스"""

    @staticmethod
    def create_message(content: str, author: str) -> Dict[str, Any]:
        """
        메시지 생성 (비즈니스 로직)

        Args:
            content: 메시지 내용
            author: 작성자 이름

        Returns:
            Dict: 생성된 메시지 정보

        Raises:
            Exception: 데이터베이스 오류 시
        """
        try:
            logger.info(f"📝 메시지 생성 시작 | 작성자: {author} | 내용길이: {len(content)}")

            with get_db_connection() as (conn, cursor):
                from database import DB_TYPE
                param_style = "%s" if DB_TYPE == "postgres" else "?"

                if DB_TYPE == "postgres":
                    cursor.execute(
                        f"""
                        INSERT INTO messages (content, author) 
                        VALUES ({param_style}, {param_style}) 
                        RETURNING id, content, author, created_at
                        """,
                        (content, author)
                    )
                    row = cursor.fetchone()
                else:
                    cursor.execute(
                        f"""
                        INSERT INTO messages (content, author) 
                        VALUES ({param_style}, {param_style})
                        """,
                        (content, author)
                    )
                    last_id = cursor.lastrowid
                    cursor.execute(f"SELECT id, content, author, created_at FROM messages WHERE id = {param_style}", (last_id,))
                    row = cursor.fetchone()

                conn.commit()

                result = {
                    'id': row['id'],
                    'content': row['content'],
                    'author': row['author'],
                    'created_at': row['created_at'].isoformat()
                }

                logger.info(f"✅ 메시지 생성 완료 | ID: {row['id']}")
                return result

        except Exception as e:
            logger.error(f"💥 메시지 생성 실패 | 작성자: {author} | 오류: {str(e)}")
            raise e

    @staticmethod
    def get_messages(limit: int = 100, offset: int = 0, author: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        메시지 목록 조회

        Args:
            limit: 조회할 메시지 수
            offset: 건너뛸 메시지 수
            author: 특정 작성자 필터

        Returns:
            List[Dict]: 메시지 목록
        """
        try:
            logger.info(f"📋 메시지 목록 조회 | limit: {limit}, offset: {offset}, author: {author}")

            with get_db_connection() as (conn, cursor):
                from database import DB_TYPE
                param_style = "%s" if DB_TYPE == "postgres" else "?"

                base_query = """
                    SELECT id, content, author, created_at 
                    FROM messages 
                """

                if author:
                    query = base_query + f"""
                        WHERE author ILIKE {param_style}
                        ORDER BY created_at DESC 
                        LIMIT {param_style} OFFSET {param_style}
                    """
                    cursor.execute(query, (f"%{author}%", limit, offset))
                else:
                    query = base_query + f"""
                        ORDER BY created_at DESC 
                        LIMIT {param_style} OFFSET {param_style}
                    """
                    cursor.execute(query, (limit, offset))

                rows = cursor.fetchall()

                messages = [
                    {
                        'id': row['id'],
                        'content': row['content'],
                        'author': row['author'],
                        'created_at': row['created_at'].isoformat()
                    }
                    for row in rows
                ]

                logger.info(f"✅ 메시지 목록 조회 완료 | 반환: {len(messages)}개")
                return messages

        except Exception as e:
            logger.error(f"💥 메시지 목록 조회 실패 | 오류: {str(e)}")
            raise e

    @staticmethod
    def get_message_by_id(message_id: int) -> Optional[Dict[str, Any]]:
        """
        특정 메시지 조회

        Args:
            message_id: 메시지 ID

        Returns:
            Optional[Dict]: 메시지 정보 또는 None
        """
        try:
            logger.info(f"🔍 메시지 조회 | ID: {message_id}")

            with get_db_connection() as (conn, cursor):
                from database import DB_TYPE
                param_style = "%s" if DB_TYPE == "postgres" else "?"

                cursor.execute(
                    f"""
                    SELECT id, content, author, created_at 
                    FROM messages 
                    WHERE id = {param_style}
                    """,
                    (message_id,)
                )

                row = cursor.fetchone()

                if not row:
                    logger.warning(f"🔍 메시지를 찾을 수 없음 | ID: {message_id}")
                    return None

                result = {
                    'id': row['id'],
                    'content': row['content'],
                    'author': row['author'],
                    'created_at': row['created_at'].isoformat()
                }

                logger.info(f"✅ 메시지 조회 완료 | ID: {message_id}")
                return result

        except Exception as e:
            logger.error(f"💥 메시지 조회 실패 | ID: {message_id} | 오류: {str(e)}")
            raise e

    @staticmethod
    def update_message(message_id: int, content: Optional[str] = None, author: Optional[str] = None) -> Optional[
        Dict[str, Any]]:
        """
        메시지 수정

        Args:
            message_id: 메시지 ID
            content: 새로운 내용 (선택사항)
            author: 새로운 작성자 (선택사항)

        Returns:
            Optional[Dict]: 수정된 메시지 정보 또는 None
        """
        try:
            if not content and not author:
                raise ValueError("수정할 내용이 없습니다")

            logger.info(f"✏️ 메시지 수정 시작 | ID: {message_id}")

            with get_db_connection() as (conn, cursor):
                from database import DB_TYPE
                param_style = "%s" if DB_TYPE == "postgres" else "?"

                # 메시지 존재 여부 확인
                cursor.execute(f"SELECT id FROM messages WHERE id = {param_style}", (message_id,))
                if not cursor.fetchone():
                    logger.warning(f"🔍 수정할 메시지를 찾을 수 없음 | ID: {message_id}")
                    return None

                # 동적 쿼리 생성
                update_fields = []
                values = []

                if content:
                    update_fields.append(f"content = {param_style}")
                    values.append(content)

                if author:
                    update_fields.append(f"author = {param_style}")
                    values.append(author)

                values.append(message_id)

                query = f"""
                    UPDATE messages 
                    SET {', '.join(update_fields)}
                    WHERE id = {param_style}
                """

                if DB_TYPE == "postgres":
                    query += " RETURNING id, content, author, created_at"
                
                cursor.execute(query, values)

                if DB_TYPE == "sqlite":
                    cursor.execute(f"SELECT id, content, author, created_at FROM messages WHERE id = {param_style}", (message_id,))
                    row = cursor.fetchone()
                else:
                    row = cursor.fetchone()
                
                conn.commit()

                result = {
                    'id': row['id'],
                    'content': row['content'],
                    'author': row['author'],
                    'created_at': row['created_at'].isoformat()
                }

                logger.info(f"✅ 메시지 수정 완료 | ID: {message_id}")
                return result

        except Exception as e:
            logger.error(f"💥 메시지 수정 실패 | ID: {message_id} | 오류: {str(e)}")
            raise e

    @staticmethod
    def delete_message(message_id: int) -> bool:
        """
        메시지 삭제

        Args:
            message_id: 메시지 ID

        Returns:
            bool: 삭제 성공 여부
        """
        try:
            logger.info(f"🗑️ 메시지 삭제 시작 | ID: {message_id}")

            with get_db_connection(use_dict_cursor=False) as (conn, cursor):
                from database import DB_TYPE
                param_style = "%s" if DB_TYPE == "postgres" else "?"

                cursor.execute(f"DELETE FROM messages WHERE id = {param_style}", (message_id,))

                if cursor.rowcount == 0:
                    logger.warning(f"🗑️ 삭제할 메시지를 찾을 수 없음 | ID: {message_id}")
                    return False

                conn.commit()
                logger.info(f"✅ 메시지 삭제 완료 | ID: {message_id}")
                return True

        except Exception as e:
            logger.error(f"💥 메시지 삭제 실패 | ID: {message_id} | 오류: {str(e)}")
            raise e

    @staticmethod
    def get_message_stats() -> Dict[str, Any]:
        """
        메시지 통계 조회

        Returns:
            Dict: 통계 정보
        """
        try:
            logger.info("📊 메시지 통계 조회 시작")

            with get_db_connection() as (conn, cursor):
                from database import DB_TYPE
                now_func = "NOW()" if DB_TYPE == "postgres" else "CURRENT_TIMESTAMP"
                interval_24h = "INTERVAL '24 hours'" if DB_TYPE == "postgres" else "'-24 hours'"

                # 전체 메시지 수
                cursor.execute("SELECT COUNT(*) as total FROM messages")
                total_messages = cursor.fetchone()['total']

                # 전체 작성자 수
                cursor.execute("SELECT COUNT(DISTINCT author) as total FROM messages")
                total_authors = cursor.fetchone()['total']

                # 최근 24시간 메시지 수
                if DB_TYPE == "postgres":
                    cursor.execute(f"""
                        SELECT COUNT(*) as recent 
                        FROM messages 
                        WHERE created_at >= {now_func} - {interval_24h}
                    """)
                else:
                    cursor.execute(f"""
                        SELECT COUNT(*) as recent 
                        FROM messages 
                        WHERE created_at >= datetime({now_func}, {interval_24h})
                    """)
                recent_message_count = cursor.fetchone()['recent']

                # 가장 활발한 작성자
                cursor.execute("""
                    SELECT author, COUNT(*) as count 
                    FROM messages 
                    GROUP BY author 
                    ORDER BY count DESC 
                    LIMIT 1
                """)
                most_active_result = cursor.fetchone()
                most_active_author = most_active_result['author'] if most_active_result else None

                result = {
                    'total_messages': total_messages,
                    'total_authors': total_authors,
                    'recent_message_count': recent_message_count,
                    'most_active_author': most_active_author
                }

                logger.info(f"✅ 메시지 통계 조회 완료 | {result}")
                return result

        except Exception as e:
            logger.error(f"💥 메시지 통계 조회 실패 | 오류: {str(e)}")
            raise e

    @staticmethod
    def delete_all_messages() -> int:
        """
        모든 메시지 삭제

        Returns:
            int: 삭제된 메시지 수
        """
        try:
            logger.warning("🗑️ 모든 메시지 삭제 시작")

            with get_db_connection(use_dict_cursor=False) as (conn, cursor):

                cursor.execute("SELECT COUNT(*) FROM messages")
                count_before = cursor.fetchone()[0]

                cursor.execute("DELETE FROM messages")
                conn.commit()

                logger.warning(f"✅ 모든 메시지 삭제 완료 | 삭제된 수: {count_before}개")
                return count_before

        except Exception as e:
            logger.error(f"💥 전체 메시지 삭제 실패 | 오류: {str(e)}")
            raise e


# 편의를 위한 함수들 (함수형 인터페이스)
def create_message(content: str, author: str) -> Dict[str, Any]:
    """메시지 생성 (함수형 인터페이스)"""
    return MessageService.create_message(content, author)


def get_messages(limit: int = 100, offset: int = 0, author: Optional[str] = None) -> List[Dict[str, Any]]:
    """메시지 목록 조회 (함수형 인터페이스)"""
    return MessageService.get_messages(limit, offset, author)


def get_message_by_id(message_id: int) -> Optional[Dict[str, Any]]:
    """특정 메시지 조회 (함수형 인터페이스)"""
    return MessageService.get_message_by_id(message_id)


def update_message(message_id: int, content: Optional[str] = None, author: Optional[str] = None) -> Optional[
    Dict[str, Any]]:
    """메시지 수정 (함수형 인터페이스)"""
    return MessageService.update_message(message_id, content, author)


def delete_message(message_id: int) -> bool:
    """메시지 삭제 (함수형 인터페이스)"""
    return MessageService.delete_message(message_id)


def get_message_stats() -> Dict[str, Any]:
    """메시지 통계 조회 (함수형 인터페이스)"""
    return MessageService.get_message_stats()