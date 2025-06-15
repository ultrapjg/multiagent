# backend/services/api_key_service.py
import secrets
import hashlib
from typing import List, Optional, Dict, Any
from psycopg2.extras import RealDictCursor
from database import get_db_connection
import logging
from datetime import datetime, timedelta

logger = logging.getLogger("api_key_service")


class APIKeyService:
    """API 키 관리 서비스"""

    @staticmethod
    def generate_api_key() -> str:
        """새로운 API 키 생성"""
        return f"ak_{secrets.token_urlsafe(32)}"

    @staticmethod
    def hash_api_key(api_key: str) -> str:
        """API 키 해싱"""
        return hashlib.sha256(api_key.encode()).hexdigest()

    @staticmethod
    def create_api_key(name: str, description: str = "", expires_days: Optional[int] = None) -> Dict[str, Any]:
        """
        새로운 API 키 생성

        Args:
            name: API 키 이름
            description: 설명
            expires_days: 만료일 (일 단위, None이면 만료되지 않음)

        Returns:
            Dict: 생성된 API 키 정보 (원본 키 포함)
        """
        try:
            logger.info(f"📝 API 키 생성 시작 | 이름: {name}")

            # 새 API 키 생성
            api_key = APIKeyService.generate_api_key()
            key_hash = APIKeyService.hash_api_key(api_key)

            # 만료일 계산
            expires_at = None
            if expires_days:
                expires_at = datetime.now() + timedelta(days=expires_days)

            with get_db_connection() as conn:
                cursor = conn.cursor(cursor_factory=RealDictCursor)

                cursor.execute(
                    """
                    INSERT INTO api_keys (name, key_hash, description, expires_at) 
                    VALUES (%s, %s, %s, %s) 
                    RETURNING id, name, description, created_at, expires_at, is_active
                    """,
                    (name, key_hash, description, expires_at)
                )

                row = cursor.fetchone()
                conn.commit()

                result = {
                    'id': row['id'],
                    'name': row['name'],
                    'description': row['description'],
                    'created_at': row['created_at'].isoformat(),
                    'expires_at': row['expires_at'].isoformat() if row['expires_at'] else None,
                    'is_active': row['is_active'],
                    'api_key': api_key  # 원본 키는 생성 시에만 반환
                }

                logger.info(f"✅ API 키 생성 완료 | ID: {row['id']}, 이름: {name}")
                return result

        except Exception as e:
            logger.error(f"💥 API 키 생성 실패 | 이름: {name} | 오류: {str(e)}")
            raise e

    @staticmethod
    def get_api_keys(include_inactive: bool = False) -> List[Dict[str, Any]]:
        """
        API 키 목록 조회

        Args:
            include_inactive: 비활성화된 키도 포함할지 여부

        Returns:
            List[Dict]: API 키 목록 (해싱된 키는 포함하지 않음)
        """
        try:
            logger.info(f"📋 API 키 목록 조회 | 비활성화 포함: {include_inactive}")

            with get_db_connection() as conn:
                cursor = conn.cursor(cursor_factory=RealDictCursor)

                if include_inactive:
                    query = """
                        SELECT id, name, description, created_at, expires_at, is_active, last_used_at
                        FROM api_keys 
                        ORDER BY created_at DESC
                    """
                    cursor.execute(query)
                else:
                    query = """
                        SELECT id, name, description, created_at, expires_at, is_active, last_used_at
                        FROM api_keys 
                        WHERE is_active = true AND (expires_at IS NULL OR expires_at > NOW())
                        ORDER BY created_at DESC
                    """
                    cursor.execute(query)

                rows = cursor.fetchall()

                api_keys = []
                for row in rows:
                    # 만료 상태 확인
                    is_expired = False
                    if row['expires_at']:
                        is_expired = datetime.now() > row['expires_at']

                    api_keys.append({
                        'id': row['id'],
                        'name': row['name'],
                        'description': row['description'],
                        'created_at': row['created_at'].isoformat(),
                        'expires_at': row['expires_at'].isoformat() if row['expires_at'] else None,
                        'is_active': row['is_active'],
                        'is_expired': is_expired,
                        'last_used_at': row['last_used_at'].isoformat() if row['last_used_at'] else None,
                        'key_preview': f"ak_{'*' * 32}..."  # 키 미리보기
                    })

                logger.info(f"✅ API 키 목록 조회 완료 | 반환: {len(api_keys)}개")
                return api_keys

        except Exception as e:
            logger.error(f"💥 API 키 목록 조회 실패 | 오류: {str(e)}")
            raise e

    @staticmethod
    def get_api_key_by_id(key_id: int) -> Optional[Dict[str, Any]]:
        """
        특정 API 키 조회

        Args:
            key_id: API 키 ID

        Returns:
            Optional[Dict]: API 키 정보 또는 None
        """
        try:
            logger.info(f"🔍 API 키 조회 | ID: {key_id}")

            with get_db_connection() as conn:
                cursor = conn.cursor(cursor_factory=RealDictCursor)

                cursor.execute(
                    """
                    SELECT id, name, description, created_at, expires_at, is_active, last_used_at
                    FROM api_keys 
                    WHERE id = %s
                    """,
                    (key_id,)
                )

                row = cursor.fetchone()

                if not row:
                    logger.warning(f"🔍 API 키를 찾을 수 없음 | ID: {key_id}")
                    return None

                # 만료 상태 확인
                is_expired = False
                if row['expires_at']:
                    is_expired = datetime.now() > row['expires_at']

                result = {
                    'id': row['id'],
                    'name': row['name'],
                    'description': row['description'],
                    'created_at': row['created_at'].isoformat(),
                    'expires_at': row['expires_at'].isoformat() if row['expires_at'] else None,
                    'is_active': row['is_active'],
                    'is_expired': is_expired,
                    'last_used_at': row['last_used_at'].isoformat() if row['last_used_at'] else None,
                    'key_preview': f"ak_{'*' * 32}..."
                }

                logger.info(f"✅ API 키 조회 완료 | ID: {key_id}")
                return result

        except Exception as e:
            logger.error(f"💥 API 키 조회 실패 | ID: {key_id} | 오류: {str(e)}")
            raise e

    @staticmethod
    def validate_api_key(api_key: str) -> Optional[Dict[str, Any]]:
        """
        API 키 유효성 검증

        Args:
            api_key: 검증할 API 키

        Returns:
            Optional[Dict]: 유효한 경우 키 정보, 아니면 None
        """
        try:
            if not api_key or not api_key.startswith('ak_'):
                return None

            key_hash = APIKeyService.hash_api_key(api_key)

            with get_db_connection() as conn:
                cursor = conn.cursor(cursor_factory=RealDictCursor)

                cursor.execute(
                    """
                    SELECT id, name, description, created_at, expires_at, is_active
                    FROM api_keys 
                    WHERE key_hash = %s
                    """,
                    (key_hash,)
                )

                row = cursor.fetchone()

                if not row:
                    logger.warning(f"🔑 유효하지 않은 API 키 사용 시도")
                    return None

                # 활성화 상태 확인
                if not row['is_active']:
                    logger.warning(f"🔑 비활성화된 API 키 사용 시도 | ID: {row['id']}")
                    return None

                # 만료 확인
                if row['expires_at'] and datetime.now() > row['expires_at']:
                    logger.warning(f"🔑 만료된 API 키 사용 시도 | ID: {row['id']}")
                    return None

                # 마지막 사용 시간 업데이트
                cursor.execute(
                    "UPDATE api_keys SET last_used_at = NOW() WHERE id = %s",
                    (row['id'],)
                )
                conn.commit()

                result = {
                    'id': row['id'],
                    'name': row['name'],
                    'description': row['description'],
                    'created_at': row['created_at'].isoformat(),
                    'expires_at': row['expires_at'].isoformat() if row['expires_at'] else None,
                    'is_active': row['is_active']
                }

                logger.info(f"✅ API 키 검증 성공 | ID: {row['id']}, 이름: {row['name']}")
                return result

        except Exception as e:
            logger.error(f"💥 API 키 검증 실패 | 오류: {str(e)}")
            return None

    @staticmethod
    def update_api_key(key_id: int, name: Optional[str] = None, 
                      description: Optional[str] = None, 
                      is_active: Optional[bool] = None) -> Optional[Dict[str, Any]]:
        """
        API 키 정보 업데이트

        Args:
            key_id: API 키 ID
            name: 새로운 이름
            description: 새로운 설명
            is_active: 활성화 상태

        Returns:
            Optional[Dict]: 업데이트된 API 키 정보 또는 None
        """
        try:
            if not any([name, description, is_active is not None]):
                raise ValueError("업데이트할 내용이 없습니다")

            logger.info(f"✏️ API 키 수정 시작 | ID: {key_id}")

            with get_db_connection() as conn:
                cursor = conn.cursor(cursor_factory=RealDictCursor)

                # 키 존재 여부 확인
                cursor.execute("SELECT id FROM api_keys WHERE id = %s", (key_id,))
                if not cursor.fetchone():
                    logger.warning(f"🔍 수정할 API 키를 찾을 수 없음 | ID: {key_id}")
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

                if is_active is not None:
                    update_fields.append("is_active = %s")
                    values.append(is_active)

                values.append(key_id)

                query = f"""
                    UPDATE api_keys 
                    SET {', '.join(update_fields)}
                    WHERE id = %s
                    RETURNING id, name, description, created_at, expires_at, is_active, last_used_at
                """

                cursor.execute(query, values)
                row = cursor.fetchone()
                conn.commit()

                # 만료 상태 확인
                is_expired = False
                if row['expires_at']:
                    is_expired = datetime.now() > row['expires_at']

                result = {
                    'id': row['id'],
                    'name': row['name'],
                    'description': row['description'],
                    'created_at': row['created_at'].isoformat(),
                    'expires_at': row['expires_at'].isoformat() if row['expires_at'] else None,
                    'is_active': row['is_active'],
                    'is_expired': is_expired,
                    'last_used_at': row['last_used_at'].isoformat() if row['last_used_at'] else None
                }

                logger.info(f"✅ API 키 수정 완료 | ID: {key_id}")
                return result

        except Exception as e:
            logger.error(f"💥 API 키 수정 실패 | ID: {key_id} | 오류: {str(e)}")
            raise e

    @staticmethod
    def delete_api_key(key_id: int, soft_delete: bool = True) -> bool:
        """
        API 키 삭제

        Args:
            key_id: API 키 ID
            soft_delete: 소프트 삭제 여부 (True: 비활성화, False: 완전 삭제)

        Returns:
            bool: 삭제 성공 여부
        """
        try:
            logger.info(f"🗑️ API 키 삭제 시작 | ID: {key_id}, 소프트삭제: {soft_delete}")

            with get_db_connection() as conn:
                cursor = conn.cursor()

                if soft_delete:
                    # 소프트 삭제: 비활성화
                    cursor.execute(
                        "UPDATE api_keys SET is_active = false WHERE id = %s", 
                        (key_id,)
                    )
                else:
                    # 하드 삭제: 완전 제거
                    cursor.execute("DELETE FROM api_keys WHERE id = %s", (key_id,))

                if cursor.rowcount == 0:
                    logger.warning(f"🗑️ 삭제할 API 키를 찾을 수 없음 | ID: {key_id}")
                    return False

                conn.commit()
                delete_type = "비활성화" if soft_delete else "삭제"
                logger.info(f"✅ API 키 {delete_type} 완료 | ID: {key_id}")
                return True

        except Exception as e:
            logger.error(f"💥 API 키 삭제 실패 | ID: {key_id} | 오류: {str(e)}")
            raise e

    @staticmethod
    def get_api_key_stats() -> Dict[str, Any]:
        """
        API 키 통계 조회

        Returns:
            Dict: API 키 사용 통계
        """
        try:
            logger.info("📊 API 키 통계 조회 시작")

            with get_db_connection() as conn:
                cursor = conn.cursor(cursor_factory=RealDictCursor)

                # 전체 키 수
                cursor.execute("SELECT COUNT(*) as total FROM api_keys")
                total_keys = cursor.fetchone()['total']

                # 활성 키 수
                cursor.execute("SELECT COUNT(*) as active FROM api_keys WHERE is_active = true")
                active_keys = cursor.fetchone()['active']

                # 만료된 키 수
                cursor.execute("""
                    SELECT COUNT(*) as expired 
                    FROM api_keys 
                    WHERE expires_at IS NOT NULL AND expires_at < NOW()
                """)
                expired_keys = cursor.fetchone()['expired']

                # 최근 24시간 내 사용된 키 수
                cursor.execute("""
                    SELECT COUNT(*) as recent_used 
                    FROM api_keys 
                    WHERE last_used_at >= NOW() - INTERVAL '24 hours'
                """)
                recent_used = cursor.fetchone()['recent_used']

                # 가장 최근에 생성된 키
                cursor.execute("""
                    SELECT name, created_at 
                    FROM api_keys 
                    ORDER BY created_at DESC 
                    LIMIT 1
                """)
                latest_key = cursor.fetchone()

                result = {
                    'total_keys': total_keys,
                    'active_keys': active_keys,
                    'expired_keys': expired_keys,
                    'recent_used_keys': recent_used,
                    'latest_key': {
                        'name': latest_key['name'] if latest_key else None,
                        'created_at': latest_key['created_at'].isoformat() if latest_key else None
                    } if latest_key else None
                }

                logger.info(f"✅ API 키 통계 조회 완료 | {result}")
                return result

        except Exception as e:
            logger.error(f"💥 API 키 통계 조회 실패 | 오류: {str(e)}")
            raise e