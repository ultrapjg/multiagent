import re
import logging
from typing import List, Dict, Any, Optional
from psycopg2.extras import RealDictCursor
from database import get_db_connection

logger = logging.getLogger(__name__)


class InputFilter:
    """Utility class to check if a given text contains sensitive information."""

    _compiled_rules = []
    _rules_loaded = False

    @classmethod
    def load_rules(cls) -> None:
        """Load filter rules from the database and compile regex patterns."""
        try:
            logger.info("📋 필터 규칙 로딩 시작")
            
            with get_db_connection() as (conn, cursor):
                cursor.execute("SELECT id, name, pattern FROM filter_rules ORDER BY id")
                rows = cursor.fetchall()
                
                cls._compiled_rules = []
                for row in rows:
                    try:
                        compiled_pattern = re.compile(row['pattern'], re.IGNORECASE)
                        cls._compiled_rules.append({
                            'id': row['id'],
                            'name': row['name'],
                            'pattern': row['pattern'],
                            'compiled': compiled_pattern
                        })
                    except re.error as e:
                        logger.error(f"❌ 정규식 컴파일 실패 - ID: {row['id']}, 패턴: {row['pattern']}, 오류: {e}")
                        continue
                
                cls._rules_loaded = True
                logger.info(f"✅ 필터 규칙 로딩 완료: {len(cls._compiled_rules)}개 규칙")
                
        except Exception as e:
            logger.error(f"💥 필터 규칙 로딩 실패: {e}")
            cls._compiled_rules = []
            cls._rules_loaded = False

    @classmethod
    def contains_sensitive(cls, text: str) -> Dict[str, Any]:
        """Return detection result if the text matches any filter rule."""
        if not cls._rules_loaded:
            cls.load_rules()
        
        if not cls._compiled_rules:
            return {
                "is_sensitive": False,
                "matched_rules": [],
                "message": "필터 규칙이 없습니다."
            }
        
        matched_rules = []
        
        try:
            for rule in cls._compiled_rules:
                if rule['compiled'].search(text):
                    matched_rules.append({
                        'id': rule['id'],
                        'name': rule['name'],
                        'pattern': rule['pattern']
                    })
            
            is_sensitive = len(matched_rules) > 0
            
            result = {
                "is_sensitive": is_sensitive,
                "matched_rules": matched_rules,
                "message": f"{len(matched_rules)}개의 민감한 패턴이 감지되었습니다." if is_sensitive else "민감한 내용이 감지되지 않았습니다."
            }
            
            if is_sensitive:
                logger.warning(f"⚠️ 민감한 내용 감지: {len(matched_rules)}개 규칙 매칭")
                for rule in matched_rules:
                    logger.warning(f"  - 규칙: {rule['name']} (ID: {rule['id']})")
            
            return result
            
        except Exception as e:
            logger.error(f"💥 민감한 내용 검사 실패: {e}")
            return {
                "is_sensitive": False,
                "matched_rules": [],
                "message": f"검사 중 오류 발생: {str(e)}"
            }

    @classmethod
    def get_rules_count(cls) -> int:
        """Return the number of loaded rules."""
        if not cls._rules_loaded:
            cls.load_rules()
        return len(cls._compiled_rules)

    @classmethod
    def get_all_rules(cls) -> List[Dict[str, Any]]:
        """Return all loaded rules information."""
        if not cls._rules_loaded:
            cls.load_rules()
        
        return [{
            'id': rule['id'],
            'name': rule['name'],
            'pattern': rule['pattern']
        } for rule in cls._compiled_rules]

    @classmethod
    def reload_rules(cls) -> bool:
        """Force reload rules from database."""
        try:
            cls._rules_loaded = False
            cls.load_rules()
            return cls._rules_loaded
        except Exception as e:
            logger.error(f"💥 규칙 강제 리로드 실패: {e}")
            return False


def init_filter_db():
    """Initialize filter rules database table."""
    try:
        with get_db_connection(use_dict_cursor=False) as (conn, cursor):
            
            # 테이블 생성
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS filter_rules (
                    id SERIAL PRIMARY KEY,
                    name VARCHAR(255) NOT NULL,
                    pattern TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # 인덱스 생성
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_filter_rules_name 
                ON filter_rules(name)
            ''')
            
            conn.commit()
            logger.info("✅ 필터 규칙 테이블 초기화 완료")
            return True
            
    except Exception as e:
        logger.error(f"💥 필터 규칙 테이블 초기화 실패: {e}")
        return False


def test_filter_connection():
    """Test filter database connection."""
    try:
        with get_db_connection(use_dict_cursor=False) as (conn, cursor):
            cursor.execute("SELECT COUNT(*) FROM filter_rules")
            count = cursor.fetchone()[0]
            logger.info(f"✅ 필터 DB 연결 테스트 성공: {count}개 규칙 존재")
            return True
    except Exception as e:
        logger.error(f"💥 필터 DB 연결 테스트 실패: {e}")
        return False