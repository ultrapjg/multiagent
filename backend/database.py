import psycopg2
from psycopg2.extras import RealDictCursor
import os
from contextlib import contextmanager
import logging

logger = logging.getLogger(__name__)

# 데이터베이스 설정
DATABASE_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": os.getenv("DB_PORT", "5432"),
    "database": os.getenv("DB_NAME", "multi_agent"),
    "user": os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD", "admin123")
}


@contextmanager
def get_db_connection():
    """데이터베이스 연결 컨텍스트 매니저"""
    conn = None
    try:
        conn = psycopg2.connect(**DATABASE_CONFIG)
        yield conn
    except Exception as e:
        if conn:
            conn.rollback()
        raise e
    finally:
        if conn:
            conn.close()


def init_db():
    """데이터베이스 초기화 - 메시지 및 필터 규칙 테이블"""
    try:
        logger.info("🚀 데이터베이스 초기화 시작")
        
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            # 메시지 테이블 생성
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS messages (
                    id SERIAL PRIMARY KEY,
                    content TEXT NOT NULL,
                    author VARCHAR(255) NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            # 메시지 테이블 인덱스 생성 (성능 향상)
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_messages_created_at 
                ON messages(created_at DESC)
            ''')

            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_messages_author 
                ON messages(author)
            ''')

            # 필터 규칙 테이블 생성
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS filter_rules (
                    id SERIAL PRIMARY KEY,
                    name VARCHAR(255) NOT NULL UNIQUE,
                    pattern TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            # 필터 규칙 테이블 인덱스 생성
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_filter_rules_name 
                ON filter_rules(name)
            ''')

            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_filter_rules_created_at 
                ON filter_rules(created_at DESC)
            ''')

            # 필터 규칙 업데이트 시간 자동 갱신을 위한 트리거 함수
            cursor.execute('''
                CREATE OR REPLACE FUNCTION update_updated_at_column()
                RETURNS TRIGGER AS $$
                BEGIN
                    NEW.updated_at = CURRENT_TIMESTAMP;
                    RETURN NEW;
                END;
                $$ language 'plpgsql';
            ''')

            # 필터 규칙 테이블에 트리거 적용
            cursor.execute('''
                DROP TRIGGER IF EXISTS update_filter_rules_updated_at ON filter_rules;
                CREATE TRIGGER update_filter_rules_updated_at
                    BEFORE UPDATE ON filter_rules
                    FOR EACH ROW
                    EXECUTE FUNCTION update_updated_at_column();
            ''')

                        # API 키 테이블 생성
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS api_keys (
                    id SERIAL PRIMARY KEY,
                    name VARCHAR(255) NOT NULL,
                    key_hash VARCHAR(64) NOT NULL UNIQUE,
                    description TEXT DEFAULT '',
                    is_active BOOLEAN DEFAULT true,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    expires_at TIMESTAMP NULL,
                    last_used_at TIMESTAMP NULL
                )
            ''')
            
            # 인덱스 생성
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_api_keys_hash 
                ON api_keys(key_hash)
            ''')
            
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_api_keys_active 
                ON api_keys(is_active)
            ''')
            
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_api_keys_expires 
                ON api_keys(expires_at)
            ''')

            conn.commit()
            logger.info("✅ 데이터베이스 초기화 완료 (메시지 + 필터 규칙 테이블)")
            
            # 테이블 생성 확인
            cursor.execute('''
                SELECT table_name FROM information_schema.tables 
                WHERE table_schema = 'public' 
                AND table_name IN ('messages', 'filter_rules')
            ''')
            tables = cursor.fetchall()
            logger.info(f"📋 생성된 테이블: {[table[0] for table in tables]}")
            
            return True
            
    except Exception as e:
        logger.error(f"💥 데이터베이스 초기화 실패: {e}")
        raise e


def test_db_connection():
    """데이터베이스 연결 테스트"""
    try:
        logger.info("🔍 데이터베이스 연결 테스트 시작")
        
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            result = cursor.fetchone()
            
            # 테이블 존재 여부 확인
            cursor.execute('''
                SELECT COUNT(*) FROM information_schema.tables 
                WHERE table_schema = 'public' 
                AND table_name IN ('messages', 'filter_rules')
            ''')
            table_count = cursor.fetchone()[0]
            
            logger.info(f"✅ 데이터베이스 연결 테스트 성공 (테이블 {table_count}개 확인)")
            return result is not None and table_count == 2
            
    except Exception as e:
        logger.error(f"💥 데이터베이스 연결 테스트 실패: {e}")
        return False


def get_database_stats():
    """데이터베이스 통계 정보"""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            stats = {}
            
            # 메시지 통계
            cursor.execute("SELECT COUNT(*) as count FROM messages")
            stats['messages_count'] = cursor.fetchone()['count']
            
            # 필터 규칙 통계
            cursor.execute("SELECT COUNT(*) as count FROM filter_rules")
            stats['filter_rules_count'] = cursor.fetchone()['count']
            
            # 데이터베이스 크기 (대략적)
            cursor.execute('''
                SELECT pg_size_pretty(pg_database_size(current_database())) as size
            ''')
            stats['database_size'] = cursor.fetchone()['size']
            
            logger.info(f"📊 데이터베이스 통계: {stats}")
            return stats
            
    except Exception as e:
        logger.error(f"💥 데이터베이스 통계 조회 실패: {e}")
        return {}


def cleanup_old_data(days: int = 30):
    """오래된 데이터 정리 (선택적)"""
    try:
        logger.info(f"🧹 {days}일 이전 데이터 정리 시작")
        
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            # 30일 이전 메시지 정리 (필요시)
            cursor.execute('''
                DELETE FROM messages 
                WHERE created_at < CURRENT_TIMESTAMP - INTERVAL '%s days'
            ''', (days,))
            
            deleted_messages = cursor.rowcount
            conn.commit()
            
            logger.info(f"✅ 데이터 정리 완료: {deleted_messages}개 메시지 삭제")
            return deleted_messages
            
    except Exception as e:
        logger.error(f"💥 데이터 정리 실패: {e}")
        return 0