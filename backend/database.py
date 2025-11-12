import psycopg2
from psycopg2.extras import RealDictCursor
import sqlite3
import os
from contextlib import contextmanager
import logging
from datetime import datetime

# SQLite datetime handling
def adapt_datetime_iso(val):
    """Adapt datetime.datetime to timezone-naive ISO 8601 format."""
    return val.isoformat()

def convert_timestamp(val):
    """Convert ISO 8601 string to datetime object."""
    return datetime.fromisoformat(val.decode('utf-8'))

sqlite3.register_adapter(datetime, adapt_datetime_iso)
sqlite3.register_converter("DATETIME", convert_timestamp)
sqlite3.register_converter("TIMESTAMP", convert_timestamp)


logger = logging.getLogger(__name__)

# 데이터베이스 유형 결정
DATABASE_URL = os.getenv("DATABASE_URL")
DB_TYPE = "sqlite" if DATABASE_URL and DATABASE_URL.startswith("sqlite") else "postgres"

# PostgreSQL 설정
POSTGRES_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": os.getenv("DB_PORT", "5432"),
    "database": os.getenv("DB_NAME", "multi_agent"),
    "user": os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD", "admin123")
}

# SQLite 설정
SQLITE_DB_PATH = DATABASE_URL.split("///")[1] if DB_TYPE == "sqlite" else "shared_data.db"


def _get_postgres_connection():
    return psycopg2.connect(**POSTGRES_CONFIG)

def _get_sqlite_connection():
    # SQLite 데이터베이스 파일이 있는 디렉터리가 없으면 생성
    db_dir = os.path.dirname(SQLITE_DB_PATH)
    if db_dir and not os.path.exists(db_dir):
        os.makedirs(db_dir)
    return sqlite3.connect(SQLITE_DB_PATH, detect_types=sqlite3.PARSE_DECLTYPES|sqlite3.PARSE_COLNAMES)

@contextmanager
def get_db_connection(use_dict_cursor=True):
    """데이터베이스 연결 컨텍스트 매니저"""
    conn = None
    cursor = None
    try:
        if DB_TYPE == "postgres":
            conn = _get_postgres_connection()
            if use_dict_cursor:
                cursor = conn.cursor(cursor_factory=RealDictCursor)
            else:
                cursor = conn.cursor()
            yield conn, cursor
        else: # sqlite
            conn = _get_sqlite_connection()
            if use_dict_cursor:
                conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            yield conn, cursor
    except Exception as e:
        logger.error(f"데이터베이스 연결 실패: {e}")
        if conn:
            conn.rollback()
        raise e
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


def init_db():
    """데이터베이스 초기화 - 모든 테이블 생성"""
    try:
        logger.info(f"🚀 데이터베이스 초기화 시작 (유형: {DB_TYPE})")
        
        with get_db_connection(use_dict_cursor=False) as (conn, cursor):
            
            # 데이터베이스 유형에 따른 SQL 문법 조정
            if DB_TYPE == "postgres":
                serial_pk = "SERIAL PRIMARY KEY"
                timestamp_def = "TIMESTAMP DEFAULT CURRENT_TIMESTAMP"
                unique_constraint = "UNIQUE"
                json_type = "JSON"
                bool_type = "BOOLEAN"
                update_trigger_func = '''
                    CREATE OR REPLACE FUNCTION update_updated_at_column()
                    RETURNS TRIGGER AS $$
                    BEGIN
                        NEW.updated_at = CURRENT_TIMESTAMP;
                        RETURN NEW;
                    END;
                    $$ language 'plpgsql';
                '''
                update_trigger = lambda table: f'''
                    DROP TRIGGER IF EXISTS update_{table}_updated_at ON {table};
                    CREATE TRIGGER update_{table}_updated_at
                        BEFORE UPDATE ON {table}
                        FOR EACH ROW
                        EXECUTE FUNCTION update_updated_at_column();
                '''
            else: # sqlite
                serial_pk = "INTEGER PRIMARY KEY AUTOINCREMENT"
                timestamp_def = "DATETIME DEFAULT CURRENT_TIMESTAMP"
                unique_constraint = "UNIQUE"
                json_type = "TEXT" # SQLite는 JSON을 TEXT로 저장
                bool_type = "INTEGER" # SQLite는 BOOLEAN을 0/1로 저장
                update_trigger_func = None
                update_trigger = lambda table: f'''
                    CREATE TRIGGER IF NOT EXISTS update_{table}_updated_at
                    AFTER UPDATE ON {table}
                    FOR EACH ROW
                    BEGIN
                        UPDATE {table}
                        SET updated_at = CURRENT_TIMESTAMP
                        WHERE id = OLD.id;
                    END;
                '''

            # 테이블 생성 쿼리 리스트
            queries = [
                f'''
                CREATE TABLE IF NOT EXISTS messages (
                    id {serial_pk},
                    content TEXT NOT NULL,
                    author VARCHAR(255) NOT NULL,
                    created_at {timestamp_def}
                )
                ''',
                '''CREATE INDEX IF NOT EXISTS idx_messages_created_at ON messages(created_at DESC)''',
                '''CREATE INDEX IF NOT EXISTS idx_messages_author ON messages(author)''',
                
                f'''
                CREATE TABLE IF NOT EXISTS filter_rules (
                    id {serial_pk},
                    name VARCHAR(255) NOT NULL {unique_constraint},
                    pattern TEXT NOT NULL,
                    created_at {timestamp_def},
                    updated_at {timestamp_def}
                )
                ''',
                '''CREATE INDEX IF NOT EXISTS idx_filter_rules_name ON filter_rules(name)''',
                
                f'''
                CREATE TABLE IF NOT EXISTS api_keys (
                    id {serial_pk},
                    name VARCHAR(255) NOT NULL,
                    key_hash VARCHAR(64) NOT NULL {unique_constraint},
                    description TEXT DEFAULT '',
                    is_active {bool_type} DEFAULT 1,
                    created_at {timestamp_def},
                    expires_at DATETIME NULL,
                    last_used_at DATETIME NULL
                )
                ''',
                '''CREATE INDEX IF NOT EXISTS idx_api_keys_hash ON api_keys(key_hash)''',
                '''CREATE INDEX IF NOT EXISTS idx_api_keys_active ON api_keys(is_active)''',
                
                f'''
                CREATE TABLE IF NOT EXISTS mcp_tools (
                    id {serial_pk},
                    name VARCHAR(255) NOT NULL {unique_constraint},
                    description TEXT DEFAULT '',
                    transport VARCHAR(50) NOT NULL DEFAULT 'stdio',
                    command TEXT,
                    args {json_type},
                    url TEXT,
                    config {json_type} NOT NULL,
                    is_active {bool_type} DEFAULT 1,
                    created_at {timestamp_def},
                    updated_at {timestamp_def}
                )
                ''',
                '''CREATE INDEX IF NOT EXISTS idx_mcp_tools_name ON mcp_tools(name)''',
                '''CREATE INDEX IF NOT EXISTS idx_mcp_tools_active ON mcp_tools(is_active)'''
            ]

            # 쿼리 실행
            for query in queries:
                cursor.execute(query)

            # 트리거 생성 (DB 유형에 따라)
            if update_trigger_func:
                cursor.execute(update_trigger_func)
            
            # SQLite에서는 한 번에 하나의 트리거만 적용 가능
            if DB_TYPE == 'sqlite':
                try:
                    cursor.execute(update_trigger('filter_rules'))
                    cursor.execute(update_trigger('mcp_tools'))
                except sqlite3.OperationalError as e:
                    logger.warning(f"트리거 생성 중 경고: {e}")
            else:
                cursor.execute(update_trigger('filter_rules'))
                cursor.execute(update_trigger('mcp_tools'))


            conn.commit()
            logger.info(f"✅ 데이터베이스 초기화 완료 (테이블: messages, filter_rules, api_keys, mcp_tools)")
            
            return True
            
    except Exception as e:
        logger.error(f"💥 데이터베이스 초기화 실패: {e}")
        import traceback
        traceback.print_exc()
        raise e


def test_db_connection():
    """데이터베이스 연결 테스트"""
    try:
        logger.info(f"🔍 데이터베이스 연결 테스트 시작 (유형: {DB_TYPE})")
        
        with get_db_connection(use_dict_cursor=False) as (conn, cursor):
            cursor.execute("SELECT 1")
            result = cursor.fetchone()
            
            if DB_TYPE == "postgres":
                cursor.execute('''
                    SELECT COUNT(*) FROM information_schema.tables 
                    WHERE table_schema = 'public' 
                    AND table_name IN ('messages', 'filter_rules', 'api_keys', 'mcp_tools')
                ''')
            else: # sqlite
                cursor.execute('''
                    SELECT COUNT(*) FROM sqlite_master 
                    WHERE type='table' 
                    AND name IN ('messages', 'filter_rules', 'api_keys', 'mcp_tools')
                ''')
            table_count = cursor.fetchone()[0]
            
            logger.info(f"✅ 데이터베이스 연결 테스트 성공 (테이블 {table_count}개 확인)")
            return result is not None and table_count >= 4
            
    except Exception as e:
        logger.error(f"💥 데이터베이스 연결 테스트 실패: {e}")
        return False


def get_database_stats():
    """데이터베이스 통계 정보"""
    try:
        with get_db_connection() as (conn, cursor):
            stats = {}
            
            cursor.execute("SELECT COUNT(*) as count FROM messages")
            stats['messages_count'] = cursor.fetchone()['count']
            
            cursor.execute("SELECT COUNT(*) as count FROM filter_rules")
            stats['filter_rules_count'] = cursor.fetchone()['count']

            cursor.execute("SELECT COUNT(*) as count FROM api_keys")
            stats['api_keys_count'] = cursor.fetchone()['count']

            cursor.execute("SELECT COUNT(*) as count FROM mcp_tools")
            stats['mcp_tools_count'] = cursor.fetchone()['count']
            
            if DB_TYPE == "postgres":
                cursor.execute("SELECT pg_size_pretty(pg_database_size(current_database())) as size")
                stats['database_size'] = cursor.fetchone()['size']
            else:
                try:
                    db_size = os.path.getsize(SQLITE_DB_PATH)
                    stats['database_size'] = f"{db_size / 1024:.2f} KB"
                except OSError:
                    stats['database_size'] = "N/A"
            
            logger.info(f"📊 데이터베이스 통계: {stats}")
            return stats
            
    except Exception as e:
        logger.error(f"💥 데이터베이스 통계 조회 실패: {e}")
        return {}


def cleanup_old_data(days: int = 30):
    """오래된 데이터 정리"""
    try:
        logger.info(f"🧹 {days}일 이전 데이터 정리 시작")
        
        with get_db_connection(use_dict_cursor=False) as (conn, cursor):
            if DB_TYPE == "postgres":
                cursor.execute("DELETE FROM messages WHERE created_at < CURRENT_TIMESTAMP - INTERVAL '%s days'", (days,))
            else:
                cursor.execute("DELETE FROM messages WHERE created_at < datetime('now', ?)", (f'-{days} days',))
            
            deleted_rows = cursor.rowcount
            conn.commit()
            
            logger.info(f"✅ 데이터 정리 완료: {deleted_rows}개 메시지 삭제")
            return deleted_rows
            
    except Exception as e:
        logger.error(f"💥 데이터 정리 실패: {e}")
        return 0
