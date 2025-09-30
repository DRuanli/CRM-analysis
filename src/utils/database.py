# ================================================================================
# FILE: src/utils/database.py
# ================================================================================
"""
Database Utilities

This module provides an industrial-grade `DatabaseManager` class for managing
connections to relational databases (via SQLAlchemy), Redis (for caching),
and MongoDB (for NoSQL storage). It simplifies database operations with:

- Connection pooling and SQLAlchemy engine management.
- Context-managed sessions with automatic commit/rollback.
- Query execution with pandas integration.
- Bulk insertion of large DataFrames.
- Table inspection utilities.
- Redis integration for caching DataFrames.
- Connection lifecycle management.

Dependencies:
    - SQLAlchemy: database engine and ORM.
    - pandas: DataFrame integration.
    - redis: caching client.
    - pymongo: MongoDB client.
    - loguru: structured logging.
"""

from typing import Optional, Dict, Any
import os
import pandas as pd
from sqlalchemy import create_engine, text, inspect
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker
from contextlib import contextmanager
import redis
from pymongo import MongoClient
from loguru import logger


class DatabaseManager:
    """
    Industrial database connection manager.

    This class provides a unified interface to multiple databases:
    - SQL databases (via SQLAlchemy engine/session).
    - Redis (for caching key-value or serialized DataFrames).
    - MongoDB (for document storage).

    Attributes:
        config (DatabaseConfig): Database configuration object with connection details.
        _engine (Optional[Engine]): SQLAlchemy engine instance (lazy-loaded).
        _redis_client (Optional[redis.Redis]): Redis client instance (lazy-loaded).
        _mongo_client (Optional[MongoClient]): MongoDB client instance (lazy-loaded).
    """

    def __init__(self, config: "DatabaseConfig"):
        self.config = config
        self._engine: Optional[Engine] = None
        self._redis_client: Optional[redis.Redis] = None
        self._mongo_client: Optional[MongoClient] = None

    # --------------------------------------------------------------------------
    # SQLAlchemy Engine
    # --------------------------------------------------------------------------
    @property
    def engine(self) -> Engine:
        """
        Get or create a SQLAlchemy engine with connection pooling.

        Returns:
            Engine: SQLAlchemy engine instance.

        Notes:
            - Uses connection pooling with 10 base connections and up to 20 overflow.
            - Connection pre-ping enabled to avoid stale connections.
        """
        if self._engine is None:
            self._engine = create_engine(
                self.config.connection_string,
                pool_size=10,
                max_overflow=20,
                pool_pre_ping=True,
                echo=False
            )
            logger.info(
                f"Created database engine for {self.config.host}:{self.config.port}/{self.config.name}"
            )
        return self._engine

    # --------------------------------------------------------------------------
    # Session Management
    # --------------------------------------------------------------------------
    @contextmanager
    def session(self):
        """
        Context manager for SQLAlchemy sessions.

        Yields:
            Session: SQLAlchemy session object.

        Ensures:
            - Automatic commit if no errors.
            - Rollback on exceptions.
            - Session closed after use.
        """
        Session = sessionmaker(bind=self.engine)
        session = Session()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Database session error: {str(e)}")
            raise
        finally:
            session.close()

    # --------------------------------------------------------------------------
    # SQL Utilities
    # --------------------------------------------------------------------------
    def execute_query(self, query: str, params: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """
        Execute an SQL query and return results as a DataFrame.

        Args:
            query (str): SQL query string.
            params (Optional[Dict[str, Any]]): Query parameters for parameterized execution.

        Returns:
            pd.DataFrame: Query result as a DataFrame.
        """
        try:
            with self.engine.connect() as conn:
                result = pd.read_sql(text(query), conn, params=params)
                logger.info(f"Query executed successfully: {len(result)} rows returned")
                return result
        except Exception as e:
            logger.error(f"Query execution failed: {str(e)}")
            raise

    def bulk_insert(self, df: pd.DataFrame, table_name: str, if_exists: str = "append"):
        """
        Bulk insert a DataFrame into a database table.

        Args:
            df (pd.DataFrame): DataFrame to insert.
            table_name (str): Name of the target table.
            if_exists (str, optional): Behavior if the table exists
                ('fail', 'replace', 'append'). Defaults to 'append'.
        """
        try:
            df.to_sql(
                table_name,
                self.engine,
                if_exists=if_exists,
                index=False,
                chunksize=10000,
                method="multi"
            )
            logger.info(f"Bulk inserted {len(df)} rows to {table_name}")
        except Exception as e:
            logger.error(f"Bulk insert failed: {str(e)}")
            raise

    def table_exists(self, table_name: str) -> bool:
        """
        Check if a table exists in the database.

        Args:
            table_name (str): Name of the table.

        Returns:
            bool: True if the table exists, False otherwise.
        """
        inspector = inspect(self.engine)
        return table_name in inspector.get_table_names()

    def get_table_info(self, table_name: str) -> Dict[str, Any]:
        """
        Get metadata information about a database table.

        Args:
            table_name (str): Name of the table.

        Returns:
            Dict[str, Any]: Dictionary containing:
                - 'columns': List of columns with metadata.
                - 'primary_keys': Primary key constraint details.
                - 'indexes': List of indexes.
        """
        inspector = inspect(self.engine)
        columns = inspector.get_columns(table_name)
        primary_keys = inspector.get_pk_constraint(table_name)
        indexes = inspector.get_indexes(table_name)

        return {
            "columns": columns,
            "primary_keys": primary_keys,
            "indexes": indexes
        }

    # --------------------------------------------------------------------------
    # Redis Utilities
    # --------------------------------------------------------------------------
    @property
    def redis_client(self) -> redis.Redis:
        """
        Get or create a Redis client.

        Reads connection info from environment variables:
            - REDIS_HOST (default: 'localhost')
            - REDIS_PORT (default: 6379)
            - REDIS_DB   (default: 0)

        Returns:
            redis.Redis: Redis client instance.
        """
        if self._redis_client is None:
            self._redis_client = redis.Redis(
                host=os.getenv("REDIS_HOST", "localhost"),
                port=int(os.getenv("REDIS_PORT", 6379)),
                db=int(os.getenv("REDIS_DB", 0)),
                decode_responses=True
            )
            logger.info("Connected to Redis cache")
        return self._redis_client

    def cache_dataframe(self, key: str, df: pd.DataFrame, expire: int = 3600):
        """
        Cache a DataFrame in Redis.

        Args:
            key (str): Redis key under which the DataFrame is cached.
            df (pd.DataFrame): DataFrame to cache.
            expire (int, optional): Expiration time in seconds. Defaults to 3600.
        """
        try:
            json_data = df.to_json(orient="split", date_format="iso")
            self.redis_client.setex(key, expire, json_data)
            logger.info(f"Cached DataFrame with key: {key}")
        except Exception as e:
            logger.warning(f"Failed to cache DataFrame: {str(e)}")

    def get_cached_dataframe(self, key: str) -> Optional[pd.DataFrame]:
        """
        Retrieve a cached DataFrame from Redis.

        Args:
            key (str): Redis key.

        Returns:
            Optional[pd.DataFrame]: Cached DataFrame if available, else None.
        """
        try:
            json_data = self.redis_client.get(key)
            if json_data:
                df = pd.read_json(json_data, orient="split")
                logger.info(f"Retrieved cached DataFrame with key: {key}")
                return df
        except Exception as e:
            logger.warning(f"Failed to retrieve cached DataFrame: {str(e)}")
        return None

    # --------------------------------------------------------------------------
    # Connection Management
    # --------------------------------------------------------------------------
    def close_connections(self):
        """
        Close all active database connections:
        - SQLAlchemy engine
        - Redis client
        - MongoDB client
        """
        if self._engine:
            self._engine.dispose()
        if self._redis_client:
            self._redis_client.close()
        if self._mongo_client:
            self._mongo_client.close()
        logger.info("All database connections closed")
