from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
import sqlite3
from typing import Any, Iterable, Mapping, Sequence
from urllib.parse import urlparse

from spectral_packet_engine.tabular import TabularDataset, TabularSource

_log = logging.getLogger(__name__)


_SQLITE_URL_PREFIX = "sqlite:///"
_SQLITE_READONLY_DENIED_ACTIONS = {
    getattr(sqlite3, "SQLITE_ATTACH", -1),
    getattr(sqlite3, "SQLITE_DETACH", -1),
    getattr(sqlite3, "SQLITE_ALTER_TABLE", -1),
    getattr(sqlite3, "SQLITE_ANALYZE", -1),
    getattr(sqlite3, "SQLITE_CREATE_INDEX", -1),
    getattr(sqlite3, "SQLITE_CREATE_TABLE", -1),
    getattr(sqlite3, "SQLITE_CREATE_TEMP_INDEX", -1),
    getattr(sqlite3, "SQLITE_CREATE_TEMP_TABLE", -1),
    getattr(sqlite3, "SQLITE_CREATE_TEMP_TRIGGER", -1),
    getattr(sqlite3, "SQLITE_CREATE_TEMP_VIEW", -1),
    getattr(sqlite3, "SQLITE_CREATE_TRIGGER", -1),
    getattr(sqlite3, "SQLITE_CREATE_VIEW", -1),
    getattr(sqlite3, "SQLITE_CREATE_VTABLE", -1),
    getattr(sqlite3, "SQLITE_DELETE", -1),
    getattr(sqlite3, "SQLITE_DROP_INDEX", -1),
    getattr(sqlite3, "SQLITE_DROP_TABLE", -1),
    getattr(sqlite3, "SQLITE_DROP_TEMP_INDEX", -1),
    getattr(sqlite3, "SQLITE_DROP_TEMP_TABLE", -1),
    getattr(sqlite3, "SQLITE_DROP_TEMP_TRIGGER", -1),
    getattr(sqlite3, "SQLITE_DROP_TEMP_VIEW", -1),
    getattr(sqlite3, "SQLITE_DROP_TRIGGER", -1),
    getattr(sqlite3, "SQLITE_DROP_VIEW", -1),
    getattr(sqlite3, "SQLITE_DROP_VTABLE", -1),
    getattr(sqlite3, "SQLITE_INSERT", -1),
    getattr(sqlite3, "SQLITE_REINDEX", -1),
    getattr(sqlite3, "SQLITE_TRANSACTION", -1),
    getattr(sqlite3, "SQLITE_UPDATE", -1),
}


def _sqlite_readonly_authorizer(action: int, _arg1, _arg2, _db_name, _trigger_name) -> int:
    if action in _SQLITE_READONLY_DENIED_ACTIONS:
        return sqlite3.SQLITE_DENY
    return sqlite3.SQLITE_OK


def sqlalchemy_is_available() -> bool:
    try:
        import sqlalchemy  # noqa: F401
    except ModuleNotFoundError:
        return False
    return True


def _require_sqlalchemy():
    try:
        import sqlalchemy
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Remote SQL backends require SQLAlchemy. Install the 'sql' extra."
        ) from exc
    return sqlalchemy


def _quote_identifier(name: str) -> str:
    text = str(name)
    if not text:
        raise ValueError("identifier must not be empty")
    if "\x00" in text:
        raise ValueError("identifier must not contain NUL bytes")
    return '"' + text.replace('"', '""') + '"'


def _quote_compound_identifier(name: str) -> str:
    parts = str(name).split(".")
    return ".".join(_quote_identifier(part) for part in parts)


def _split_table_reference(name: str) -> tuple[str | None, str]:
    text = str(name).strip()
    if not text:
        raise ValueError("table name must not be empty")
    parts = text.split(".")
    if any(not part for part in parts):
        raise ValueError(f"invalid table reference: {name!r}")
    if len(parts) == 1:
        return None, parts[0]
    return ".".join(parts[:-1]), parts[-1]


def _literal_sql(value: Any) -> str:
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, (int, float)):
        return str(value)
    return "'" + str(value).replace("'", "''") + "'"


def _count_sql_statements(script: str) -> int:
    statement_count = 0
    buffer = ""
    for line in str(script).splitlines():
        buffer = f"{buffer}\n{line}" if buffer else line
        candidate = buffer.strip()
        if not candidate:
            continue
        if sqlite3.complete_statement(candidate):
            if candidate.rstrip(";").strip():
                statement_count += 1
            buffer = ""
    trailing = buffer.strip().rstrip(";").strip()
    if trailing:
        statement_count += 1
    return statement_count


def _normalize_sql_dtype(name: str) -> str:
    dtype = str(name).strip().upper()
    aliases = {
        "INT": "INTEGER",
        "INTEGER": "INTEGER",
        "BIGINT": "INTEGER",
        "SMALLINT": "INTEGER",
        "TINYINT": "INTEGER",
        "REAL": "REAL",
        "FLOAT": "REAL",
        "DOUBLE": "REAL",
        "DOUBLE_PRECISION": "REAL",
        "DECIMAL": "REAL",
        "NUMERIC": "NUMERIC",
        "TEXT": "TEXT",
        "VARCHAR": "TEXT",
        "CHAR": "TEXT",
        "STRING": "TEXT",
        "DATE": "TEXT",
        "TIME": "TEXT",
        "DATETIME": "TEXT",
        "TIMESTAMP": "TEXT",
        "BOOLEAN": "BOOLEAN",
        "BOOL": "BOOLEAN",
        "BLOB": "BLOB",
        "BINARY": "BLOB",
        "VARBINARY": "BLOB",
        "LARGEBINARY": "BLOB",
    }
    if dtype in aliases:
        return aliases[dtype]
    raise ValueError(f"unsupported SQL type '{name}'")


def _sqlalchemy_column_type(dtype: str):
    sqlalchemy = _require_sqlalchemy()
    mapping = {
        "INTEGER": sqlalchemy.Integer,
        "REAL": sqlalchemy.Float,
        "NUMERIC": sqlalchemy.Numeric,
        "TEXT": sqlalchemy.Text,
        "BOOLEAN": sqlalchemy.Boolean,
        "BLOB": sqlalchemy.LargeBinary,
    }
    return mapping[_normalize_sql_dtype(dtype)]


@dataclass(frozen=True, slots=True)
class DatabaseConfig:
    url: str
    read_only: bool = False
    create_if_missing: bool = True
    connect_timeout_seconds: float = 30.0
    pool_pre_ping: bool = True
    pool_recycle_seconds: int | None = 1800
    pool_size: int | None = None
    max_overflow: int | None = None
    pool_timeout_seconds: float | None = None

    def __post_init__(self) -> None:
        if "://" not in self.url:
            raise ValueError("database url must contain a scheme such as sqlite:///path.db")
        if self.connect_timeout_seconds <= 0:
            raise ValueError("connect_timeout_seconds must be positive")
        if self.pool_recycle_seconds is not None and self.pool_recycle_seconds <= 0:
            raise ValueError("pool_recycle_seconds must be positive when set")
        if self.pool_size is not None and self.pool_size <= 0:
            raise ValueError("pool_size must be positive when set")
        if self.max_overflow is not None and self.max_overflow < -1:
            raise ValueError("max_overflow must be -1 or greater when set")
        if self.pool_timeout_seconds is not None and self.pool_timeout_seconds <= 0:
            raise ValueError("pool_timeout_seconds must be positive when set")
        if self.read_only and self.create_if_missing:
            object.__setattr__(self, "create_if_missing", False)

    @classmethod
    def from_reference(
        cls,
        reference: str | Path,
        *,
        read_only: bool = False,
        create_if_missing: bool = True,
        connect_timeout_seconds: float = 30.0,
        pool_pre_ping: bool = True,
        pool_recycle_seconds: int | None = 1800,
        pool_size: int | None = None,
        max_overflow: int | None = None,
        pool_timeout_seconds: float | None = None,
    ) -> DatabaseConfig:
        raw = str(reference)
        if "://" in raw:
            return cls(
                url=raw,
                read_only=read_only,
                create_if_missing=create_if_missing,
                connect_timeout_seconds=connect_timeout_seconds,
                pool_pre_ping=pool_pre_ping,
                pool_recycle_seconds=pool_recycle_seconds,
                pool_size=pool_size,
                max_overflow=max_overflow,
                pool_timeout_seconds=pool_timeout_seconds,
            )
        return cls.sqlite(
            raw,
            read_only=read_only,
            create_if_missing=create_if_missing,
            connect_timeout_seconds=connect_timeout_seconds,
            pool_pre_ping=pool_pre_ping,
            pool_recycle_seconds=pool_recycle_seconds,
            pool_size=pool_size,
            max_overflow=max_overflow,
            pool_timeout_seconds=pool_timeout_seconds,
        )

    @classmethod
    def sqlite(
        cls,
        path: str | Path,
        *,
        read_only: bool = False,
        create_if_missing: bool = True,
        connect_timeout_seconds: float = 30.0,
        pool_pre_ping: bool = True,
        pool_recycle_seconds: int | None = 1800,
        pool_size: int | None = None,
        max_overflow: int | None = None,
        pool_timeout_seconds: float | None = None,
    ) -> DatabaseConfig:
        sqlite_path = Path(path)
        if str(sqlite_path) == ":memory:":
            url = "sqlite:///:memory:"
        else:
            resolved = sqlite_path if sqlite_path.is_absolute() else sqlite_path.resolve()
            url = f"{_SQLITE_URL_PREFIX}{resolved.as_posix()}"
        return cls(
            url=url,
            read_only=read_only,
            create_if_missing=create_if_missing,
            connect_timeout_seconds=connect_timeout_seconds,
            pool_pre_ping=pool_pre_ping,
            pool_recycle_seconds=pool_recycle_seconds,
            pool_size=pool_size,
            max_overflow=max_overflow,
            pool_timeout_seconds=pool_timeout_seconds,
        )

    @property
    def backend(self) -> str:
        return self.url.split("://", 1)[0].lower()

    @property
    def is_sqlite(self) -> bool:
        return self.backend == "sqlite"

    @property
    def redacted_url(self) -> str:
        parsed = urlparse(self.url)
        if parsed.password is None:
            return self.url
        username = "" if parsed.username is None else parsed.username
        host = "" if parsed.hostname is None else parsed.hostname
        port = "" if parsed.port is None else f":{parsed.port}"
        netloc = f"{username}:***@{host}{port}"
        return parsed._replace(netloc=netloc).geturl()

    @property
    def sqlite_path(self) -> Path | None:
        if not self.is_sqlite:
            return None
        raw_path = self.url[len(_SQLITE_URL_PREFIX):]
        if raw_path == ":memory:":
            return None
        if raw_path.startswith("/") and len(raw_path) >= 3 and raw_path[2] == ":":
            raw_path = raw_path[1:]
        return Path(raw_path)

    @property
    def sqlalchemy_engine_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "future": True,
            "pool_pre_ping": bool(self.pool_pre_ping),
        }
        if self.pool_recycle_seconds is not None:
            kwargs["pool_recycle"] = int(self.pool_recycle_seconds)
        if self.pool_size is not None:
            kwargs["pool_size"] = int(self.pool_size)
        if self.max_overflow is not None:
            kwargs["max_overflow"] = int(self.max_overflow)
        if self.pool_timeout_seconds is not None:
            kwargs["pool_timeout"] = float(self.pool_timeout_seconds)
        return kwargs


@dataclass(frozen=True, slots=True)
class DatabaseCapabilityReport:
    backend: str
    redacted_url: str
    local_bootstrap_supported: bool
    remote_backends_supported: bool
    sqlalchemy_available: bool


@dataclass(frozen=True, slots=True)
class TableColumnSpec:
    name: str
    dtype: str
    nullable: bool = True
    primary_key: bool = False
    default: Any | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "dtype", _normalize_sql_dtype(self.dtype))
        if self.primary_key and self.nullable:
            object.__setattr__(self, "nullable", False)


@dataclass(frozen=True, slots=True)
class TableSchemaSummary:
    table_name: str
    columns: tuple[TableColumnSpec, ...]
    row_count: int | None


@dataclass(frozen=True, slots=True)
class QueryResult:
    query: str
    parameters: dict[str, Any]
    dataset: TabularDataset

    @property
    def row_count(self) -> int:
        return self.dataset.row_count


@dataclass(frozen=True, slots=True)
class StatementExecutionResult:
    statement: str
    parameters: dict[str, Any]
    row_count: int


@dataclass(frozen=True, slots=True)
class ScriptExecutionResult:
    script: str
    statement_count: int


def _table_column_spec_from_python(name: str, values) -> TableColumnSpec:
    dtype_kind = values.dtype.kind
    if dtype_kind in {"i", "u"}:
        dtype = "INTEGER"
    elif dtype_kind == "f":
        dtype = "REAL"
    elif dtype_kind == "b":
        dtype = "BOOLEAN"
    else:
        dtype = "TEXT"
    nullable = False
    if dtype_kind == "f":
        import numpy as np

        nullable = bool(np.isnan(values).any())
    elif dtype_kind in {"U", "S"}:
        nullable = bool(any(value == "" for value in values.tolist()))
    else:
        nullable = bool(any(value is None for value in values.tolist()))
    return TableColumnSpec(name=name, dtype=dtype, nullable=nullable)


class DatabaseConnection:
    def __init__(self, config: DatabaseConfig | str | Path) -> None:
        if isinstance(config, DatabaseConfig):
            self.config = config
        else:
            self.config = DatabaseConfig.from_reference(config)
        self._sqlite_connection: sqlite3.Connection | None = None
        self._engine = None

    def __enter__(self) -> DatabaseConnection:
        self.connect()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    @property
    def capability_report(self) -> DatabaseCapabilityReport:
        return DatabaseCapabilityReport(
            backend=self.config.backend,
            redacted_url=self.config.redacted_url,
            local_bootstrap_supported=self.config.is_sqlite,
            remote_backends_supported=sqlalchemy_is_available(),
            sqlalchemy_available=sqlalchemy_is_available(),
        )

    def connect(self) -> DatabaseConnection:
        if self.config.is_sqlite:
            self._connect_sqlite()
            return self
        sqlalchemy = _require_sqlalchemy()
        self._engine = sqlalchemy.create_engine(self.config.url, **self.config.sqlalchemy_engine_kwargs)
        return self

    def close(self) -> None:
        if self._sqlite_connection is not None:
            self._sqlite_connection.close()
            self._sqlite_connection = None
        if self._engine is not None:
            self._engine.dispose()
            self._engine = None

    def _connect_sqlite(self) -> None:
        if self._sqlite_connection is not None:
            return
        path = self.config.sqlite_path
        if path is not None:
            if path.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
            elif self.config.create_if_missing:
                path.parent.mkdir(parents=True, exist_ok=True)
            else:
                raise FileNotFoundError(f"SQLite database does not exist: {path}")
        database = ":memory:" if path is None else str(path)
        if self.config.read_only and path is not None:
            uri = f"file:{path.as_posix()}?mode=ro"
            connection = sqlite3.connect(
                uri,
                timeout=self.config.connect_timeout_seconds,
                uri=True,
            )
        else:
            connection = sqlite3.connect(
                database,
                timeout=self.config.connect_timeout_seconds,
            )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        self._sqlite_connection = connection

    def _ensure_connected(self) -> None:
        if self.config.is_sqlite:
            self._connect_sqlite()
        elif self._engine is None:
            self.connect()

    def list_tables(self) -> tuple[str, ...]:
        self._ensure_connected()
        if self.config.is_sqlite:
            cursor = self._sqlite_connection.execute(
                "SELECT name FROM sqlite_master WHERE type IN ('table', 'view') AND name NOT LIKE 'sqlite_%' ORDER BY name"
            )
            return tuple(row["name"] for row in cursor.fetchall())
        sqlalchemy = _require_sqlalchemy()
        inspector = sqlalchemy.inspect(self._engine)
        return tuple(sorted(inspector.get_table_names()))

    def _table_exists(self, table_name: str) -> bool:
        self._ensure_connected()
        if self.config.is_sqlite:
            normalized_name = _quote_compound_identifier(table_name)
            cursor = self._sqlite_connection.execute(f"PRAGMA table_info({normalized_name})")
            return bool(cursor.fetchall())
        sqlalchemy = _require_sqlalchemy()
        schema_name, resolved_table_name = _split_table_reference(table_name)
        inspector = sqlalchemy.inspect(self._engine)
        return bool(inspector.has_table(resolved_table_name, schema=schema_name))

    def describe_table(self, table_name: str) -> TableSchemaSummary:
        self._ensure_connected()
        normalized_name = _quote_compound_identifier(table_name)
        if self.config.is_sqlite:
            cursor = self._sqlite_connection.execute(f"PRAGMA table_info({normalized_name})")
            rows = cursor.fetchall()
            if not rows:
                raise ValueError(f"unknown table: {table_name}")
            columns = tuple(
                TableColumnSpec(
                    name=row["name"],
                    dtype=row["type"] or "TEXT",
                    nullable=not bool(row["notnull"]),
                    primary_key=bool(row["pk"]),
                    default=row["dflt_value"],
                )
                for row in rows
            )
            row_count = self._sqlite_connection.execute(
                f"SELECT COUNT(*) AS count FROM {normalized_name}"
            ).fetchone()["count"]
            return TableSchemaSummary(
                table_name=table_name,
                columns=columns,
                row_count=int(row_count),
            )

        sqlalchemy = _require_sqlalchemy()
        schema_name, resolved_table_name = _split_table_reference(table_name)
        table = sqlalchemy.Table(
            resolved_table_name,
            sqlalchemy.MetaData(),
            schema=schema_name,
            autoload_with=self._engine,
        )
        if not table.columns:
            raise ValueError(f"unknown table: {table_name}")
        columns = tuple(
            TableColumnSpec(
                name=column.name,
                dtype=getattr(column.type, "__visit_name__", "TEXT").upper(),
                nullable=bool(column.nullable),
                primary_key=bool(column.primary_key),
                default=None if column.server_default is None else str(column.server_default.arg),
            )
            for column in table.columns
        )
        row_count_statement = sqlalchemy.select(sqlalchemy.func.count()).select_from(table)
        with self._engine.connect() as connection:
            row_count = int(connection.execute(row_count_statement).scalar_one())
        return TableSchemaSummary(table_name=table_name, columns=columns, row_count=row_count)

    def query(
        self,
        query: str,
        *,
        parameters: Mapping[str, Any] | None = None,
    ) -> QueryResult:
        self._ensure_connected()
        bound_parameters = {} if parameters is None else dict(parameters)
        if self.config.is_sqlite:
            self._sqlite_connection.execute("PRAGMA query_only = ON")
            self._sqlite_connection.set_authorizer(_sqlite_readonly_authorizer)
            try:
                cursor = self._sqlite_connection.execute(query, bound_parameters)
                rows = cursor.fetchall()
                if cursor.description is None:
                    raise ValueError("query did not return a result set")
                column_names = [item[0] for item in cursor.description]
            finally:
                self._sqlite_connection.set_authorizer(None)
                try:
                    self._sqlite_connection.execute("PRAGMA query_only = OFF")
                except sqlite3.DatabaseError:
                    _log.warning("Failed to reset PRAGMA query_only after read-only query")
        else:
            sqlalchemy = _require_sqlalchemy()
            with self._engine.connect() as connection:
                result = connection.execute(sqlalchemy.text(query), bound_parameters)
                rows = result.mappings().all()
                column_names = list(result.keys())
        materialized_rows = [
            {column_name: row[column_name] for column_name in column_names}
            for row in rows
        ]
        if not materialized_rows:
            dataset = TabularDataset(
                columns={name: [] for name in column_names},
                source=TabularSource(kind="database", location=self.config.redacted_url, description=query),
            )
        else:
            dataset = TabularDataset.from_rows(
                materialized_rows,
                source=TabularSource(kind="database", location=self.config.redacted_url, description=query),
            )
        return QueryResult(query=query, parameters=bound_parameters, dataset=dataset)

    def execute(
        self,
        statement: str,
        *,
        parameters: Mapping[str, Any] | None = None,
    ) -> StatementExecutionResult:
        self._ensure_connected()
        bound_parameters = {} if parameters is None else dict(parameters)
        if self.config.is_sqlite:
            cursor = self._sqlite_connection.execute(statement, bound_parameters)
            self._sqlite_connection.commit()
            row_count = int(cursor.rowcount if cursor.rowcount is not None else 0)
            return StatementExecutionResult(
                statement=statement,
                parameters=bound_parameters,
                row_count=row_count,
            )
        sqlalchemy = _require_sqlalchemy()
        with self._engine.begin() as connection:
            result = connection.execute(sqlalchemy.text(statement), bound_parameters)
            row_count = int(result.rowcount if result.rowcount is not None else 0)
            return StatementExecutionResult(
                statement=statement,
                parameters=bound_parameters,
                row_count=row_count,
            )

    def execute_script(self, script: str) -> ScriptExecutionResult:
        self._ensure_connected()
        if not self.config.is_sqlite:
            raise NotImplementedError(
                "Multi-statement SQL scripts are only supported for SQLite connections. "
                "Use execute() for a single statement on remote SQL backends."
            )
        self._sqlite_connection.executescript(script)
        self._sqlite_connection.commit()
        return ScriptExecutionResult(
            script=script,
            statement_count=_count_sql_statements(script),
        )

    def create_table(
        self,
        table_name: str,
        columns: Sequence[TableColumnSpec],
        *,
        if_not_exists: bool = True,
    ) -> TableSchemaSummary:
        if not columns:
            raise ValueError("columns must not be empty")
        self._ensure_connected()
        normalized_name = _quote_compound_identifier(table_name)
        if self.config.is_sqlite:
            definitions = []
            for column in columns:
                parts = [_quote_identifier(column.name), column.dtype]
                if not column.nullable:
                    parts.append("NOT NULL")
                if column.primary_key:
                    parts.append("PRIMARY KEY")
                if column.default is not None:
                    parts.append(f"DEFAULT {_literal_sql(column.default)}")
                definitions.append(" ".join(parts))
            if_clause = "IF NOT EXISTS " if if_not_exists else ""
            statement = f"CREATE TABLE {if_clause}{normalized_name} ({', '.join(definitions)})"
            self.execute(statement)
        else:
            sqlalchemy = _require_sqlalchemy()
            schema_name, resolved_table_name = _split_table_reference(table_name)
            metadata = sqlalchemy.MetaData()
            sqlalchemy.Table(
                resolved_table_name,
                metadata,
                *[
                    sqlalchemy.Column(
                        column.name,
                        _sqlalchemy_column_type(column.dtype),
                        nullable=column.nullable,
                        primary_key=column.primary_key,
                        server_default=(
                            None
                            if column.default is None
                            else sqlalchemy.text(_literal_sql(column.default))
                        ),
                    )
                    for column in columns
                ],
                schema=schema_name,
            )
            metadata.create_all(self._engine, checkfirst=if_not_exists)
        return self.describe_table(table_name)

    def add_columns(
        self,
        table_name: str,
        columns: Sequence[TableColumnSpec],
    ) -> TableSchemaSummary:
        if not columns:
            raise ValueError("columns must not be empty")
        self._ensure_connected()
        normalized_name = _quote_compound_identifier(table_name)
        for column in columns:
            parts = [_quote_identifier(column.name), column.dtype]
            if not column.nullable:
                parts.append("NOT NULL")
            if column.default is not None:
                parts.append(f"DEFAULT {_literal_sql(column.default)}")
            statement = f"ALTER TABLE {normalized_name} ADD COLUMN {' '.join(parts)}"
            self.execute(statement)
        return self.describe_table(table_name)

    def drop_table(self, table_name: str, *, if_exists: bool = True) -> None:
        self._ensure_connected()
        if self.config.is_sqlite:
            normalized_name = _quote_compound_identifier(table_name)
            if_clause = "IF EXISTS " if if_exists else ""
            self.execute(f"DROP TABLE {if_clause}{normalized_name}")
            return
        sqlalchemy = _require_sqlalchemy()
        schema_name, resolved_table_name = _split_table_reference(table_name)
        metadata = sqlalchemy.MetaData()
        table = sqlalchemy.Table(resolved_table_name, metadata, schema=schema_name, autoload_with=self._engine)
        table.drop(self._engine, checkfirst=if_exists)

    def delete_rows(
        self,
        table_name: str,
        *,
        where: str | None = None,
        parameters: Mapping[str, Any] | None = None,
    ) -> int:
        normalized_name = _quote_compound_identifier(table_name)
        statement = f"DELETE FROM {normalized_name}"
        if where:
            statement += f" WHERE {where}"
        return self.execute(statement, parameters=parameters).row_count

    def update_rows(
        self,
        table_name: str,
        values: Mapping[str, Any],
        *,
        where: str | None = None,
        parameters: Mapping[str, Any] | None = None,
    ) -> int:
        if not values:
            raise ValueError("values must not be empty")
        normalized_name = _quote_compound_identifier(table_name)
        bound_parameters = {} if parameters is None else dict(parameters)
        assignments = []
        for index, (name, value) in enumerate(values.items()):
            parameter_name = f"set_{index}"
            assignments.append(f"{_quote_identifier(name)} = :{parameter_name}")
            bound_parameters[parameter_name] = value
        statement = f"UPDATE {normalized_name} SET {', '.join(assignments)}"
        if where:
            statement += f" WHERE {where}"
        return self.execute(statement, parameters=bound_parameters).row_count

    def write_dataset(
        self,
        table_name: str,
        dataset: TabularDataset,
        *,
        if_exists: str = "fail",
    ) -> TableSchemaSummary:
        if if_exists not in {"fail", "replace", "append"}:
            raise ValueError("if_exists must be 'fail', 'replace', or 'append'")
        self._ensure_connected()
        existing_tables = set(self.list_tables())
        if table_name in existing_tables:
            if if_exists == "fail":
                raise ValueError(f"table already exists: {table_name}")
            if if_exists == "replace":
                self.drop_table(table_name)
                existing_tables.remove(table_name)

        if table_name not in existing_tables:
            self.create_table(
                table_name,
                [_table_column_spec_from_python(name, values) for name, values in dataset.columns.items()],
            )

        normalized_name = _quote_compound_identifier(table_name)
        column_names = list(dataset.column_names)
        placeholder_names = [f"p{index}" for index in range(len(column_names))]
        placeholders = ", ".join(f":{name}" for name in placeholder_names)
        statement = (
            f"INSERT INTO {normalized_name} "
            f"({', '.join(_quote_identifier(name) for name in column_names)}) "
            f"VALUES ({placeholders})"
        )
        rows = [
            {
                placeholder_name: row[column_name]
                for placeholder_name, column_name in zip(placeholder_names, column_names)
            }
            for row in dataset.to_rows()
        ]
        if self.config.is_sqlite:
            with self._sqlite_connection:
                self._sqlite_connection.executemany(statement, rows)
        else:
            sqlalchemy = _require_sqlalchemy()
            schema_name, resolved_table_name = _split_table_reference(table_name)
            metadata = sqlalchemy.MetaData()
            table = sqlalchemy.Table(resolved_table_name, metadata, schema=schema_name, autoload_with=self._engine)
            with self._engine.begin() as connection:
                connection.execute(table.insert(), rows)
        return self.describe_table(table_name)

    def coerce_column_types(self, table_name: str) -> TableSchemaSummary:
        """Detect actual data types in a table and rebuild with proper affinity.

        SQLite's ``CREATE TABLE AS SELECT`` often stores numeric values with
        TEXT affinity.  This method samples all rows, infers the real type for
        each column, and rebuilds the table with explicit column types so that
        subsequent comparisons (e.g. ``WHERE col = 1``) use the correct
        affinity.
        """
        self._ensure_connected()
        if not self.config.is_sqlite:
            return self.describe_table(table_name)

        normalized_name = _quote_compound_identifier(table_name)
        schema = self.describe_table(table_name)

        needs_coercion = False
        coerced_columns: list[TableColumnSpec] = []
        for column in schema.columns:
            if column.dtype != "TEXT":
                coerced_columns.append(column)
                continue
            cursor = self._sqlite_connection.execute(
                f"SELECT {_quote_identifier(column.name)} FROM {normalized_name} LIMIT 500"
            )
            values = [row[0] for row in cursor.fetchall()]
            if not values or all(v is None for v in values):
                coerced_columns.append(column)
                continue
            non_null = [v for v in values if v is not None]
            all_integer = True
            all_numeric = True
            for v in non_null:
                try:
                    f = float(v)
                    if not f.is_integer():
                        all_integer = False
                except (TypeError, ValueError):
                    all_numeric = False
                    all_integer = False
                    break
            if all_numeric and non_null:
                inferred = "INTEGER" if all_integer else "REAL"
                coerced_columns.append(TableColumnSpec(
                    name=column.name, dtype=inferred,
                    nullable=column.nullable, primary_key=column.primary_key,
                    default=column.default,
                ))
                needs_coercion = True
            else:
                coerced_columns.append(column)

        if not needs_coercion:
            return schema

        tmp_name = f"_coerce_tmp_{table_name.replace('.', '_')}"
        normalized_tmp = _quote_compound_identifier(tmp_name)

        col_defs = []
        cast_exprs = []
        for spec in coerced_columns:
            parts = [_quote_identifier(spec.name), spec.dtype]
            if not spec.nullable:
                parts.append("NOT NULL")
            if spec.primary_key:
                parts.append("PRIMARY KEY")
            if spec.default is not None:
                parts.append(f"DEFAULT {_literal_sql(spec.default)}")
            col_defs.append(" ".join(parts))
            cast_exprs.append(
                f"CAST({_quote_identifier(spec.name)} AS {spec.dtype})"
            )
        col_names = ", ".join(_quote_identifier(c.name) for c in coerced_columns)

        with self._sqlite_connection:
            self._sqlite_connection.execute(f"DROP TABLE IF EXISTS {normalized_tmp}")
            self._sqlite_connection.execute(
                f"CREATE TABLE {normalized_tmp} ({', '.join(col_defs)})"
            )
            self._sqlite_connection.execute(
                f"INSERT INTO {normalized_tmp} ({col_names}) "
                f"SELECT {', '.join(cast_exprs)} FROM {normalized_name}"
            )
            self._sqlite_connection.execute(f"DROP TABLE {normalized_name}")
            self._sqlite_connection.execute(
                f"ALTER TABLE {normalized_tmp} RENAME TO {_quote_identifier(table_name)}"
            )
        _log.info("Coerced column types for table %s", table_name)
        return self.describe_table(table_name)

    def create_table_from_query(
        self,
        target_table: str,
        query: str,
        *,
        parameters: Mapping[str, Any] | None = None,
        replace: bool = False,
        coerce_types: bool = True,
    ) -> TableSchemaSummary:
        self._ensure_connected()
        normalized_target = _quote_compound_identifier(target_table)
        bound_parameters = {} if parameters is None else dict(parameters)
        if self.config.is_sqlite:
            try:
                with self._sqlite_connection:
                    if replace:
                        self._sqlite_connection.execute(f"DROP TABLE IF EXISTS {normalized_target}")
                    self._sqlite_connection.execute(
                        f"CREATE TABLE {normalized_target} AS {query}",
                        bound_parameters,
                    )
            except sqlite3.DatabaseError as exc:
                if self._table_exists(target_table):
                    raise ValueError(f"table already exists: {target_table}") from exc
                raise
            if coerce_types:
                return self.coerce_column_types(target_table)
        else:
            sqlalchemy = _require_sqlalchemy()
            schema_name, resolved_table_name = _split_table_reference(target_table)
            try:
                with self._engine.begin() as connection:
                    if replace:
                        sqlalchemy.Table(
                            resolved_table_name,
                            sqlalchemy.MetaData(),
                            schema=schema_name,
                        ).drop(connection, checkfirst=True)
                    connection.execute(
                        sqlalchemy.text(f"CREATE TABLE {normalized_target} AS {query}"),
                        bound_parameters,
                    )
            except sqlalchemy.exc.SQLAlchemyError as exc:
                if self._table_exists(target_table):
                    raise ValueError(f"table already exists: {target_table}") from exc
                raise
        return self.describe_table(target_table)


    def pivot_query(
        self,
        table_name: str,
        index_column: str,
        pivot_column: str,
        value_column: str,
        *,
        aggregate: str = "MAX",
    ) -> str:
        """Generate a SQL pivot query for the given table.

        Returns the raw SQL string that can be executed or materialized.
        """
        self._ensure_connected()
        normalized_name = _quote_compound_identifier(table_name)
        cursor_sql = (
            f"SELECT DISTINCT {_quote_identifier(pivot_column)} "
            f"FROM {normalized_name} ORDER BY {_quote_identifier(pivot_column)}"
        )
        if self.config.is_sqlite:
            cursor = self._sqlite_connection.execute(cursor_sql)
            distinct_values = [row[0] for row in cursor.fetchall()]
        else:
            sqlalchemy = _require_sqlalchemy()
            with self._engine.connect() as connection:
                result = connection.execute(sqlalchemy.text(cursor_sql))
                distinct_values = [row[0] for row in result.fetchall()]

        agg = aggregate.upper()
        case_clauses = []
        for val in distinct_values:
            alias = f"x={val}" if isinstance(val, (int, float)) else str(val)
            case_clauses.append(
                f"{agg}(CASE WHEN {_quote_identifier(pivot_column)} = {_literal_sql(val)} "
                f"THEN CAST({_quote_identifier(value_column)} AS REAL) END) AS {_quote_identifier(alias)}"
            )
        return (
            f"SELECT {_quote_identifier(index_column)}, {', '.join(case_clauses)} "
            f"FROM {normalized_name} GROUP BY {_quote_identifier(index_column)}"
        )

    def unpivot_query(
        self,
        table_name: str,
        id_columns: Sequence[str],
        value_columns: Sequence[str] | None = None,
    ) -> str:
        """Generate a SQL unpivot (melt) query for the given table.

        Returns the raw SQL string.
        """
        self._ensure_connected()
        if value_columns is None:
            schema = self.describe_table(table_name)
            id_set = set(id_columns)
            value_columns = [c.name for c in schema.columns if c.name not in id_set]
        if not value_columns:
            raise ValueError("no value columns to unpivot")

        normalized_name = _quote_compound_identifier(table_name)
        id_select = ", ".join(_quote_identifier(c) for c in id_columns)
        unions = []
        for col in value_columns:
            unions.append(
                f"SELECT {id_select}, {_literal_sql(col)} AS variable, "
                f"CAST({_quote_identifier(col)} AS REAL) AS value "
                f"FROM {normalized_name}"
            )
        return " UNION ALL ".join(unions)

    def interpolate_time_series(
        self,
        table_name: str,
        time_column: str,
        value_columns: Sequence[str],
        *,
        step: float = 1.0,
        method: str = "linear",
    ) -> str:
        """Generate SQL for linear interpolation over time gaps.

        Returns a query that fills missing time steps with linearly
        interpolated values using window functions.
        """
        normalized_name = _quote_compound_identifier(table_name)
        tc = _quote_identifier(time_column)

        series_cte = (
            f"WITH bounds AS ("
            f"  SELECT MIN(CAST({tc} AS REAL)) AS t_min, MAX(CAST({tc} AS REAL)) AS t_max "
            f"  FROM {normalized_name}"
            f"), "
            f"series(t) AS ("
            f"  SELECT t_min FROM bounds "
            f"  UNION ALL SELECT t + {_literal_sql(step)} FROM series, bounds WHERE t < t_max"
            f"), "
            f"src AS ("
            f"  SELECT CAST({tc} AS REAL) AS t, "
            + ", ".join(
                f"CAST({_quote_identifier(c)} AS REAL) AS {_quote_identifier(c)}"
                for c in value_columns
            )
            + f" FROM {normalized_name}"
            f")"
        )

        interp_cols = []
        for col in value_columns:
            qc = _quote_identifier(col)
            interp_cols.append(
                f"CASE WHEN src.{qc} IS NOT NULL THEN src.{qc} ELSE "
                f"prev.{qc} + (next_.{qc} - prev.{qc}) * "
                f"(series.t - prev.t) / NULLIF(next_.t - prev.t, 0) "
                f"END AS {qc}"
            )

        return (
            f"{series_cte} "
            f"SELECT series.t AS {tc}, {', '.join(interp_cols)} "
            f"FROM series "
            f"LEFT JOIN src ON src.t = series.t "
            f"LEFT JOIN src AS prev ON prev.t = ("
            f"  SELECT MAX(s2.t) FROM src s2 WHERE s2.t <= series.t AND s2.{_quote_identifier(value_columns[0])} IS NOT NULL"
            f") "
            f"LEFT JOIN src AS next_ ON next_.t = ("
            f"  SELECT MIN(s3.t) FROM src s3 WHERE s3.t >= series.t AND s3.{_quote_identifier(value_columns[0])} IS NOT NULL"
            f") "
            f"ORDER BY series.t"
        )

    def upsert_rows(
        self,
        table_name: str,
        rows: Iterable[Mapping[str, Any]],
        *,
        conflict_columns: Sequence[str],
    ) -> int:
        """Insert rows, updating on conflict (INSERT OR REPLACE for SQLite,
        ON CONFLICT DO UPDATE for PostgreSQL)."""
        rows_list = list(rows)
        if not rows_list:
            return 0
        self._ensure_connected()
        column_names = list(rows_list[0].keys())
        normalized_name = _quote_compound_identifier(table_name)

        if self.config.is_sqlite:
            placeholders = ", ".join(f":{name}" for name in column_names)
            col_list = ", ".join(_quote_identifier(c) for c in column_names)
            statement = (
                f"INSERT OR REPLACE INTO {normalized_name} ({col_list}) VALUES ({placeholders})"
            )
            with self._sqlite_connection:
                self._sqlite_connection.executemany(statement, rows_list)
            return len(rows_list)
        else:
            sqlalchemy = _require_sqlalchemy()
            from sqlalchemy.dialects.postgresql import insert as pg_insert
            schema_name, resolved_table_name = _split_table_reference(table_name)
            metadata = sqlalchemy.MetaData()
            table = sqlalchemy.Table(
                resolved_table_name, metadata, schema=schema_name, autoload_with=self._engine
            )
            with self._engine.begin() as connection:
                stmt = pg_insert(table).values(rows_list)
                update_cols = {c.name: c for c in stmt.excluded if c.name not in conflict_columns}
                stmt = stmt.on_conflict_do_update(
                    index_elements=conflict_columns, set_=update_cols
                )
                connection.execute(stmt)
            return len(rows_list)

    def window_aggregate_query(
        self,
        table_name: str,
        value_column: str,
        *,
        order_by: str,
        window_size: int = 3,
        functions: Sequence[str] = ("AVG", "SUM", "COUNT"),
    ) -> str:
        """Generate a query with sliding window aggregates."""
        normalized_name = _quote_compound_identifier(table_name)
        vc = _quote_identifier(value_column)
        ob = _quote_identifier(order_by)
        window_spec = f"OVER (ORDER BY {ob} ROWS BETWEEN {window_size // 2} PRECEDING AND {window_size // 2} FOLLOWING)"

        cols = [f"{ob}", f"{vc}"]
        for fn in functions:
            alias = f"{fn.lower()}_{value_column}_w{window_size}"
            cols.append(f"{fn.upper()}(CAST({vc} AS REAL)) {window_spec} AS {_quote_identifier(alias)}")
        return f"SELECT {', '.join(cols)} FROM {normalized_name} ORDER BY {ob}"

    def batch_query(
        self,
        query_template: str,
        parameter_sets: Sequence[Mapping[str, Any]],
    ) -> list[QueryResult]:
        """Execute a parameterized query with multiple parameter sets."""
        results = []
        for params in parameter_sets:
            results.append(self.query(query_template, parameters=params))
        return results


def inspect_database_capabilities(config: DatabaseConfig | str | Path) -> DatabaseCapabilityReport:
    with DatabaseConnection(config) as connection:
        return connection.capability_report


__all__ = [
    "DatabaseCapabilityReport",
    "DatabaseConfig",
    "DatabaseConnection",
    "QueryResult",
    "ScriptExecutionResult",
    "StatementExecutionResult",
    "TableColumnSpec",
    "TableSchemaSummary",
    "inspect_database_capabilities",
    "sqlalchemy_is_available",
]
