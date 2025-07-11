# from datetime import UTC, datetime
from datetime import datetime, timezone
from typing import Optional

import sqlalchemy as sa
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import CHAR, TypeDecorator
from sqlalchemy import MetaData
from sqlalchemy import func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.orm import DeclarativeBase


class StringUUID(TypeDecorator):
	impl = CHAR
	cache_ok = True
	
	def process_bind_param(self, value, dialect):
		if value is None:
			return value
		elif dialect.name == "postgresql":
			return str(value)
		else:
			return value.hex
	
	def load_dialect_impl(self, dialect):
		if dialect.name == "postgresql":
			return dialect.type_descriptor(UUID())
		else:
			return dialect.type_descriptor(CHAR(36))
	
	def process_result_value(self, value, dialect):
		if value is None:
			return value
		return str(value)


POSTGRES_INDEXES_NAMING_CONVENTION = {
	"ix": "%(column_0_label)s_idx",
	"uq": "%(table_name)s_%(column_0_name)s_key",
	"ck": "%(table_name)s_%(constraint_name)s_check",
	"fk": "%(table_name)s_%(column_0_name)s_fkey",
	"pk": "%(table_name)s_pkey",
}

metadata = MetaData(naming_convention=POSTGRES_INDEXES_NAMING_CONVENTION)
db = SQLAlchemy(metadata=metadata)


class Base(DeclarativeBase):
	metadata = metadata


class Workflow(Base):
	"""
	Workflow, for `Workflow App` and `Chat App workflow mode`.

	Attributes:

	- id (uuid) Workflow ID, pk
	- tenant_id (uuid) Workspace ID
	- app_id (uuid) App ID
	- type (string) Workflow type

		`workflow` for `Workflow App`

		`chat` for `Chat App workflow mode`

	- version (string) Version

		`draft` for draft version (only one for each app), other for version number (redundant)

	- graph (text) Workflow canvas configuration (JSON)

		The entire canvas configuration JSON, including Node, Edge, and other configurations

		- nodes (array[object]) Node list, see Node Schema

		- edges (array[object]) Edge list, see Edge Schema

	- created_by (uuid) Creator ID
	- created_at (timestamp) Creation time
	- updated_by (uuid) `optional` Last updater ID
	- updated_at (timestamp) `optional` Last update time
	"""
	__tablename__ = "workflows"
	__table_args__ = (
		db.PrimaryKeyConstraint("id", name="workflow_pkey"),
		db.Index("workflow_version_idx", "tenant_id", "app_id", "version"),
	)
	
	id: Mapped[str] = mapped_column(StringUUID, server_default=db.text("uuid_generate_v4()"))
	tenant_id: Mapped[str] = mapped_column(StringUUID, nullable=False)
	app_id: Mapped[str] = mapped_column(StringUUID, nullable=False)
	type: Mapped[str] = mapped_column(db.String(255), nullable=False)
	version: Mapped[str] = mapped_column(db.String(255), nullable=False)
	marked_name: Mapped[str] = mapped_column(default="", server_default="")
	marked_comment: Mapped[str] = mapped_column(default="", server_default="")
	graph: Mapped[str] = mapped_column(sa.Text)
	features: Mapped[str] = mapped_column("features", sa.TEXT)
	created_by: Mapped[str] = mapped_column(StringUUID, nullable=False)
	created_at: Mapped[datetime] = mapped_column(db.DateTime, nullable=False, server_default=func.current_timestamp())
	updated_by: Mapped[Optional[str]] = mapped_column(StringUUID)
	updated_at: Mapped[datetime] = mapped_column(
		db.DateTime,
		nullable=False,
		default=datetime.now(timezone.utc).replace(tzinfo=None),
		server_onupdate=func.current_timestamp(),
	)
	_environment_variables: Mapped[str] = mapped_column(
		"environment_variables", db.Text, nullable=False, server_default="{}"
	)
	_conversation_variables: Mapped[str] = mapped_column(
		"conversation_variables", db.Text, nullable=False, server_default="{}"
	)
	
	VERSION_DRAFT = "draft"
