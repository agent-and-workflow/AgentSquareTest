from sqlalchemy import create_engine
from sqlalchemy import desc
from sqlalchemy.orm import sessionmaker, scoped_session
from contextlib import contextmanager


class DataBase:
	def __init__(self, user: str, password: str, host: str, port: str, dbname: str):
		"""
		初始化数据库连接
		:param db_url: 例如 postgresql://user:password@localhost:port/dbname
		"""
		self.user = user
		self.password = password
		self.host = host
		self.port = port
		self.dbname = dbname
		self.url = f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.dbname}"
		self.engine = create_engine(self.url, pool_pre_ping=True)
		self.Session = scoped_session(sessionmaker(bind=self.engine))
	
	@contextmanager
	def get_session(self):
		"""
		获取数据库 session，会自动提交或回滚
		用法：with dbio.get_session() as session:
		"""
		session = self.Session()
		try:
			yield session
			session.commit()
		except Exception as e:
			session.rollback()
			raise e
		finally:
			session.close()
	
	def insert(self, instance):
		"""
		插入一条数据对象
		"""
		with self.get_session() as session:
			session.add(instance)
	
	def update_by_id(self, model, record_id, update_data: dict):
		"""
		根据主键 ID 更新字段
		"""
		with self.get_session() as session:
			obj = session.get(model, record_id)
			if not obj:
				raise ValueError("Record not found.")
			for key, value in update_data.items():
				setattr(obj, key, value)
			session.add(obj)  # 主键存在：更新；主键不存在：插入
	
	def query(self, model, filters: dict = None):
		"""
		查询记录（可选条件过滤）
		"""
		with self.get_session() as session:
			query = session.query(model)
			if filters:
				for attr, val in filters.items():
					query = query.filter(getattr(model, attr) == val)
			results = query.all()
			return [
				{
					attr: getattr(row, attr)
					for attr in row.__mapper__.c.keys()  # 仅访问映射的字段
				}
				for row in results
			]
	
	def get_latest(self, model):
		"""
		获取指定表中按 created_at 排序的最新一条记录的主键 id
		:param model: ORM 映射的表类，如 Workflow
		:return: 最新记录的 id，若无记录则返回 None
		"""
		with self.get_session() as session:
			record = session.query(model).order_by(desc(model.created_at)).first()
			if record:
				return {
					attr: getattr(record, attr)
					for attr in record.__mapper__.c.keys()
				}
			return None
	
	def shutdown(self):
		"""
		清理连接池
		"""
		self.Session.remove()
		self.engine.dispose()

