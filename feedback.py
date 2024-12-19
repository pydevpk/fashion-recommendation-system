import os

from sqlalchemy import Column
from sqlalchemy import Integer
from sqlalchemy import String
from sqlalchemy import JSON
from sqlalchemy import DateTime
from sqlalchemy import Text
from sqlalchemy import func
from sqlalchemy import create_engine

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.engine.url import URL

from dotenv import load_dotenv
load_dotenv('env.txt')

# database config
def get_db_url(aio=True):
	driver = 'mysql+pymysql' 
	if aio: driver = 'mysql+aiomysql'
	return URL.create(
		driver,
		host=os.getenv('DB_HOST'),
		port=3306,
		username=os.getenv('DB_USER'),
		password=os.getenv('DB_PASSWORD'),
		database= os.getenv('DB_NAME'),
	)


engine = create_async_engine(get_db_url(), future=True, echo=True)

async_session = sessionmaker(
	engine, expire_on_commit=False, class_=AsyncSession)
Base = declarative_base()
	 

class Feedback(Base):
	__tablename__ = 'feedback'

	id = Column(Integer, primary_key=True)
	user_id = Column(String(255))
	remove = Column(JSON, default=list)
	addon = Column(JSON, default=list)
	comment = Column(Text, default="No Comments")
	timestamp = Column(DateTime, nullable=False)
	utimestamp = Column(DateTime, default=func.now(), nullable=False)
	

class FeedbackHistory(Base):
	__tablename__ = 'feedback_history'

	id = Column(Integer, primary_key=True, autoincrement=True)
	item_id = Column(Integer)
	user_id = Column(String(255))
	remove = Column(JSON, default=list)
	addon = Column(JSON, default=list)
	comment = Column(Text, default="No Comments")
	timestamp = Column(DateTime, nullable=False)	 
	utimestamp = Column(DateTime, default=func.now(), nullable=False)	 
		  
	 
"""
dynamodb = boto3.resource('dynamodb')

table = dynamodb.Table('feedback_for_styles')

async def get_item(item_id: str):
	response = table.get_item(Key={"pk": str(item_id)})
	if 'Item' in response:
		return response['Item']
	return None


async def put_item(remove, addon, item_id, comment):
	if item_id:
		item = await get_item(item_id)
		if item:
			table.update_item(
				Key={'pk': str(item_id)},
				ConditionExpression= 'attribute_exists(pk)',
				UpdateExpression="SET #r = :remove, #a = :addon, #c = :comment",
				ExpressionAttributeNames={
						'#r': 'remove',
						'#a': 'addon',
						'#c': 'comment'
					},
				ExpressionAttributeValues={
						':remove': remove, 
						':addon': addon, 
						':comment': comment
					},
				ReturnValues="UPDATED_NEW"
			)
			return f"Item {item_id} feedback updated"
	table.put_item(
		Item={
			'pk': str(item_id),
			'remove': remove,
			'addon': addon,
			'comment': comment
		}
	)

	return f"Item {item_id} feedback created"

"""

if __name__ == "__main__":
	engine = create_engine(get_db_url(False), echo=True)
	Base.metadata.create_all(engine)

