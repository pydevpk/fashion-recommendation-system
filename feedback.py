import boto3
from typing import Any
from typing import List
from pydantic import BaseModel


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