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


async def put_item(remove, addon, item_id):
    if item_id:
        item = await get_item(item_id)
        if item:
            table.update_item(
                Key={'pk': str(item_id)},
                ConditionExpression= 'attribute_exists(pk)',
                UpdateExpression="SET #remove = :s_remove, addon = :s_addon",
                ExpressionAttributeNames={
                    '#remove': 'remove'  # Map the reserved keyword to a placeholder
                },
                ExpressionAttributeValues={':s_remove': remove, ':s_addon': addon},
                ReturnValues="UPDATED_NEW"
            )
            return f"Item {item_id} feedback updated"
    table.put_item(
        Item={
            'pk': str(item_id),
            'remove': remove,
            'addon': addon
        }
    )

    return f"Item {item_id} feedback created"