import boto3
from typing import Any

dynamodb = boto3.resource('dynamodb')

table = dynamodb.Table('feedback_for_styles')

def dynamo_to_python(dynamo_object: dict) -> dict:
    deserializer = TypeDeserializer()
    return {
        k: deserializer.deserialize(v) 
        for k, v in dynamo_object.items()
    }  
  
def python_to_dynamo(python_object: dict) -> dict:
    serializer = TypeSerializer()
    return {
        k: serializer.serialize(v)
        for k, v in python_object.items()
    }


def get_item(item_id: str):
    response = table.get_item(Key={"pk": str(item_id)})
    if 'Item' in response:
        return response['Item']
    return None


def put_item(remove, addon, item_id=None):
    if item_id:
        item = get_item(item_id)
        if item:
            table.update_item(
                Key={'pk': item_id},
                UpdateExpression="SET remove = :s_remove, allowed = :s_addon",
                ExpressionAttributeValues={':s_remove': remove, ':s_addon': addon},
                ReturnValues="UPDATED_NEW"
            )
    else:
        table.put_item(
            Item={
                'pk': item_id,
                'remove': remove,
                'addon': addon
            }
        )