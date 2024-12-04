import boto3

dynamodb = boto3.resource('dynamodb')

table = dynamodb.Table('restriction')

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


def get_item(item_id: int):
    response = table.get_item(Key={"pk": item_id})
    if 'Item' in response:
        print("Retrieved item:", response['Item'])
    return None


def put_item(item_id: int, restictions: str, allowed: str):
    allowed = allowed.split(',')
    restictions = restictions.split(',')
    item = get_item(item_id)
    f_restictions = list(set(list(restictions+item['restictions'])))
    f_allowed = list(set(list(allowed+item['allowed'])))
    s_restictions = ",".join(f_restictions)
    s_allowed = ",".join(f_allowed)
    if item:
        table.update_item(
            Key={'pk': item_id},
            UpdateExpression="SET restictions = :s_restictions, allowed = :s_allowed",
            ExpressionAttributeValues={':s_restictions': s_restictions, ':s_allowed': s_allowed},
            ReturnValues="UPDATED_NEW"
        )
    else:
        table.put_item(
            Item={
                'pk': item_id,
                'restictions': 'cart#123',
                'allowed': ''
            }
        )