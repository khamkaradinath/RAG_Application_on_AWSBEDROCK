import boto3
client = boto3.client("bedrock", region_name="us-east-1")
response = client.list_foundation_models()
print(response)
