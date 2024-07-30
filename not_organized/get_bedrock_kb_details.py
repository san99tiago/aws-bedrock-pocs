import boto3

client = boto3.client("bedrock-agent")


response = client.get_knowledge_base(knowledgeBaseId="QBZT39OO2N")

print(response)
