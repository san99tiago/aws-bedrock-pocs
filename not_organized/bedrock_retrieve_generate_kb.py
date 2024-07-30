import boto3

client = boto3.client("bedrock-agent-runtime")


response = client.retrieve_and_generate(
    input={
        "text": "How many loans are saved? Please provide the borrower, and coborrower name for each one"
    },
    retrieveAndGenerateConfiguration={
        "knowledgeBaseConfiguration": {
            "generationConfiguration": {
                "inferenceConfig": {
                    "textInferenceConfig": {
                        "maxTokens": 2048,
                        "temperature": 0,
                        "topP": 1,
                    }
                },
            },
            "knowledgeBaseId": "QBZT39OO2N",
            "modelArn": "arn:aws:bedrock:us-west-2::foundation-model/anthropic.claude-v2:1",
            "retrievalConfiguration": {
                "vectorSearchConfiguration": {
                    "numberOfResults": 100,
                }
            },
        },
        "type": "KNOWLEDGE_BASE",
    },
)

print("\n\n")
print(response)

print("\n\n")
print(response.keys())

print("\n\n")
print(response["output"]["text"])


# # Output:
# There are 2 loans saved.
# The first loan has borrower Rick Sanchez and coborrower Morty Smith.
# The second loan has borrower Santiago Garcia Arango and coborrower Diego Puducay.
