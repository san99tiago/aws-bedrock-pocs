# AWS-BEDROCK-POCS

My initial Generative-AI experiments on top of Amazon BedRock and the latest most amazing LLMs.

> Disclaimer: some of these projects are open-source and I just "enhanced" them. Do not use them in production.

<img src="assets/Gen_AI_Drawing_Santi.png" width=50%><img src="assets/Gen_AI_Representation.png" width=50%> <br>

## DEMO 01: RAG SPELLS SIMPLE

Demo inspired by the "Generative AI: Architectures and applications in depth (BOA308)" re:Invent 2023 session.

Technical Details:

- RAG implementation without using Amazon Bedrock KBs (low-level).
- Developed on top of an in-memory vector database powered by [faiss](https://github.com/facebookresearch/faiss) and euclidean distance (IndexFlatL2).
- Embeddings with `amazon.titan-embed-text-v1`.
- Templates for prompts enhanced with [Jinja2](https://github.com/pallets/jinja/).

## DEMO 02: BEDROCK OUTFIT AGENT

Demo inspired by "Mike Chambers" that is an Amazon Bedrock Agent with usage of multiple Agent Groups for Time, Weather and Locations APIs.

Technical Details:

- Infrastructure as Code on top of [Serverless Application Model](https://aws.amazon.com/serverless/sam/).
- Leveraged Amazon Bedrock Agents backed with the Functions:
  - getCurrentTime --> (Lambda Function that connects to external API)
  - getCoordinates --> (Lambda Function that connects to external API)
  - getCurrentWeather --> (Lambda Function that connects to external API)

## DEMO 03: BEDROCK SIMPLE WEATHER AGENT

Demo inspired by "Marcia Villalba" that is an Amazon Bedrock Agent with usage of multiple Agent Groups for interacting with external APIs.

Technical Details:

- Infrastructure as Code on top of [Serverless Application Model](https://aws.amazon.com/serverless/sam/).
- Leveraged Amazon Bedrock Agents backed with the Functions:
  - getWeather --> (Lambda Function that connects to external API)

## LICENSE

Copyright 2024 Santiago Garcia Arango
