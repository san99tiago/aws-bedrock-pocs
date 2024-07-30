####################################################################################################
# NOTE: This demo was inspired from AWS re:Invent 2023 session with my own touch/ideas
# >> "Generative AI: Architectures and applications in depth (BOA308)"
####################################################################################################

# Use the json library to parse the JSON input
import json
import boto3

# Use the faiss library to perform similarity search on the embeddings
import faiss

# Use the numpy library to perform numerical operations
import numpy as np

# Use the jinja2 library to render the template with the results
from jinja2 import Template

# Initialize bedrock client
bedrock_runtime_client = boto3.client("bedrock-runtime")

# Set the simple list of spells (simple example to have some fun)
spells = [
    "To fly, you can nod your head and say 'fly fly fly'",
    "To teleport, you close your eyes and say 'may I be there with my atoms rearranged'",
    "To run fast, you must say 'I am as fast as the wind'",
    "To turn the enemy into a fly, you must say 'I do not fight with useless beings'",
    "To turn yourself into a dragon, you must scream loud 'I am the dragon of the sky'",
    "To turn a paper into money, you must say 'Warren Buffet, give me your power'",
    "To turn sand into water, you must focus and say 'My precious sand you will become H2O'",
    "To turn water into ice, you must say 'This water will become solid form'",
    "To turn water into food, you must say 'This water will become a delicious meal'",
    "To make an animal talk, you must say 'You are now polyglot'",
    "To make yourself invisible, you must say 'I am the ghost of the night'",
    "To make yourself invincible, you must say 'I will be the last one standing for eternity'",
    "To make yourself a giant, you must say 'I am huge with wisdom and strength'",
    "To kill a demon, you must say 'I am taking you to the underworld'",
    "To go back in time 5 minutes, you must say 'I am the master of time and I will go back'",
    "To improve your sight, you must say 'I am the healthy eagle of the sky'",
    "To make your enemy weak, you must say 'You are not worthy of my power'",
    "To open a locked door, you must say 'Open the hidden mysteries of this door'",
    "To hide an object, you must say 'This object is now invisible to the world'",
    "To gain wisdom, you must say 'I am as wise as the owls of the night'",
]
print("\n\n<<< Spells in the magic book... >>>")
print(f"Total number of spells: {len(spells)}")
print(f"Spells: {spells}")


def embed(spell: str) -> list:
    """
    Embeds a text (spell) using the Amazon titan-embed-text-v1 model.
    """

    kwargs = {
        "modelId": "amazon.titan-embed-text-v1",
        "contentType": "application/json",
        "accept": "*/*",
        "body": json.dumps({"inputText": spell}),
    }

    response = bedrock_runtime_client.invoke_model(**kwargs)
    response_body = json.loads(response.get("body").read().decode("utf-8"))

    return response_body.get("embedding")


# Create an array to store the embeddings of the spells
spell_embeddings = np.array([]).reshape(0, 1536)

# Embed all the spells
print("\n\n<<< Embedding spells... >>>")
for spell in spells:
    single_spell_embedding = embed(spell)
    spell_embeddings = np.append(
        spell_embeddings, np.array(single_spell_embedding).reshape(1, -1), axis=0
    )
print(f"Spell embeddings shape: {spell_embeddings.shape}")
print(f"Spell embeddings: {spell_embeddings}")

# Create a vector store (this is just an in memory vector store for the demo)
magic_book_shelf_index = faiss.IndexFlatL2(1536)

# Add the spell embeddings to the vector store
print("\n\n<<< Adding spells to the magic book shelf... >>>")
magic_book_shelf_index.add(spell_embeddings)
print(f"Number of spells in the magic book shelf: {magic_book_shelf_index.ntotal}")
print(f"Magic book shelf index: {magic_book_shelf_index}")

# NOTE: until this point, we have done NOTHING with LLMs (only embeddings model)
# ... Now, we will use the LLMs to find the most similar spells

###################
# Here is where the magic (RAG) happens
###################

# Define the query spell
# NOTE: UPDATE AS NEEDED FOR THE DEMO!!!!
# query_spell = "How do I fly?"
# query_spell = "How do I convert water into food?"
# query_spell = "How do I convert water into ice?"
# query_spell = "How do I get rich?"
# query_spell = "How do I become invisible?"
# query_spell = "How do I become invincible?"
query_spell = "Are you happy?"

print(f"Query spell (the one to answer): {query_spell}")

# Embed the query spell (*Note that we are using the same embed function as before)
embedded_query_spell = embed(query_spell)
print(f"Embedded query spell: {embedded_query_spell}")

# Perform similarity search on the query spell
print(
    "\n\n<<< Searching for the 5 most similar spells with embeddings proximity search... >>>"
)
k = 5
distances, indices = magic_book_shelf_index.search(np.array([embedded_query_spell]), k)
print(f"Distances: {distances}")
print(f"Indices: {indices}")

# Create a prompt for the LLM (adding the query spell and the most similar spells found -- RAG)
prompt_template = """
Given the spells provided in the following list of spells between the <spell> tags,
find the answer to the question written between the <question> tags.

<spells>
{%- for spell in spells %}
    `{{ spell }}`{% endfor %}
</spells>

<question>{{ question }}</question>

You must provide an answer that is the COMPLETE spell (including the description and spell).

Provide an answer in full.
If the spells are not relevant to the question being asked,
you must NOT provide any other opinion and you must respond with:
"Given the information I know, I cannot provide an answer to the question."

"""

# Use Jinja2 to render the prompt template adding the query spell and the most similar spells found
print("\n\n<<< Rendering the prompt template... >>>")
data = {
    "spells": [spells[i] for i in indices[0]],
    "question": query_spell,
}
template = Template(prompt_template)
prompt = template.render(data)

print(f"Prompt for the LLM (with RAG): {prompt}")

# Send the prompt to the LLM (RAG) to get the answer
print("\n\n<<< Sending the prompt to the LLM (RAG) to get the answer... >>>")
kwargs = {
    "modelId": "amazon.titan-text-express-v1",
    "contentType": "application/json",
    "accept": "*/*",
    "body": json.dumps(
        {
            "inputText": prompt,
            "textGenerationConfig": {
                "maxTokenCount": 4096,
                "stopSequences": [],
                "temperature": 0,
                "topP": 1,
            },
        }
    ),
}

response = bedrock_runtime_client.invoke_model(**kwargs)
response_body = json.loads(response.get("body").read().decode("utf-8"))

generation = response_body["results"][0]["outputText"]
print(f"Answer: {generation}")
