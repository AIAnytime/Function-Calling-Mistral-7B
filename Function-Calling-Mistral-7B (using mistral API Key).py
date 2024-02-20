import locale
import gc, json, re
import xml.etree.ElementTree as ET
from functools import partial
import os
from dotenv import load_dotenv

import transformers
import torch

from langchain.utils.openai_functions import convert_pydantic_to_openai_function
from langchain.pydantic_v1 import BaseModel, Field, validator
from langchain_mistralai.chat_models import ChatMistralAI

# Load API key from .env file
load_dotenv()
mistral_api_key = os.getenv("MISTRAL_API_KEY")

if mistral_api_key is None:
    raise ValueError("MISTRAL_API_KEY not found in .env file")

tokenizer = transformers.AutoTokenizer.from_pretrained("teknium/OpenHermes-2.5-Mistral-7B")
model = ChatMistralAI(mistral_api_key=mistral_api_key)

def delete_model(*args):
    for var in args:
        if var in globals():
            del globals()[var]

    gc.collect()
    torch.cuda.empty_cache()

class BookRecommendation(BaseModel):
    """Provides book recommendations based on specified interest."""
    interest: str = Field(description="question of user interest about a book.")
    recommended_book: str = Field(description="answer to recommend a book")

    @validator("interest")
    def interests_must_not_be_empty(cls, field):
        if not field:
            raise ValueError("Interest cannot be empty.")
        return field

class Joke(BaseModel):
    """Get a joke that includes the setup and punchline"""
    setup: str = Field(description="question to set up a joke")
    punchline: str = Field(description="answer to resolve the joke")

    # You can add custom validation logic easily with Pydantic.
    @validator("setup")
    def question_ends_with_question_mark(cls, field):
        if field[-1] != "?":
            raise ValueError("Badly formed question!")
        return field

class SongRecommendation(BaseModel):
    """Provides song recommendations based on specified genre."""
    genre: str = Field(description="question to recommend a song.")
    song: str = Field(description="answer to recommend a song")

    @validator("genre")
    def genre_must_not_be_empty(cls, field):
        if not field:
            raise ValueError("genre cannot be empty.")
        return field

convert_pydantic_to_openai_function(SongRecommendation)

def extract_function_calls(completion):
    if isinstance(completion, str):
        content = completion
    else:
        content = completion.content

    pattern = r"<multiplefunctions>(.*?)</multiplefunctions>"
    match = re.search(pattern, content, re.DOTALL)
    if not match:
        return None

    multiplefn = match.group(1)
    functions = []
    for fn_match in re.finditer(r"<functioncall>(.*?)</functioncall>", multiplefn, re.DOTALL):
        fn_text = fn_match.group(1)
        try:
            functions.append(json.loads(fn_text))
        except json.JSONDecodeError:
            pass  # Ignore invalid JSON

    return functions

def generate_hermes(prompt, model, tokenizer, generation_config_overrides={}):
    fn = """{"name": "function_name", "arguments": {"arg_1": "value_1", "arg_2": value_2, ...}}"""
    prompt = f"""system
You are a helpful assistant with access to the following functions:

{convert_pydantic_to_openai_function(Joke)}

{convert_pydantic_to_openai_function(BookRecommendation)}

{convert_pydantic_to_openai_function(SongRecommendation)}

To use these functions respond with:
<multiplefunctions>
    <functioncall> {fn} </functioncall>
    <functioncall> {fn} </functioncall>
    ...
</multiplefunctions>

Edge cases you must handle:
- If there are no functions that match the user request, you will respond politely that you cannot help.
user
{prompt}
assistant"""

    with torch.inference_mode():
        completion = model.invoke([{"role": "user", "content": prompt}])

    if isinstance(completion, str):
        # Handle the case where completion is a string
        content = completion.strip()
    else:
        # Handle the case where completion is an AIMessage object
        content = completion.content.strip()

    functions = extract_function_calls(content)

    if functions:
        print(functions)
    else:
        print(content)
    print("="*100)

generation_func = partial(generate_hermes, model=model, tokenizer=tokenizer)

prompts = [
    "Tell me a joke about kenyan athletes",
    "Song for working out",
    "Recommend me a book on singularity."
]

for prompt in prompts:
    generation_func(prompt)
