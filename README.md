## Function Calling Mistral 7B Integration

This project demonstrates two different approaches to utilizing the MistralAI ChatMistralAI API for generating responses based on user prompts. 

1. The first approach, implemented in [Function-Calling-Open-Source.ipynb](https://colab.research.google.com/drive/1CgRaeM0RxO1DFNldHMF1ZgzKXpRCnEKX), uses an open-source language model (`teknium/OpenHermes-2.5-Mistral-7B`). It demonstrates how to define Pydantic models for different types of responses, such as book recommendations, jokes, and song recommendations. It also includes functions for loading the model, generating responses based on prompts, and extracting function calls from the generated responses.

2. The second approach, implemented in [Function-Calling-Mistral-7B (using mistral API Key).py](#), demonstrates integration with MistralAI API key using langchain ChatMistralAI API. This integration allows the system to interact with MistralAI to generate responses based on user prompts. Similar to the local model approach, Pydantic models are defined for different types of responses, and functions are provided for loading the MistralAI model, generating responses, and extracting function calls from the responses. This approach uses an API key for authentication.

### Features

- **Pydantic Models**: Both approaches use Pydantic models to define the structure of the input prompts and the expected response formats. This allows for easy validation of input data and generation of response objects.

- **Function Call Extraction**: Both approaches include functions for extracting function calls from the generated responses. This functionality allows the system to identify specific actions or functions requested by the user in the prompts.

- **API Key Handling**: The MistralAI integration includes a mechanism for loading the API key from a `.env` file, ensuring that the API key is kept secure and not exposed in the code.

- **Response Generation**: Both approaches demonstrate how to generate responses based on user prompts using the respective language models. The responses include text as well as function calls that can be executed based on the user's request.

### Usage

1. **Open-Source Approach**: In `Function-Calling-Open-Source.ipynb`, you can use the defined Pydantic models to create instances of requests (such as a book recommendation request or a joke request) and generate responses based on these requests.

2. **MistralAI API Key Approach**: In `Function-Calling-Mistral-7B (using mistral API Key).py`, the integration with MistralAI API allows for more sophisticated responses based on the MistralAI model's capabilities. Requests can be made to the MistralAI model, and responses containing text and function calls can be generated.

Both approaches demonstrate how language models can be used to create interactive systems that can understand user requests and provide relevant responses.
