from openai import OpenAI
import time
import os
import tiktoken

def addDict(messages, content, role):
    element = {"role": role, "content": content+"\n"}
    messages.append(element)
    return messages


def num_tokens_from_messages(messages, model):
    # Pasted from https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print("Warning: model not found. Using o200k_base encoding.")
        encoding = tiktoken.get_encoding("o200k_base")
    if model in {
        "gpt-3.5-turbo-0125",
        "gpt-4-0314",
        "gpt-4-32k-0314",
        "gpt-4-0613",
        "gpt-4-32k-0613",
        "gpt-4o-mini-2024-07-18",
        "gpt-4o-2024-08-06"
        }:
        tokens_per_message = 3
        tokens_per_name = 1
    elif "gpt-3.5-turbo" in model:
        print("Warning: gpt-3.5-turbo may update over time. Returning num tokens assuming gpt-3.5-turbo-0125.")
        return num_tokens_from_messages(messages, model="gpt-3.5-turbo-0125")
    elif "gpt-4o-mini" in model:
        print("Warning: gpt-4o-mini may update over time. Returning num tokens assuming gpt-4o-mini-2024-07-18.")
        return num_tokens_from_messages(messages, model="gpt-4o-mini-2024-07-18")
    elif "gpt-4o" in model:
        print("Warning: gpt-4o and gpt-4o-mini may update over time. Returning num tokens assuming gpt-4o-2024-08-06.")
        return num_tokens_from_messages(messages, model="gpt-4o-2024-08-06")
    elif "gpt-4" in model:
        print("Warning: gpt-4 may update over time. Returning num tokens assuming gpt-4-0613.")
        return num_tokens_from_messages(messages, model="gpt-4-0613")
    else:
        raise NotImplementedError(
            f"""num_tokens_from_messages() is not implemented for model {model}."""
        )
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens


client = OpenAI()

CHUNK_SIZE = 2048
MODEL = "gpt-4o"
TOKEN_LIMIT = 30000
input_path = "./ACA_lesson1-1.txt"
base_name, ext = os.path.splitext(input_path)  # Split into name and extension
output_path = f"{base_name}_sbobina{ext}" 
infile = open(input_path, 'r', encoding='utf-8')
messages = []
DEV_MESSAGE = "I am going to provide you with an audio transcription of a university lecture. Your task is to correct any grammatical, spelling, or formatting errors and ensure the text is readable and coherent with the context. Do not summarize, shorten, or remove any part of the content. Preserve the original meaning and structure of the text while making it clear and consistent."
REFRESH_MESSAGE = "Remember that your task is to correct any grammatical, spelling, or formatting errors and ensure the text is readable and coherent with the context. Do not summarize, shorten, or remove any part of the content. Preserve the original meaning and structure of the text while making it clear and consistent."
messages.append({"role": "developer", "content": DEV_MESSAGE})

start_time = time.time()

with open(output_path, 'w',encoding='utf-8') as outfile:
    while True:
        chunk = infile.read(CHUNK_SIZE)
        if not chunk:
            break
        messages.append({"role": "developer", "content": DEV_MESSAGE})
        messages = addDict(messages, chunk, "user")
        if num_tokens_from_messages(messages, MODEL) > TOKEN_LIMIT:
            messages = messages[len(messages)//2:]
        completion = client.chat.completions.create(
        model= MODEL,
        messages = messages
        )
        chatGPT_answer = completion.choices[0].message.content
        outfile.write(chatGPT_answer)
        messages = addDict(messages, chatGPT_answer, "assistant")

end_time = time.time()
print(f"Execution time= {(end_time-start_time): .2f}")
infile.close()
