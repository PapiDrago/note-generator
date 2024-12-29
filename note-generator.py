from openai import OpenAI
import time
import os
import sys
import tiktoken

def getChunk(infile): ##it does not address situation like very long words or file text length < CHUNK_SIZE
    chunk = infile.read(CHUNK_SIZE)
    while chunk and chunk[-1] not in ['.', '\n']:
        new_char = infile.read(1)
        if not new_char:
            break
        chunk = chunk + new_char
    return chunk


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
    print(f'Number of tokens used by the prompt= {num_tokens}')
    return num_tokens


client = OpenAI()

CHUNK_SIZE = 2048
MODEL = "gpt-4o"
STOP_DURATION = 300
TOKEN_LIMIT = 30000-6000 
#Input-Output token limit is 30000. I do not know why when I prompt with about 24000 token then the answer fail despite of knowing that the output token are about 400.
#Actually 30000 is the token per minute limit! Is this number cumulating from output token?
#TPM = 30000

if len(sys.argv) < 2 or len(sys.argv) > 3:
    print("Usage: python note-generator.py <input_filepath>")
else:
    input_path = sys.argv[1]
    base_name, ext = os.path.splitext(input_path)  # Split into name and extension
    output_path = f"{base_name}_sbobina{ext}" 
    infile = open(input_path, 'r', encoding='utf-8')
    messages = []
    DEV_MESSAGE = "I am going to provide you with an audio transcription of a university lecture. Your task is to correct any grammatical, spelling, or formatting errors and ensure the text is readable and coherent with the context. Do not summarize, shorten, or remove any part of the content. Preserve the original meaning and structure of the text while making it clear and consistent."
    REFRESH_MESSAGE = "Remember that your task is to correct any grammatical, spelling, or formatting errors and ensure the text is readable and coherent with the context. Do not summarize, shorten, or remove any part of the content. Preserve the original meaning and structure of the text while making it clear and consistent."


    ex_start_time = time.time()
    start_time = ex_start_time

    token_bucket = 0
    #i = 0
    with open(output_path, 'w',encoding='utf-8') as outfile:
        while True:
            chunk = getChunk(infile)
            if not chunk:
                break
            # if i % 10 and i !=0:
            #     messages = messages[len(messages)//2:]
            # messages.append({"role": "developer", "content": DEV_MESSAGE})
            # messages = addDict(messages, chunk, "user")
            tokens_number = num_tokens_from_messages(messages, MODEL)
            if tokens_number > TOKEN_LIMIT:
                print("Number of tokens would have exceeded!")
                messages = messages[len(messages)//2:]
            token_bucket += (tokens_number + 500)
            print(f'token bucket= {token_bucket}')

            completion = client.chat.completions.create( #completion = response
            model= MODEL,
            messages = [{"role": 'developer', "content": DEV_MESSAGE},
                        {"role": 'user', "content": chunk}]
            )
            print(f'Number of tokens used in obtaining the answer= {completion.usage.total_tokens}')
            token_bucket -= (num_tokens_from_messages(messages, MODEL) + 500)
            token_bucket += completion.usage.total_tokens
            chatGPT_answer = completion.choices[0].message.content
            outfile.write(chatGPT_answer)
            #messages = addDict(messages, chatGPT_answer, "assistant")
            #i +=1

    ex_end_time = time.time()
    print(f"Execution time= {(ex_end_time-ex_start_time): .2f}")
    infile.close()

'''Notice that requests get disabled after reaching token per minute limit
without any control after about 10 minutes within the execution of the program.
For this reason I choose to pause it after 5 minutes. It is a lazy solution.
Actually it still does not work! Coming back to halve the size of messages.''' 
