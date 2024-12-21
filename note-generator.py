from openai import OpenAI
import time
import os

def addDict(messages, content, role):
    element = {"role": role, "content": content+"\n"}
    messages.append(element)
    return messages


client = OpenAI()

CHUNK_SIZE = 2048
input_path = "./process171024-1.txt"
base_name, ext = os.path.splitext(input_path)  # Split into name and extension
output_path = f"{base_name}_sbobina{ext}" 
infile = open(input_path, 'r', encoding='utf-8')
messages = []
DEV_MESSAGE = "I am going to provide you with an audio transcription of a university lecture. Your task is to correct any grammatical, spelling, or formatting errors and ensure the text is readable and coherent with the context. Do not summarize, shorten, or remove any part of the content. Preserve the original meaning and structure of the text while making it clear and consistent."
REFRESH_MESSAGE = "Remember that your task is to correct any grammatical, spelling, or formatting errors and ensure the text is readable and coherent with the context. Do not summarize, shorten, or remove any part of the content. Preserve the original meaning and structure of the text while making it clear and consistent."
messages.append({"role": "developer", "content": DEV_MESSAGE})

start_time = time.time()
i = 1
with open(output_path, 'w',encoding='utf-8') as outfile:
    while True:
        chunk = infile.read(CHUNK_SIZE)
        if not chunk:
            break
        messages.append({"role": "developer", "content": DEV_MESSAGE})
        messages = addDict(messages, chunk, "user")
        completion = client.chat.completions.create(
        model="gpt-4o",
        messages = messages
        )
        chatGPT_answer = completion.choices[0].message.content
        outfile.write(chatGPT_answer)
        messages = addDict(messages, chatGPT_answer, "assistant")
        i = i + 1
        if i % 10 == 0:
            messages = addDict(messages, REFRESH_MESSAGE, "user")
            completion = client.chat.completions.create(
                model="gpt-4o",
                messages = messages
                )

end_time = time.time()
print(f"Execution time= {(end_time-start_time): .2f}")
infile.close()
