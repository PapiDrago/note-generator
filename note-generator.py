from openai import OpenAI



def addDict(messages, chunk):
    if (len(messages) % 2 == 0):
        element = {"role": "assistant", "content": chunk+"\n"}
    else:
        element = {"role": "user", "content": chunk+"\n"}
    messages.append(element)
    return messages


client = OpenAI()

CHUNK_SIZE = 2048
input_path = "./input.txt"
output_path = "./output.txt"
infile = open(input_path, 'r', encoding='utf-8')
messages = []
messages.append({"role": "developer", "content": "I am going to provide you with an audio transcription of a university lecture.  I ask you to correct any errors, and to make the whole text coherent. Please do not summarize it. Keep it whole"})

with open(output_path, 'w',encoding='utf-8') as outfile:
    while True:
        chunk = infile.read(CHUNK_SIZE)
        if not chunk:
            break
        messages = addDict(messages, chunk)
        completion = client.chat.completions.create(
        model="gpt-4o",
        messages = messages
        )
        outfile.write(completion.choices[0].message.content)
infile.close()