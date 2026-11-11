import os
import json
import tiktoken
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY")
)
MODEL = "openai/gpt-4o-mini"
def ask_llm(messages):
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages
    )
    return response.choices[0].message.content
print("\nSTATELESS LLM TEST")
response1 = ask_llm([
    {"role": "user", "content": "My name is Arun"}
])
print("First response:", response1)
response2 = ask_llm([
    {"role": "user", "content": "What is my name?"}
])
print("Second response:", response2)
class Conversation:
    def __init__(self):
        self.messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant. "
                           "Remember information from the "
                           "conversation history."
            }
        ]
    def count_tokens(self):
        encoding = tiktoken.get_encoding("cl100k_base")
        text = json.dumps(self.messages)
        tokens = encoding.encode(text)
        return len(tokens)
    def send_message(self, user_message):
        if user_message == "/reset":
            system_message = self.messages[0]
            self.messages = [system_message]
            print("\nConversation history reset!")
            print("Token count:", self.count_tokens())
            return
        self.messages.append({
            "role": "user",
            "content": user_message
        })
        response = client.chat.completions.create(
            model=MODEL,
            messages=self.messages
        )
        assistant_reply = response.choices[0].message.content
        self.messages.append({
            "role": "assistant",
            "content": assistant_reply
        })
        print("\nAssistant:", assistant_reply)
        print("Total history tokens:", self.count_tokens())
    def save(self, filename):
        with open(filename, "w") as file:
            json.dump(self.messages, file, indent=4)
        print("\nConversation saved successfully!")
    def load(self, filename):
        with open(filename, "r") as file:
            self.messages = json.load(file)
        print("\nConversation loaded successfully!")
        print("Total history tokens:", self.count_tokens())
print("\nCONVERSATION MEMORY TEST")
chat = Conversation()
questions = [
    "My name is Arun.",
    "I live in Chennai.",
    "I am learning Python programming.",
    "What is my name?",
    "Where do I live?",
    "What programming language am I learning?"
]
for i, question in enumerate(questions, start=1):
    print("\nTURN", i)
    print("User:", question)
    chat.send_message(question)
chat.save("conversation.json")
new_chat = Conversation()
new_chat.load("conversation.json")
print("\nCHECK LOADED MEMORY")
new_chat.send_message("What is my name and where do I live?")
