import os
import json
import tiktoken
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("OPENROUTER_API_KEY")
if not API_KEY:
    raise ValueError("OPENROUTER_API_KEY is missing in .env file")
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=API_KEY
)
MODEL = "openai/gpt-4o-mini"
SYSTEM_BUDGET = 500
HISTORY_BUDGET = 3000
CONTEXT_BUDGET = 2000
OUTPUT_BUDGET = 1000
SUMMARY_THRESHOLD = 2000
LAST_N_TURNS = 5
encoding = tiktoken.get_encoding("cl100k_base")

def count_tokens(messages):
    text = json.dumps(messages, ensure_ascii=False)
    return len(encoding.encode(text))
class Conversation:
    def __init__(self):
        self.system_prompt = {
            "role": "system",
            "content": (
                "You are a helpful AI assistant. "
                "Answer clearly and remember important "
                "information from the conversation."
            )
        }
        self.messages = [self.system_prompt]
        self.summary = ""
        self.important_facts = {
            "name": "Arun",
            "order_id": "1001"
        }
    def build_history(self, recent_messages):
        messages = [self.system_prompt]
        if self.summary:
            messages.append({
                "role": "system",
                "content": (
                    "Previous conversation summary: "
                    + self.summary
                )
            })
        messages.append({
            "role": "system",
            "content": (
                "Important facts: "
                + json.dumps(self.important_facts)
            )
        })
        messages.extend(recent_messages)
        return messages
    def sliding_window(self, keep_turns=5):
        recent_messages = [
            message for message in self.messages
            if message["role"] != "system"
        ]
        recent_messages = recent_messages[-(keep_turns * 2):]
        self.messages = self.build_history(recent_messages)
    def check_budget(self):
        system_tokens = count_tokens([
            self.system_prompt
        ])
        history_messages = [
            message for message in self.messages
            if message["role"] != "system"
        ]
        history_tokens = count_tokens(history_messages)
        context_tokens = 0
        print("\nTOKEN BUDGET CHECK")
        print("System tokens:", system_tokens)
        print("History tokens:", history_tokens)
        print("Context tokens:", context_tokens)
        print("Output budget:", OUTPUT_BUDGET)
        if system_tokens > SYSTEM_BUDGET:
            raise ValueError("System token budget exceeded")
        if history_tokens > HISTORY_BUDGET:
            raise ValueError("History token budget exceeded")
        if context_tokens > CONTEXT_BUDGET:
            raise ValueError("Context token budget exceeded")
        total_tokens = (system_tokens+ history_tokens+ context_tokens+ OUTPUT_BUDGET)
        maximum_tokens = (SYSTEM_BUDGET+ HISTORY_BUDGET+ CONTEXT_BUDGET+ OUTPUT_BUDGET)
        if total_tokens > maximum_tokens:
            raise ValueError("Total token budget exceeded")
        print("Token budget: OK")
    def summarise_old_messages(self, keep_turns=5):
        old_messages = [
            message for message in self.messages
            if message["role"] != "system"
        ]
        keep_count = keep_turns * 2
        if len(old_messages) <= keep_count:
            return
        recent_messages = old_messages[-keep_count:]
        messages_to_summarise = old_messages[:-keep_count]
        print("\nSummarising old messages...")
        summary_prompt = [
            {
                "role": "system",
                "content": (
                    "You are a conversation summariser. "
                    "Create a short summary. "
                    "Preserve names, order IDs, locations "
                    "and important facts. "
                    "Never remove important information."
                )
            },
            {
                "role": "user",
                "content": (
                    "Previous summary: "
                    + self.summary
                    + "\n\nOld conversation:\n"
                    + json.dumps(messages_to_summarise)
                )
            }
        ]
        summary_tokens = count_tokens(summary_prompt)
        if summary_tokens + 250 > (
            SYSTEM_BUDGET + HISTORY_BUDGET
        ):
            raise ValueError(
                "Summarisation request exceeds token budget"
            )
        response = client.chat.completions.create(
            model=MODEL,
            messages=summary_prompt,
            max_tokens=250
        )
        new_summary = response.choices[0].message.content
        if not new_summary:
            raise ValueError("Empty summary received")
        self.summary = new_summary
        self.messages = self.build_history(recent_messages)
        print("Summary created successfully!")
        print("Saved summary:", self.summary)
    def prepare_history(self):
        before_tokens = count_tokens(self.messages)
        print("\nHistory before trimming:", before_tokens)
        if before_tokens > SUMMARY_THRESHOLD:
            print("\nHistory exceeded 2000 tokens!")
            for keep_turns in range(LAST_N_TURNS, 0, -1):
                self.summarise_old_messages(keep_turns)
                self.sliding_window(keep_turns)
                history_tokens = count_tokens([
                    message for message in self.messages
                    if message["role"] != "system"
                ])
                print("Trying last",keep_turns,"turns:",history_tokens,"tokens")
                if history_tokens <= HISTORY_BUDGET:
                    break
        final_tokens = count_tokens(self.messages)
        print("History after trimming:", final_tokens)
        self.check_budget()
    def send_message(self, user_message):
        self.messages.append({
            "role": "user",
            "content": user_message
        })
        self.prepare_history()
        response = client.chat.completions.create(
            model=MODEL,
            messages=self.messages,
            max_tokens=OUTPUT_BUDGET
        )
        assistant_reply = response.choices[0].message.content
        if not assistant_reply:
            assistant_reply = "Sorry, I could not generate a response."
        self.messages.append({
            "role": "assistant",
            "content": assistant_reply
        })
        print("\nAssistant:", assistant_reply)
        print("Current history tokens:",count_tokens(self.messages))
    def test_important_facts(self):
        print("\nIMPORTANT FACT TEST")
        self.send_message(
            "What is my name and what is my important order ID?"
        )
chat = Conversation()
questions = [
    "My name is Arun. My important order ID is 1001.",
    "I live in Chennai.",
    "I am learning Python.",
    "What is an LLM?",
    "Explain machine learning.",
    "What is an API?",
    "Explain Python functions.",
    "What is FastAPI?",
    "Explain dictionaries in Python.",
    "What is JSON?",
    "Explain list comprehension.",
    "What is an asynchronous function?",
    "Explain API requests.",
    "What is a database?",
    "Explain SQL.",
    "What is tokenization?",
    "Explain prompt engineering.",
    "What is context management?",
    "Explain sliding windows.",
    "What is conversation summarisation?"
]
for turn, question in enumerate(questions, start=1):
    print("\nTURN:", turn)
    print("User:", question)
    chat.send_message(question)
chat.test_important_facts()
print("\n20 TURN CONVERSATION COMPLETED!")
print("Final token count:",count_tokens(chat.messages))
