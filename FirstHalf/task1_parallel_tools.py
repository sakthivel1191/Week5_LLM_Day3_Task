import os
import json
import asyncio
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY")
)
MODEL = "openai/gpt-4o-mini"
orders = {
    1001: {"status": "Shipped"},
    1002: {"status": "Processing"}
}
customers = {
    5: {"name": "Arun", "city": "Chennai"}
}
async def get_order(order_id):
    await asyncio.sleep(1)
    if not isinstance(order_id, int):
        raise ValueError("order_id must be an integer")
    if order_id not in orders:
        raise ValueError(f"Order {order_id} not found")
    return orders[order_id]
async def get_customer(customer_id):
    await asyncio.sleep(1)
    if not isinstance(customer_id, int):
        raise ValueError("customer_id must be an integer")
    if customer_id not in customers:
        raise ValueError(f"Customer {customer_id} not found")
    return customers[customer_id]
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_order",
            "description": "Get order status using order ID",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "integer"}
                },
                "required": ["order_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_customer",
            "description": "Get customer details using customer ID",
            "parameters": {
                "type": "object",
                "properties": {
                    "customer_id": {"type": "integer"}
                },
                "required": ["customer_id"]
            }
        }
    }
]
async def execute_tool(tool_call):
    tool_id = tool_call.id
    tool_name = tool_call.function.name
    try:
        arguments = json.loads(tool_call.function.arguments)
        if tool_name == "get_order":
            result = await get_order(**arguments)
        elif tool_name == "get_customer":
            result = await get_customer(**arguments)
        else:
            raise ValueError(f"Tool '{tool_name}' does not exist")
        return {
            "tool_call_id": tool_id,
            "result": json.dumps(result)
        }
    except Exception as e:
        return {
            "tool_call_id": tool_id,
            "result": json.dumps({"error": str(e)})
        }
async def ask_model(question):
    messages = [
        {
            "role": "system",
            "content": "You are a helpful order support assistant. "
                       "If a tool returns an error, explain it politely."
        },
        {
            "role": "user",
            "content": question
        }
    ]
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        tools=tools,
        parallel_tool_calls=True
    )
    assistant_message = response.choices[0].message
    if assistant_message.tool_calls:
        print("\nTool calls in one round:",
              len(assistant_message.tool_calls))
        messages.append(assistant_message)
        results = await asyncio.gather(
            *[
                execute_tool(call)
                for call in assistant_message.tool_calls
            ]
        )
        for item in results:
            print("Tool result:",
                  item["tool_call_id"],
                  item["result"])
            messages.append({
                "role": "tool",
                "tool_call_id": item["tool_call_id"],
                "content": item["result"]
            })
        final_response = client.chat.completions.create(
            model=MODEL,
            messages=messages
        )
        print("\nFinal Assistant Message:")
        print(final_response.choices[0].message.content)
    else:
        print(assistant_message.content)
async def main():
    question = "Status of orders 1001 and 1002 and customer 5"
    await ask_model(question)
asyncio.run(main())
