from python_a2a import A2AClient, Message, TextContent, MessageRole
import asyncio

async def main():
    # Create a client connected to an A2A-compatible agent
    client = A2AClient("http://localhost:8001")
    
    # View agent information
    print(f"Connected to: {client.agent_card.name}")
    print(f"Description: {client.agent_card.description}")
    print(f"Skills: {[skill.name for skill in client.agent_card.skills]}")
    
    # Ask a question
    message = Message(role=MessageRole.USER, content=TextContent(text="What is the most common diagnosis among female patients under 40?"))
    response = client.send_message(message)
    
    # Get the text content from the response
    if response.content.type == "text":
        print(f"Response: {response.content.text}")
    else:
        print(f"Response: {response.content}")

if __name__ == "__main__":
    asyncio.run(main())