import requests

url = "http://localhost:11434/api/chat"

data = {
    "model": "llama3.2",
    "messages": [
        {"role": "user", "content": "What is an API? Answer in one sentence."}
    ],
    "stream": False
}

def chatbot():
    # Initialize the LLaMa model
    model = llama3.Model("ollama3.2")

    # Main loop
    while True:
        # Prompt user to enter a message
        user_input = input("You: ")

        # Check if user wants to exit
        if user_input.lower() == "quit" or user_input.lower() == "exit":
            print("Exiting chatbot. Goodbye!")
            break

        try:
            # Generate response from LLaMa model
            response = model(user_input)

            # Display user's message and LLaMa response
            print("You:", user_input)
            print("LLaMa:", response)
        except llama3.exceptions.LlamaException as e:
            # Handle errors in communication with LLaMa service
            print("Error:", e)
            print("Please try again later.")
        except Exception as e:
            # Handle any other unexpected errors
            print("Error:", e)

# Run the chatbot
if __name__ == "__main__":
    chatbot()