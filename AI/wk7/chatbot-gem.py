import sys
import ollama

# Define the model to use. Ensure you have pulled this model via 'ollama pull <model_name>'
MODEL_NAME = "llama3.2"


def start_chatbot():
  print(f"--- Local Ollama Chatbot (Model: {MODEL_NAME}) ---")
  print("Type your message below. Type 'exit', 'quit', or 'q' to end the chat.\n")

  # Maintain conversation history so the model remembers context
  messages = []

  while True:
    try:
      # Get user input from the command line
      user_input = input("You: ").strip()

      # Check for exit commands
      if user_input.lower() in ["exit", "quit", "q"]:
        print("\nGoodbye! Have a great day.")
        break

      # Skip empty inputs
      if not user_input:
        continue

      # Append the user's message to the conversation history
      messages.append({"role": "user", "content": user_input})

      print("Assistant: ", end="", flush=True)

      # Send the chat history to the local Ollama service
      # Using streaming provides a better user experience for text generation
      stream = ollama.chat(model=MODEL_NAME, messages=messages, stream=True)

      # Accumulate response content to update history later
      full_response = ""
      for chunk in stream:
        content = chunk["message"]["content"]
        print(content, end="", flush=True)
        full_response += content

      print("\n")  # Add a newline after the full response finishes

      # Append the assistant's response to the conversation history
      messages.append({"role": "assistant", "content": full_response})

    except KeyboardInterrupt:
      # Handle Ctrl+C gracefully
      print("\n\nChat interrupted. Goodbye!")
      sys.exit(0)

    except ConnectionError as ce:
      # Handle specific network/connection failures to Ollama service
      print(
          "\n[Error] Could not connect to the Ollama service. Make sure Ollama"
          f" is running.\nDetails: {ce}\n"
      )
    except Exception as e:
      # Handle other unexpected errors (e.g., model not found, invalid API usage)
      print(
          f"\n[Error] An unexpected error occurred: {e}\nIf the model '{MODEL_NAME}'"
          " is missing, try running 'ollama pull "
          f"{MODEL_NAME}' in your terminal.\n"
      )


if __name__ == "__main__":
  start_chatbot()