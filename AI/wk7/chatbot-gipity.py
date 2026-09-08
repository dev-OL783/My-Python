import requests

# Ollama's default local API endpoint.
OLLAMA_URL = "http://localhost:11434/api/chat"

# Change this to any model you have installed in Ollama.
# MODEL = "llama3.2:1b"
MODEL = "llama3.2"


def get_response(message):
    # Send a message to Ollama and return the model's response.
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "user", "content": message}
        ],
        "stream": False
    }

    try:
        response = requests.post(
            OLLAMA_URL,
            json=payload,
            timeout=60
        )

        # Raise an exception for HTTP errors such as 404 or 500.
        response.raise_for_status()

        data = response.json()
        return data["message"]["content"]

    except requests.exceptions.ConnectionError:
        return "Error: Could not connect to Ollama. Is Ollama running?"

    except requests.exceptions.Timeout:
        return "Error: Ollama took too long to respond."

    except requests.exceptions.RequestException as error:
        return f"Error communicating with Ollama: {error}"

    except (KeyError, ValueError):
        return "Error: Ollama returned an unexpected response."


def main():
    # Run the chatbot's command-line interface.
    print("Local Ollama Chatbot")
    print("Type 'exit' or 'quit' to end the program.")
    print("--------------------------------------------------------")

    # Keep accepting messages until the user chooses to exit.
    while True:
        try:
            user_message = input("User (You): ").strip()

            # Provide a clear way to leave the chatbot.
            if user_message.lower() in {"exit", "quit"}:
                print("Goodbye!")
                break

            # Ignore empty messages rather than sending them to the model.
            if not user_message:
                continue

            answer = get_response(user_message)
            print(f"Bot: {answer}")

        except KeyboardInterrupt:
            # Allow Ctrl+C to exit cleanly.
            print("\nGoodbye!")
            break

        except EOFError:
            # Handle Ctrl+D (or an equivalent end-of-input signal).
            print("\nGoodbye!")
            break


if __name__ == "__main__":
    main() 