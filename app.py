import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from langsmith import traceable
from flask import Flask, render_template, request

app = Flask(__name__)
# Load environment variables
load_dotenv()

# Validate API key
if "GOOGLE_API_KEY" not in os.environ:
    raise ValueError("Missing GOOGLE_API_KEY. Please add it to your .env file.")

# Initialize Gemini model
chat = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",  # or gemini-pro
    google_api_key=os.environ["GOOGLE_API_KEY"]
)

# Define translation logic
@traceable(name="Language Translator")
def translate_text(text: str, target_lang: str):
    system_instruction = f"You are an advanced language translator, proficient in all languages. Translate all input text into {target_lang}. Ensure the translation is accurate, clear, and tailored to both bilingual and non-bilingual users."
    messages = [
        SystemMessage(content=system_instruction),
        HumanMessage(content=text),
    ]
    response = chat.invoke(messages)
    return response.content

# Main CLI logic
if __name__ == "__main__":
    print("🌍 Language Translator using Gemini\n")
    text = input("Enter text to translate: ")
    target_lang = input("Translate to (e.g., French, Spanish, Italian): ")
    translated = translate_text(text, target_lang)
    print(f"\n🗣 Translated to {target_lang}: {translated}")


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/translate', methods=['POST'])
def translate():
    text = request.form['text']
    target_lang = request.form['target_lang']
    translated = translate_text(text, target_lang)
    return render_template('index.html', translated=translated)


if __name__ == '__main__':
    app.run(debug=True)
