pip install -r requirements.txt
python text_generator.py

### Task 4: Text Generation Model
# This script generates text using GPT-2 model.

from transformers import pipeline

generator = pipeline("text-generation", model="gpt2")

def generate_text(prompt, max_length=100):
    """Generates text based on the given prompt using GPT-2."""
    return generator(prompt, max_length=max_length, num_return_sequences=1)[0]["generated_text"]

if __name__ == "__main__":
    prompt = "AI is transforming the world by"
    generated_text = generate_text(prompt)
    print("Generated Text:\n", generated_text)
