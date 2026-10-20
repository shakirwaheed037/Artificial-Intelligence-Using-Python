# TASK 4: TEXT TOKENIZATION USING HUGGING FACE TRANSFORMERS
# . Generative AI Basics: Marks = 2
# Write a Python program to load the following provided exercise_text.json file and tokenize its text 
# using a Hugging Face Transformers tokenizer.
# You can use following json text file.


import json
from transformers import AutoTokenizer

#  1: Load JSON file
with open("exercise_text.json", "r", encoding="utf-8") as file:
    data = json.load(file)

print("JSON file loaded successfully.")
print("Data type:", type(data))
print("Data preview:")
print(str(data)[:1000])


#  2: Extract text from common JSON structures
def extract_text(obj):
    texts = []

    if isinstance(obj, str):
        texts.append(obj)

    elif isinstance(obj, dict):
        for key, value in obj.items():
            if key.lower() in ["text", "content", "sentence", "prompt"]:
                if isinstance(value, str):
                    texts.append(value)
                elif isinstance(value, (dict, list)):
                    texts.extend(extract_text(value))
            elif isinstance(value, (dict, list)):
                texts.extend(extract_text(value))

    elif isinstance(obj, list):
        for item in obj:
            texts.extend(extract_text(item))

    return texts


text_data = extract_text(data)

if not text_data:
    raise ValueError(
        "No text was extracted. Check the JSON file structure."
    )

print("\nNumber of text entries:", len(text_data))
print("\nFirst text entry:")
print(text_data[0])


#  3: Load pretrained Hugging Face tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")


# 4: Tokenize text
encoded = tokenizer(
    text_data,
    padding=True,
    truncation=True,
    max_length=128
)

#  5: Display tokens, IDs, and attention mask
print("\nTokens:")
print(tokenizer.convert_ids_to_tokens(encoded["input_ids"][0]))

print("\nToken IDs:")
print(encoded["input_ids"][0])

print("\nAttention Mask:")
print(encoded["attention_mask"][0])

print("\nTokenization completed successfully.")
