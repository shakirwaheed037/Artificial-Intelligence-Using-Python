import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load dataset
df = pd.read_csv("toxic_comments_dataset.csv")

# Clean missing values and duplicates
df = df.dropna(subset=["comment"])
df = df.drop_duplicates(subset=["comment"])

# Text cleaning
import re

def clean_text(text):
    text = text.lower()
    text = re.sub(r"<.*?>", " ", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"http\S+|www\S+|https\S+", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

df["clean_comment"] = df["comment"].apply(clean_text)

# Separate input and target
X = df["clean_comment"]
y = df["label"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

# Create tokenizer using training text only
tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train)

# Convert text into sequences
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

# Maximum sequence length
max_length = 100

# Apply padding and truncation
X_train_pad = pad_sequences(
    X_train_seq,
    maxlen=max_length,
    padding="post",
    truncating="post"
)

X_test_pad = pad_sequences(
    X_test_seq,
    maxlen=max_length,
    padding="post",
    truncating="post"
)

print("\nVocabulary Size:", len(tokenizer.word_index))
print("Maximum Sequence Length:", max_length)

print("\nTraining Sequence Shape:", X_train_pad.shape)
print("Testing Sequence Shape:", X_test_pad.shape)

print("\nExample Tokenized Sequence:")
print(X_train_seq[0])

print("\nExample Padded Sequence:")
print(X_train_pad[0])