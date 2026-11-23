# AAI Practical Exam – Copy-Paste Cheat Sheet

Each practical = one section. Each code block = one notebook cell (same order as your journal). Paste cells top to bottom.
Runs on Google Colab / Jupyter. Datasets download automatically (internet needed) except `intents.json` (2a Part B – fallback included).

**Quick fixes if a cell errors**
- `ModuleNotFoundError` → run `%pip install -q <package>` in a cell, then restart runtime.
- Keras 3: use `keras.Input(shape=...)` as first layer (not `input_shape=`).
- Run cells in order; the variables (`x_train`, `model`, `df`...) come from earlier cells.
- Slow training → cut `epochs` to 1–2 (accuracy drops slightly, code stays the same).

---

## Practical 1a – CNN and RNN (LSTM) with TensorFlow on MNIST

```python
# Cell 1 – imports
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

print("TensorFlow:", tf.__version__, "| Keras:", keras.__version__)
tf.random.set_seed(42)
np.random.seed(42)
```

```python
# Cell 2 – load + prepare data
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
print("Train:", x_train.shape, y_train.shape, "| Test:", x_test.shape, y_test.shape)

# scale 0-255 -> 0-1 and add channel dimension (28,28) -> (28,28,1)
x_train = (x_train / 255.0).astype("float32")[..., np.newaxis]
x_test = (x_test / 255.0).astype("float32")[..., np.newaxis]

fig, ax = plt.subplots(1, 8, figsize=(12, 2))
for i in range(8):
    ax[i].imshow(x_train[i].squeeze(), cmap="gray")
    ax[i].set_title(y_train[i])
    ax[i].axis("off")
plt.suptitle("Sample training images")
plt.show()
```

```python
# Cell 3 – CNN model
cnn = keras.Sequential([
    keras.Input(shape=(28, 28, 1)),
    layers.Conv2D(32, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dropout(0.3),
    layers.Dense(64, activation="relu"),
    layers.Dense(10, activation="softmax"),   # 10 digits: 0-9
], name="mnist_cnn")
cnn.summary()
```

```python
# Cell 4 – compile + train CNN
cnn.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
history = cnn.fit(x_train, y_train, epochs=5, batch_size=128, validation_split=0.1, verbose=2)
```

```python
# Cell 5 – evaluate CNN + curves
test_loss, test_acc = cnn.evaluate(x_test, y_test, verbose=0)
print(f"CNN test loss: {test_loss:.4f} | CNN test accuracy: {test_acc:.4f}")

fig, ax = plt.subplots(1, 2, figsize=(11, 3.5))
ax[0].plot(history.history["loss"], "o-", label="train")
ax[0].plot(history.history["val_loss"], "o-", label="validation")
ax[0].set_title("Loss"); ax[0].set_xlabel("Epoch"); ax[0].legend()
ax[1].plot(history.history["accuracy"], "o-", label="train")
ax[1].plot(history.history["val_accuracy"], "o-", label="validation")
ax[1].set_title("Accuracy"); ax[1].set_xlabel("Epoch"); ax[1].legend()
plt.tight_layout()
plt.show()
```

```python
# Cell 6 – predictions, report, confusion matrix
predictions = cnn.predict(x_test, verbose=0)
pred_labels = predictions.argmax(axis=1)

fig, ax = plt.subplots(1, 8, figsize=(13, 2.2))
for i in range(8):
    ax[i].imshow(x_test[i].squeeze(), cmap="gray")
    ax[i].set_title(f"P:{pred_labels[i]} / A:{y_test[i]}", fontsize=9)
    ax[i].axis("off")
plt.suptitle("CNN predictions (P = predicted, A = actual)")
plt.show()

from sklearn.metrics import confusion_matrix, classification_report
print(classification_report(y_test, pred_labels, digits=4))
cm = confusion_matrix(y_test, pred_labels)
plt.figure(figsize=(5.5, 4.5))
plt.imshow(cm, cmap="Blues"); plt.colorbar()
plt.xticks(range(10)); plt.yticks(range(10))
plt.xlabel("Predicted"); plt.ylabel("Actual")
plt.title("CNN confusion matrix")
plt.show()
```

```python
# Cell 7 – RNN (LSTM): treat each image as 28 time steps x 28 features
x_train_seq = x_train.squeeze(-1)   # (60000, 28, 28)
x_test_seq = x_test.squeeze(-1)

rnn = keras.Sequential([
    keras.Input(shape=(28, 28)),
    layers.LSTM(128, return_sequences=True),
    layers.LSTM(64),
    layers.Dense(64, activation="relu"),
    layers.Dense(10, activation="softmax"),
], name="mnist_lstm")
rnn.summary()
rnn.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
rnn_history = rnn.fit(x_train_seq, y_train, epochs=3, batch_size=128, validation_split=0.1, verbose=2)
```

```python
# Cell 8 – compare CNN vs LSTM
rnn_loss, rnn_acc = rnn.evaluate(x_test_seq, y_test, verbose=0)
print(f"LSTM test accuracy: {rnn_acc:.4f}")
print(f"CNN test accuracy: {test_acc:.4f}")
print("CNN parameters :", cnn.count_params(), "| LSTM parameters:", rnn.count_params())
```

---

## Practical 1b – NLP sentiment analysis / text classification (IMDB)

```python
# Cell 1 – imports
import os, re, string
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

print("TensorFlow:", tf.__version__)
tf.random.set_seed(42)
```

```python
# Cell 2 – toy example: tokenise, pad, embedding classifier
texts = ["I love this product", "This is the worst movie", "I am so happy",
         "I hate this book", "What a great day", "I am very disappointed"]
labels = np.array([1, 0, 1, 0, 1, 0])   # 1 = Positive, 0 = Negative

vectorizer = layers.TextVectorization(output_mode="int", output_sequence_length=5)
vectorizer.adapt(texts)
print("Vocabulary:", vectorizer.get_vocabulary())
padded = vectorizer(np.array(texts))
print("Padded sequences:\n", padded.numpy())
vocab_size = vectorizer.vocabulary_size()

toy = keras.Sequential([
    keras.Input(shape=(5,)),
    layers.Embedding(input_dim=vocab_size, output_dim=8),
    layers.Flatten(),
    layers.Dense(6, activation="relu"),
    layers.Dense(1, activation="sigmoid"),
])
toy.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
h = toy.fit(padded, labels, epochs=100, verbose=0)
print("Final training accuracy:", h.history["accuracy"][-1], "| loss:", round(h.history["loss"][-1], 4))
```

```python
# Cell 3 – test toy model on new sentences
test_texts = ["I love this product", "worst movie", "I am so happy", "I hate this book",
              "great day", "I am very disappointed", "so happy", "What a worst day"]
predictions = toy.predict(vectorizer(np.array(test_texts)), verbose=0)
for text, p in zip(test_texts, predictions):
    print(f"{text:<26} -> {'Positive' if p[0] > 0.5 else 'Negative'} ({p[0]:.2f})")
```

```python
# Cell 4 – load IMDB dataset
csv = "IMDB-Dataset.csv"
url = "https://raw.githubusercontent.com/Ankit152/IMDB-sentiment-analysis/master/IMDB-Dataset.csv"
df = pd.read_csv(csv if os.path.exists(csv) else url)
df["label"] = (df["sentiment"] == "positive").astype(int)
print(df.shape)
print(df["sentiment"].value_counts())
print("\nSample review:\n", df["review"][1][:400], "...")
```

```python
# Cell 5 – train / validation / test split + tf.data
from sklearn.model_selection import train_test_split

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["label"])
train_df, val_df = train_test_split(train_df, test_size=0.125, random_state=42, stratify=train_df["label"])
print("Train:", len(train_df), "Validation:", len(val_df), "Test:", len(test_df))

batch_size = 32

def make_ds(d, shuffle=False):
    ds = tf.data.Dataset.from_tensor_slices((d["review"].values, d["label"].values))
    return (ds.shuffle(10000, seed=42) if shuffle else ds).batch(batch_size)

raw_train_ds, raw_val_ds, raw_test_ds = make_ds(train_df, True), make_ds(val_df), make_ds(test_df)
```

```python
# Cell 6 – text vectorisation layer
def custom_standardization(input_data):
    lowercase = tf.strings.lower(input_data)
    stripped_html = tf.strings.regex_replace(lowercase, "<br />", " ")
    return tf.strings.regex_replace(stripped_html, "[%s]" % re.escape(string.punctuation), "")

max_features, sequence_length = 10000, 250
vectorize_layer = layers.TextVectorization(
    standardize=custom_standardization, max_tokens=max_features,
    output_mode="int", output_sequence_length=sequence_length)
vectorize_layer.adapt(raw_train_ds.map(lambda x, y: x))

def vectorize_text(text, label):
    return vectorize_layer(text), label

text_batch, label_batch = next(iter(raw_train_ds))
print("Review:", text_batch[0].numpy()[:150], "...")
print("Label:", label_batch[0].numpy())
print("Vectorized:", vectorize_layer(text_batch[:1]).numpy()[0][:20], "...")
print("Vocabulary size:", len(vectorize_layer.get_vocabulary()))

AUTOTUNE = tf.data.AUTOTUNE
train_ds = raw_train_ds.map(vectorize_text).cache().prefetch(AUTOTUNE)
val_ds = raw_val_ds.map(vectorize_text).cache().prefetch(AUTOTUNE)
test_ds = raw_test_ds.map(vectorize_text).cache().prefetch(AUTOTUNE)
```

```python
# Cell 7 – embedding + global-average-pooling model
embedding_dim = 16
model = keras.Sequential([
    keras.Input(shape=(sequence_length,)),
    layers.Embedding(max_features, embedding_dim),
    layers.Dropout(0.2),
    layers.GlobalAveragePooling1D(),
    layers.Dropout(0.2),
    layers.Dense(1, activation="sigmoid"),
])
model.summary()
model.compile(loss=keras.losses.BinaryCrossentropy(), optimizer="adam",
              metrics=[keras.metrics.BinaryAccuracy(threshold=0.5)])
history = model.fit(train_ds, validation_data=val_ds, epochs=10, verbose=2)
```

```python
# Cell 8 – evaluate + plots
loss, accuracy = model.evaluate(test_ds, verbose=0)
print("Test loss:", round(loss, 4), "| Test accuracy:", round(accuracy, 4))

hd = history.history
ep = range(1, len(hd["loss"]) + 1)
fig, ax = plt.subplots(1, 2, figsize=(11, 3.5))
ax[0].plot(ep, hd["loss"], "bo", label="Training loss")
ax[0].plot(ep, hd["val_loss"], "b", label="Validation loss")
ax[0].set_title("Training and validation loss"); ax[0].set_xlabel("Epochs"); ax[0].legend()
ax[1].plot(ep, hd["binary_accuracy"], "bo", label="Training acc")
ax[1].plot(ep, hd["val_binary_accuracy"], "b", label="Validation acc")
ax[1].set_title("Training and validation accuracy"); ax[1].set_xlabel("Epochs"); ax[1].legend(loc="lower right")
plt.tight_layout()
plt.show()
```

```python
# Cell 9 – end-to-end model that accepts raw text + predictions
export_model = keras.Sequential([keras.Input(shape=(1,), dtype="string"), vectorize_layer, model])
export_model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
raw_test_2d = raw_test_ds.map(lambda x, y: (tf.expand_dims(x, -1), y))
print(export_model.evaluate(raw_test_2d, return_dict=True, verbose=0))

examples = tf.constant([["The movie was great!"], ["The movie was okay."], ["The movie was terrible..."],
                        ["The movie was worse..."], ["The movie was good..."],
                        ["An absolute masterpiece with brilliant acting"],
                        ["Boring, predictable and far too long"]])
scores = export_model.predict(examples, verbose=0).ravel()
for t, s in zip(examples.numpy().ravel(), scores):
    print(f"{t.decode():<48} {s:.3f} {'Positive' if s >= 0.5 else 'Negative'}")
```

```python
# Cell 10 – Bidirectional LSTM model
lstm_model = keras.Sequential([
    keras.Input(shape=(sequence_length,)),
    layers.Embedding(max_features, 32, mask_zero=True),
    layers.Bidirectional(layers.LSTM(32)),
    layers.Dense(32, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(1, activation="sigmoid"),
])
lstm_model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
lstm_model.summary()
lstm_hist = lstm_model.fit(train_ds, validation_data=val_ds, epochs=2, verbose=2)
print("BiLSTM test accuracy:", round(lstm_model.evaluate(test_ds, verbose=0)[1], 4))
```

---

## Practical 2a – Chatbot with transformer models

### Part A – BERT question-answering chatbot

```python
# Cell A1 – install + load pretrained BERT QA model
%pip install -q transformers torch
import transformers
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline

print("Transformers version:", transformers.__version__)
model_name = "deepset/bert-base-cased-squad2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
qa_model = AutoModelForQuestionAnswering.from_pretrained(model_name)
nlp = pipeline("question-answering", model=qa_model, tokenizer=tokenizer)
```

```python
# Cell A2 – context + questions + chatbot function
context = """
The Bengal tiger was chosen as the national animal in a meeting of the Indian wildlife board in 1972 and was adopted
officially in April 1973. It was chosen over the Asiatic lion due to the wider presence of the tiger across India.
Indian peacock was designated as the national bird of India in February 1963.
Indian elephant is the largest terrestrial mammal in India and a cultural symbol throughout its range, appearing in
various religious traditions and mythologies.

Indian banyan is a large tree native to the Indian subcontinent and produces aerial roots from the
branches which grow downwards, eventually becoming trunks.
Mango is a large fruit tree with many varieties, believed to have originated in northeast India.
Lotus is an aquatic plant adapted to grow in the flood plains. Lotus seeds can remain dormant and viable
for many years, therefore the plant is regarded as a symbol of longevity.
"""
questions = ["What is the National animal of India?", "What is the National bird of India?",
             "What is the largest terrestrial mammal in India?", "Which tree produces aerial roots?",
             "Which fruit originated in northeast India?", "What is the symbol of longevity?"]
print("Encoded first question:", tokenizer.encode(questions[0]))

def chatbot(question, context):
    output = nlp(question=question, context=context)
    return output["answer"], output["score"]

for q in questions:
    ans, score = chatbot(q, context)
    print(f"Q: {q}\nA: {ans}   (confidence {score:.2f})\n")
```

### Part B – Transformer-encoder intent chatbot (needs `intents.json`)

```python
# Cell B0 (ONLY if you do not have intents.json) – creates a small one so the rest runs
import json, os
if not os.path.exists("intents.json"):
    data = {"intents": [
        {"tag": "greeting", "patterns": ["hi", "hello", "hey there", "good morning", "is anyone there"],
         "responses": ["Hello! I am CampusBot. How can I help you today?"]},
        {"tag": "goodbye", "patterns": ["bye", "see you later", "goodbye", "talk to you soon", "ok bye"],
         "responses": ["Goodbye! Have a great day."]},
        {"tag": "thanks", "patterns": ["thanks", "thank you", "thanks a lot", "that was helpful", "thanks for helping me"],
         "responses": ["You're welcome!"]},
        {"tag": "courses", "patterns": ["what courses do you offer", "list the courses", "which programs are available",
                                         "do you have msc it", "is data science offered here"],
         "responses": ["We offer BSc IT, BSc CS, MSc IT and MSc Data Science."]},
        {"tag": "fees", "patterns": ["what is the fee", "how much are the fees", "course fees", "fee structure",
                                      "how much do I have to pay for the course"],
         "responses": ["The MSc IT fee is Rs 45,000 per year, payable online in two installments."]},
        {"tag": "admission", "patterns": ["how to apply", "admission process", "how do I get admission",
                                           "last date for admission", "what is the procedure to join the college"],
         "responses": ["Apply online on the college portal and upload your marksheets."]},
        {"tag": "timings", "patterns": ["college timings", "what time does college start", "when does college close",
                                         "working hours", "at what time does the college start"],
         "responses": ["College runs from 8:00 AM to 2:30 PM, Monday to Saturday."]},
        {"tag": "library", "patterns": ["library timings", "how many books can I borrow", "library rules",
                                         "is there a library", "how long can I keep a library book"],
         "responses": ["The library is open 9 AM to 5 PM; you can borrow 3 books for 14 days."]},
        {"tag": "exam", "patterns": ["when are the exams", "exam dates", "hall ticket", "result date",
                                      "when is the semester exam"],
         "responses": ["Semester exams are held in November and April. Hall tickets are on the portal."]},
        {"tag": "placement", "patterns": ["placement details", "do companies visit", "campus recruitment",
                                           "job opportunities", "do companies come for campus recruitment"],
         "responses": ["The placement cell arranges campus drives every year."]},
        {"tag": "contact", "patterns": ["contact number", "phone number", "email address", "how can I contact you",
                                         "give me your phone number"],
         "responses": ["Call 022-12345678 or mail info@college.edu."]},
        {"tag": "out_of_scope", "patterns": ["what is the weather today", "who won the match", "tell me a joke",
                                              "what is the capital of France", "what is the weather in Paris"],
         "responses": ["Sorry, I can only answer questions about the college."]},
    ]}
    with open("intents.json", "w") as f:
        json.dump(data, f)
    print("intents.json created")
```

```python
# Cell B1 – load intents
import os, json, random
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

tf.random.set_seed(7); np.random.seed(7); random.seed(7)

path = "intents.json"
if not os.path.exists(path):
    raise FileNotFoundError("Upload intents.json to the notebook folder (or run cell B0)")
with open(path) as f:
    intents = json.load(f)["intents"]

sentences, tags = [], []
for it in intents:
    for p in it["patterns"]:
        sentences.append(p)
        tags.append(it["tag"])
classes = sorted(set(tags))
tag2id = {t: i for i, t in enumerate(classes)}
y = np.array([tag2id[t] for t in tags])
print("Intents:", classes)
print("Training sentences:", len(sentences))
```

```python
# Cell B2 – vectorise sentences
MAX_LEN = 12
vectorizer = layers.TextVectorization(max_tokens=1000, output_sequence_length=MAX_LEN)
vectorizer.adapt(sentences)
X = vectorizer(np.array(sentences)).numpy()
VOCAB = vectorizer.vocabulary_size()
print("Vocabulary size:", VOCAB)
print(sentences[0], "->", X[0])
```

```python
# Cell B3 – transformer encoder model
class TokenAndPositionEmbedding(layers.Layer):
    """Token embedding + learned positional embedding."""
    def __init__(self, maxlen, vocab_size, embed_dim, **kw):
        super().__init__(**kw)
        self.token_emb = layers.Embedding(vocab_size, embed_dim)
        self.pos_emb = layers.Embedding(maxlen, embed_dim)

    def call(self, x):
        positions = tf.range(start=0, limit=tf.shape(x)[-1], delta=1)
        return self.token_emb(x) + self.pos_emb(positions)


class TransformerBlock(layers.Layer):
    """Multi-head self-attention + feed-forward network, each with residual connection and layer norm."""
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, **kw):
        super().__init__(**kw)
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential([layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim)])
        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)
        self.drop1, self.drop2 = layers.Dropout(rate), layers.Dropout(rate)

    def call(self, x, training=False):
        attn = self.att(x, x)
        x = self.norm1(x + self.drop1(attn, training=training))
        return self.norm2(x + self.drop2(self.ffn(x), training=training))


EMBED, HEADS, FF = 32, 2, 64
inputs = keras.Input(shape=(MAX_LEN,), dtype="int64")
x = TokenAndPositionEmbedding(MAX_LEN, VOCAB, EMBED)(inputs)
x = TransformerBlock(EMBED, HEADS, FF)(x)
x = layers.GlobalAveragePooling1D()(x)
x = layers.Dropout(0.2)(x)
x = layers.Dense(32, activation="relu")(x)
outputs = layers.Dense(len(classes), activation="softmax")(x)
model = keras.Model(inputs, outputs, name="transformer_intent_classifier")
model.summary()
```

```python
# Cell B4 – train
model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
hist = model.fit(X, y, epochs=150, batch_size=16, verbose=0)
print("Final training accuracy:", round(hist.history["accuracy"][-1], 4), "| loss:", round(hist.history["loss"][-1], 4))
```

```python
# Cell B5 – test on unseen questions
test = [("hello there, anybody around?", "greeting"), ("how much do I have to pay for the course", "fees"),
        ("what is the procedure to join the college", "admission"), ("is data science offered here", "courses"),
        ("at what time does the college start", "timings"), ("how long can I keep a library book", "library"),
        ("when is the semester exam", "exam"), ("do companies come for campus recruitment", "placement"),
        ("give me your phone number", "contact"), ("thanks for helping me", "thanks"), ("ok bye", "goodbye"),
        ("where can I download the hall ticket", "exam")]
probs = model.predict(vectorizer(np.array([t for t, _ in test])), verbose=0)
correct = 0
for (q, true), p in zip(test, probs):
    pred = classes[p.argmax()]
    correct += pred == true
    print(f"{q:<45} -> {pred:<10} ({p.max():.2f}) {'OK' if pred == true else 'expected ' + true}")
print(f"\nAccuracy on unseen questions: {correct}/{len(test)}")
```

```python
# Cell B6 – chatbot function + demo conversation
responses = {it["tag"]: it["responses"] for it in intents}

def chat(message, threshold=0.45):
    p = model.predict(vectorizer(np.array([message])), verbose=0)[0]
    if p.max() < threshold:
        return "Sorry, I did not understand. Please ask about admissions, fees, courses, exams, library or placements."
    return random.choice(responses[classes[p.argmax()]])

conversation = ["Hi", "What courses do you offer?", "how much are the fees for MSc IT?",
                "when are the exams held?", "what is the weather in Paris?", "thank you", "bye"]
for m in conversation:
    print("You    :", m)
    print("Chatbot:", chat(m), "\n")
```

```python
# Cell B7 – interactive chat (type quit to stop)
while True:
    msg = input("You: ")
    if msg.lower() in ("quit", "exit"):
        break
    print("Chatbot:", chat(msg))
```

---

## Practical 2b – Recommendation system (embeddings / matrix factorisation, MovieLens)

```python
# Cell 1 – imports
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

tf.random.set_seed(42); np.random.seed(42)
```

```python
# Cell 2 – Part A: toy example (user x item dot product)
ratings = np.array([[0, 0, 5.0], [0, 1, 3.0], [1, 0, 4.0], [1, 2, 2.0], [2, 1, 4.0], [2, 2, 5.0]])
num_users = int(ratings[:, 0].max()) + 1
num_items = int(ratings[:, 1].max()) + 1
embedding_size = 2

user_input = keras.Input(shape=(1,), name="user")
item_input = keras.Input(shape=(1,), name="item")
user_embedding = layers.Embedding(num_users, embedding_size)(user_input)
item_embedding = layers.Embedding(num_items, embedding_size)(item_input)
dot_product = layers.Flatten()(layers.Dot(axes=2)([user_embedding, item_embedding]))
toy = keras.Model([user_input, item_input], dot_product)

# lr 0.05 and 500 epochs: with default 0.001/100 epochs tiny embeddings barely move
toy.compile(optimizer=keras.optimizers.Adam(0.05), loss="mse")
X_users, X_items = ratings[:, 0].astype("int32"), ratings[:, 1].astype("int32")
y_ratings = ratings[:, 2].astype("float32")
h = toy.fit([X_users, X_items], y_ratings, epochs=500, verbose=0)
print("Final training MSE:", round(h.history["loss"][-1], 4))

pred = toy.predict([X_users, X_items], verbose=0).ravel()
print("\nKnown ratings vs fitted:")
for u, i, r, p in zip(X_users, X_items, y_ratings, pred):
    print(f" user {u} - item {i}: actual {r:.0f} predicted {p:.2f}")

print("\nPredictions for unrated pairs:")
for u, i in [(0, 2), (1, 1), (2, 0)]:
    print(f" user {u} - item {i}: predicted {toy.predict([np.array([u]), np.array([i])], verbose=0).item():.2f}")
```

```python
# Cell 3 – Part B: load MovieLens 100k + index mapping
path = "u.data"
url = "https://files.grouplens.org/datasets/movielens/ml-100k/u.data"
df = pd.read_csv(path if os.path.exists(path) else url, sep="\t",
                 names=["user_id", "item_id", "rating", "timestamp"])
print(df.head()); print("Shape:", df.shape)
print("Rating distribution:\n", df["rating"].value_counts().sort_index())

# map ids to zero-based indices
user_ids, item_ids = df["user_id"].unique(), df["item_id"].unique()
user2idx = {u: i for i, u in enumerate(user_ids)}
item2idx = {m: i for i, m in enumerate(item_ids)}
idx2item = {i: m for m, i in item2idx.items()}
df["user"] = df["user_id"].map(user2idx)
df["item"] = df["item_id"].map(item2idx)
num_users, num_items = len(user2idx), len(item2idx)
print("Users:", num_users, "| Items:", num_items,
      f"| Sparsity: {1 - len(df) / (num_users * num_items):.4f}")
```

```python
# Cell 4 – train / validation / test split
train, test = train_test_split(df, test_size=0.2, random_state=42)
train_df, val_df = train_test_split(train, test_size=0.1, random_state=42)
X_train, y_train = [train_df["user"].values, train_df["item"].values], train_df["rating"].values.astype("float32")
X_val, y_val = [val_df["user"].values, val_df["item"].values], val_df["rating"].values.astype("float32")
X_test, y_test = [test["user"].values, test["item"].values], test["rating"].values.astype("float32")
print("Train:", len(train_df), "Validation:", len(val_df), "Test:", len(test))
```

```python
# Cell 5 – embedding model with biases + training
def build_model(embedding_size=32, l2=1e-4, with_bias=True):
    u_in = keras.Input(shape=(1,), name="user")
    i_in = keras.Input(shape=(1,), name="item")
    reg = keras.regularizers.l2(l2)
    u_vec = layers.Flatten()(layers.Embedding(num_users, embedding_size,
                                              embeddings_initializer="he_normal",
                                              embeddings_regularizer=reg)(u_in))
    i_vec = layers.Flatten()(layers.Embedding(num_items, embedding_size,
                                              embeddings_initializer="he_normal",
                                              embeddings_regularizer=reg)(i_in))
    out = layers.Dot(axes=1)([u_vec, i_vec])
    if with_bias:   # rating = global mean + user bias + item bias + u.v
        u_b = layers.Flatten()(layers.Embedding(num_users, 1)(u_in))
        i_b = layers.Flatten()(layers.Embedding(num_items, 1)(i_in))
        out = layers.Add()([out, u_b, i_b])
        out = layers.Lambda(lambda t: t + float(y_train.mean()))(out)
    return keras.Model([u_in, i_in], out)

model = build_model()
model.summary()
model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse",
              metrics=[keras.metrics.RootMeanSquaredError(name="rmse")])
early = keras.callbacks.EarlyStopping(monitor="val_rmse", patience=3, restore_best_weights=True)
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=30,
                    batch_size=1024, callbacks=[early], verbose=2)
```

```python
# Cell 6 – learning curve + test RMSE vs baseline
plt.figure(figsize=(8, 4.5))
plt.plot(history.history["rmse"], label="Training RMSE")
plt.plot(history.history["val_rmse"], label="Validation RMSE")
plt.xlabel("Epoch"); plt.ylabel("RMSE"); plt.title("Training vs Validation RMSE"); plt.legend()
plt.show()

pred_test = model.predict(X_test, batch_size=4096, verbose=0).ravel().clip(1, 5)
rmse = np.sqrt(mean_squared_error(y_test, pred_test))
baseline = np.sqrt(mean_squared_error(y_test, np.full_like(y_test, y_train.mean())))
print(f"Test RMSE (embedding model): {rmse:.4f}")
print(f"Test RMSE (predict global mean {y_train.mean():.2f}): {baseline:.4f}")
print(f"Test MAE: {np.abs(y_test - pred_test).mean():.4f}")
print("\nSample predictions:")
print(pd.DataFrame({"user_id": test["user_id"].values[:8], "item_id": test["item_id"].values[:8],
                    "actual": y_test[:8], "predicted": pred_test[:8].round(2)}).to_string(index=False))
```

```python
# Cell 7 – top-N recommendations for a user
def recommend(user_id, n=10):
    u = user2idx[user_id]
    seen = set(df.loc[df["user"] == u, "item"])
    candidates = np.array([i for i in range(num_items) if i not in seen])
    scores = model.predict([np.full(len(candidates), u), candidates], batch_size=4096, verbose=0).ravel()
    top = candidates[np.argsort(-scores)[:n]]
    return pd.DataFrame({"item_id": [idx2item[i] for i in top],
                         "predicted_rating": np.sort(scores)[::-1][:n].round(2)})

print("User 196 rated", (df.user_id == 196).sum(), "movies. Top-10 recommendations:")
print(recommend(196).to_string(index=False))
```

```python
# Cell 8 – similar items from learned embeddings
emb_layers = [l for l in model.layers if isinstance(l, layers.Embedding)]
item_vectors = emb_layers[1].get_weights()[0]   # (num_items, embedding_size)
norm = item_vectors / np.linalg.norm(item_vectors, axis=1, keepdims=True)
target = item2idx[50]                           # item 50 = Star Wars (1977)
sims = norm @ norm[target]
nearest = np.argsort(-sims)[1:6]
print("Items most similar to item_id 50:", [(int(idx2item[i]), round(float(sims[i]), 3)) for i in nearest])
```

---

## Practical 3 – Computer vision: object detection + image segmentation

```python
# Cell 1 – imports + synthetic shapes generator
import numpy as np
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

print("TensorFlow:", tf.__version__, "| OpenCV:", cv2.__version__)
rng = np.random.default_rng(0)
tf.random.set_seed(0)
CLASSES = ["circle", "square", "triangle"]
IMG = 64

def draw_shape(img, mask, cls, cx, cy, r, colour):
    if cls == 0:
        cv2.circle(img, (cx, cy), r, colour, -1)
        cv2.circle(mask, (cx, cy), r, 1, -1)
    elif cls == 1:
        cv2.rectangle(img, (cx - r, cy - r), (cx + r, cy + r), colour, -1)
        cv2.rectangle(mask, (cx - r, cy - r), (cx + r, cy + r), 2, -1)
    else:
        pts = np.array([[cx, cy - r], [cx - r, cy + r], [cx + r, cy + r]], np.int32)
        cv2.fillPoly(img, [pts], colour)
        cv2.fillPoly(mask, [pts], 3)

def background():
    base = rng.integers(0, 80, 3)
    return np.clip(base + rng.normal(0, 12, (IMG, IMG, 3)), 0, 255).astype(np.uint8)

def colour():
    return tuple(int(c) for c in rng.integers(120, 256, 3))
```

```python
# Cell 2 – detection dataset (class + bounding box) and box plotting helper
def make_detection_data(n):
    X = np.zeros((n, IMG, IMG, 3), np.float32)
    cls = np.zeros(n, np.int32)
    box = np.zeros((n, 4), np.float32)
    for i in range(n):
        img = background().copy()
        m = np.zeros((IMG, IMG), np.uint8)
        c = int(rng.integers(0, 3))
        r = int(rng.integers(6, 16))
        cx, cy = int(rng.integers(r, IMG - r)), int(rng.integers(r, IMG - r))
        draw_shape(img, m, c, cx, cy, r, colour())
        X[i] = img / 255.0
        cls[i] = c
        box[i] = [(cx - r) / IMG, (cy - r) / IMG, (cx + r) / IMG, (cy + r) / IMG]   # x1, y1, x2, y2
    return X, cls, box

Xd_train, cd_train, bd_train = make_detection_data(4000)
Xd_test, cd_test, bd_test = make_detection_data(500)
print("Detection train:", Xd_train.shape, "| test:", Xd_test.shape)

def show_boxes(X, cls, boxes, pred_cls=None, pred_boxes=None, n=6, title=""):
    fig, ax = plt.subplots(1, n, figsize=(2.2 * n, 2.6))
    for i in range(n):
        ax[i].imshow(X[i])
        ax[i].axis("off")
        x1, y1, x2, y2 = boxes[i] * IMG
        ax[i].add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, ec="lime", lw=2))
        t = CLASSES[cls[i]]
        if pred_boxes is not None:
            px1, py1, px2, py2 = pred_boxes[i] * IMG
            ax[i].add_patch(plt.Rectangle((px1, py1), px2 - px1, py2 - py1, fill=False, ec="red", lw=2, ls="--"))
            t = f"{CLASSES[pred_cls[i]]} (true {CLASSES[cls[i]]})"
        ax[i].set_title(t, fontsize=8)
    plt.suptitle(title)
    plt.show()

show_boxes(Xd_train, cd_train, bd_train, title="Training samples with ground-truth boxes")
```

```python
# Cell 3 – detector model (2 outputs: class + box) and training
inp = keras.Input(shape=(IMG, IMG, 3))
x = inp
for f in [32, 64, 128]:
    x = layers.Conv2D(f, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)
x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(128, activation="relu")(x)
class_out = layers.Dense(3, activation="softmax", name="class")(x)   # what?
box_out = layers.Dense(4, activation="sigmoid", name="box")(x)       # where?
detector = keras.Model(inp, [class_out, box_out], name="shape_detector")
detector.compile(optimizer="adam",
                 loss={"class": "sparse_categorical_crossentropy", "box": "mse"},
                 loss_weights={"class": 1.0, "box": 10.0},
                 metrics={"class": "accuracy"})
detector.summary()
hist = detector.fit(Xd_train, {"class": cd_train, "box": bd_train}, validation_split=0.1,
                    epochs=12, batch_size=64, verbose=2)
```

```python
# Cell 4 – IoU metric + evaluation
def iou(a, b):
    x1, y1 = np.maximum(a[:, 0], b[:, 0]), np.maximum(a[:, 1], b[:, 1])
    x2, y2 = np.minimum(a[:, 2], b[:, 2]), np.minimum(a[:, 3], b[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = lambda t: (t[:, 2] - t[:, 0]) * (t[:, 3] - t[:, 1])
    return inter / (area(a) + area(b) - inter)

pc, pb = detector.predict(Xd_test, verbose=0)
pc = pc.argmax(1)
ious = iou(bd_test, pb)
print(f"Classification accuracy : {(pc == cd_test).mean():.4f}")
print(f"Mean IoU of boxes       : {ious.mean():.4f}")
print(f"Detections with IoU>=0.5: {(ious >= 0.5).mean():.4f}")
print(f"Correct class AND IoU>=0.5 (detection accuracy @0.5): {((pc == cd_test) & (ious >= 0.5)).mean():.4f}")
show_boxes(Xd_test, cd_test, bd_test, pc, pb, title="Test predictions (green = truth, red dashed = predicted)")
```

```python
# Cell 5 – segmentation dataset (per-pixel masks)
def make_seg_data(n):
    X = np.zeros((n, IMG, IMG, 3), np.float32)
    Y = np.zeros((n, IMG, IMG), np.uint8)
    for i in range(n):
        img = background().copy()
        m = np.zeros((IMG, IMG), np.uint8)
        for _ in range(int(rng.integers(1, 4))):
            r = int(rng.integers(5, 13))
            draw_shape(img, m, int(rng.integers(0, 3)), int(rng.integers(r, IMG - r)),
                       int(rng.integers(r, IMG - r)), r, colour())
        X[i] = img / 255.0
        Y[i] = m
    return X, Y

Xs_train, Ys_train = make_seg_data(3000)
Xs_test, Ys_test = make_seg_data(300)
print("Segmentation train:", Xs_train.shape, Ys_train.shape,
      "| classes: 0=background, 1=circle, 2=square, 3=triangle")
```

```python
# Cell 6 – U-Net + training
def conv_block(x, f):
    x = layers.Conv2D(f, 3, padding="same", activation="relu")(x)
    return layers.Conv2D(f, 3, padding="same", activation="relu")(x)

def build_unet(n_classes=4):
    inp = keras.Input(shape=(IMG, IMG, 3))
    # encoder (contracting path)
    c1 = conv_block(inp, 16); p1 = layers.MaxPooling2D()(c1)
    c2 = conv_block(p1, 32);  p2 = layers.MaxPooling2D()(c2)
    c3 = conv_block(p2, 64);  p3 = layers.MaxPooling2D()(c3)
    b = conv_block(p3, 128)   # bottleneck
    # decoder (expanding path) with skip connections
    u3 = layers.Conv2DTranspose(64, 2, strides=2, padding="same")(b)
    c4 = conv_block(layers.Concatenate()([u3, c3]), 64)
    u2 = layers.Conv2DTranspose(32, 2, strides=2, padding="same")(c4)
    c5 = conv_block(layers.Concatenate()([u2, c2]), 32)
    u1 = layers.Conv2DTranspose(16, 2, strides=2, padding="same")(c5)
    c6 = conv_block(layers.Concatenate()([u1, c1]), 16)
    out = layers.Conv2D(n_classes, 1, activation="softmax")(c6)   # per-pixel class probabilities
    return keras.Model(inp, out, name="unet")

unet = build_unet()
unet.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
print("U-Net parameters:", unet.count_params())
seg_hist = unet.fit(Xs_train, Ys_train, validation_split=0.1, epochs=10, batch_size=32, verbose=2)
```

```python
# Cell 7 – segmentation metrics + visualisation
P = unet.predict(Xs_test, verbose=0).argmax(-1)
print(f"Pixel accuracy: {(P == Ys_test).mean():.4f}")
ious = []
for k in range(4):
    inter = ((P == k) & (Ys_test == k)).sum()
    union = ((P == k) | (Ys_test == k)).sum()
    ious.append(inter / union)
    print(f"IoU {(['background'] + CLASSES)[k]:<10}: {inter / union:.4f}")
print(f"Mean IoU: {np.mean(ious):.4f}")

cmap = plt.matplotlib.colors.ListedColormap(["black", "tab:red", "tab:green", "tab:blue"])
fig, ax = plt.subplots(3, 5, figsize=(11, 7))
for j in range(5):
    ax[0, j].imshow(Xs_test[j])
    ax[1, j].imshow(Ys_test[j], cmap=cmap, vmin=0, vmax=3)
    ax[2, j].imshow(P[j], cmap=cmap, vmin=0, vmax=3)
    for i in range(3):
        ax[i, j].axis("off")
ax[0, 0].set_title("Image", loc="left")
ax[1, 0].set_title("True mask", loc="left")
ax[2, 0].set_title("U-Net prediction", loc="left")
plt.suptitle("Segmentation (red = circle, green = square, blue = triangle)")
plt.tight_layout()
plt.show()
```

```python
# Cell 8 (optional) – pretrained YOLO detector on a real photo
%pip install -q ultralytics
from ultralytics import YOLO

yolo = YOLO("yolo11n.pt")
results = yolo("https://ultralytics.com/images/bus.jpg")
r = results[0]
for c, conf, b in zip(r.boxes.cls, r.boxes.conf, r.boxes.xyxy):
    print(f"{r.names[int(c)]:<10} confidence {float(conf):.2f} box {[round(float(v)) for v in b]}")
plt.figure(figsize=(7, 9))
plt.imshow(r.plot()[..., ::-1])
plt.axis("off")
plt.title("YOLO11n detections")
plt.show()
```

---

## Practical 4 – GAN (DCGAN) generating Fashion-MNIST images

```python
# Cell 1 – imports
import time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

print("TensorFlow:", tf.__version__, "| GPU:", tf.config.list_physical_devices("GPU"))
tf.random.set_seed(1)
```

```python
# Cell 2 – data (scaled to [-1, 1] to match generator tanh output)
(train_images, train_labels), _ = keras.datasets.fashion_mnist.load_data()
train_images = train_images.reshape(-1, 28, 28, 1).astype("float32")
train_images = (train_images - 127.5) / 127.5

BUFFER_SIZE, BATCH_SIZE, NOISE_DIM = 60000, 128, 100
dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)
print("Training images:", train_images.shape)

fig, ax = plt.subplots(1, 10, figsize=(13, 1.6))
for i in range(10):
    ax[i].imshow(train_images[i, :, :, 0], cmap="gray")
    ax[i].axis("off")
plt.suptitle("Real Fashion-MNIST images")
plt.show()
```

```python
# Cell 3 – generator + discriminator
def make_generator():
    return keras.Sequential([
        keras.Input(shape=(NOISE_DIM,)),
        layers.Dense(7 * 7 * 256, use_bias=False), layers.BatchNormalization(), layers.LeakyReLU(0.2),
        layers.Reshape((7, 7, 256)),
        layers.Conv2DTranspose(128, 5, strides=1, padding="same", use_bias=False),
        layers.BatchNormalization(), layers.LeakyReLU(0.2),
        layers.Conv2DTranspose(64, 5, strides=2, padding="same", use_bias=False),
        layers.BatchNormalization(), layers.LeakyReLU(0.2),
        layers.Conv2DTranspose(1, 5, strides=2, padding="same", activation="tanh"),
    ], name="generator")

def make_discriminator():
    return keras.Sequential([
        keras.Input(shape=(28, 28, 1)),
        layers.Conv2D(64, 5, strides=2, padding="same"), layers.LeakyReLU(0.2), layers.Dropout(0.3),
        layers.Conv2D(128, 5, strides=2, padding="same"), layers.LeakyReLU(0.2), layers.Dropout(0.3),
        layers.Flatten(),
        layers.Dense(1),   # logit: real (>0) or fake (<0)
    ], name="discriminator")

generator, discriminator = make_generator(), make_discriminator()
generator.summary()
discriminator.summary()
```

```python
# Cell 4 – losses, optimisers, training step, sample viewer
cross_entropy = keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output) * 0.9, real_output)   # one-sided label smoothing
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    return real_loss + fake_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)   # fool the discriminator

generator_optimizer = keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_optimizer = keras.optimizers.Adam(2e-4, beta_1=0.5)

@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, NOISE_DIM])
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)
        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)
        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)
    generator_optimizer.apply_gradients(
        zip(gen_tape.gradient(gen_loss, generator.trainable_variables), generator.trainable_variables))
    discriminator_optimizer.apply_gradients(
        zip(disc_tape.gradient(disc_loss, discriminator.trainable_variables), discriminator.trainable_variables))
    return gen_loss, disc_loss

seed = tf.random.normal([25, NOISE_DIM])   # fixed noise to watch progress

def show_samples(epoch):
    pred = generator(seed, training=False)
    fig = plt.figure(figsize=(5, 5))
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        plt.imshow(pred[i, :, :, 0] * 127.5 + 127.5, cmap="gray")
        plt.axis("off")
    plt.suptitle(f"Generated images after epoch {epoch}")
    plt.show()
```

```python
# Cell 5 – training loop
EPOCHS = 15
g_hist, d_hist = [], []
for epoch in range(1, EPOCHS + 1):
    start = time.time()
    gl, dl = [], []
    for image_batch in dataset:
        g, d = train_step(image_batch)
        gl.append(float(g)); dl.append(float(d))
    g_hist.append(np.mean(gl)); d_hist.append(np.mean(dl))
    print(f"Epoch {epoch:2d}/{EPOCHS} generator loss {g_hist[-1]:.4f} "
          f"discriminator loss {d_hist[-1]:.4f} ({time.time() - start:.0f}s)")
    if epoch in (1, 5, EPOCHS):
        show_samples(epoch)
```

```python
# Cell 6 – loss curves
plt.figure(figsize=(7, 3.5))
plt.plot(range(1, EPOCHS + 1), g_hist, "o-", label="Generator")
plt.plot(range(1, EPOCHS + 1), d_hist, "o-", label="Discriminator")
plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("GAN training losses"); plt.legend()
plt.show()
```

```python
# Cell 7 – generate new images, check with discriminator, save generator
new = generator(tf.random.normal([8, NOISE_DIM]), training=False)
d_fake = tf.sigmoid(discriminator(new, training=False)).numpy().ravel()
d_real = tf.sigmoid(discriminator(train_images[:8], training=False)).numpy().ravel()
print("Discriminator P(real) for 8 real images     :", d_real.round(2))
print("Discriminator P(real) for 8 generated images:", d_fake.round(2))

fig, ax = plt.subplots(1, 8, figsize=(12, 1.8))
for i in range(8):
    ax[i].imshow(new[i, :, :, 0], cmap="gray")
    ax[i].set_title(f"{d_fake[i]:.2f}", fontsize=8)
    ax[i].axis("off")
plt.suptitle("Brand-new generated images (title = discriminator's P(real))")
plt.show()

generator.save("fashion_dcgan_generator.keras")
print("Generator saved to fashion_dcgan_generator.keras")
```

---

## Practical 5 – Reinforcement learning (Q-learning on Taxi, DQN on CartPole)

```python
# Cell 1 – imports
%pip install -q gymnasium
import random, time
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim

print("Gymnasium:", gym.__version__, "| PyTorch:", torch.__version__)
SEED = 42
random.seed(SEED); np.random.seed(SEED)
_ = torch.manual_seed(SEED)
torch.set_num_threads(1)   # small networks train faster on one CPU thread
```

```python
# Cell 2 – Part A: tabular Q-learning on Taxi
# newest registered Taxi version (Taxi-v3 on Colab, Taxi-v4 on newer Gymnasium)
TAXI_ID = max((k for k in gym.registry if k.startswith("Taxi-v")), key=lambda k: int(k.split("-v")[-1]))
print("Environment:", TAXI_ID)
env = gym.make(TAXI_ID)
n_states, n_actions = env.observation_space.n, env.action_space.n
print("States:", n_states, "| Actions:", n_actions, "(0 south, 1 north, 2 east, 3 west, 4 pickup, 5 dropoff)")

Q = np.zeros((n_states, n_actions))
alpha, gamma = 0.1, 0.99                      # learning rate, discount factor
epsilon, eps_min, eps_decay = 1.0, 0.01, 0.9995
episodes = 6000
rewards = []
for ep in range(episodes):
    state, _ = env.reset(seed=SEED + ep)
    total, done = 0, False
    while not done:
        # epsilon-greedy action selection (exploration vs exploitation)
        action = env.action_space.sample() if random.random() < epsilon else int(np.argmax(Q[state]))
        next_state, reward, terminated, truncated, _ = env.step(action)
        # Bellman update: Q(s,a) <- Q(s,a) + alpha [r + gamma max_a' Q(s',a') - Q(s,a)]
        target = reward + gamma * np.max(Q[next_state]) * (not terminated)
        Q[state, action] += alpha * (target - Q[state, action])
        state, total, done = next_state, total + reward, terminated or truncated
    epsilon = max(eps_min, epsilon * eps_decay)
    rewards.append(total)
    if (ep + 1) % 1000 == 0:
        print(f"Episode {ep + 1:5d} average reward (last 1000) {np.mean(rewards[-1000:]):7.2f} epsilon {epsilon:.3f}")
```

```python
# Cell 3 – Taxi: learning curve, greedy evaluation, text demo
plt.figure(figsize=(8, 3.5))
plt.plot(np.convolve(rewards, np.ones(100) / 100, mode="valid"))
plt.xlabel("Episode"); plt.ylabel("Reward (100-episode moving average)")
plt.title("Q-learning on " + TAXI_ID)
plt.show()

# evaluate the greedy policy
test_rewards, steps = [], []
for ep in range(100):
    s, _ = env.reset(seed=10000 + ep)
    done, tot, n = False, 0, 0
    while not done:
        s, r, term, trunc, _ = env.step(int(np.argmax(Q[s])))
        tot += r; n += 1; done = term or trunc
    test_rewards.append(tot); steps.append(n)
print(f"Greedy policy over 100 test episodes: mean reward {np.mean(test_rewards):.2f}, "
      f"mean steps {np.mean(steps):.1f}, success rate {np.mean(np.array(test_rewards) > 0):.0%}")

# show one episode as text
env_vis = gym.make(TAXI_ID, render_mode="ansi")
s, _ = env_vis.reset(seed=7)
print(env_vis.render())
done, n = False, 0
while not done:
    s, r, term, trunc, _ = env_vis.step(int(np.argmax(Q[s])))
    n += 1; done = term or trunc
print(env_vis.render())
print("Passenger delivered in", n, "steps, final reward", r)
```

```python
# Cell 4 – Part B: DQN on CartPole – network + replay buffer + settings
env = gym.make("CartPole-v1")
obs_dim, n_actions = env.observation_space.shape[0], env.action_space.n
print("Observation: cart position, cart velocity, pole angle, pole angular velocity | Actions:", n_actions)

class QNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(obs_dim, 128), nn.ReLU(),
                                 nn.Linear(128, 128), nn.ReLU(),
                                 nn.Linear(128, n_actions))
    def forward(self, x):
        return self.net(x)

from collections import deque

class ReplayBuffer:
    def __init__(self, capacity=50000):
        self.buf = deque(maxlen=capacity)   # oldest transitions discarded automatically
    def push(self, *t):
        self.buf.append(t)
    def sample(self, n):
        s, a, r, s2, d = zip(*random.sample(self.buf, n))
        return (torch.tensor(np.array(s), dtype=torch.float32), torch.tensor(a),
                torch.tensor(r, dtype=torch.float32),
                torch.tensor(np.array(s2), dtype=torch.float32), torch.tensor(d, dtype=torch.float32))

random.seed(SEED); np.random.seed(SEED)
_ = torch.manual_seed(SEED)
policy_net, target_net = QNetwork(), QNetwork()
target_net.load_state_dict(policy_net.state_dict())
optimizer = optim.AdamW(policy_net.parameters(), lr=3e-4, amsgrad=True)
memory = ReplayBuffer(20000)
GAMMA, BATCH, TAU = 0.99, 128, 0.005    # TAU: soft update rate of the target network
eps, EPS_MIN, EPS_DECAY = 1.0, 0.01, 0.98

def evaluate(net, n=5, seed0=9000):
    scores = []
    for k in range(n):
        s, _ = env.reset(seed=seed0 + k)
        done, tot = False, 0
        while not done:
            with torch.no_grad():
                a = int(net(torch.tensor(s, dtype=torch.float32)).argmax())
            s, r, term, trunc, _ = env.step(a)
            tot += r; done = term or trunc
        scores.append(tot)
    return np.mean(scores)
```

```python
# Cell 5 – DQN training loop (keeps best checkpoint)
import copy
episode_rewards, start = [], time.time()
best_score, best_weights = -1, None
for episode in range(1, 501):
    state, _ = env.reset(seed=SEED + episode)
    total, done = 0, False
    while not done:
        if random.random() < eps:                       # explore
            action = env.action_space.sample()
        else:                                           # exploit
            with torch.no_grad():
                action = int(policy_net(torch.tensor(state, dtype=torch.float32)).argmax())
        next_state, reward, terminated, truncated, _ = env.step(action)
        memory.push(state, action, reward, next_state, float(terminated))
        state, total, done = next_state, total + reward, terminated or truncated

        if len(memory.buf) >= 500:
            s, a, r, s2, d = memory.sample(BATCH)
            q = policy_net(s).gather(1, a.unsqueeze(1)).squeeze(1)                 # Q(s, a)
            with torch.no_grad():
                q_target = r + GAMMA * target_net(s2).max(1).values * (1 - d)      # r + gamma max Q_target(s', a')
            loss = nn.functional.smooth_l1_loss(q, q_target)
            optimizer.zero_grad(); loss.backward()
            nn.utils.clip_grad_value_(policy_net.parameters(), 100)
            optimizer.step()
            with torch.no_grad():                                                  # soft update of target network
                for tp, pp in zip(target_net.parameters(), policy_net.parameters()):
                    tp.mul_(1 - TAU).add_(TAU * pp)
    eps = max(EPS_MIN, eps * EPS_DECAY)
    episode_rewards.append(total)
    if episode % 25 == 0:
        score = evaluate(policy_net)
        if score >= best_score:
            best_score, best_weights = score, copy.deepcopy(policy_net.state_dict())
        if episode % 50 == 0:
            print(f"Episode {episode:3d} mean reward (last 50) {np.mean(episode_rewards[-50:]):6.1f} "
                  f"greedy eval {score:6.1f} epsilon {eps:.3f} ({time.time() - start:.0f}s)")

policy_net.load_state_dict(best_weights)
print("Restored best checkpoint, greedy evaluation score:", best_score)
```

```python
# Cell 6 – CartPole: reward curve + final test
plt.figure(figsize=(8, 3.5))
plt.plot(episode_rewards, alpha=0.4, label="episode reward")
plt.plot(np.convolve(episode_rewards, np.ones(20) / 20, mode="valid"), label="20-episode average")
plt.axhline(500, color="grey", ls="--", label="maximum (500)")
plt.xlabel("Episode"); plt.ylabel("Reward"); plt.title("DQN on CartPole-v1"); plt.legend()
plt.show()

test = []
for ep in range(20):
    s, _ = env.reset(seed=5000 + ep)
    done, tot = False, 0
    while not done:
        with torch.no_grad():
            a = int(policy_net(torch.tensor(s, dtype=torch.float32)).argmax())
        s, r, term, trunc, _ = env.step(a)
        tot += r; done = term or trunc
    test.append(tot)
print("Greedy DQN policy over 20 test episodes:", test)
print(f"Mean test reward: {np.mean(test):.1f} (CartPole is considered solved at an average of 475)")
```

---

## Practical 6a – Transfer learning on limited data

### Part A – digits 0-4 → digits 5-9 (only 50 labelled images)

```python
# Cell A1 – imports
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

print("TensorFlow:", tf.__version__)
tf.random.set_seed(3); np.random.seed(3)
```

```python
# Cell A2 – source task (digits 0-4), target task (digits 5-9, 10 images per class), model parts
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = (x_train / 255.0).astype("float32")[..., None]
x_test = (x_test / 255.0).astype("float32")[..., None]

src_tr, src_te = y_train < 5, y_test < 5
Xs, ys, Xs_te, ys_te = x_train[src_tr], y_train[src_tr], x_test[src_te], y_test[src_te]

# target task: digits 5-9 relabelled 0-4, only 10 labelled images per class
rng = np.random.default_rng(0)
idx = np.concatenate([rng.choice(np.where(y_train == d)[0], 10, replace=False) for d in range(5, 10)])
Xt, yt = x_train[idx], y_train[idx] - 5
tgt_te = y_test >= 5
Xt_te, yt_te = x_test[tgt_te], y_test[tgt_te] - 5
print("Source train:", Xs.shape, "| Target train:", Xt.shape, "| Target test:", Xt_te.shape)

def conv_base():
    return keras.Sequential([
        keras.Input(shape=(28, 28, 1)),
        layers.Conv2D(32, 3, activation="relu"), layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation="relu"), layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation="relu"), layers.Flatten()], name="conv_base")

def head(base, name):
    return keras.Sequential([base, layers.Dense(64, activation="relu"), layers.Dropout(0.3),
                             layers.Dense(5, activation="softmax")], name=name)
```

```python
# Cell A3 – pre-train on the source task
source_base = conv_base()
source_model = head(source_base, "source_model")
source_model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
source_model.fit(Xs, ys, epochs=3, batch_size=128, verbose=2)
print("Source-task test accuracy (digits 0-4):", round(source_model.evaluate(Xs_te, ys_te, verbose=0)[1], 4))
```

```python
# Cell A4 – scratch vs feature extraction vs fine-tuning
results = {}

def train_target(model, name, epochs=60, lr=1e-3):
    model.compile(optimizer=keras.optimizers.Adam(lr), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    h = model.fit(Xt, yt, epochs=epochs, batch_size=10, verbose=0, validation_data=(Xt_te, yt_te))
    acc = model.evaluate(Xt_te, yt_te, verbose=0)[1]
    results[name] = (acc, h.history["val_accuracy"])
    print(f"{name:<38} target test accuracy: {acc:.4f}")

# baseline: same architecture trained from scratch on 50 images
train_target(head(conv_base(), "scratch"), "From scratch (random weights)")

# feature extraction: copy pre-trained base, freeze it, train only the new head
fe_base = conv_base()
fe_base.set_weights(source_base.get_weights())
fe_base.trainable = False
fe_model = head(fe_base, "feature_extraction")
train_target(fe_model, "Transfer: feature extraction (frozen)")
print("Trainable parameters - frozen model:", sum(np.prod(w.shape) for w in fe_model.trainable_weights),
      "of", fe_model.count_params())

# fine-tuning: unfreeze the base and continue with a small learning rate
fe_base.trainable = True
train_target(fe_model, "Transfer: fine-tuned (unfrozen, lr=1e-4)", epochs=30, lr=1e-4)
```

```python
# Cell A5 – comparison plots
plt.figure(figsize=(8, 4))
for name, (acc, curve) in results.items():
    plt.plot(curve, label=f"{name} ({acc:.3f})")
plt.xlabel("Epoch"); plt.ylabel("Target test accuracy"); plt.title("Learning digits 5-9 from 50 images"); plt.legend()
plt.show()

names = list(results)
accs = [results[n][0] for n in names]
plt.figure(figsize=(7, 3))
plt.barh(names, accs, color=["grey", "tab:blue", "tab:green"])
plt.xlim(0, 1)
for i, a in enumerate(accs):
    plt.text(a + 0.01, i, f"{a:.3f}", va="center")
plt.title("Target-task accuracy")
plt.show()
```

### Part B – MobileNetV2 on flower photos (real transfer learning)

```python
# Cell B1 – load flowers dataset + MobileNetV2 transfer model + fine-tuning
import pathlib
url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
data_dir = pathlib.Path(keras.utils.get_file("flower_photos", origin=url, untar=True)) / "flower_photos"
IMG = (160, 160)
train_ds = keras.utils.image_dataset_from_directory(data_dir, validation_split=0.2, subset="training",
                                                    seed=1, image_size=IMG, batch_size=32)
val_ds = keras.utils.image_dataset_from_directory(data_dir, validation_split=0.2, subset="validation",
                                                  seed=1, image_size=IMG, batch_size=32)
class_names = train_ds.class_names
print(class_names)
train_ds, val_ds = train_ds.prefetch(tf.data.AUTOTUNE), val_ds.prefetch(tf.data.AUTOTUNE)

base = keras.applications.MobileNetV2(input_shape=IMG + (3,), include_top=False, weights="imagenet")
base.trainable = False

inputs = keras.Input(shape=IMG + (3,))
x = layers.RandomFlip("horizontal")(inputs)
x = layers.RandomRotation(0.1)(x)
x = keras.applications.mobilenet_v2.preprocess_input(x)
x = base(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(len(class_names), activation="softmax")(x)
flower_model = keras.Model(inputs, outputs)
flower_model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
flower_model.fit(train_ds, validation_data=val_ds, epochs=5)

# fine-tune only the top layers
base.trainable = True
for layer in base.layers[:100]:
    layer.trainable = False
flower_model.compile(optimizer=keras.optimizers.Adam(1e-5), loss="sparse_categorical_crossentropy",
                     metrics=["accuracy"])
flower_model.fit(train_ds, validation_data=val_ds, epochs=5)
print("Validation accuracy after fine-tuning:", flower_model.evaluate(val_ds, verbose=0)[1])
```

---

## Practical 6b – Time-series forecasting (MLP, SimpleRNN, LSTM) + anomaly detection

```python
# Cell 1 – imports
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

tf.random.set_seed(42); np.random.seed(42)
```

```python
# Cell 2 – Part A: sales forecasting with an MLP on sliding windows
sales = np.array([10, 12, 13, 12, 14, 15, 16, 18, 17, 19]).reshape(-1, 1)
scaler = MinMaxScaler()
sales_scaled = scaler.fit_transform(sales).flatten()
n_steps = 3
X = np.array([sales_scaled[i:i + n_steps] for i in range(len(sales_scaled) - n_steps)])
y = np.array([sales_scaled[i + n_steps] for i in range(len(sales_scaled) - n_steps)])
print("Sliding windows (scaled):\n", X.round(3), "\nTargets:", y.round(3))

mlp = keras.Sequential([keras.Input(shape=(n_steps,)), layers.Dense(8, activation="relu"), layers.Dense(1)])
mlp.compile(optimizer=keras.optimizers.Adam(0.01), loss="mse")
mlp.fit(X, y, epochs=500, verbose=0)
x_input = scaler.transform(np.array([18, 17, 19]).reshape(-1, 1)).reshape(1, n_steps)
y_pred = scaler.inverse_transform(mlp.predict(x_input, verbose=0))[0][0]
print(f"Predicted sales on day 11 (from days 8-10 = 18, 17, 19): {y_pred:.2f}")
```

```python
# Cell 3 – Part B: load daily minimum temperature data
path = "daily-min-temperatures.csv"
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv"
df = pd.read_csv(path if os.path.exists(path) else url, parse_dates=["Date"])
print(df.head())
print("Rows:", len(df), "| from", df.Date.min().date(), "to", df.Date.max().date())
print(df["Temp"].describe().round(2))
plt.figure(figsize=(12, 3))
plt.plot(df.Date, df.Temp, lw=0.6)
plt.title("Daily minimum temperature, Melbourne"); plt.ylabel("°C")
plt.show()
```

```python
# Cell 4 – chronological split, normalise (train stats only), windows
data = df["Temp"].values.astype("float32")
split = int(len(data) * 0.8)                          # first 8 years train, last 2 test
mean, std = data[:split].mean(), data[:split].std()   # statistics from training data only (no leakage)
norm = (data - mean) / std
print(f"Train mean {mean:.3f}, std {std:.3f}")

seq_length = 14   # previous 14 days -> next day

def windows(series):
    X = np.array([series[i:i + seq_length] for i in range(len(series) - seq_length)])
    return X[..., None], series[seq_length:]

X_train, y_train = windows(norm[:split])
X_test, y_test = windows(norm[split - seq_length:])
print("X_train:", X_train.shape, "| X_test:", X_test.shape)
```

```python
# Cell 5 – SimpleRNN vs LSTM + metrics vs naive baseline
def build(kind):
    rnn = layers.SimpleRNN(32, activation="tanh") if kind == "SimpleRNN" else layers.LSTM(32)
    m = keras.Sequential([keras.Input(shape=(seq_length, 1)), rnn, layers.Dense(1)], name=kind)
    m.compile(optimizer="adam", loss="mse")
    return m

models, preds = {}, {}
for kind in ["SimpleRNN", "LSTM"]:
    m = build(kind)
    m.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.1, verbose=0,
          callbacks=[keras.callbacks.EarlyStopping(patience=4, restore_best_weights=True)])
    p = m.predict(X_test, verbose=0).ravel() * std + mean
    models[kind], preds[kind] = m, p

true = y_test * std + mean
naive = data[split - 1:-1]   # persistence baseline: tomorrow = today
print(f"{'Model':<22}{'RMSE':>8}{'MAE':>8}")
print(f"{'Naive (yesterday)':<22}{np.sqrt(mean_squared_error(true, naive)):8.3f}{mean_absolute_error(true, naive):8.3f}")
for k, p in preds.items():
    print(f"{k:<22}{np.sqrt(mean_squared_error(true, p)):8.3f}{mean_absolute_error(true, p):8.3f}")
for t, p in zip(true[:5], preds["LSTM"][:5]):
    print(f"True: {t:.2f}, Predicted (LSTM): {p:.2f}")
```

```python
# Cell 6 – true vs predicted plot
dates = df.Date.values[split:]
plt.figure(figsize=(12, 4.5))
plt.plot(dates, true, label="True", lw=1)
plt.plot(dates, preds["LSTM"], label="LSTM predicted", lw=1)
plt.xlabel("Date"); plt.ylabel("Temperature (°C)")
plt.title("Daily Minimum Temperature (test period): True vs Predicted")
plt.legend()
plt.show()
```

```python
# Cell 7 – Part C: anomaly detection (error > mean + 3 std of training errors)
train_pred = models["LSTM"].predict(X_train, verbose=0).ravel() * std + mean
train_err = np.abs(y_train * std + mean - train_pred)
threshold = train_err.mean() + 3 * train_err.std()

test_err = np.abs(true - preds["LSTM"])
anom = test_err > threshold
print(f"Error threshold: {threshold:.2f} °C | anomalies found in test period: {anom.sum()} of {len(anom)} days")
print(pd.DataFrame({"Date": pd.to_datetime(dates[anom]).date, "Actual": true[anom].round(1),
                    "Predicted": preds["LSTM"][anom].round(1), "Error": test_err[anom].round(1)}).to_string(index=False))

plt.figure(figsize=(12, 4))
plt.plot(dates, true, lw=0.8, label="Temperature")
plt.scatter(dates[anom], true[anom], color="red", zorder=3, label="Anomaly")
plt.title("Anomalous days detected by the LSTM forecaster"); plt.ylabel("°C"); plt.legend()
plt.show()
```

---

## Practical 7 – ML pipeline: automated feature engineering + model selection (Titanic)

```python
# Cell 1 – imports
import os, re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib, sklearn
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, accuracy_score

print("scikit-learn:", sklearn.__version__)
```

```python
# Cell 2 – load data + split
path = "titanic.csv"
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(path if os.path.exists(path) else url)
print(df.shape)
print(df.head())
print("\nMissing values:\n", df.isna().sum()[df.isna().sum() > 0])
print("\nSurvival rate:", round(df.Survived.mean(), 3))

X = df.drop(columns=["Survived", "PassengerId"])
y = df["Survived"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
```

```python
# Cell 3 – feature engineering + preprocessing + base pipeline
def engineer_features(d):
    d = d.copy()
    d["Title"] = d["Name"].str.extract(r",\s*([^\.]+)\.", expand=False).str.strip()
    d["Title"] = d["Title"].replace({"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"})
    d.loc[~d["Title"].isin(["Mr", "Mrs", "Miss", "Master"]), "Title"] = "Rare"
    d["FamilySize"] = d["SibSp"] + d["Parch"] + 1
    d["IsAlone"] = (d["FamilySize"] == 1).astype(int)
    d["FarePerPerson"] = d["Fare"] / d["FamilySize"]
    d["HasCabin"] = d["Cabin"].notna().astype(int)
    d["Deck"] = d["Cabin"].str[0].fillna("U")
    d["TicketPrefix"] = d["Ticket"].str.contains(r"[A-Za-z]").astype(int)
    d["AgeMissing"] = d["Age"].isna().astype(int)
    return d.drop(columns=["Name", "Ticket", "Cabin"])

fe = FunctionTransformer(engineer_features)
print(fe.fit_transform(X_train).head())

numeric = make_column_selector(dtype_include=np.number)
categorical = make_column_selector(dtype_exclude=np.number)

preprocess = ColumnTransformer([
    ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric),
    ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                      ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), categorical)])

pipe = Pipeline([("features", fe), ("preprocess", preprocess),
                 ("select", SelectKBest(mutual_info_classif, k=15)),
                 ("model", LogisticRegression(max_iter=1000))])
```

```python
# Cell 4 – automated model selection (GridSearchCV over several algorithms)
param_grid = [
    {"model": [LogisticRegression(max_iter=2000)], "model__C": [0.1, 1, 10], "select__k": [10, 15, "all"]},
    {"model": [RandomForestClassifier(random_state=42)], "model__n_estimators": [200],
     "model__max_depth": [4, 6, None], "select__k": [15, "all"]},
    {"model": [HistGradientBoostingClassifier(random_state=42)], "model__learning_rate": [0.05, 0.1],
     "model__max_depth": [3, None], "select__k": [15, "all"]},
    {"model": [SVC(probability=True, random_state=42)], "model__C": [0.5, 1, 5], "model__gamma": ["scale"],
     "select__k": [15, "all"]},
    {"model": [KNeighborsClassifier()], "model__n_neighbors": [5, 9, 15], "select__k": [10, 15]},
]
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
search = GridSearchCV(pipe, param_grid, cv=cv, scoring="accuracy", n_jobs=-1)
search.fit(X_train, y_train)
print("Pipelines evaluated:", len(search.cv_results_["params"]), "x 5 folds")
print("Best CV accuracy :", round(search.best_score_, 4))
print("Best pipeline    :", {k: (type(v).__name__ if k == "model" else v) for k, v in search.best_params_.items()})
```

```python
# Cell 5 – best configuration of each algorithm
res = pd.DataFrame(search.cv_results_)
res["algorithm"] = res["param_model"].apply(lambda m: type(m).__name__)
best_per_algo = res.sort_values("mean_test_score", ascending=False).groupby("algorithm").head(1)
print(best_per_algo[["algorithm", "mean_test_score", "std_test_score"]].round(4).to_string(index=False))
plt.figure(figsize=(7, 3))
plt.barh(best_per_algo["algorithm"], best_per_algo["mean_test_score"],
         xerr=best_per_algo["std_test_score"], color="tab:blue")
plt.xlim(0.7, 0.9)
plt.xlabel("5-fold CV accuracy")
plt.title("Best configuration of each algorithm")
plt.show()
```

```python
# Cell 6 – final test, selected features, save + reload pipeline
best = search.best_estimator_
y_pred = best.predict(X_test)
print("Test accuracy:", round(accuracy_score(y_test, y_pred), 4))
print(classification_report(y_test, y_pred, target_names=["Died", "Survived"]))
ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=["Died", "Survived"], cmap="Blues")
plt.show()

names = best.named_steps["preprocess"].get_feature_names_out()
selected = names[best.named_steps["select"].get_support()]
print("Features kept by SelectKBest:", [s.split("__")[1] for s in selected])

joblib.dump(best, "titanic_best_pipeline.joblib")
new_passenger = pd.DataFrame([{"Pclass": 1, "Name": "Doe, Mrs. Jane", "Sex": "female", "Age": 30,
                               "SibSp": 1, "Parch": 0, "Ticket": "PC 17599", "Fare": 80.0,
                               "Cabin": "C85", "Embarked": "C"}])
print("Saved pipeline; survival probability for a new passenger:",
      round(joblib.load("titanic_best_pipeline.joblib").predict_proba(new_passenger)[0, 1], 3))
```

---

## Practical 8 – Hyper-parameter tuning: Grid, Random, Bayesian (Optuna) and Genetic algorithm (DEAP)

```python
# Cell 1 – imports, data, helper functions
%pip install -q optuna deap
import time, random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import optuna, deap
from deap import base, creator, tools, algorithms
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from scipy.stats import randint, uniform, loguniform

print("Optuna:", optuna.__version__, "| DEAP:", deap.__version__)
optuna.logging.set_verbosity(optuna.logging.WARNING)

data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.25,
                                                    stratify=data.target, random_state=0)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
print("Train:", X_train.shape, "| Test:", X_test.shape)

def cv_score(params):
    model = GradientBoostingClassifier(random_state=0, **params)
    return cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy", n_jobs=-1).mean()

def test_score(params):
    return GradientBoostingClassifier(random_state=0, **params).fit(X_train, y_train).score(X_test, y_test)

summary = []
default_cv = cv_score({})
summary.append(("Default parameters", 1, default_cv, test_score({}), 0.0))
print(f"Default GradientBoosting: CV accuracy {default_cv:.4f}")
```

```python
# Cell 2 – Grid search + Random search
t = time.time()
grid = GridSearchCV(GradientBoostingClassifier(random_state=0),
                    {"n_estimators": [50, 150, 300], "learning_rate": [0.01, 0.1, 0.3], "max_depth": [2, 3, 5]},
                    cv=cv, scoring="accuracy", n_jobs=-1).fit(X_train, y_train)
summary.append(("Grid search", len(grid.cv_results_["params"]), grid.best_score_,
                grid.score(X_test, y_test), time.time() - t))
print("Grid search best:", grid.best_params_, round(grid.best_score_, 4))

t = time.time()
rand = RandomizedSearchCV(GradientBoostingClassifier(random_state=0),
                          {"n_estimators": randint(50, 400), "learning_rate": loguniform(0.005, 0.5),
                           "max_depth": randint(1, 6), "subsample": uniform(0.5, 0.5),
                           "min_samples_leaf": randint(1, 20)},
                          n_iter=30, cv=cv, scoring="accuracy", n_jobs=-1, random_state=0).fit(X_train, y_train)
summary.append(("Random search", 30, rand.best_score_, rand.score(X_test, y_test), time.time() - t))
print("Random search best:", {k: (round(float(v), 4) if isinstance(v, float) else int(v))
                              for k, v in rand.best_params_.items()}, round(rand.best_score_, 4))
```

```python
# Cell 3 – Bayesian optimisation with Optuna (TPE)
def objective(trial):
    params = {"n_estimators": trial.suggest_int("n_estimators", 50, 400),
              "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.5, log=True),
              "max_depth": trial.suggest_int("max_depth", 1, 5),
              "subsample": trial.suggest_float("subsample", 0.5, 1.0),
              "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20)}
    return cv_score(params)

t = time.time()
study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=0))
study.optimize(objective, n_trials=30)
summary.append(("Bayesian optimisation (Optuna TPE)", 30, study.best_value,
                test_score(study.best_params), time.time() - t))
print("Optuna best CV accuracy:", round(study.best_value, 4))
print("Optuna best parameters :", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in study.best_params.items()})

trials = study.trials_dataframe()
plt.figure(figsize=(8, 3.5))
plt.plot(trials["number"], trials["value"], "o", alpha=0.5, label="trial")
plt.plot(trials["number"], trials["value"].cummax(), "r-", label="best so far")
plt.xlabel("Trial"); plt.ylabel("CV accuracy"); plt.title("Optuna optimisation history"); plt.legend()
plt.show()
imp = optuna.importance.get_param_importances(study)
print("Hyper-parameter importance:", {k: round(v, 3) for k, v in imp.items()})
```

```python
# Cell 4 – Genetic algorithm with DEAP
def decode(ind):
    g = [min(max(v, 0.0), 1.0) for v in ind]
    return {"n_estimators": int(50 + g[0] * 350),
            "learning_rate": float(10 ** (np.log10(0.005) + g[1] * 2)),
            "max_depth": int(1 + round(g[2] * 4)),
            "subsample": 0.5 + g[3] * 0.5,
            "min_samples_leaf": int(1 + g[4] * 19)}

random.seed(0); np.random.seed(0)
if not hasattr(creator, "FitnessMax"):
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("gene", random.random)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.gene, n=5)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", lambda ind: (cv_score(decode(ind)),))
toolbox.register("mate", tools.cxBlend, alpha=0.3)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.15, indpb=0.3)
toolbox.register("select", tools.selTournament, tournsize=3)

t = time.time()
pop = toolbox.population(n=8)
hof = tools.HallOfFame(1)
stats = tools.Statistics(lambda ind: ind.fitness.values[0])
stats.register("avg", np.mean)
stats.register("max", np.max)
pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.6, mutpb=0.3, ngen=4, stats=stats, halloffame=hof, verbose=True)
evals = sum(rec["nevals"] for rec in log)
ga_params = decode(hof[0])
summary.append(("Genetic algorithm (DEAP)", evals, hof[0].fitness.values[0], test_score(ga_params), time.time() - t))
print("GA best CV accuracy:", round(hof[0].fitness.values[0], 4))
print("GA best parameters :", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in ga_params.items()})

plt.figure(figsize=(7, 3))
plt.plot(log.select("gen"), log.select("max"), "o-", label="best")
plt.plot(log.select("gen"), log.select("avg"), "s--", label="average")
plt.xlabel("Generation"); plt.ylabel("CV accuracy"); plt.title("Genetic algorithm progress"); plt.legend()
plt.show()
```

```python
# Cell 5 – comparison table of all methods
cmp = pd.DataFrame(summary, columns=["Method", "Models evaluated", "Best CV accuracy", "Test accuracy", "Time (s)"])
print(cmp.round(4).to_string(index=False))
```

---

## Practical 9 – Deploy an ML model: FastAPI + Docker (+ Cloud Run)

```python
# Cell 1 – imports
%pip install -q fastapi uvicorn scikit-learn joblib requests
import os, sys, time, json, subprocess, textwrap
import joblib, requests, sklearn
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

print("scikit-learn:", sklearn.__version__)
```

```python
# Cell 2 – train + save model
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2,
                                                    stratify=iris.target, random_state=1)
model = Pipeline([("scale", StandardScaler()), ("clf", RandomForestClassifier(n_estimators=200, random_state=1))])
model.fit(X_train, y_train)
print(f"Test accuracy: {accuracy_score(y_test, model.predict(X_test)):.4f}")

os.makedirs("iris_app", exist_ok=True)
joblib.dump({"model": model, "classes": list(iris.target_names), "version": "1.0.0"}, "iris_app/model.joblib")
print("Saved iris_app/model.joblib,", os.path.getsize("iris_app/model.joblib"), "bytes")
```

```python
# Cell 3 – write the FastAPI app (iris_app/main.py)
app_code = '''
from contextlib import asynccontextmanager
from typing import List
import joblib, numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

bundle = {}

@asynccontextmanager
async def lifespan(app: FastAPI):          # replaces the deprecated @app.on_event("startup")
    bundle.update(joblib.load("model.joblib"))
    yield

app = FastAPI(title="Iris Species Classifier", version="1.0.0", lifespan=lifespan)

class Flower(BaseModel):
    sepal_length: float = Field(..., gt=0, examples=[5.1])
    sepal_width: float = Field(..., gt=0, examples=[3.5])
    petal_length: float = Field(..., gt=0, examples=[1.4])
    petal_width: float = Field(..., gt=0, examples=[0.2])

class Batch(BaseModel):
    flowers: List[Flower]

@app.get("/health")
def health():
    return {"status": "ok", "model_version": bundle.get("version")}

def _predict(flowers):
    X = np.array([[f.sepal_length, f.sepal_width, f.petal_length, f.petal_width] for f in flowers])
    proba = bundle["model"].predict_proba(X)
    return [{"species": bundle["classes"][int(p.argmax())], "confidence": round(float(p.max()), 3)} for p in proba]

@app.post("/predict")
def predict(flower: Flower):
    return _predict([flower])[0]

@app.post("/predict_batch")
def predict_batch(batch: Batch):
    if not batch.flowers:
        raise HTTPException(status_code=400, detail="empty batch")
    return {"predictions": _predict(batch.flowers)}
'''
with open("iris_app/main.py", "w") as f:
    f.write(textwrap.dedent(app_code).lstrip())
print(open("iris_app/main.py").read()[:400], "...")
```

```python
# Cell 4 – requirements.txt, Dockerfile, .dockerignore
requirements = f"""fastapi>=0.110
uvicorn[standard]>=0.29
scikit-learn=={sklearn.__version__}
joblib>=1.3
numpy>=1.26
"""
dockerfile = """# ---- Dockerfile ----
FROM python:3.12-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY main.py model.joblib ./
# Cloud Run sends traffic to the port given in $PORT (default 8080)
ENV PORT=8080
EXPOSE 8080
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT}"]
"""
dockerignore = "__pycache__/\n*.ipynb\n.ipynb_checkpoints/\n"

for name, content in [("requirements.txt", requirements), ("Dockerfile", dockerfile), (".dockerignore", dockerignore)]:
    with open(f"iris_app/{name}", "w") as f:
        f.write(content)
print(open("iris_app/requirements.txt").read())
print(open("iris_app/Dockerfile").read())
print("Files in iris_app:", sorted(os.listdir("iris_app")))
```

```python
# Cell 5 – run the API locally and test every endpoint
server = subprocess.Popen([sys.executable, "-m", "uvicorn", "main:app", "--port", "8000"],
                          cwd="iris_app", stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
BASE = "http://127.0.0.1:8000"
for _ in range(30):
    try:
        if requests.get(f"{BASE}/health", timeout=1).ok:
            break
    except requests.ConnectionError:
        time.sleep(0.5)

print("GET /health ->", requests.get(f"{BASE}/health").json())
sample = {"sepal_length": 6.7, "sepal_width": 3.0, "petal_length": 5.2, "petal_width": 2.3}
print("POST /predict", sample, "->", requests.post(f"{BASE}/predict", json=sample).json())

batch = {"flowers": [
    {"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2},
    {"sepal_length": 5.9, "sepal_width": 2.8, "petal_length": 4.4, "petal_width": 1.3},
    {"sepal_length": 7.7, "sepal_width": 2.6, "petal_length": 6.9, "petal_width": 2.3}]}
print("POST /predict_batch ->")
for p in requests.post(f"{BASE}/predict_batch", json=batch).json()["predictions"]:
    print("     ", p)

bad = requests.post(f"{BASE}/predict", json={"sepal_length": -1, "sepal_width": 3})
print("Invalid input -> HTTP", bad.status_code, "|", bad.json()["detail"][0]["msg"])
print("Interactive API docs (Swagger UI) available at", f"{BASE}/docs")

t = time.time()
for _ in range(200):
    requests.post(f"{BASE}/predict", json=sample)
print(f"Average latency over 200 requests: {(time.time() - t) / 200 * 1000:.1f} ms")

server.terminate()
server.wait()
print("Local server stopped")
```

```bash
# Cell 6 – build + deploy commands (run in a terminal, NOT in the notebook; needs Docker / gcloud)
cd iris_app
docker build -t iris-api .
docker run -p 8080:8080 iris-api

# Google Cloud Run
gcloud run deploy iris-api --source . --region asia-south1 --allow-unauthenticated
```

```python
# Cell 7 – call the deployed cloud service (replace the URL with the one gcloud prints)
CLOUD_URL = "https://iris-api-xxxxx-el.a.run.app"
print(requests.get(f"{CLOUD_URL}/health").json())
print(requests.post(f"{CLOUD_URL}/predict", json=sample).json())
```

---

## Practical 10a – Text generation: char-RNN (LSTM) and GPT-2

### Part A – character-level LSTM on Tiny Shakespeare

```python
# Cell A1 – imports + load text
import os, time
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

print("TensorFlow:", tf.__version__)
tf.random.set_seed(0); np.random.seed(0)

path = "tinyshakespeare.txt"
url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
if not os.path.exists(path):
    path = keras.utils.get_file("tinyshakespeare.txt", url)
text = open(path, encoding="utf-8").read()
print("Corpus length:", len(text), "characters")
print(text[:300])
```

```python
# Cell A2 – vocabulary, dataset, model
vocab = sorted(set(text))
char2idx = {c: i for i, c in enumerate(vocab)}
idx2char = np.array(vocab)
print("Unique characters:", len(vocab))
encoded = np.array([char2idx[c] for c in text], dtype=np.int32)

SEQ_LEN, BATCH = 100, 64
CORPUS = encoded[:600_000]   # first 600k characters (use all on a GPU)
ds = tf.data.Dataset.from_tensor_slices(CORPUS).batch(SEQ_LEN + 1, drop_remainder=True)
ds = ds.map(lambda chunk: (chunk[:-1], chunk[1:]))   # input = text, target = text shifted by one char
ds = ds.shuffle(10000, seed=0).batch(BATCH, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
for x, y in ds.take(1):
    print("Input :", repr("".join(idx2char[x[0][:40].numpy()])))
    print("Target:", repr("".join(idx2char[y[0][:40].numpy()])))

model = keras.Sequential([
    keras.Input(shape=(None,)),
    layers.Embedding(len(vocab), 64),
    layers.LSTM(384, return_sequences=True),
    layers.Dense(len(vocab)),   # logits for the next character
])
model.compile(optimizer=keras.optimizers.Adam(2e-3),
              loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True))
model.summary()
```

```python
# Cell A3 – text generator + training loop
def generate(seed, length=400, temperature=0.8):
    ids = [char2idx[c] for c in seed]
    out = list(seed)
    for _ in range(length):
        logits = model(np.array([ids[-SEQ_LEN:]]), training=False)[0, -1].numpy() / temperature
        nxt = tf.random.categorical(logits[None, :], 1)[0, 0].numpy()
        ids.append(int(nxt))
        out.append(idx2char[nxt])
    return "".join(out)

EPOCHS = 8
for epoch in range(1, EPOCHS + 1):
    t = time.time()
    h = model.fit(ds, epochs=1, verbose=0)
    print(f"Epoch {epoch}/{EPOCHS} - loss {h.history['loss'][0]:.4f} ({time.time() - t:.0f}s)")
    if epoch in (1, EPOCHS):
        print("Sample:", generate("ROMEO:", 200, 0.8).replace("\n", " / ")[:260], "\n")
```

```python
# Cell A4 – effect of temperature + save
for temp in [0.3, 0.8, 1.2]:
    print(f"===== temperature {temp} =====")
    print(generate("JULIET:\n", 350, temp), "\n")
model.save("char_lstm_shakespeare.keras")
```

### Part B – GPT-2 (pretrained generation + fine-tuning)

```python
# Cell B1 – pretrained GPT-2 text generation
%pip install -q transformers datasets accelerate
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, DataCollatorForLanguageModeling, Trainer, TrainingArguments
from transformers import set_seed
from datasets import Dataset

device = 0 if torch.cuda.is_available() else -1
generator = pipeline("text-generation", model="gpt2", device=device)
set_seed(42)
for out in generator("The future of artificial intelligence in education", max_new_tokens=60,
                     num_return_sequences=2, do_sample=True, top_p=0.95):
    print(out["generated_text"], "\n---")
```

```python
# Cell B2 – fine-tune GPT-2 on the Shakespeare corpus (uses `text` from Cell A1)
tok = AutoTokenizer.from_pretrained("gpt2")
tok.pad_token = tok.eos_token
gpt2 = AutoModelForCausalLM.from_pretrained("gpt2")
blocks = [text[i:i + 2000] for i in range(0, 400_000, 2000)]
ds_hf = Dataset.from_dict({"text": blocks}).map(
    lambda b: tok(b["text"], truncation=True, max_length=256), batched=True, remove_columns=["text"])
args = TrainingArguments("gpt2-shakespeare", num_train_epochs=1, per_device_train_batch_size=8,
                         learning_rate=5e-5, logging_steps=20, save_strategy="no", report_to="none")
trainer = Trainer(model=gpt2, args=args, train_dataset=ds_hf,
                  data_collator=DataCollatorForLanguageModeling(tok, mlm=False))
trainer.train()
ft = pipeline("text-generation", model=gpt2, tokenizer=tok, device=device)
print(ft("ROMEO:", max_new_tokens=120, do_sample=True, temperature=0.8, top_p=0.95)[0]["generated_text"])
```

---

## Practical 10b – Conditional GAN (cGAN) in PyTorch on MNIST

```python
# Cell 1 – imports
import time
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)
print("PyTorch:", torch.__version__, "| torchvision:", torchvision.__version__, "| device:", device)
```

```python
# Cell 2 – data loader (pixels scaled to [-1, 1])
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
mnist = datasets.MNIST(root="data", train=True, download=True, transform=transform)
loader = DataLoader(mnist, batch_size=128, shuffle=True, drop_last=True, num_workers=0)
print("Training images:", len(mnist), "| batches per epoch:", len(loader))

imgs, labels = next(iter(loader))
grid = torchvision.utils.make_grid(imgs[:32], nrow=16, normalize=True)
plt.figure(figsize=(12, 2))
plt.imshow(grid.permute(1, 2, 0))
plt.axis("off")
plt.title("Real MNIST digits")
plt.show()
```

```python
# Cell 3 – conditional generator + discriminator
NZ, NCLASS = 64, 10

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.label_emb = nn.Embedding(NCLASS, 16)
        self.fc = nn.Sequential(nn.Linear(NZ + 16, 128 * 7 * 7), nn.BatchNorm1d(128 * 7 * 7), nn.ReLU(True))
        self.net = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.ReLU(True),   # 7 -> 14
            nn.ConvTranspose2d(64, 1, 4, 2, 1), nn.Tanh())                              # 14 -> 28
    def forward(self, z, y):
        h = self.fc(torch.cat([z, self.label_emb(y)], 1)).view(-1, 128, 7, 7)
        return self.net(h)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.label_emb = nn.Embedding(NCLASS, 28 * 28)
        self.net = nn.Sequential(
            nn.Conv2d(2, 64, 4, 2, 1), nn.LeakyReLU(0.2, True),                          # 28 -> 14
            nn.Conv2d(64, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2, True),   # 14 -> 7
            nn.Flatten(), nn.Linear(128 * 7 * 7, 1))                                     # real/fake logit
    def forward(self, x, y):
        return self.net(torch.cat([x, self.label_emb(y).view(-1, 1, 28, 28)], 1)).squeeze(1)

G, D = Generator().to(device), Discriminator().to(device)

def init_weights(m):   # DCGAN initialisation N(0, 0.02)
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        nn.init.normal_(m.weight, 0.0, 0.02)

G.apply(init_weights); D.apply(init_weights)
print("Generator parameters:", sum(p.numel() for p in G.parameters()))
print("Discriminator parameters:", sum(p.numel() for p in D.parameters()))
```

```python
# Cell 4 – losses, optimisers, training loop
criterion = nn.BCEWithLogitsLoss()
optG = torch.optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
optD = torch.optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))
fixed_z = torch.randn(100, NZ, device=device)
fixed_y = torch.arange(10, device=device).repeat_interleave(10)   # 10 samples of each digit

def show_grid(title):
    G.eval()
    with torch.no_grad():
        g = torchvision.utils.make_grid(G(fixed_z, fixed_y).cpu(), nrow=10, normalize=True)
    G.train()
    plt.figure(figsize=(6, 6))
    plt.imshow(g.permute(1, 2, 0))
    plt.axis("off")
    plt.title(title)
    plt.show()

EPOCHS = 6
lossG_hist, lossD_hist = [], []
for epoch in range(1, EPOCHS + 1):
    t = time.time()
    lg, ld = [], []
    for real, y in loader:
        real, y = real.to(device), y.to(device)
        b = real.size(0)
        z = torch.randn(b, NZ, device=device)
        fake = G(z, y)
        # 1) Discriminator: maximise log D(x|y) + log(1 - D(G(z|y)|y))
        lossD = criterion(D(real, y), torch.full((b,), 0.9, device=device)) + \
                criterion(D(fake.detach(), y), torch.zeros(b, device=device))
        optD.zero_grad(); lossD.backward(); optD.step()
        # 2) Generator: maximise log D(G(z|y)|y) (non-saturating loss)
        lossG = criterion(D(fake, y), torch.ones(b, device=device))
        optG.zero_grad(); lossG.backward(); optG.step()
        lg.append(lossG.item()); ld.append(lossD.item())
    lossG_hist.append(np.mean(lg)); lossD_hist.append(np.mean(ld))
    print(f"Epoch {epoch}/{EPOCHS} loss_D {lossD_hist[-1]:.4f} loss_G {lossG_hist[-1]:.4f} ({time.time() - t:.0f}s)")
    if epoch in (1, EPOCHS):
        show_grid(f"Generated digits after epoch {epoch} (row = requested digit)")
```

```python
# Cell 5 – loss curves
plt.figure(figsize=(7, 3.5))
plt.plot(range(1, EPOCHS + 1), lossD_hist, "o-", label="Discriminator")
plt.plot(range(1, EPOCHS + 1), lossG_hist, "o-", label="Generator")
plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("cGAN training losses"); plt.legend()
plt.show()
```

```python
# Cell 6 – experiments: generate a phone number + latent-space interpolation + save
G.eval()
number = "9876543210"
with torch.no_grad():
    digits = G(torch.randn(len(number), NZ, device=device),
               torch.tensor([int(c) for c in number], device=device)).cpu()
plt.figure(figsize=(10, 1.5))
plt.imshow(torchvision.utils.make_grid(digits, nrow=len(number), normalize=True).permute(1, 2, 0))
plt.axis("off")
plt.title(f"Generated on request: {number}")
plt.show()

with torch.no_grad():
    z0, z1 = torch.randn(1, NZ, device=device), torch.randn(1, NZ, device=device)
    alphas = torch.linspace(0, 1, 10, device=device).view(-1, 1)
    z = (1 - alphas) * z0 + alphas * z1
    rows = [G(z, torch.full((10,), d, device=device)).cpu() for d in (3, 7)]
plt.figure(figsize=(10, 2.2))
plt.imshow(torchvision.utils.make_grid(torch.cat(rows), nrow=10, normalize=True).permute(1, 2, 0))
plt.axis("off")
plt.title("Latent interpolation z0 -> z1 for digits 3 and 7 (style changes, identity kept)")
plt.show()

torch.save(G.state_dict(), "cgan_mnist_generator.pt")
print("Generator weights saved to cgan_mnist_generator.pt")
```
