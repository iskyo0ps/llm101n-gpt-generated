### 1. Prepare the Data
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Sample text data
text = "This is a sample text for N-gram language modeling using a multi-layer perceptron."
# text = "This is a sample text for N-gram language modeling using a multi-layer perceptron. I love jisoo, she is the best."

# Tokenize text
tokens = text.lower().split()

# Create N-gram sequences
N = 3  # Example: Trigram model
ngrams = [tokens[i:i+N] for i in range(len(tokens)-N+1)]

# Convert tokens to integers
encoder = LabelEncoder()
encoder.fit(tokens)
encoded_ngrams = [encoder.transform(ngram) for ngram in ngrams]

# Split into input (X) and output (y)
X = np.array([ngram[:-1] for ngram in encoded_ngrams])
y = np.array([ngram[-1] for ngram in encoded_ngrams])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

### 2. Build the Model
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class NGramModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(NGramModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.hidden1 = nn.Linear(embedding_dim * (N-1), hidden_dim)
        self.hidden2 = nn.Linear(hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, vocab_size)
        self.gelu = nn.GELU()

    def forward(self, x):
        x = self.embedding(x).view(x.size(0), -1)
        x = self.gelu(self.hidden1(x))
        x = self.gelu(self.hidden2(x))
        x = self.output(x)
        return x

# Hyperparameters
embedding_dim = 10
hidden_dim = 50
vocab_size = len(encoder.classes_)

# Model, loss function, and optimizer
model = NGramModel(vocab_size, embedding_dim, hidden_dim)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


#### 3. Train the Model
# Training loop
epochs = 100
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    
    inputs = torch.LongTensor(X_train)
    targets = torch.LongTensor(y_train)
    
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()
    
    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item()}')

print("Training complete.")



#### 4. Generate Text
# Function to generate text
def generate_text(model, start_text, length):
    model.eval()
    result = start_text.split()
    for _ in range(length):
        input_seq = torch.LongTensor(encoder.transform(result[-(N-1):])).unsqueeze(0)
        output = model(input_seq)
        _, predicted = torch.max(output, 1)
        next_word = encoder.inverse_transform(predicted.cpu().numpy())[0]
        result.append(next_word)
    return ' '.join(result)

# Generate text
start_text = "this is a"
generated_text = generate_text(model, start_text, 10)
print("Generated Text:", generated_text)

# Generate text_1
# start_text_1 = "I love jisoo"
# generated_text_1 = generate_text(model, start_text_1, 10)
# print("Generated Text_1:", generated_text_1)

