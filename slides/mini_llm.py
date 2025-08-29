#minimal example of how to train a LLM
# of course the vocabulary and corpus is too small to 
# produce intersting result.
# It is just a minimal example of how to set it up in
# PyTorch

import torch
import torch.optim as optim
import torch.nn as nn
tokens = [".", "chat", "le", "mange", "matou",  "mulot"]
#           0     1       2        3         4      5
token2id = {token: i for i, token in enumerate(tokens)}
id2token = {v: k for k, v in token2id.items()}


def sentence_to_indices(sentence):
    words = sentence.split()
    return [token2id[word] for word in words]


class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, sentences):
        self.sentences = sentences

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        input_indices = sentence_to_indices(sentence)
        # shift input by one and append '.' at the end
        output_indices = input_indices[1:] + [token2id['.']]
        return torch.tensor(input_indices), torch.tensor(output_indices)


sentences = [". . le chat mange .", ". . le mulot mange .",
             "le matou mange le mulot .", "le chat mange le mulot ."]
dataset = SimpleDataset(sentences)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)

# example usage:
for batch in dataloader:
    inputs, targets = batch
    print(inputs, targets)

input_dim = 6  # size of the input vocabulary
output_dim = 6  # size of the output vocabulary
embedding_dim = 4  # size of the token embeddings
hidden_dim = 30  # size of the hidden layer in the transformer encoder
num_layers = 2  # number of transformer encoder layers
num_heads = 2  # number of attention heads in each transformer encoder layer
dropout_prob = 0.1  # dropout probability in the transformer encoder


class TransformerModel(nn.Module):
    def __init__(self, input_dim, output_dim, embedding_dim, hidden_dim, num_layers, num_heads, dropout_prob):
        super(TransformerModel, self).__init__()

        # create the input embedding layer
        self.embedding = nn.Embedding(input_dim, embedding_dim)

        # create the transformer encoder layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim, nhead=num_heads, dim_feedforward=hidden_dim, dropout=dropout_prob
        )

        # create the transformer encoder
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers)

        # create the output linear layer
        self.linear = nn.Linear(embedding_dim, output_dim)

    def forward(self, src):
        # embed the input tokens
        embedded = self.embedding(src)

        # apply the transformer encoder
        encoded = self.transformer_encoder(embedded)

        # project the output to the output dimension and return
        output = self.linear(encoded)
        return output


model = TransformerModel(input_dim, output_dim, embedding_dim,
                         hidden_dim, num_layers, num_heads, dropout_prob)


# define the model and optimizer
optimizer = optim.Adam(model.parameters())

# define the loss function
criterion = nn.CrossEntropyLoss()

# define the training loop


def train(model, iterator, optimizer, criterion):
    model.train()
    total_loss = 0
    for batch in iterator:
        optimizer.zero_grad()
        inputs, targets = batch
        outputs = model(inputs)
        loss = criterion(outputs.view(-1, outputs.shape[-1]), targets.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(iterator)


# train the model for a number of epochs
train_iterator = iter(dataloader)
num_epochs = 10
for epoch in range(num_epochs):
    train_loss = train(model, train_iterator, optimizer, criterion)
    print('Epoch {} loss: {:.4f}'.format(epoch+1, train_loss))


def generate_next_word(model, start_sentence):
    # tokenize the starting sentence and convert to indices
    input_indices = sentence_to_indices(start_sentence)
    print("Input indices:", input_indices)
    # add a dummy index at the end to use as a placeholder for the generated output
    input_indices.append(0)
    # convert to a tensor and add a batch dimension
    inputs = torch.tensor(input_indices).unsqueeze(0)
    print("Input sentence:", start_sentence)
    print("Input tensor:", inputs)
    # get the output from the model
    outputs = model(inputs)
    print("Output tensor:", outputs)
    # get the last token from the output
    output_token = outputs[0, -1, :].argmax().item()
    # convert the index to a token
    output_word = id2token[output_token]
    return output_word


# define the test sentence
start_sentence = "le chat mange le "

# generate the next word

for iter in range(1):
    next_word = generate_next_word(model, start_sentence)
    print("Next word:", next_word)

    # add the next word to the sentence
    start_sentence += next_word + " "

print("Generated sentence:", start_sentence)
