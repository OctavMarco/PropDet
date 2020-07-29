# -*- coding: utf-8 -*-


import pickle
import pathlib
from pathlib import Path
from typing import List, Tuple, Dict
import numpy as np
import torch
import torch.nn as nn
from torch.optim import SGD, Adam
from torch.utils.data import Dataset, DataLoader
from torchtext.data import get_tokenizer
from matplotlib import pyplot as plt

"""### **Preprocesare**"""


def read_data(directory):
    ids = []
    texts = []
    labels = []
    for f in directory.glob('*.txt'):
        id = f.name.replace('article', '').replace('.txt', '')
        ids.append(id)
        texts.append(f.read_text('utf8'))
        labels.append(parse_label(f.as_posix().replace('.txt', '.labels.tsv')))
    # labels can be empty
    return ids, texts, labels


def parse_label(label_path):
    labels = []
    f = Path(label_path)

    if not f.exists():
        return labels

    for line in open(label_path):
        parts = line.strip().split('\t')
        labels.append([int(parts[2]), int(parts[3]), parts[1], 0, 0])
    labels = sorted(labels)

    if labels:
        length = max([label[1] for label in labels])
        visit = np.zeros(length)
        res = []
        for label in labels:
            if sum(visit[label[0]:label[1]]):
                label[3] = 1
            else:
                visit[label[0]:label[1]] = 1
            res.append(label)
        return res
    else:
        return labels


def clean_text(articles, ids):
    texts = []
    for article, id in zip(articles, ids):
        sentences = article.split('\n')
        end = -1
        res = []
        for sentence in sentences:
            start = end + 1
            end = start + len(sentence)  # length of sequence
            if sentence != "":  # if not empty line
                res.append([id, sentence, start, end])
        texts.append(res)
    return texts


def make_dataset(texts, lbls):
    txt = []
    lbl = []
    for text, label in zip(texts, lbls):
        for Text in text:
            txt.append(Text[1])
            k = 0
            for l in label:
                if Text[2] < l[0] < Text[3]:
                    lbl.append(1)
                    k = 1
                    break
                elif Text[2] < l[1] < Text[3]:
                    lbl.append(1)
                    k = 1
                    break
            if k == 0:
                lbl.append(0)
    return txt, lbl


directory = pathlib.Path('data/protechn_corpus_eval/train')
ids, texts,lbl = read_data(directory)

ids_train = ids
texts_train = texts
lbl_train = lbl
directory = pathlib.Path('data/protechn_corpus_eval/test')
ids_test, texts_test,lbl_test = read_data(directory)
directory = pathlib.Path('data/protechn_corpus_eval/dev')
ids_dev, texts_dev,lbl_dev = read_data(directory)

txt_train = clean_text(texts_train, ids_train)
txt_test = clean_text(texts_test, ids_test)
txt_dev =clean_text(texts_dev, ids_dev)

train_txt, train_lbl = make_dataset(txt_train, lbl_train)
test_txt, test_lbl = make_dataset(txt_test, lbl_test)
dev_txt, dev_lbl = make_dataset(txt_dev, lbl_dev)

pickle.dump([dev_txt,dev_lbl], open("savedata/dev.txt", "wb"))
pickle.dump([test_txt,test_lbl], open("savedata/test.txt", "wb"))
pickle.dump([train_txt,train_lbl], open("savedata/train.txt", "wb"))

train_txt, train_lbl = pickle.load(open("savedata/train.txt", "rb"))
test_txt, test_lbl = pickle.load(open("savedata/test.txt", "rb"))
dev_txt, dev_lbl = pickle.load(open("savedata/dev.txt", "rb"))

"""### **Dataset+ data_loader**"""


class Vocabulary:
    """
    Helper class that maps words to unique indices and the other way around
    """

    def __init__(self, tokens: List[str]):
        # dictionary that maps words to indices
        self.word_to_idx = {'<PAD>': 0}

        for idx, tok in enumerate(tokens, 1):
            self.word_to_idx[tok] = idx

        # dictionary that maps indices to words
        self.idx_to_word = {}
        for tok, idx in self.word_to_idx.items():
            self.idx_to_word[idx] = tok

    def get_token_at_index(self, idx: int):
        return self.idx_to_word[idx]

    def get_index_of_token(self, token: str):
        return self.word_to_idx[token]

    def size(self):
        return len(self.word_to_idx)


class PropagandaDataset(Dataset):
    def __init__(self,
                 fold: str,
                 examples: List[str],
                 labels: List[int],
                 vocab: Vocabulary):
        """
        :type vocab: object
        :param fold: 'train'/'eval'/'test'
        :param examples: List of sentences/paragraphs
        :param labels: List of labels (1 if propaganda, 0 otherwise)
        """
        self.fold = fold
        self.examples = examples
        self.labels = labels
        self.vocab = vocab

    def __getitem__(self, index: int) -> (torch.Tensor, torch.Tensor):
        """
        This function converts an example to a Tensor containing the indices

        :param index: position of example to be retrieved.
        """
        # retrieve sentence and label (correct class index)
        example, label = self.examples[index], self.labels[index]

        # tokenize sentence into words and other symbols
        tokenizer = get_tokenizer("spacy")
        tokens = tokenizer(example)

        # convert tokens to their corresponding indices, according to
        # vocabulary
        token_indices = []
        for i in tokens:
            token_indices.append(self.vocab.get_index_of_token(i))

        return torch.LongTensor(token_indices), torch.LongTensor(label)

    def __len__(self):
        """
        Return the size of this dataset. This is given by the number
        of sentences.
        """
        return len(self.examples)


def collate_sentences(batch: List[Tuple]):
    """
    This function converts a list of batch_size examples to
    a Tensor of size batch_size x max_len
    batch: [(example_1_tensor, example_1_label),
             ...
            (example_batch_size_tensor, example_batch_size_label)]
    """
    # fill this list with all the labels in the batch
    batch_labels = []

    # we need to find the maximum length of a sentence in this batch
    max_len = 0
    for i in batch:
        if len(i[0]) > max_len:
            max_len = len(i[0])
    batch_size = len(batch)

    # print('batch size',batch_size)
    # initialize a Tensor filled with zeros (aka index of <PAD>)
    batch_sentences = torch.LongTensor(batch_size, max_len).fill_(0)

    # fill each row idx in batch_sentences with the corresponding
    # sequence tensor
    #
    # ... batch_sentences[idx, ...] = ...
    for idx in range(0, batch_size):
        # print(idx)
        # print(len(batch[idx][0]))
        # print(len(batch_sentences[idx]))
        batch_sentences[idx][0:len(batch[idx][0])] = batch[idx][0]
        print(batch[idx])
        batch_labels.append(batch[idx][1])
        # print(batch_sentences[idx])
    print(type(batch_labels))
    # batch_labels = [torch.LongTensor(x) for x in batch_labels]
    batch_labels = torch.tensor(batch_labels)
    # print(batch_labels)
    return batch_sentences, batch_labels


def fill_vocab(txt: List[Tuple]):
    tokenizer = get_tokenizer("spacy")
    list_v = []
    for i in txt:
        tok = tokenizer(i)
        for j in tok:
            if list_v.count(j) == 0:
                list_v.append(j)
    vocab = Vocabulary(tokens=list_v)
    return vocab

full_text = train_txt + dev_txt
vocab = fill_vocab(full_text)

test_vocab = fill_vocab(test_txt)
train_vocab = fill_vocab(train_txt)
dev_vocab = fill_vocab(dev_txt)

pickle.dump(dev_vocab, open("savedata/dev_vocab.txt", "wb"))
pickle.dump(test_vocab, open("savedata/test_vocab.txt", "wb"))
pickle.dump(train_vocab, open("savedata/train_vocab.txt", "wb"))

pickle.dump(vocab, open("savedata/vocab.txt", "wb"))

dev_vocab = pickle.load(open("savedata/dev_vocab.txt","rb"))
test_vocab = pickle.load(open("savedata/test_vocab.txt","rb"))
train_vocab = pickle.load(open("savedata/train_vocab.txt","rb"))

vocab = pickle.load(open("savedata/vocab.txt", "rb"))

dataset_train = PropagandaDataset('train', train_txt,  train_lbl,  train_vocab)
train_loader = DataLoader(dataset_train, batch_size=16, collate_fn=collate_sentences)

dataset_test = PropagandaDataset('train', test_txt,  test_lbl,  test_vocab)
test_loader = DataLoader(dataset_test, batch_size=16, collate_fn=collate_sentences)

dataset_dev = PropagandaDataset('train', dev_txt,  dev_lbl,  dev_vocab)
dev_loader = DataLoader(dataset_dev, batch_size=16, collate_fn=collate_sentences)

pickle.dump(train_loader, open("savedata/train_loaded.txt", "wb"))
pickle.dump(test_loader, open("savedata/test_loaded.txt", "wb"))
pickle.dump(dev_loader, open("savedata/dev_loaded.txt", "wb"))

train_loader = pickle.load(open("savedata/train_loaded.txt", "rb"))
test_loader = pickle.load(open("savedata/test_loaded.txt", "rb"))
dev_loader = pickle.load(open("savedata/dev_loaded.txt", "rb"))

"""### model"""

############################## PARAMETERS ######################################
_hyperparameters_dict = {
    "batch_size": 64,
    "num_epochs": 10,  # 10,
    "max_len": 250,
    "embedding_size": 128,  # 256,
    "rnn_size": 256,  # 1024,
    "learning_algo": "adam",
    "learning_rate": 0.001,
    "max_grad_norm": 5.0
}


class RNN(nn.Module):
    def __init__(self, vocab_size: int, char_embedding_size: int,
                 rnn_size: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.char_embedding_size = char_embedding_size
        self.rnn_size = rnn_size
        self.dropout = nn.Dropout(p=0.3)
        # instantiate Modules with the correct arguments
        self.embedding = nn.Embedding(num_embeddings=vocab_size,
                                      embedding_dim=char_embedding_size)
        self.rnn = nn.LSTM(input_size=char_embedding_size,
                           hidden_size=rnn_size, bidirectional=True)

        # self.rnn_cell = nn.GRUCell(input_size = char_embedding_size,
        #                            hidden_size = rnn_size)
        self.logits = nn.Linear(in_features=2 * rnn_size, out_features=2)
        # self.softmax = nn.Softmax(dim = 2)

        self.loss = nn.CrossEntropyLoss()

    def get_loss(self, logits: torch.FloatTensor, y: torch.FloatTensor):
        """
        Computes loss for a batch of sequences. The sequence loss is the
        average of the individual losses at each timestep. The batch loss is
        the average of sequence losses across all batches.

        :param logits: unnormalized probabilities for T timesteps, size
                       batch_size x max_timesteps x vocab_size
        :param y: ground truth values (index of correct characters), size
                  batch_size x max_timesteps
        :returns: loss as a scalar
        """
        #
        # logits: B x T x vocab_size
        #        B x T

        # cross entropy: B x vocab_size x T
        #                B x T
        # vision: B x num_classes
        #        B
        return self.loss(logits, y)

    def get_logits(self, hidden_states: torch.FloatTensor,
                   temperature: float = 1.0):
        """
        Computes the unnormalized probabilities from hidden states. Optionally
        divide logits by a temperature, in order to influence predictions at
        test time (https://www.quora.com/What-is-Temperature-in-LSTM)

        :param hidden_states: tensor of size batch_size x timesteps x rnn_size
        :param temperature: coefficient that scales outputs before turning them
        to probabilities. A low temperature (0.1) results in more conservative
        predictions, while a higher temperature (0.9) results in more diverse
        predictions

        :return: tensor of size batch_size x timesteps x vocab_size
        """
        return self.logits(hidden_states) / temperature

    def forward(self, batch: torch.LongTensor,
                hidden_start: torch.FloatTensor = None) -> torch.FloatTensor:
        """
        Computes the hidden states for the current batch (x, y).
        :param x: input of size batch_size x max_len
        :param hidden_start: hidden state at time step t = 0,
                             size batch_size x rnn_size
        :return: hidden states at all timesteps,
                 size batch_size x timesteps x rnn_size
        """

        # max_len = x.size(1)
        # x,label = batch
        # batch_size x max_len x embedding_dim
        x_embedded = self.embedding(batch)
        # x_drop = self.dropout
        x_drop = self.dropout(x_embedded)

        # compute hidden states and logits for each time step
        # hidden_states_list = []
        # prev_hidden = hidden_start
        hidden_state = self.rnn(x_drop)[0]
        # print(hidden_state)
        # print(hidden_state[0].shape)
        # print(hidden_state[1].shape)

        # hidden_state = hidden_state.permute(2,1,0)
        # hidden_state_maxPooled = F.max_pool1d(hidden_state,hidden_state.shape[2])
        # hidden_state_maxPooled = hidden_state.permute(2,1,0)
        hidden_state_pooled, _ = torch.max(hidden_state, dim=1)

        output = self.get_logits(hidden_state_pooled)

        # Loss = self.loss(output, y)

        # hidden_state = softmax(logits(hidden_state))

        # batch_size x max_len x rnn_size
        # hidden_states = torch.stack(hidden_states_list, dim=1)

        return output


# instantiate the RNNLM module
network = RNN(vocab.size(),
              _hyperparameters_dict['embedding_size'],
              _hyperparameters_dict['rnn_size'])

# if torch.cuda.is_available():
#     device = torch.device('cuda:0')
# else:
#     device = torch.device('cpu')

# move network to GPU if available
# network = network.to(device)
# device = torch.device('cpu')
# network = network.to(device)
optimizer = Adam(params=network.parameters(), lr=0.001)

# CHECKPOINT: make sure you understand each parameter size
print("Neural network parameters: ")
for param_name, param in network.named_parameters():
    print("\t" + param_name, " size: ", param.size())

"""# Training/evaluation loop"""


# Commented out IPython magic to ensure Python compatibility.
class Trainer:
    def __init__(self, model: nn.Module,
                 train_data: torch.LongTensor,
                 dev_data: torch.LongTensor,
                 vocab: Vocabulary,
                 hyperparams: Dict):
        self.model = model
        self.train_data = train_data
        self.dev_data = dev_data
        self.vocab = vocab
        # self.device = torch.device('cuda:0')
        if hyperparams['learning_algo'] == 'adam':
            self.optimizer = Adam(params=self.model.parameters(),
                                  lr=hyperparams['learning_rate'])
        else:
            self.optimizer = SGD(params=self.model.parameters(),
                                 lr=hyperparams['learning_rate'])
        self.num_epochs = hyperparams['num_epochs']
        self.max_len = hyperparams['max_len']
        self.batch_size = hyperparams['batch_size']
        self.rnn_size = hyperparams['rnn_size']
        self.max_grad_norm = hyperparams['max_grad_norm']

        # number of characters in training/dev data
        self.train_size = len(train_data)
        self.dev_size = len(dev_data)

        # number of sequences (X, Y) used for training
        self.num_train_examples = \
            self.train_size // (self.batch_size * self.max_len) * self.batch_size

    def train_epoch(self, epoch_num: int) -> float:
        """
        Compute the loss on the training set
        :param epoch_num: number of current epoch
        """
        self.model.train()
        epoch_loss = 0.0
        # hidden_start = torch.zeros(self.batch_size, self.rnn_size)
        # for batch_num, (x, y) in enumerate(make_batches(self.train_data,
        #                                                self.batch_size,
        #                                                self.max_len)):

        for batch_num, batch_tuple in enumerate(self.train_data):
            print('batch: ', batch_num)
            # reset gradients in train epoch
            self.optimizer.zero_grad()
            x = len(batch_tuple[0])
            y = len(batch_tuple[0][0])
            # compute hidden states
            # batch x timesteps x hidden_size
            x, y = batch_tuple
            # x = x.to(self.device)
            # y = y.to(self.device)
            hidden_states = self.model(x)
            # compute unnormalized probabilities
            # batch x timesteps x vocab_size
            # logits = self.model.get_logits(hidden_states)

            # compute loss
            # scalar
            batch_loss = self.model.get_loss(hidden_states, y)
            epoch_loss += batch_loss.item()

            # backpropagation (gradient of loss wrt parameters)
            batch_loss.backward()

            # clip gradients if they get too large
            torch.nn.utils.clip_grad_norm_(list(self.model.parameters()),
                                           self.max_grad_norm)

            # update parameters
            self.optimizer.step()

            # we use a stateful RNN, which means the first hidden state for the
            # next batch is the last hidden state of the current batch
            # hidden_states.detach_()
            # hidden_start = hidden_states[:,-1,:] # add comment
            if batch_num % 100 == 0:
                print("epoch %d, %d/%d examples, batch loss = %f"
                      % (epoch_num, (batch_num + 1) * self.batch_size,
                         self.num_train_examples, batch_loss.item()))
                epoch_loss /= (batch_num + 1)

        return epoch_loss

    def eval_epoch(self, epoch_num: int) -> float:
        """
        Compute the loss on the validation set
        :param epoch_num: number of current epoch
        """
        epoch_loss = 0.0
        # hidden_start = torch.zeros(self.batch_size, self.rnn_size).to(device)
        with torch.no_grad():
            # for batch_num, (x, y) in enumerate(make_batches(self.dev_data,
            #                                                 self.batch_size,
            #                                                 self.max_len)):
            acc = 0;
            for batch_num, batch_tuple in enumerate(self.train_data):
                print('batch: ', batch_num)
                # reset gradients
                # self.optimizer.zero_grad()
                # x = len(batch_tuple[0])
                # y = len(batch_tuple[0][0])
                # batch x timesteps x hidden_size
                x, y = batch_tuple
                # x = x.to(self.device)
                # y = y.to(self.device)
                hidden_states = self.model(x)
                # batch x timesteps x vocab_size
                # logits = self.model.get_logits(hidden_states)

                batch_loss = self.model.get_loss(hidden_states, y)
                epoch_loss += batch_loss.item()
                hidden_states_m = torch.argmax(hidden_states, dim=1)
                acc += sum(hidden_states_m == y).item()
                # we use a stateful RNN, which means the first hidden state for
                # the next batch is the last hidden state of the current batch
                # hidden_states.detach_()
                # hidden_start = hidden_states[:,-1,:]

            epoch_loss /= (batch_num + 1)

        return epoch_loss, acc

    def train(self) -> Dict:
        train_losses, dev_losses, dev_acc = [], [], []
        for epoch in range(self.num_epochs):
            epoch_train_loss = self.train_epoch(epoch)
            epoch_dev_loss, epoch_dev_train = self.eval_epoch(epoch)
            train_losses.append(epoch_train_loss)
            dev_losses.append(epoch_dev_loss)
            dev_acc.append(epoch_dev_train)
        return {"train_losses": train_losses,
                "dev_losses": dev_losses,
                "dev_acc": epoch_dev_train}


def plot_losses(metrics: Dict):
    """
    Plots training/validation losses.
    :param metrics: dictionar
    """
    plt.figure()
    plt.plot(metrics['train_losses'], c='b', label='Train')
    plt.plot(metrics['dev_losses'], c='g', label='Valid')
    plt.ylabel('Loss')
    plt.xlabel('Iteration')
    plt.legend()
    plt.show()


# op= torch.rand(4)
# thx = torch.rand(4)
# thx[0] = op[0]
# t = thx==op
# print(t)
# print(sum(t).item())

# train network for some epoch
trainer = Trainer(network, train_loader, dev_loader, vocab, _hyperparameters_dict)
metrics = trainer.train()

# plot training and validations losses each epoch
plot_losses(metrics)

# for i in train_loader:
#   print(len(i[0][0]))
#   print(len(i[0]))
#   print(i[0])
# x = 1
# while (True)
#     x = 0
