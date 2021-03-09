import torch
import torch.nn as nn
import torch.nn.functional as F
# import onnxruntime
# import onnx
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

from similarity_check.data_process.load_data import load_data_with_tokens, load_data_with_tokens_filter, load_idfs
from similarity_check.data_process.encode_data import encode_data_with_tokens_pos
from similarity_check.data_process.process_idf import process_idf
from similarity_check.feature_process.wordVocabulary import Vocabulary
from similarity_check.feature_process.pinyinVocabulary import PinYinVocabulary
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from similarity_check.utils.tools import loadWordVecfromVec, loadWordVecfromText

from torch.autograd import Variable
import pickle
import time
import random
import json
import math


class Model(nn.Module):
    def __init__(self, hidden_size, batch_size, max_length, embed_dim, embed_dim_pinyin, embed_dim_pos,
                 num_class, vocab_size, pinyin_vocab_size, bidirectional, n_layers, kernel_sizes, conv_dim, dropout, use_cuda):
        super().__init__()
        self.embed_dim = embed_dim
        self.embed_dim_pinyin = embed_dim_pinyin
        self.embed_dim_pos = embed_dim_pos
        self.batch_size = batch_size
        self.num_class = num_class
        self.max_length = max_length
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.embedding_pinyin = nn.Embedding(pinyin_vocab_size, embed_dim)
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.n_directions = 2 if bidirectional else 1
        self.output_size = hidden_size * self.n_directions
        self.use_cuda = use_cuda
        self.dropout = 0 if n_layers == 1 else dropout
        self.kernel_sizes = kernel_sizes
        self.conv_dim = conv_dim
        
        self.rnn_word = nn.GRU(input_size=embed_dim + embed_dim_pos * 5 + embed_dim_pinyin, hidden_size=hidden_size, num_layers=n_layers,
                               batch_first=True, dropout=self.dropout, bidirectional=bidirectional)
        '''
        self.rnn_pinyin = nn.GRU(input_size=embed_dim_pinyin, hidden_size=hidden_size,
                                 num_layers=n_layers,
                                 batch_first=True, dropout=self.dropout, bidirectional=bidirectional)
        '''
        convs = [nn.Sequential(nn.Conv1d(in_channels=2, out_channels=conv_dim, kernel_size=kernel_size),
                               nn.BatchNorm1d(conv_dim),
                               nn.ReLU(inplace=True),
                               nn.MaxPool1d(kernel_size=(max_length - kernel_size + 1))) for kernel_size in
                 kernel_sizes]
        self.convs = nn.ModuleList(convs)
        # self.similarity_function = torch.nn.CosineSimilarity(dim =-1,eps=1e-6)
        input_pos = torch.tensor([i for i in range(self.max_length)])
        self.eps = 1e-8
        # self.fc = nn.Linear(max_length * 4 + self.embed_dim_pos * self.max_length * 2, num_class)
        self.fc = nn.Linear(2 * len(kernel_sizes) * conv_dim, num_class)
        self.init_weights()
        self.register_buffer('input_pos', input_pos)
        
        
        pe = torch.zeros(self.max_length, self.embed_dim_pos)
        position = torch.arange(0, self.max_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.embed_dim_pos, 2).float() * (-math.log(10000.0) / self.embed_dim_pos))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        #print(pe.shape)
        self.encode_pos = nn.Embedding.from_pretrained(pe)
        #print(self.encode_pos)

        #self.register_buffer('pe', pe)



    def cal_similarity(self, input_1, input_2, input_dim):
        input_tiled_1 = torch.transpose(torch.reshape(input_1.repeat((1, self.max_length, 1)),
                                                      (-1, self.max_length, self.max_length, input_dim)),
                                        1, 2)
        input_tiled_2 = torch.reshape(input_2.repeat((1, self.max_length, 1)),
                                      (-1, self.max_length, self.max_length, input_dim))
        similarity = F.cosine_similarity(input_tiled_1, input_tiled_2, dim=-1, eps=self.eps)
        return similarity

    def forward(self, word_input1, word_input2, pinyin_input1, pinyin_input2, len_1, len_2, 
                input_token_b_1, input_token_e_1,input_token_b_2, input_token_e_2):

        embedded_word_1 = self.embedding(word_input1)
        embedded_word_2 = self.embedding(word_input2)
        embedded_pinyin_1 = self.embedding_pinyin(pinyin_input1)
        embedded_pinyin_2 = self.embedding_pinyin(pinyin_input2)
        # print(self.input_pos)
        input_pos_1 = self.input_pos.repeat((word_input1.size()[0], 1))
        input_pos_2 = self.input_pos.repeat((word_input2.size()[0], 1))

        embeded_input_1_pos = self.encode_pos(input_pos_1)  # ,  self.embed_dim_pos)
        embeded_input_2_pos = self.encode_pos(input_pos_2)  # ,  self.embed_dim_pos)
        embeded_input_token_b_1 = self.encode_pos(input_token_b_1)  # ,  self.embed_dim_pos)
        embeded_input_token_b_2 = self.encode_pos(input_token_b_2)  # ,  self.embed_dim_pos)
        embeded_input_token_e_1 = self.encode_pos(input_token_e_1)  # ,  self.embed_dim_pos)
        embeded_input_token_e_2 = self.encode_pos(input_token_e_2)  # ,  self.embed_dim_pos)
        # embeded_input_token_e_b_1 = self.encode_pos(input_token_e_1 - input_token_b_1)#,  self.embed_dim_pos)
        # embeded_input_token_e_b_2 = self.encode_pos(input_token_e_2 - input_token_b_2)#,  self.embed_dim_pos)
        embeded_input_token_e_p_1 = self.encode_pos(input_token_e_1 - input_pos_1)  # ,  self.embed_dim_pos)
        embeded_input_token_e_p_2 = self.encode_pos(input_token_e_2 - input_pos_2)  # ,  self.embed_dim_pos)
        embeded_input_token_p_b_1 = self.encode_pos(input_pos_1 - input_token_b_1)  # ,  self.embed_dim_pos)
        embeded_input_token_p_b_2 = self.encode_pos(input_pos_2 - input_token_b_2)  # ,  self.embed_dim_pos)

        embedded_1 = torch.cat([embedded_word_1, embedded_pinyin_1, embeded_input_1_pos,
                                embeded_input_token_b_1, embeded_input_token_e_1, embeded_input_token_e_p_1,
                                embeded_input_token_p_b_1], dim=-1)
        embedded_2 = torch.cat([embedded_word_2, embedded_pinyin_2, embeded_input_2_pos,
                                embeded_input_token_b_2, embeded_input_token_e_2, embeded_input_token_e_p_2,
                                embeded_input_token_p_b_2], dim=-1)


        packed_embeds_1 = pack_padded_sequence(embedded_1,
                                               len_1,
                                               batch_first=True, enforce_sorted=False)
        encoder_output_1, _ = self.rnn_word(packed_embeds_1)
        encoder_output_1, _ = pad_packed_sequence(encoder_output_1, batch_first=True, total_length=self.max_length)
        packed_embeds_2 = pack_padded_sequence(embedded_2,
                                               len_2,
                                               batch_first=True, enforce_sorted=False)
        encoder_output_2, _ = self.rnn_word(packed_embeds_2)
        encoder_output_2, _ = pad_packed_sequence(encoder_output_2, batch_first=True, total_length=self.max_length)

        word_similarity = self.cal_similarity(encoder_output_1, encoder_output_2, self.output_size)
        word_similarity_transpose = torch.transpose(word_similarity, 1, 2)
        max_word_similarity, max_word_similarity_index = torch.max(word_similarity, dim=-1)
        max_word_similarity_transpose, max_word_similarity_index_transpose = torch.max(word_similarity_transpose, dim=-1)
        similar_relative_positions = input_pos_1 - max_word_similarity_index
        similar_relative_positions_transpose = input_pos_2 - max_word_similarity_index_transpose
        #similar_relative_positions = self.gaussion(similar_relative_positions)
        #similar_relative_positions_transpose = self.gaussion(similar_relative_positions_transpose)
        #print(similar_relative_positions.unsqueeze(2).shape)
        #print(max_word_similarity.unsqueeze(2).shape)
        conv_in_1 = torch.cat([max_word_similarity.unsqueeze(2), similar_relative_positions.unsqueeze(2)], dim=-1)
        conv_in_2 = torch.cat([max_word_similarity_transpose.unsqueeze(2), similar_relative_positions_transpose.unsqueeze(2)], dim=-1)

        conv_out_1 = [conv(conv_in_1.permute(0, 2, 1)) for conv in self.convs]
        conv_out_2 = [conv(conv_in_2.permute(0, 2, 1)) for conv in self.convs]
        conv_out_1 = torch.cat(conv_out_1, dim=1)
        conv_out_2 = torch.cat(conv_out_2, dim=1)

        conv_out_reshaped_1 = conv_out_1.view(-1, len(self.kernel_sizes) * self.conv_dim)
        conv_out_reshaped_2 = conv_out_2.view(-1, len(self.kernel_sizes) * self.conv_dim)
        feature = torch.cat([conv_out_reshaped_1,conv_out_reshaped_2],dim=1)

        output = self.fc(feature)
        # print(output.size())
        return F.softmax(output, dim=-1)

    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

    def init_hidden(self, batch_size):
        hidden = torch.zeros(batch_size, self.n_layers * self.n_directions,
                             self.hidden_size, requires_grad=True)
        # hidden.data.normal_(std=0.01)
        # if self.use_cuda:
        # hidden = hidden.cuda()
        return hidden


def generate_batch(batch):
    # print(batch[:3])
    label = torch.tensor([entry[4] for entry in batch])
    word_input1 = torch.tensor([entry[0] for entry in batch])
    word_input2 = torch.tensor([entry[1] for entry in batch])
    pinyin_input1 = torch.tensor([entry[2] for entry in batch])
    pinyin_input2 = torch.tensor([entry[3] for entry in batch])
    len_1 = torch.tensor([entry[5] for entry in batch])
    len_2 = torch.tensor([entry[6] for entry in batch])
    input_token_b_1 = torch.tensor([entry[7] for entry in batch])
    input_token_e_1 = torch.tensor([entry[8] for entry in batch])
    input_token_b_2 = torch.tensor([entry[9] for entry in batch])
    input_token_e_2 = torch.tensor([entry[10] for entry in batch])

    return word_input1, word_input2, pinyin_input1, pinyin_input2, len_1, len_2, input_token_b_1, input_token_e_1, \
           input_token_b_2, input_token_e_2,label


def train_func(train_data, model):
    # Train the model
    model.train()
    torch.autograd.set_detect_anomaly(True)
    train_loss = 0
    train_acc = 0
    data = DataLoader(train_data, batch_size=batch_size, shuffle=True,
                      collate_fn=generate_batch)
    for i, (word_input1, word_input2, pinyin_input1, pinyin_input2, len_1, len_2,\
        input_token_b_1, input_token_e_1,input_token_b_2, input_token_e_2,label ) in enumerate(data):
        optimizer.zero_grad()
        # print('text1',text1.size())
        word_input1, word_input2, pinyin_input1, pinyin_input2, len_1, len_2,\
        input_token_b_1, input_token_e_1, input_token_b_2, input_token_e_2, label = word_input1.to(device), word_input2.to(
            device), \
                                                                             pinyin_input1.to(device), pinyin_input2.to(
            device), \
                                                                             len_1.to(device), len_2.to(device), \
                                                                             input_token_b_1.to(
                                                                                 device), input_token_e_1.to(device), \
                                                                             input_token_b_2.to(
                                                                                 device), input_token_e_2.to(device),\
                label.to(device)
        output = model(word_input1, word_input2, pinyin_input1, pinyin_input2, len_1, len_2,
                           input_token_b_1, input_token_e_1, input_token_b_2, input_token_e_2)
        # print(output)
        # print(label)
        loss = criterion(output, label)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
        train_acc += (output.argmax(1) == label).sum().item()

    # Adjust the learning rate
    # scheduler.step()

    return train_loss / len(train_data), train_acc / len(train_data)


def eval(data_, model):
    model.eval()
    loss = 0
    acc = 0
    data = DataLoader(data_, batch_size=batch_size, collate_fn=generate_batch)
    for word_input1, word_input2, pinyin_input1, pinyin_input2, len_1, len_2,\
        input_token_b_1, input_token_e_1,input_token_b_2, input_token_e_2, label in data:
        word_input1, word_input2, pinyin_input1, pinyin_input2, len_1, len_2,\
        input_token_b_1, input_token_e_1, input_token_b_2, input_token_e_2,label = word_input1.to(device), word_input2.to(
            device), \
                                                                             pinyin_input1.to(device), pinyin_input2.to(
            device), \
                                                                             len_1.to(device), len_2.to(device), \
                                                                             input_token_b_1.to(
                                                                                 device), input_token_e_1.to(device), \
                                                                             input_token_b_2.to(
                                                                                 device), input_token_e_2.to(device),\
        label.to(device)
        with torch.no_grad():
            output = model(word_input1, word_input2, pinyin_input1, pinyin_input2, len_1, len_2,
                           input_token_b_1, input_token_e_1, input_token_b_2, input_token_e_2)
            loss = criterion(output, label)
            loss += loss.item()
            acc += (output.argmax(1) == label).sum().item()

    return loss / len(data_), acc / len(data_)


def test(data_, model):
    model.eval()
    loss = 0
    acc = 0
    data = DataLoader(data_, batch_size=batch_size, collate_fn=generate_batch)
    preds = []
    labels = []
    for word_input1, word_input2, pinyin_input1, pinyin_input2, len_1, len_2,\
        input_token_b_1, input_token_e_1,input_token_b_2, input_token_e_2, label in data:
        word_input1, word_input2, pinyin_input1, pinyin_input2, len_1, len_2,\
        input_token_b_1, input_token_e_1, input_token_b_2, input_token_e_2,label = word_input1.to(device), word_input2.to(device),\
                                                                            pinyin_input1.to(device), pinyin_input2.to(device),\
                                                                            len_1.to(device), len_2.to(device),\
                                                                            input_token_b_1.to(device), input_token_e_1.to(device),\
                                                                            input_token_b_2.to(device), input_token_e_2.to(device),label.to(device)
        with torch.no_grad():
            output = model(word_input1, word_input2, pinyin_input1, pinyin_input2, len_1, len_2,
                           input_token_b_1, input_token_e_1, input_token_b_2, input_token_e_2)
            loss = criterion(output, label)
            loss += loss.item()
            pred = output.argmax(1)
            acc += (pred == label).sum().item()
            # print(pred, label)
            preds.extend(pred.data.tolist())
            labels.extend(label.data.tolist())
    p, r, f, _ = precision_recall_fscore_support(labels, preds, labels=[0, 1], average='macro')
    print("test results")
    print(p, r, f)

    return loss / len(data_), acc / len(data_)


def predict(data, model):
    model.eval()
    with torch.no_grad():
        text = torch.tensor(data[0])
        output = model(text, torch.tensor([0]))
        return output.argmax(1).item()


if __name__ == '__main__':
    config = json.load(open("./model_configs.json", 'r'))

    print(torch.cuda.is_available())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    max_length = config['max_length']
    embed_dim = config['embed_dim']
    embed_dim_pinyin = config['embed_dim_pinyin']
    embed_dim_pos = config['embed_dim_pos']
    kernel_sizes = config['kernel_sizes']
    conv_dim = config['conv_dim']
    hidden_size = config['hidden_size']
    num_class = config['num_class']
    N_EPOCHS = config['N_EPOCHS']
    min_epochs = config['min_epochs']
    batch_size = config['batch_size']
    bidirectional = config['bidirectional']
    n_layers = config['n_layers']
    dropout = config['dropout']
    dir_path = config['dir_path']
    p_data_file_name = config['p_data_file_name']
    n_data_file_name = config['n_data_file_name']
    saved_config_path = config["saved_config_path"]
    num_of_data = config['num_of_data']
    json.dump(config, open(saved_config_path, "w"))

    p_data_file = dir_path + p_data_file_name
    n_data_file_1 = dir_path + n_data_file_name
    p_data = load_data_with_tokens_filter(p_data_file, max_length)
    print('p_data', len(p_data))
    n_data_1 = load_data_with_tokens_filter(n_data_file_1, max_length)
    print('n_data', len(n_data_1))

    random.seed(19)
    #balance dataset
    random.shuffle(p_data)
    random.shuffle(n_data_1)
    p_data = p_data[:num_of_data]
    n_data_1 = n_data_1[:num_of_data]

    saved_model_path = config['saved_model_path']
    word_vocabulary_path = config['word_vocabulary_path']
    pinyin_vocabulary_path = config['pinyin_vocabulary_path']

    use_cuda = torch.cuda.is_available()
    # all_data = build_data(raw_data)
    all_data = p_data + n_data_1  # + n_data_2[:len(p_data) - len(n_data_1)]
    idfs = process_idf(all_data)
    all_data = load_idfs(all_data, idfs)
    vocabulary = Vocabulary()
    vocabulary.buildVocabulary(all_data)
    vocab_size = vocabulary.size

    pingyin_vocabulary = PinYinVocabulary()
    pingyin_vocabulary.buildVocabulary(all_data)
    pinyin_vocab_size = pingyin_vocabulary.size
    '''
    pre_trained_word_vectors_path = config['pre_trained_word_vectors_path']

    if pre_trained_word_vectors_path == '':
        pretrained_word_vectors = None
    elif pre_trained_word_vectors_path.endswith('.bin'):
        pretrained_word_vectors = gensim.models.KeyedVectors.load_word2vec_format(
            os.path.join(os.path.dirname(__file__), pre_trained_word_vectors_path), binary=True)
    elif pre_trained_word_vectors_path.endswith('.txt'):
        pretrained_word_vectors = loadWordVecfromText(pre_trained_word_vectors_path, embed_dim)
    elif pre_trained_word_vectors_path.endswith('.vec'):
        pretrained_word_vectors = loadWordVecfromVec(pre_trained_word_vectors_path)
    else:
        raise Exception('loading word embeddings with unknown format')

    pre_trained_pinyin_vectors_path = config['pre_trained_pinyin_vectors_path']
    if pre_trained_pinyin_vectors_path == '':
        pretrained_pinyin_vectors = None
    elif pre_trained_pinyin_vectors_path.endswith('.bin'):
        pretrained_word_vectors = gensim.models.KeyedVectors.load_word2vec_format(
            os.path.join(os.path.dirname(__file__), pre_trained_pinyin_vectors_path), binary=True)
    elif pre_trained_pinyin_vectors_path.endswith('.txt'):
        pretrained_pinyin_vectors = loadWordVecfromText(pre_trained_pinyin_vectors_path, embed_dim_pinyin)
    elif pre_trained_pinyin_vectors_path.endswith('.vec'):
        pretrained_pinyin_vectors = loadWordVecfromVec(pre_trained_pinyin_vectors_path)
    else:
        raise Exception('loading word embeddings with unknown format')

    vocabulary.loadWordVectors(pretrained_word_vectors, embed_dim)
    
    pingyin_vocabulary.loadWordVectors(pretrained_pinyin_vectors, embed_dim_pinyin)

    word_vectors = vocabulary.vectors
    pinyin_vectors = pingyin_vocabulary.vectors
    '''
    word_vectors = None
    pinyin_vectors = None
    model = Model(hidden_size, batch_size, max_length, embed_dim, embed_dim_pinyin, embed_dim_pos,num_class,
                  vocab_size, pinyin_vocab_size, bidirectional, n_layers, kernel_sizes, conv_dim, dropout, use_cuda).to(device)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0,
                                 amsgrad=False)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.9)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 15)
    encoded_all_data = encode_data_with_tokens_pos(all_data, vocabulary.word2idx,
                                                                  pingyin_vocabulary.word2idx,
                                                                  None, max_length)
    encoded_all_data = encoded_all_data
    random.shuffle(encoded_all_data)
    split_ratio = 0.8
    num_data = len(encoded_all_data)
    train_data, test_data = encoded_all_data[:int(num_data * split_ratio)], encoded_all_data[
                                                                            int(num_data * split_ratio):]
    num_train_data = len(train_data)
    train_data, valid_data = train_data[:int(num_train_data * split_ratio)], train_data[
                                                                             int(num_train_data * split_ratio):]

    best_valid_acc = 0
    early_stop = 0
    for epoch in range(N_EPOCHS):

        start_time = time.time()
        train_loss, train_acc = train_func(train_data, model)
        valid_loss, valid_acc = eval(valid_data, model)
        early_stop += 1
        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            torch.save(model.state_dict(), saved_model_path)
            early_stop = 0
        secs = int(time.time() - start_time)
        mins = secs / 60
        secs = secs % 60

        print('train', train_loss, train_acc)
        print('valid', valid_loss, valid_acc)
        if early_stop >= config['early_stop'] and epoch >= min_epochs:
            print("early_stop at %d epoch" % epoch)
            break
    model.eval()
    loss, acc = test(test_data, model)
    print('test', loss, acc)
    device = torch.device("cpu")
    model.to(device)
    torch.save(model.state_dict(), saved_model_path)
    pickle.dump(vocabulary, open(word_vocabulary_path, "wb"))
    pickle.dump(pingyin_vocabulary, open(pinyin_vocabulary_path, "wb"))
