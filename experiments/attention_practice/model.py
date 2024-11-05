import torch
from torch import nn
from torch.nn import Module, Parameter
import datetime
import numpy as np
import math
import torch.nn.functional as F
from tqdm import tqdm

class GNN(Module):
    def __init__(self, hidden_size, step = 1):
        super(GNN, self).__init__()
        self.step = step
        self.hidden_size = hidden_size
        self.input_size = hidden_size * 2
        self.gate_size = 3 * hidden_size
        self.w_ih = Parameter(torch.Tensor(self.gate_size, self.input_size)) # 3d * 2d
        self.w_hh = Parameter(torch.Tensor(self.gate_size, self.hidden_size)) # 3d * d
        self.b_iah = Parameter(torch.Tensor(self.hidden_size)) # d
        self.b_oah = Parameter(torch.Tensor(self.hidden_size)) # d

        self.b_ih = Parameter(torch.Tensor(self.gate_size)) # 3d
        self.b_hh = Parameter(torch.Tensor(self.gate_size)) # 3d

        self.linear_edge_in = nn.Linear(self.hidden_size, self.hidden_size, bias = True) # n * d
        self.linear_edge_out = nn.Linear(self.hidden_size, self.hidden_size, bias = True) # n * d
        self.linear_edge_f = nn.Linear(self.hidden_size, self.hidden_size, bias = True) # n * d

    def GNNCell(self, A, hidden):
        #print('hidden: ', hidden) n x d
        #linear_edge_in = self.linear_edge_in(hidden)
        #print('A[:, :, :A.shape[1]]: ', A[:, :, :A.shape[1]])
        #print('linear_edge_in: ', linear_edge_in)
        #print('self.b_iah: ', self.b_iah)
        
        input_in = torch.matmul(A[:, :, :A.shape[1]], self.linear_edge_in(hidden)) + self.b_iah # b [ (n x n) x (n x d) + (n x d) ]
        #print('input_in: ', input_in)
        input_out = torch.matmul(A[:, :,A.shape[1] : 2* A.shape[1]], self.linear_edge_out(hidden)) + self.b_oah # b [ (n x n) x (n x d) + (n x d) ]

        inputs = torch.cat([input_in, input_out], 2) # b [(n x 2d)], a^t_{s}


        gi = F.linear(inputs, self.w_ih, self.b_ih) # b [(n x 2d) x (2d x 3d) + (n x 3d)] - > b [(n x 3d)]; a^t_{s} * (W_r||W_z||W_o)
        gh = F.linear(hidden, self.w_hh, self.b_hh) # b [(n x d) x (d x 3d) + (n x 3d)] - > b [(n x 3d)]; v^(t-1) * (U_r||U_z||U_o)

        i_r, i_i, i_n = gi.chunk(3, 2) # b[n x d], b[n x d], b[n x d]; W_r * a^t_{s}, W_z * a^t_{s}, W_o * a^t_{s}
        h_r, h_i, h_n = gh.chunk(3, 2) # b[n x d], b[n x d], b[n x d]; U_r * v^(t-1), U_z * v^(t-1), U_o * v^(t-1)
        resetgate = torch.sigmoid(i_r + h_r) # b[n x d]; r^t
        inputgate = torch.sigmoid(i_i + h_i) # b[n x d]; z^t
        newgate = torch.tanh(i_n + resetgate * h_n) # b[n x d]; ~v^t
        hy = newgate + inputgate * (hidden - newgate) # b[n x d]; v^t

        #print('hy: ', hy)
        return hy


    def forward(self, A, hidden):
        for i in range(self.step):
            hidden = self.GNNCell(A, hidden)
            print(f'{i+1}th hidden: {hidden}')
        return hidden
    


class SessionGraph(Module):
    def __init__(self, opt, n_node):
        super(SessionGraph, self).__init__()
        self.hidden_size = opt.hiddenSize
        self.n_node = n_node
        self.batch_size = opt.batchSize
        self.nonhybrid = opt.nonhybrid
        self.embedding = nn.Embedding(self.n_node, self.hidden_size)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=opt.lr, weight_decay = opt.l2)
        self.reset_parameters()

        self.gnn = GNN(self.hidden_size, step=opt.step)
        self.linear_one = nn.Linear(self.hidden_size, self.hidden_size, bias = True) # W1
        self.linear_two = nn.Linear(self.hidden_size, self.hidden_size, bias = True) # W2
        self.linear_three = nn.Linear(self.hidden_size, 1, bias=False) # q
        self.linear_transform = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=True) # W3

        self.loss_function = nn.CrossEntropyLoss()


        # Define learnable parameters
        self.Wq = nn.Linear(self.hidden_size, self.hidden_size)
        self.Wk = nn.Linear(self.hidden_size, self.hidden_size)
        self.Wv = nn.Linear(self.hidden_size, self.hidden_size)

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, inputs, A):
        hidden = self.embedding(inputs)
        # print('before gnn hidden:', hidden)
        #print('before self.embedding.weight:', self.embedding.weight)
        #print('A:', A)
        #print('A.shape:', A.shape)
        #print('A.shape[1]:', A.shape[1])
        #print('A[:, :, :A.shape[1]]:', A[:, :, :A.shape[1]])
        hidden = self.gnn(A, hidden)
        #print('after gnn hidden:', hidden)
        #print('before self.embedding.weight:', self.embedding.weight)

        return hidden
    
    def compute_attention_score(self, seq_hidden, mask):
        # Apply linear transformations
        seq_hidden_transformed = self.Wq(seq_hidden)  # Transform hidden states
        ht = seq_hidden_transformed[torch.arange(mask.shape[0]), torch.sum(mask, 1) - 1]  # Transform last hidden state

        # Compute dot products
        dp = torch.einsum('bnh,bh->bn', seq_hidden_transformed, ht)

        attention_weights = F.softmax(dp, dim=1)
        attention_weights = attention_weights.unsqueeze(-1)  # Shape: (batch_size, sequence_length, 1)

        masked_attention = attention_weights * seq_hidden_transformed * mask.unsqueeze(-1).float()

        # Sum the weighted values
        a = torch.sum(masked_attention, dim=1)

        if not self.nonhybrid:
            a = self.linear_transform(torch.cat([a, ht], 1))  # (b x d); a = sg, ht = sl; sh = W3[sg || sl]

        b = self.embedding.weight[1:]  # (n_nodes - 1) x d (except for the 0 index)
        scores = torch.matmul(a, b.transpose(1, 0))  # (b x d) * (d x (n_nodes-1)) -> (b x (n_nodes - 1)); \hat z

        return scores
    
    def compute_scores(self, seq_hidden, mask, is_last_epoch):
        #print('seq_hidden: ', seq_hidden)
        #print('mask: ', mask)
        #print('torch.sum(mask, 1): ', torch.sum(mask, 1))
        #print('mask.view(mask.shape[0], -1, 1): ', mask.view(mask.shape[0], -1, 1))

        ht = seq_hidden[torch.arange(mask.shape[0]), torch.sum(mask, 1) - 1] # seq_hidden[b, s_length] (b x d); v_n (last item embedding)
        
        q1 = self.linear_one(ht).view(ht.shape[0], 1, ht.shape[1]) # (b x 1 x d); W1 * vn

        q2 = self.linear_one(seq_hidden) # (b x g max x d); W2 * vi
        alpha = self.linear_three(torch.sigmoid(q1 + q2)) # (b x g max x d) -> (b x g max x 1); q*(W1 * vn + W2 * vi); alpha
        a = torch.sum(alpha * seq_hidden * mask.view(mask.shape[0], -1, 1).float(), 1) # sg
                                                                    # (b x d)
                                                                    # = Sigma (alpha (b x g max x 1) 
                                                                    # * vi (b x g max x d)
                                                                    # * (b x g max x 1))

        

        if not self.nonhybrid:
            a = self.linear_transform(torch.cat([a, ht], 1)) # (b x d); a = sg, ht = sl; sh = W3[sg || sl]

        b = self.embedding.weight[1:] # (n_nodes - 1) x d (except for the 0 index); b (item embeddings) is not updated

        scores = torch.matmul(a, b.transpose(1, 0)) # (b x d) * (d x (n_nodes-1)) -> (b x (n_nodes - 1)); \hat z
        #print('a: ' ,a)
        #print('scores: ' , scores)

        #--------------------------------------------------------------------------------

        if is_last_epoch:

            attention_tensor = (alpha * mask.view(mask.shape[0], -1, 1)).float()
            attention_tensor = attention_tensor.view(attention_tensor.shape[0], -1)

            return scores, attention_tensor
        
        #--------------------------------------------------------------------------------
        else:

            return scores, None


def trans_to_cuda(variable):
    if torch.cuda.is_available():
        return variable.cuda()
    else:
        return variable    
    
def trans_to_cpu(variable):
    if torch.cuda.is_available():
        return variable.cpu()
    else:
        return variable
    
    
def forward(model, i, data, is_last_epoch):
    alias_inputs, A, items, mask, targets = data.get_slice(i)
    # alias_inputs: item index transition -> indicator in all
    # A: incoming edge || outgoing edge |max n node| x 2|max n node| in b
    # item: unique item set
    # mask: padded in all
    alias_inputs = trans_to_cuda(torch.Tensor(alias_inputs).long()) 
        # it lies in their type and potentially their device (CPU or GPU):
        # The .long() method converts the tensor to have integer dtype (torch.int64), suitable for indexing and integer operations in PyTorch
    #print('after long, alias_inputs: ', alias_inputs)
    items = trans_to_cuda(torch.Tensor(items).long())
    A = trans_to_cuda(torch.Tensor(A).float())
    mask = trans_to_cuda(torch.Tensor(mask).long())
    targets = trans_to_cuda(torch.Tensor(targets).long())
    hidden = model(items, A) # b [b max x d]
    get = lambda i: hidden[i][alias_inputs[i]] # hidden[0][alias_inputs[0]] = hidden[0][[2, 1, 0, ... ,0]]

    seq_hidden = torch.stack([get(i) for i in torch.arange(len(alias_inputs)).long()]) # b [g max x d]

    #print('seq_hidden: ', seq_hidden)
    #print('seq_hidden.shape: ', seq_hidden.shape)

    score, attention_score = model.compute_scores(seq_hidden, mask, is_last_epoch)

    return targets, score, attention_score

def train_test(model, train_data, test_data, is_last_epoch):
    print('start training: ', datetime.datetime.now())
    model.train()
    total_loss = 0.0
    
    slices = train_data.generate_batch(model.batch_size)
    #print('slices.shape: ', len(slices))
    #print('train_data.len_max: ', train_data.len_max)

    attention_scores = torch.empty((len(slices), model.batch_size, train_data.len_max))
    #for i, j in zip(slices, np.arange(len(slices))): # i: item indices in a batch, j: jth-batch 
    for i, j in tqdm(zip(slices, np.arange(len(slices))), total=len(slices), desc="Training Batches"): # i: item indices in a batch, j: jth-batch 

        model.optimizer.zero_grad() # resets the gradients of the model parameters. 
                                    # This is because the optimizer does not want to mistakenly accumulate gradients from previous iterations.
        targets, scores, attention_score = forward(model, i, train_data, is_last_epoch)

        loss = model.loss_function(scores, targets - 1)

        loss.backward()
        model.optimizer.step()
        total_loss += loss

        if is_last_epoch:
            try:
                #print(f"Attempting to assign attention_score to attention_scores at index {i}")
                #print(f"attention_score shape: {attention_score.shape}")
                #print(f"attention_scores shape: {attention_scores.shape}")
                attention_scores[j] = attention_score
            except IndexError as e:
                print(f"IndexError: {e}")
                print(f"Invalid index: {j}")
                print(f"Attention_scores shape: {attention_scores.shape}")
                print(f"Attention_score shape: {attention_score.shape}")
                raise  # Re-raise the exception after logging

        if j % int(len(slices) / 5 + 1) == 0:
            print('[%d/%d] Loss: %.4f' % (j, len(slices), loss.item()))

    if is_last_epoch:
        print('attention_scores: ', attention_scores)

    print('\tLoss:\t%.3f' % total_loss)

    print('start prdeicting: ', datetime.datetime.now())
    model.eval()

    hit, mrr = [], []

    slices = test_data.generate_batch(model.batch_size)
    #for i in slices: # i: i-th batch indicators [1 2 ... batchsize]
    for i in tqdm(slices, desc="Predicting Batches"): # i: i-th batch indicators [1 2 ... batchsize]
        #targets, scores = forward(model, i, test_data, is_last_epoch) 
        targets, scores, attention_score = forward(model, i, train_data, is_last_epoch)
        sub_scores = scores.topk(20)[1] # it returns topk item indices for every session in a batch; (b x s x k)
        #print('scores.topk(20): ', scores.topk(20)[1])
        sub_scores = trans_to_cpu(sub_scores)

        #print('sub_scores: ', sub_scores)
        #print('targets: ', targets)
        #print('len(b): ',len(b))

        for score, target, mask in zip(sub_scores, targets, test_data.mask):
            #print('score: ', score)
            #print('target: ', target)
            #print('np.isin(target - 1, score): ', np.isin(target - 1, score))
            #print('np.where(target - 1, score)[0]: ', np.where(score == target - 1)[0])
            #print('np.where(target - 1, score)[0][0]: ', np.where(score == target - 1)[0][0])

            hit.append(np.isin(target - 1, score))
            if len(np.where(score == target - 1)[0]) == 0:
                mrr.append(0)
            else:
                mrr.append(1 / np.where(score == target - 1)[0][0] + 1)
                #print('np.where(target - 1, score)[0][0]: ', np.where(score == target - 1)[0][0])

    hit = np.mean(hit) * 100            
    mrr = np.mean(mrr) * 100

    return hit, mrr
                  


        
