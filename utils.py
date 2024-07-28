import numpy as np

def data_masks(all_usr_pois, item_tail): # pois: points of interest.
    us_lens = [len(upois) for upois in all_usr_pois] # the lengths of each pois (input sequence)
    len_max = max(us_lens) # the longest input sequence in the data for padding others
    us_pois = [upois + item_tail * (len_max - le) for upois, le in zip(all_usr_pois, us_lens)] # padding other session sequences
    us_msks = [[1] * le + [0] * (len_max-le) for le in us_lens] # item existance indicator
    
    return us_pois, us_msks, len_max
    
def split_validation(train_set, valid_portion):
    train_set_x, train_set_y = train_set
    n_samples = len(train_set_x)
    sidx = np.arange(n_samples, dtype = 'int32')
    n_train = int(np.round(n_samples * (1. - valid_portion)))
    valid_set_x = [train_set_x[s] for s in sidx[n_train:]]
    valid_set_y = [train_set_y[s] for s in sidx[n_train:]]
    train_set_x = [train_set_x[s] for s in sidx[:n_train]]
    train_set_y = [train_set_y[s] for s in sidx[:n_train]]

    return (train_set_x, train_set_y), (valid_set_x, valid_set_y)

class Data():
    def __init__(self, data, shuffle = False, graph = False):
        inputs = data[0] # input sequence
        inputs, mask, len_max = data_masks(inputs, [0])
        self.inputs = np.asarray(inputs) # tuple -> narrary
        self.mask = np.asarray(mask) # list -> narrary
        self.targets = np.asarray(data[1]) # tuple -> narrary
        self.len_max = len_max
        self.length = len(inputs)
        self.shuffle = shuffle
        self.graph = graph

    def generate_batch(self, batch_size):
        if self.shuffle:
            shuffled_arg = np.arange(self.length) # len(inputs): the # of sessions
            np.random.shuffle(shuffled_arg)  # index shuffle
            self.inputs = self.inputs[shuffled_arg] # input sequence shuffle w.r.t shuffled indices
            self.mask = self.mask[shuffled_arg] # mask sequence shuffle w.r.t shuffled indices
            self.targets = self.targets[shuffled_arg] # target shuffle w.r.t shuffled indices

        n_batch = int(self.length / batch_size)

        if self.length % batch_size != 0: # if there is no remainder
            n_batch += 1 # the batch for remainders
        slices = np.split(np.arange(n_batch * batch_size), n_batch) # indices splits in batchs slices[0]: [1 ... 100], slices[0]: [101 ... 200]...
        slices[-1] = np.arange(self.length - batch_size, self.length) # slice for the remainer sequences: last session index - batchsize ~ last session index

        return slices
    
    def get_slice(self, i): # indices of batchl index is in an array form
        inputs, mask, targets = self.inputs[i], self.mask[i], self.targets[i]
        items, n_node, A, alias_inputs = [], [], [], []

        for u_input in inputs:
            n_node.append(len(np.unique(u_input))) # n_node holds the # of unique items session-wise in a batch
        max_n_node = np.max(n_node) # it is for padding other sessions in the batch |0 + longest session|
        #print('max_n_node: ', max_n_node)    

        for u_input in inputs:
            node = np.unique(u_input) # e.g. [1 2 3 0] (ndarray); unique item set
            items.append(node.tolist() + [0] * (max_n_node - len(node))) # [[1 2 3 0 0 0 0], [1 2 3 4 0 0 0]] (list)
            u_A = np.zeros((max_n_node, max_n_node)) # |unique items the longest session| x |unique items the longest session|

            #print('u_input: ', u_input)
            for i in range(len(u_input)-1):
                if u_input[i + 1] == 0: # the last item before padding
                    break
                u = np.where(node == u_input[i])[0][0] # Where is the i-th item of u_input[i]located in ordered set 'node'? 
                v = np.where(node == u_input[i+1])[0][0]
                # print('node:', node)
                # print('u_input:', u_input)
                # print('u: ', u)
                # print('v: ', v)

                u_A[u][v] = 1 # 1 for item transition i -> i + 1 in u_input; [u][v] = 1
            #print('u_A: ', u_A)    
            u_sum_in = np.sum(u_A, 0) # row-wise sum (incoming degree)
            u_sum_in[np.where(u_sum_in == 0)] = 1 # it is 0/1 instead of 0/0 (underflow)
            u_A_in = np.divide(u_A, u_sum_in) # incoming edge: |3| -> 0.333..., |2| -> 0.5, |1| -> 1
            #print('u_sum_in: ', u_sum_in)
            #print('u_A_in: ', u_A_in)

            u_sum_out = np.sum(u_A, 1) # column-wise sum (outgoing degree)
            u_sum_out[np.where(u_sum_out == 0)] = 1 # it is 0/1 instead of 0/0 (underflow)
            u_A_out = np.divide(u_A, u_sum_out) # outgoing edge: |3| -> 0.333..., |2| -> 0.5, |1| -> 1
            #print('u_A_out: ', u_A_out)

            u_A = np.concatenate([u_A_in, u_A_out]).transpose()

            A.append(u_A)

            alias_inputs.append([np.where(node == i)[0][0] for i in u_input]) # item transition indicator

        return alias_inputs, A, items, mask, targets








