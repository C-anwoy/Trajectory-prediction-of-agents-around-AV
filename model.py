import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import numpy as np

from torch.autograd import Variable

import time

device = torch.device("cuda")

### GCN Layer

class GraphConvolution ( nn.Module ):

    def __init__ ( self , in_features , out_features , bias=True ):
        super ( GraphConvolution , self ).__init__ ()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter ( torch.FloatTensor ( in_features , out_features ).cuda () )
        if bias:
            self.bias = Parameter ( torch.FloatTensor ( out_features ).cuda () )
        else:
            self.register_parameter ( 'bias' , None )
        self.reset_parameters ()

    def reset_parameters ( self ):
        stdv = 1. / math.sqrt ( self.weight.size ( 1 ) )
        self.weight.data.uniform_ ( -stdv , stdv )
        if self.bias is not None:
            self.bias.data.uniform_ ( -stdv , stdv )

    def forward ( self , input , adj ):
        support = torch.mm ( input , self.weight )
        output = torch.spmm ( adj , support )
        n, kc, t, v = adj.size()

        #x = x.view(n, self.kernel_size, kc//self.kernel_size, t, v)
        x = torch.einsum('nkctv,nkvw->nctw', (x, A))

        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__ ( self ):
        return self.__class__.__name__ + ' (' \
               + str ( self.in_features ) + ' -> ' \
               + str ( self.out_features ) + ')'


class GCN ( nn.Module ):
    def __init__ ( self , nfeat , nhid , nclass , dropout ):
        super ( GCN , self ).__init__ ()

        self.gc1 = GraphConvolution ( nfeat , nhid )
        self.gc2 = GraphConvolution ( nhid , nclass )
        self.dropout = dropout

    def forward ( self , x , adj ):
        x = F.relu ( self.gc1 ( x , adj ) )
        x = F.dropout ( x , self.dropout , training=self.training )
        x = self.gc2 ( x , adj )
        # return F.log_softmax(x, dim=1)
        return x

### Temporal Convolution Layer

class Temp_Conv(nn.Module):
	def __init__(self,
				 in_channels,
				 out_channels,
				 kernel_size=3,
				 stride=1,
				 dropout=0):
		super().__init__()

		#assert len(kernel_size) == 2
		#assert kernel_size[0] % 2 == 1
		padding = 1
		
		self.tcn = nn.Sequential(
			nn.BatchNorm1d(out_channels),
			nn.ReLU(inplace=False),
			nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=padding),
			nn.Dropout(dropout, inplace=False),
		)

		self.relu = nn.ReLU(inplace=False)

	def forward(self, x):
		#res = self.residual(x)
		#x, A = self.gcn(x, A)
		x = self.tcn(x)
		return self.relu(x)

class TCN(nn.Module):
  def __init__(self, in_channels, hidden_channels, out_channels):
		  super().__init__()

		  self.tcn_block = nn.ModuleList((
			                    Temp_Conv(in_channels, hidden_channels),
                          Temp_Conv(hidden_channels, hidden_channels),
                          Temp_Conv(hidden_channels, out_channels)
	                	))
  def forward(self, x):
    x = x.cpu().detach().numpy()
    x = x.transpose(0, 2, 1)
    x = torch.Tensor(x).cuda()
    for tcl in self.tcn_block:
      x = tcl(x)
    x = x.cpu().detach().numpy()
    x = x.transpose(0, 2, 1)
    x = torch.Tensor(x).cuda()
    return x

### Encoder

class Encoder ( nn.Module ):
    def __init__ ( self , input_size , cell_size , hidden_size ):
        """
        cell_size is the size of cell_state.
        hidden_size is the size of hidden_state, or say the output_state of each step
        """
        super ( Encoder , self ).__init__ ()

        self.cell_size = cell_size
        self.hidden_size = hidden_size
        self.fl = nn.Linear ( input_size + hidden_size , hidden_size )
        self.il = nn.Linear ( input_size + hidden_size , hidden_size )
        self.ol = nn.Linear ( input_size + hidden_size , hidden_size )
        self.Cl = nn.Linear ( input_size + hidden_size , hidden_size )

    def computeDist ( self , x1 , y1 ):
        return np.abs ( x1 - y1 )
        # return sqrt ( pow ( x1 - x2 , 2 ) + pow ( y1 - y2 , 2 ) )

    def computeKNN ( self , curr_dict , ID , k ):
        import heapq
        from operator import itemgetter

        ID_x = curr_dict[ ID ]
        dists = {}
        for j in range ( len ( curr_dict ) ):
            if j != ID:
                dists[ j ] = self.computeDist ( ID_x , curr_dict[ j ] )
        KNN_IDs = dict ( heapq.nsmallest ( k , dists.items () , key=itemgetter ( 1 ) ) )
        neighbors = list ( KNN_IDs.keys () )

        return neighbors
        # return [1,2,3]

    def compute_A ( self , xt ):
        # return Variable(torch.Tensor(np.ones([xt.shape[0],xt.shape[0]])).cuda())
        xt = xt.cpu ().detach ().numpy ()
        A = np.zeros ( [ xt.shape[ 0 ] , xt.shape[ 0 ] ] )
        for i in range ( len ( xt ) ):
            #if xt[ i ] is not None:
            if xt[i][0] and xt[i][1] :
                neighbors = self.computeKNN ( xt , i , 4 )
                for neighbor in neighbors:
                    # if neighbor in labels:
                    # if idx < labels.index ( neighbor ):
                    A[ i ][ neighbor ] = 1
        return Variable ( torch.Tensor ( A ).cuda () )

    def forward ( self , input , Hidden_State , Cell_State ):
        graph = False

        if graph is True:
            gcn_feat = [ ]
            gcn_model = GCN ( nfeat=1 , nhid=16 , nclass=1 , dropout=0.5 )
            for j in range ( input.shape[ 0 ] ):
                features = input[ j , : ]
                gcn_feat.append ( gcn_model ( torch.unsqueeze ( features , dim=1 ) ,
                                              self.compute_A ( features ) ).cpu ().detach ().numpy () )

            input = Parameter ( torch.FloatTensor ( np.asarray ( gcn_feat ) ).cuda () )
            input = torch.squeeze ( input )
        #print(input.shape)
        combined = torch.cat ( (input , Hidden_State) , 1 )
        f = torch.sigmoid ( self.fl ( combined ) )
        i = torch.sigmoid ( self.il ( combined ) )
        o = torch.sigmoid ( self.ol ( combined ) )
        C = torch.tanh ( self.Cl ( combined ) )
        Cell_State = f * Cell_State + i * C
        Hidden_State = o * torch.tanh ( Cell_State )

        return Hidden_State , Cell_State

    def loop ( self , inputs ):
        batch_size = inputs.size ( 0 )
        time_step = inputs.size ( 1 )
        Hidden_State , Cell_State = self.initHidden ( batch_size )
        
        for i in range ( time_step ):
            Hidden_State , Cell_State = self.forward(torch.squeeze(inputs[:, i:i+1,:]), Hidden_State, Cell_State)
        return Hidden_State , Cell_State

    def initHidden ( self , batch_size ):
        Hidden_State = Variable ( torch.zeros ( batch_size , self.hidden_size ).to ( device ) )
        Cell_State = Variable ( torch.zeros ( batch_size , self.hidden_size ).to ( device ) )
        return Hidden_State , Cell_State

### Decoder

class Decoder(nn.Module):
    def __init__(self, stream, input_size , cell_size , hidden_size, batchsize, timestep):
        super(Decoder, self).__init__()
        self.cell_size = cell_size
        self.hidden_size = hidden_size
        self.batch_size = batchsize
        self.time_step = timestep
        self.num_mog_params = 5
        self.sampled_point_size = 2
        self.stream = stream
        self.stream_specific_param = self.num_mog_params
        self.stream_specific_param = input_size if self.stream=='s2' else self.num_mog_params
        self.fl = nn.Linear ( self.stream_specific_param + hidden_size , hidden_size )
        self.il = nn.Linear ( self.stream_specific_param + hidden_size , hidden_size )
        self.ol = nn.Linear ( self.stream_specific_param + hidden_size , hidden_size )
        self.Cl = nn.Linear ( self.stream_specific_param + hidden_size , hidden_size )
        self.linear1 = nn.Linear ( cell_size ,  self.stream_specific_param )
        # self.one_lstm = nn.LSTMCell
        # self.linear2 = nn.Linear ( self.sampled_point_size ,  hidden_size )


    def forward(self, input , Hidden_State , Cell_State):
        graph = False

        if graph is True:
            gcn_feat = [ ]
            gcn_model = GCN ( nfeat=1 , nhid=16 , nclass=1 , dropout=0.5 )
            for j in range ( input.shape[ 0 ] ):
                features = input[ j , : ]
                gcn_feat.append ( gcn_model ( torch.unsqueeze ( features , dim=1 ) ,
                                              self.compute_A ( features ) ).cpu ().detach ().numpy () )

            input = Parameter ( torch.FloatTensor ( np.asarray ( gcn_feat ) ).cuda () )
            input = torch.squeeze ( input )
        # print(input)
        combined = torch.cat ( (input , Hidden_State) , 1 )
        f = torch.sigmoid ( self.fl ( combined ) )
        i = torch.sigmoid ( self.il ( combined ) )
        o = torch.sigmoid ( self.ol ( combined ) )
        C = torch.tanh ( self.Cl ( combined ) )
        Cell_State = f * Cell_State + i * C
        Hidden_State = o * torch.tanh ( Cell_State )

        return Hidden_State , Cell_State

    def loop ( self, hidden_vec_from_encoder ):
        batch_size = self.batch_size
        time_step = self.time_step
        if self.stream =='s2':
            Cell_State, out, stream2_output = self.initHidden()
        else:
            Cell_State , out  = self.initHidden ()
        mu1_all , mu2_all , sigma1_all , sigma2_all , rho_all = self.initMogParams()
        for i in range ( time_step ):
            if i == 0:
                Hidden_State = hidden_vec_from_encoder
            Hidden_State , Cell_State = self.forward( out , Hidden_State , Cell_State )
            # print(Hidden_State.data)
            mog_params = self.linear1(Hidden_State)
            # mog_params = params.narrow ( -1 , 0 , params.size ()[ -1 ] - 1 )
            out = mog_params
            if self.stream == 's2':
                stream2_output[:,i,:] = out
            if self.stream == 's1':
                mu_1 , mu_2 , log_sigma_1 , log_sigma_2 , pre_rho = mog_params.chunk ( 6 , dim=-1 )
                rho = torch.tanh ( pre_rho )
                log_sigma_1 = torch.exp(log_sigma_1)
                log_sigma_2 = torch.exp(log_sigma_2)
                mu1_all[:,i,:] = mu_1
                mu2_all[:,i,:] = mu_2
                sigma1_all[:,i,:] = log_sigma_1
                sigma2_all[:,i,:] = log_sigma_2
                rho_all[:,i,:] = rho
            # print(mu1_all.grad_fn)
            # out = self.sample(mu_1 , mu_2 , log_sigma_1 , log_sigma_2, rho)
        if self.stream == 's1':
            return Hidden_State , Cell_State, mu1_all , mu2_all , sigma1_all , sigma2_all , rho_all
        else:
            return stream2_output, Hidden_State , Cell_State , mu1_all , mu2_all , sigma1_all , sigma2_all , rho_all

    def initHidden(self):
        out = torch.randn(self.batch_size, self.num_mog_params, device=device) if self.stream == 's1' else torch.randn(self.batch_size, self.hidden_size, device=device)
        if self.stream == 's2':
            output =  torch.randn(self.batch_size, self.time_step, self.hidden_size, device=device)
            return torch.randn(self.batch_size, self.hidden_size, device=device), out, output
        else:
            return torch.randn ( self.batch_size , self.hidden_size , device=device ) , out

    def initMogParams(self):
        mu1_all = torch.rand(self.batch_size, self.time_step, 1, device=device) * 1000
        mu2_all = torch.rand(self.batch_size, self.time_step, 1, device=device) * 1000
        sigma1_all = torch.rand(self.batch_size, self.time_step, 1, device=device) * 100
        sigma2_all = torch.rand(self.batch_size, self.time_step, 1, device=device) * 100
        rho_all = torch.randn(self.batch_size, self.time_step, 1, device=device)
        
        return mu1_all, mu2_all, sigma1_all, sigma2_all, rho_all

## Training and evaluation

### Helper functions


import sys
import os
sys.path.append('..')
import time
import torch.utils.data as utils
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import numpy as np
#from models import *
from sklearn.cluster import SpectralClustering , KMeans
import matplotlib.pyplot as plt
from scipy.sparse.linalg import eigs
from torch.autograd import Variable

device = torch.device("cuda")
BATCH_SIZE=128
MU = 5
MODEL_LOC = '/content/drive/MyDrive/Graph LSTM/resources/trained_models/Modified/{}/'


def load_batch(index, size, seq_ID, train_sequence_stream1, pred_sequence_stream_1):
    '''
    to load a batch of data
    :param index: index of the batch
    :param size: size of the batch of data
    :param seq_ID: either train sequence or a pred sequence, give as a str
    :param train_sequence: list of dicts of train sequences
    :param pred_sequence: list of dicts of pred sequences
    :return: Batch of data
    '''

    i = index
    batch_size = size
    start_index = i * batch_size
    stop_index = (i+1) * batch_size

    if stop_index >= len(train_sequence_stream1):
        stop_index = len(train_sequence_stream1)
        start_index = stop_index - batch_size
    if seq_ID == 'train':
        stream1_train_batch = train_sequence_stream1[start_index:stop_index]
        single_batch = stream1_train_batch

    elif seq_ID == 'pred':
        stream1_pred_batch = pred_sequence_stream_1[start_index:stop_index]
        single_batch = stream1_pred_batch
    else:
        single_batch = None
        print('please enter the sequence ID. enter train for train sequence or pred for pred sequence')
    return single_batch


def train_stream1(input_tensor, target_tensor, temporal_conv, encoder, decoder, temporal_conv_optimizer, encoder_optimizer, decoder_optimizer):
    '''
    train stream 1 network
    :param input_tensor: tensor for input data
    :param target_tensor: tensor for ground truth labels/ original data
    :param temporal_conv: temporal convolution layer
    :param encoder: encoder layer
    :param decoder: decoder layer
    :param temporal_conv_optimizer: optimizer for temporal convolution
    :param encoder_optimizer: optimizer for encoder
    :param decoder_optimizer: optimizer for decoder
    :return: loss
    '''

    target_length = target_tensor.size(0)

    intermidiate_tensor = temporal_conv.forward(input_tensor)
    Hidden_State , _ = encoder.loop(intermidiate_tensor)
    _, _, mu_1, mu_2, log_sigma_1, log_sigma_2, rho = decoder.loop(Hidden_State)

    [ batch_size , step_size , fea_size ] = mu_1.size ()
    out = []
    for i in range(batch_size):
          mu1_current = mu_1[ i , : , : ]
          mu2_current = mu_2[ i , : , : ]
          sigma1_current = log_sigma_1[ i , : , : ]
          sigma2_current = log_sigma_2[ i , : , : ]
          rho_current = rho[ i , : , : ]
          out.append(sample(mu1_current , mu2_current , sigma1_current , sigma2_current , rho_current))

    out = np.array(out)
    #out=torch.Tensor(out).cuda()

    temporal_conv_optimizer.zero_grad()
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    
    #loss = -log_likelihood(mu_1, mu_2, log_sigma_1, log_sigma_2, rho, target_tensor)

    #loss = loss if loss >0 else -1*loss

    loss = (torch.Tensor(MSE(out,target_tensor)).cuda()).mean()
    loss=Variable(loss, requires_grad=True)
    #print(loss)
    loss.backward()

    temporal_conv_optimizer.step()
    encoder_optimizer.step()
    decoder_optimizer.step()
    return loss.item()

def save_model(temporal_conv, encoder_stream1, decoder_stream1, data, sufix, loc=MODEL_LOC): 
    '''
    save trained model
    :param encoder_stream1: trained encoder layer
    :param decoder_stream1: trained decoder layer
    :param data: 'APOL'/'LYFT', give as a str
    :param sufix: a suffix to uniquely identify the saved model on a dataset, give as a str
    :return: none
    '''

    torch.save(temporal_conv.state_dict(), loc.format(data) + 'temporal_conv_{}{}.pt'.format(data, sufix))
    torch.save(encoder_stream1.state_dict(), loc.format(data) + 'encoder_stream1_{}{}.pt'.format(data, sufix))
    torch.save(decoder_stream1.state_dict(), loc.format(data) + 'decoder_stream1_{}{}.pt'.format(data, sufix))
    
    print('model saved at {}'.format(loc.format(data)))

def generate(inputs, temporal_conv, encoder, decoder):
    with torch.no_grad():
        intermidiate = temporal_conv.forward(inputs)
        Hidden_State , Cell_State = encoder.loop(intermidiate)
        print(Hidden_State.shape)
        print(Cell_State.shape)
        decoder_hidden , decoder_cell , mu_1 , mu_2 , log_sigma_1 , log_sigma_2 , rho = decoder.loop(Hidden_State)
        [ batch_size , step_size , fea_size ] = mu_1.size ()
        out = []
        for i in range(batch_size):
            mu1_current = mu_1[ i , : , : ]
            mu2_current = mu_2[ i , : , : ]
            sigma1_current = log_sigma_1[ i , : , : ]
            sigma2_current = log_sigma_2[ i , : , : ]
            rho_current = rho[ i , : , : ]
            out.append(sample(mu1_current , mu2_current , sigma1_current , sigma2_current , rho_current))

        return np.array(out)


def log_likelihood(mu_1, mu_2, log_sigma_1, log_sigma_2, rho, y):

    [batch_size, step_size, fea_size] = y.size()

    epoch_loss = 0

    for i in range(step_size):
        mu1_current = mu_1[:,i,:]
        mu2_current = mu_2[:,i,:]
        
        sigma1_current = log_sigma_1[:,i,:]
        sigma2_current = log_sigma_2[:,i,:]
        rho_current = rho[:,i,:]
        y_current = y[:,i,:]
        
        batch_loss = compute_sample_loss(mu1_current, mu2_current, sigma1_current, sigma2_current, rho_current, y_current).sum()
        
        batch_loss = batch_loss/batch_size
        epoch_loss += batch_loss
    return epoch_loss

def compute_sample_loss(mu_1, mu_2,log_sigma_1, log_sigma_2, rho, y):
    const = 1E-20 # to prevent numerical error
    pi_term = torch.Tensor([2*np.pi]).to(device)

    y_1 = y[:,0]
    # y_1 = (y_1-torch.mean(y_1))/y_1.max()
    y_2 = y[:,1]
    # y_2 = (y_2 - torch.mean(y_2))/y_2.max()
    mu_1 = torch.mean(y_1) + (y_1 -torch.mean(mu_1))
    # mu_1 = torch.mean(y_1) + (y_1 -torch.mean(mu_1)) * (torch.std(y_1)/torch.std(mu_1))
    mu_2 = torch.mean(y_2) + (y_2 -torch.mean(mu_2))
    # mu_2 = torch.mean(y_2) + (y_2 -torch.mean(mu_2)) * ((torch.std(y_2))/(torch.std(mu_2)))
    z = ( (y_1 - mu_1)**2/(log_sigma_1**2) + ((y_2 - mu_2)**2/(log_sigma_2**2)) - 2*rho*(y_1-mu_1)*(y_2-mu_2)/((log_sigma_1 *log_sigma_2)) )
    mog_lik2 = torch.exp( (-1*z)/(2*(1-rho**2)) )
    mog_lik1 =  1/(pi_term * log_sigma_1 * log_sigma_2 * (1-rho**2).sqrt() )
    mog_lik = (mog_lik1*(mog_lik2+1e-8)).log()
    return mog_lik

# ====================================== OTHER HELPER FUNCTIONS =========================================

import time
import math


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

def sample(mu_1 , mu_2 , log_sigma_1 , log_sigma_2, rho):

    sample = []
    for i in range(len(mu_1)):
        mu = np.array([mu_1[i][0].item(), mu_2[i][0].item()])
        sigma_1 = log_sigma_1[i][0].item()
        sigma_2 = log_sigma_2[i][0].item()
        c = rho[i][0].item() * sigma_1 * sigma_2
        cov = np.array([[sigma_1**2, c],[c, sigma_2**2]])
        sample.append(np.random.multivariate_normal(mu, cov))

    return sample

def computeDist ( x1 , y1, x2, y2 ):
    return np.sqrt( pow ( x1 - x2 , 2 ) + pow ( y1 - y2 , 2 ) )

def computeKNN ( curr_dict , ID , k ):
    import heapq
    from operator import itemgetter

    ID_x = curr_dict[ ID ][0]
    ID_y = curr_dict[ ID ][1]
    dists = {}
    for j in range ( len ( curr_dict ) ):
        if j != ID:
            dists[ j ] = computeDist ( ID_x , ID_y, curr_dict[ j ][0],curr_dict[ j ][1] )
    KNN_IDs = dict ( heapq.nsmallest ( k , dists.items () , key=itemgetter ( 1 ) ) )
    neighbors = list ( KNN_IDs.keys () )

    return neighbors

def compute_A ( frame ):
    A = np.zeros ( [ frame.shape[ 0 ] , frame.shape[ 0 ] ] )
    for i in range ( len ( frame ) ):
        if frame[ i ] is not None:
            neighbors = computeKNN ( frame , i , 4 )
        for neighbor in neighbors:
            A[ i ][ neighbor ] = 1
    return A


def compute_accuracy_stream1(traindataloader, labeldataloader, temporal_conv, encoder, decoder, n_epochs):
    ade = 0
    fde = 0
    count = 0

    train_raw = traindataloader
    pred_raw = labeldataloader
    
    batch = load_batch(0, BATCH_SIZE, 'pred', train_raw, pred_raw)
    batch_in_form = np.asarray([batch[i]['sequence'] for i in range(BATCH_SIZE)])
    batch_in_form = torch.Tensor(batch_in_form)
    [ batch_size , step_size , fea_size ] = np.shape(batch_in_form)

    print('computing accuracy...')
    for epoch in range(0, n_epochs):
        # Prepare train and test batch
        if epoch % (int(n_epochs/10) + 1) == 0:
            print("{}/{} in computing accuracy...".format(epoch, n_epochs))
        trainbatch = load_batch ( epoch , BATCH_SIZE , 'train' , train_raw , pred_raw)
        trainbatch_in_form = np.asarray([trainbatch[i]['sequence'] for i in range(len(trainbatch))])
        trainbatch_in_form = torch.Tensor ( trainbatch_in_form )

        testbatch = load_batch ( epoch , BATCH_SIZE , 'pred' , train_raw , pred_raw)
        testbatch_in_form = np.asarray([testbatch[i]['sequence'] for i in range(len(trainbatch))])
        testbatch_in_form = torch.Tensor ( testbatch_in_form )

        train = trainbatch_in_form.to(device)
        label = testbatch_in_form.to(device)

        t0 = time.time()

        pred = generate(train, temporal_conv, encoder, decoder)

        print("###TIME TAKEN - {} seconds".format(time.time()-t0))
        print("BATCH SIZE: ", len(trainbatch))
        #print("##LABEL##")
        #print((label[0]-(label.mean(dim=0)))/(label.std(dim=0)))
        
        #print("##PRED##")
        #print(pred[0])
        #print((label[0]-(label[0].mean()))/(label[0].std()))
        mse = MSE(pred, label)
        # print(mse)
        #print("##LABEL##")
        #print(x_n[0])
        #print(y_n[0])
        mse = np.sqrt(mse)
        ade += mse
        fde += mse[-1]
        # count += testbatch_in_form.size()[0]
        count +=1  
    
    ade = ade/count
    fde = fde/count
    print('RMSE: {}'.format(ade))
    print("ADE: {} FDE: {}".format(np.mean(ade), fde))
    return np.mean(ade)

def MSE(y_pred, y_gt, device=device):
    # y_pred = y_pred.numpy()
    y_gt = y_gt.cpu().detach().numpy()
    acc = np.zeros(np.shape(y_pred)[:-1])
    muX = y_pred[:,:,0]
    muY = y_pred[:,:,1]
    x = np.array(y_gt[:,:, 0])
    x = (x-np.mean(x))/x.std()
    y = np.array(y_gt[:,:, 1])
    # muX = np.mean(x) + (x - np.mean(muX)) * (np.std(x))/(np.std(muX))
    # muX = np.mean(x) + (x - np.mean(muX))
    y = (y-np.mean(y))/y.std()
    #print("##PRED##")
    #print(muX[0])
    #print("##LABEL##")
    #print(x[0])
    #plt.scatter(x[0], y[0], c="red")
    #plt.scatter(x[0]+muX[0], y[0]+muY[0])
    #plt.show()
    # muY = np.mean(y) + (y -np.mean(muY)) * (np.std(y))/(np.std(muY))
    # muY = np.mean(y) + (y -np.mean(muY))
    acc = np.power(x-muX, 2) + np.power(y-muY, 2)
    lossVal = np.sum(acc, axis=0)/len(acc)
    return lossVal

### Training and evaluation main functions

def trainIters(n_epochs, train_dataloader, valid_dataloader, data, sufix, print_every=1, plot_every=1000, learning_rate=0.001, save_every=5):
    '''
    for training the model
    :param n_epochs: no. of training epochs
    :param train_dataloader: list of dicts of train sequences
    :param valid_dataloader: list of dicts of val sequences
    :param data: 'APOL'/'LYFT', give as a str
    :param sufix: a suffix to uniquely identify the saved model on a dataset, give as a str
    :return: encoder_stream1, decoder_stream1
    '''

    start = time.time()
    plot_losses_stream1 = []
    val_losses=[]
    val_epochs=[]
  
    num_batches = int(len(train_dataloader)/BATCH_SIZE)
   
    temp_conv = None   
    encoder_stream1 = None 
    decoder_stream1 = None
    
    tempconvloc = os.path.join(MODEL_LOC.format(data), 'temporal_conv_{}{}.pt'.format(data, sufix))
    encoder1loc = os.path.join(MODEL_LOC.format(data), 'encoder_stream1_{}{}.pt'.format(data, sufix))
    decoder1loc = os.path.join(MODEL_LOC.format(data), 'decoder_stream1_{}{}.pt'.format(data, sufix)) 
    
    train_raw = train_dataloader
    pred_raw = valid_dataloader

    DIR = '/content/drive/MyDrive/Graph LSTM/resources/data/{}/'.format(DATA)
        
    f1_obs = open ( DIR + 'stream1_obs_data_val.pkl', 'rb')  # 'r' for reading; can be omitted
    f1_pred = open ( DIR + 'stream1_pred_data_val.pkl', 'rb')  # 'r' for reading; can be omitted

    val_obs = pickle.load ( f1_obs )  # load file content as mydict
    val_pred = pickle.load ( f1_pred )  # load file content as mydict
    
    f1_obs.close()
    f1_pred.close()
   
    # Initialize encoder, decoders for both streams
    batch = load_batch ( 0 , BATCH_SIZE , 'pred' , train_raw , pred_raw)
    batch_in_form = np.asarray ( [ batch[ i ][ 'sequence' ] for i in range ( BATCH_SIZE ) ] )
    batch_in_form = torch.Tensor ( batch_in_form )
    [ batch_size , step_size , fea_size ] = np.shape(batch_in_form)
    input_dim = fea_size
    hidden_dim = fea_size
    output_dim = fea_size
 
    temp_conv = TCN (input_dim , hidden_dim , output_dim ).to ( device )
    encoder_stream1 = Encoder ( input_dim , hidden_dim , output_dim ).to ( device )
    decoder_stream1 = Decoder ( 's1' , input_dim , hidden_dim , output_dim, batch_size, step_size ).to ( device )
    temp_conv_optimizer = optim.Adam(temp_conv.parameters(), lr=learning_rate)
    encoder_stream1_optimizer = optim.RMSprop(encoder_stream1.parameters(), lr=learning_rate)
    decoder_stream1_optimizer = optim.RMSprop(decoder_stream1.parameters(), lr=learning_rate)
    print("loading {}...".format(encoder1loc))
    #encoder_stream1.load_state_dict(torch.load(encoder1loc))
    #encoder_stream1.eval()       
    #decoder_stream1.load_state_dict(torch.load(decoder1loc))        
    #decoder_stream1.eval()
    

    for epoch in range(0, n_epochs):
        #print("epoch: ", epoch)
        print_loss_total_stream1 = 0  # Reset every print_every

        # Prepare train and test batch
        for bch in range(num_batches):
            print('============ {}/{} epoch {}/{} batch ============'.format(epoch, n_epochs, bch, num_batches))
            trainbatch = load_batch ( bch , BATCH_SIZE , 'train' , train_raw , pred_raw)
            trainbatch_in_form = np.asarray([trainbatch[i]['sequence'] for i in range(BATCH_SIZE)])
            trainbatch_in_form = torch.Tensor( trainbatch_in_form ).to(device)

            testbatch = load_batch ( bch , BATCH_SIZE , 'pred' , train_raw , pred_raw)
            testbatch_in_form = np.asarray([testbatch[i]['sequence'] for i in range(BATCH_SIZE)])
            testbatch_in_form =  torch.Tensor(testbatch_in_form ).to(device)
            # for data in train_dataloader:

            input_stream1_tensor = trainbatch_in_form
            batch_agent_ids = [trainbatch[i]['agent_ID'] for i in range(BATCH_SIZE)]
            target_stream1_tensor = testbatch_in_form
            #print(input_stream1_tensor.shape)
            
            loss_stream1 = train_stream1(input_stream1_tensor, target_stream1_tensor, temp_conv, encoder_stream1, decoder_stream1, temp_conv_optimizer, encoder_stream1_optimizer, decoder_stream1_optimizer)
            print('Loss: ' + str(loss_stream1))
            print_loss_total_stream1 += loss_stream1
                # print(loss_stream1)

            # print_loss_avg_stream1 = print_loss_total_stream1 / print_every
            # print_loss_total_stream1 = 0
        print( '========================\n Average loss in the epoch: ', print_loss_total_stream1/num_batches)
        print('========================')
        plot_losses_stream1.append(print_loss_total_stream1/num_batches)
            # print('%s (%d %d%%) %.4f' % (timeSince(start, epoch / n_epochs),epoch, epoch / n_epochs * 100, print_loss_avg_stream1))
       
        if epoch % save_every == 0:
            save_model(temp_conv, encoder_stream1, decoder_stream1, data, sufix)

        if epoch % 10 == 0:
            ade_val=compute_accuracy_stream1(val_obs, val_pred, temp_conv, encoder_stream1, decoder_stream1, n_epochs)
            val_losses.append(ade_val)
            val_epochs.append(epoch)

    print("validation loss: ", val_losses)
    print("validation epoch: ", val_epochs)
    ade_eval=compute_accuracy_stream1(train_dataloader, valid_dataloader, temp_conv, encoder_stream1, decoder_stream1, n_epochs)
    #showPlot(plot_losses)
    x =[i for i in range(1,n_epochs+1)]
    make_training_plot(x, plot_losses_stream1, val_epochs, val_losses, MODEL_LOC.format(data)+'plot_{}{}_'.format(data, sufix))
    save_model(temp_conv, encoder_stream1, decoder_stream1, data, sufix)
    return temp_conv, encoder_stream1, decoder_stream1

def eval(epochs, tr_seq_1, pred_seq_1, data, sufix, learning_rate=1e-3, loc=MODEL_LOC):
    '''
    for training the model
    :param epochs: no. of epochs
    :param tr_seq_1: list of dicts of train sequences
    :param pred_seq_1: list of dicts of pred sequences
    :param data: 'APOL'/'LYFT', give as a str
    :param sufix: a suffix to uniquely identify the saved model on a dataset, give as a str
    :return: 
    '''
    
    temp_conv = None
    encoder_stream1 = None
    decoder_stream1 = None

    tempconvloc = os.path.join(MODEL_LOC.format(data), 'temporal_conv_{}{}.pt'.format(data, sufix))
    encoder1loc = os.path.join(loc.format(data), 'encoder_stream1_{}{}.pt'.format(data, sufix))
    decoder1loc = os.path.join(loc.format(data), 'decoder_stream1_{}{}.pt'.format(data, sufix))

    train_raw = tr_seq_1
    pred_raw = pred_seq_1

    # Initialize encoder, decoders
    batch = load_batch ( 0 , BATCH_SIZE , 'pred' , train_raw , pred_raw)
    #print(batch)
    #print(np.array(batch).shape)
    batch_in_form = np.asarray ( [ batch[ i ][ 'sequence' ] for i in range ( BATCH_SIZE ) ] )
    batch_in_form = torch.Tensor ( batch_in_form )

    [ batch_size , step_size , fea_size ] = np.shape(batch_in_form)
    input_dim = fea_size
    hidden_dim = fea_size
    output_dim = fea_size

    temp_conv = TCN (input_dim , hidden_dim , output_dim ).to ( device )
    encoder_stream1 = Encoder ( input_dim , hidden_dim , output_dim ).to ( device )
    decoder_stream1 = Decoder ( 's1' , input_dim , hidden_dim , output_dim, batch_size, step_size ).to ( device )

    temp_conv_optimizer = optim.Adam(temp_conv.parameters(), lr=learning_rate)
    encoder_stream1_optimizer = optim.RMSprop(encoder_stream1.parameters(), lr=learning_rate)
    decoder_stream1_optimizer = optim.RMSprop(decoder_stream1.parameters(), lr=learning_rate)

    temp_conv.load_state_dict(torch.load(tempconvloc))
    temp_conv.eval() 
    encoder_stream1.load_state_dict(torch.load(encoder1loc))
    encoder_stream1.eval()       
    decoder_stream1.load_state_dict(torch.load(decoder1loc))        
    decoder_stream1.eval()

    compute_accuracy_stream1(tr_seq_1, pred_seq_1, temp_conv, encoder_stream1, decoder_stream1, epochs)

## Main

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import numpy as np
import pandas as pd
import time


import pickle            


DATA = 'LYFT'
SUFIX = '_with_tcn_MSELoss'


device = torch.device("cuda")

TRAIN = False
EVAL = True


DIR = '/content/drive/MyDrive/Graph LSTM/resources/data/{}/'.format(DATA)
MODEL_DIR = '/content/drive/MyDrive/Graph LSTM/resources/trained_models/'

epochs = 100

save_per_epochs = 5

train_seq_len = 20
pred_seq_len = 30

    
if TRAIN:
        
      f1 = open ( DIR + 'stream1_obs_data_train.pkl', 'rb')  # 'r' for reading; can be omitted
      g1 = open ( DIR + 'stream1_pred_data_train.pkl', 'rb')  # 'r' for reading; can be omitted

      tr_seq_1 = pickle.load ( f1 )  # load file content as mydict
      pred_seq_1 = pickle.load ( g1 )  # load file content as mydict

      f1.close()
      g1.close()

      temporal_conv, encoder1, decoder1 = trainIters(epochs, tr_seq_1 , pred_seq_1, DATA, SUFIX, learning_rate=0.002, print_every=1, save_every=save_per_epochs)
    
if EVAL:
    print("================================================")
    print('start evaluating {}{}...'.format(DATA, SUFIX))
  
    f1 = open ( DIR + 'stream1_obs_data_test.pkl', 'rb')  # 'r' for reading; can be omitted
    g1 = open ( DIR + 'stream1_pred_data_test.pkl', 'rb')  # 'r' for reading; can be omitted

    tr_seq_1 = pickle.load ( f1 )  # load file content as mydict
    pred_seq_1 = pickle.load ( g1 )  # load file content as mydict

    f1.close ()
    g1.close ()

    #print(tr_seq_1[10]['sequence'])
    
    print(pred_seq_1[10])
    #X=pred_seq_1[2000]['sequence']

    #t0 = time.time()
    # code

    eval(1, tr_seq_1, pred_seq_1, DATA, SUFIX)

    #print("time - {}".format(time.time()-t0))

