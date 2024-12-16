import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from model.AGCRNCell import AGCRNCell


class TimeAttention(nn.Module):
    def __init__(self,
                 outfea,  # 128
                 d):  # 16
        super(TimeAttention, self).__init__()
        self.qff = nn.Linear(outfea, outfea)
        self.kff = nn.Linear(outfea, outfea)
        self.vff = nn.Linear(outfea, outfea)

        self.ln = nn.LayerNorm(outfea)
        self.lnff = nn.LayerNorm(outfea)

        self.d = d

    def forward(self,
                x):  # (64,12,170,128)
        query = self.qff(x) # (64,12,170,128)
        key = self.kff(x) # (64,12,170,128)
        value = self.vff(x) # (64,12,170,128)

        query = torch.cat(torch.split(query, self.d, -1), 0).permute(0, 2, 1, 3)  # (512,170,12,16)
        key = torch.cat(torch.split(key, self.d, -1), 0).permute(0, 2, 3, 1)# (512,170,12,16)
        value = torch.cat(torch.split(value, self.d, -1), 0).permute(0, 2, 1, 3)# (512,170,12,16)

        A = torch.matmul(query, key) # (512,170,12,12)
        A /= (self.d ** 0.5) # (512,170,12,12)
        A = torch.softmax(A, -1) # (512,170,12,12)

        value = torch.matmul(A, value) # (512,170,12,16)
        value = torch.cat(torch.split(value, x.shape[0], 0), -1).permute(0, 2, 1, 3)  # (64,12,170,128)
        value += x # (64,12,170,128)

        value = self.ln(value) # (64,12,170,128)
        return value


class PGCRN(nn.Module):
    def __init__(self,
                 args,
                 node_num,  # 170
                 dim_in,  # 1
                 dim_out,  # 64
                 cheb_k):  # 2
        super(PGCRN, self).__init__()

        self.node_num = node_num
        self.input_dim = dim_in

        self.dcrnn_cells = AGCRNCell(args, node_num, dim_in, dim_out, cheb_k)

    def forward(self,
                x,  # (64,12,170,1)
                init_state,  # (64,170,64)
                adj_matrix):  # (170,120)
        # shape of x: (B, T, N, D)
        # shape of init_state: (num_layers, B, N, hidden_dim)
        assert x.shape[2] == self.node_num

        # temporal_attention_state  = self.Tat(x)
        seq_length = x.shape[1]  # 12
        current_inputs = x  # (64,12,170,1)
        output_hidden = []


        state = init_state # (64,170,64)
        inner_states = []
        for t in range(seq_length):
            state = self.dcrnn_cells(current_inputs[:, t, :, :], state, adj_matrix)  # (64,170,64)

            inner_states.append(state)  # 最终：(2,64,170,64)
        output_hidden.append(state)  # (64,170,64)
        current_inputs = x + torch.stack(inner_states, dim=1)  # (64,12,170,64)
            # current_inputs = torch.stack(inner_states, dim=1)
        # current_inputs: the outputs of last layer: (B, T, N, hidden_dim)
        # output_hidden: the last state for each layer: (num_layers, B, N, hidden_dim)
        # last_state: (B, N, hidden_dim)
        return current_inputs, output_hidden  # (64,12,170,64),list:2*(64,170,64)

    def init_hidden(self,
                    batch_size):  # 64
        init_states = self.dcrnn_cells.init_hidden_state(batch_size)
        return init_states  # (num_layers, B, N, hidden_dim)

class BiPGCRN(nn.Module):
    def __init__(self,
                 args,
                 node_num,  # 170
                 dim_in,  # 1
                 dim_out,  # 64
                 cheb_k,  # 2
                 num_layers=2 ):  # 2
        super(BiPGCRN, self).__init__()

        self.node_num = node_num
        self.input_dim = dim_in

        self.dim_out = dim_out
        self.PGCRNS = nn.ModuleList()
        self.PGCRNS.append(PGCRN(args, node_num, dim_in, dim_out, cheb_k))
        for _ in range(1, num_layers):
            self.PGCRNS.append(PGCRN(args, node_num,dim_in, dim_out, cheb_k))

    def forward(self,
                x,  # (64,12,170,1)
                adj_matrix,  # (170,120)
                ):
        init_state_R = self.PGCRNS[0].init_hidden(x.shape[0])  # (64,170,64)
        init_state_L = self.PGCRNS[1].init_hidden(x.shape[0])  # (64,170,64)

        # print("adj:", adj.shape)
        h_out = torch.zeros(x.shape[0], x.shape[1], x.shape[2], self.dim_out * 2).to(
            x.device)  # (64,12,170,128)  初始化一个输出（状态）矩阵
        out1 = self.PGCRNS[0](x, init_state_R,adj_matrix,)[0]  # (64,12,170,64)
        out2 = self.PGCRNS[1](torch.flip(x, [1]), init_state_L, adj_matrix)[0]  # (64,12,170,64)

        h_out[:, :, :, :self.dim_out] = out1
        h_out[:, :, :, self.dim_out:] = out2
        return h_out  # (64,12,170,128)

class Chomp1d(nn.Module):
    """
    extra dimension will be added by padding, remove it
    """
    def __init__(self,
                 chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self,
                x):
        return x[:, :, :, :-self.chomp_size].contiguous()


class TemporalConvNet(nn.Module):
    """
    time dilation convolution
    """
    def __init__(self,
                 num_inputs,
                 num_channels,
                 kernel_size=2,
                 dropout=0.2):
        """
        Args:
            num_inputs : channel's number of input data's feature
            num_channels : numbers of data feature tranform channels, the last is the output channel
            kernel_size : using 1d convolution, so the real kernel is (1, kernel_size)
        """
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            padding = (kernel_size - 1) * dilation_size
            self.conv = nn.Conv2d(in_channels, out_channels, (1, kernel_size), dilation=(1, dilation_size), padding=(0, padding))
            self.conv.weight.data.normal_(0, 0.01)
            self.chomp = Chomp1d(padding)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(dropout)

            layers += [nn.Sequential(self.conv, self.chomp, self.relu, self.dropout)]

        self.network = nn.Sequential(*layers)
        self.downsample = nn.Conv2d(num_inputs, num_channels[-1], (1, 1)) if num_inputs != num_channels[-1] else None
        if self.downsample:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self,
                x):
        """
        like ResNet
        Args:
            X : input data of shape (B, N, T, F)
        """
        # permute shape to (B, F, N, T)

        y = x.permute(0, 3, 2, 1)
        y = F.relu(self.network(y) + self.downsample(y) if self.downsample else y)

        y = y.permute(0, 2, 3, 1)
        return y

class PSTCGCN(nn.Module):
    def __init__(self,
                 args):  # 2
        super(PSTCGCN, self).__init__()
        self.adj_matrix = torch.from_numpy(args.adj_matrix).cuda()

        self.BiPGCRN = BiPGCRN(args, args.num_nodes, args.input_dim, args.rnn_units, args.cheb_k, args.num_layers)

        #predictor
        # self.timeAtt = TimeAttention(args.rnn_units * 2, args.at_filter)
        # self.out_emd = nn.Linear(args.rnn_units * 2, args.output_dim)
        
        self.temporal1 = TemporalConvNet(num_inputs=args.rnn_units * 2,
                                         num_channels=args.num_channels)

        self.pred = nn.Sequential(
            nn.Linear(args.horizon * args.rnn_units, args.horizon *args.gat_hiden),
            nn.ReLU(),
            nn.Linear(args.horizon * args.gat_hiden, args.horizon)
        )
        

    def forward(self,
                source,
                targets,# (64,170,1,12)
                teacher_forcing_ratio=0.5):  # (170,170)
        #source: B, T_1, N, D
        #target: B, T_2, N, D
        #supports = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec1.transpose(0,1))), dim=1)
        source = source[:, :, :, :1]  # (64,12,170,1)

        ####################-- BiPGCRN
        output = self.BiPGCRN(source, self.adj_matrix) # (64,12,170,128)  B, T, N, hidden  output_1: torch.Size([25, 12, 170, 32])

        # ##################-- TimeAttention
        # trans = self.timeAtt(output)  # (64,12,170,128)

        # ##############-- nn.Linear
        # out = self.out_emd(trans).transpose(1, 2) # (64,12,170,1)
        # out = out.permute(0,2,1,3) # (64,170,12,1)
        t = self.temporal1(output)  # torch.Size([6, 307, 12, 64])

        x = t.reshape((t.shape[0], t.shape[1], -1))

        x_out = self.pred(x)  # (64,307,12)
        x_out = x_out.unsqueeze(3)# (64,307,12,1)
        x_out = x_out.permute(0, 2, 1, 3)


        return x_out

