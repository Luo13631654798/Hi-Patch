from einops import rearrange, repeat



import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import softmax, add_remaining_self_loops
import math

from torch_scatter import scatter
import torch
from torch_geometric.utils.num_nodes import maybe_num_nodes

def softmax(src, index):
    # norm
    N = maybe_num_nodes(index)

    # out = src
    # out = out.exp()
    # out_sum = scatter(out, index, dim=0, dim_size=N, reduce='sum')[index]
    # local  max
    # local_out = src - scatter(src, index, dim=0, dim_size=N, reduce='max')[index]
    # local_out = local_out.exp()
    # local_out_sum = scatter(local_out, index, dim=0, dim_size=N, reduce='sum')[index]
    #
    # # global  max
    global_out = src - src.max()
    global_out = global_out.exp()
    global_out_sum = scatter(global_out, index, dim=0, dim_size=N, reduce='sum')[index]
    #
    # a = out / (out_sum + 1e-8)
    # b = local_out / (local_out_sum + 1e-16)
    c =  global_out / (global_out_sum + 1e-16)
    # eps = 0.0000001
    # print((torch.abs(a-c)>eps).sum())
    # print((torch.abs(a-b)>eps).sum())
    return c

class Intra_Inter_Graph_Layer(MessagePassing):

    def __init__(self, n_heads=2, d_input=6, d_k=6, alpha=0.9, patch_layer=1, res=1, **kwargs):
        super(Intra_Inter_Graph_Layer, self).__init__(aggr='add', **kwargs)
        self.n_heads = n_heads
        # self.dropout = nn.Dropout(dropout)
        self.patch_layer = patch_layer
        self.res = res
        self.d_input = d_input
        self.d_k = d_k // n_heads
        self.d_q = d_k // n_heads
        self.d_e = d_input // n_heads
        self.d_sqrt = math.sqrt(d_k // n_heads)
        self.alpha = alpha
        # Attention Layer Initialization
        # self.w_k_list = nn.ModuleList([nn.Linear(self.d_input, self.d_k, bias=True) for i in range(self.n_heads)])
        self.w_k_list = nn.ParameterList([nn.Parameter(torch.FloatTensor(patch_layer, 3, self.d_input, self.d_k)) for i in range(self.n_heads)])
        self.bias_k_list = nn.ParameterList([nn.Parameter(torch.FloatTensor(patch_layer, 3, self.d_k)) for i in range(self.n_heads)])
        for param in self.w_k_list:
            nn.init.xavier_uniform_(param)
        for param in self.bias_k_list:
            nn.init.uniform_(param)

        self.w_q_list = nn.ParameterList([nn.Parameter(torch.FloatTensor(patch_layer, 3, self.d_input, self.d_q)) for i in range(self.n_heads)])
        self.bias_q_list = nn.ParameterList([nn.Parameter(torch.FloatTensor(patch_layer, 3, self.d_q)) for i in range(self.n_heads)])
        for param in self.w_q_list:
            nn.init.xavier_uniform_(param)
        for param in self.bias_q_list:
            nn.init.uniform_(param)

        self.w_v_list = nn.ParameterList([nn.Parameter(torch.FloatTensor(patch_layer, 3, self.d_input, self.d_e)) for i in range(self.n_heads)])
        self.bias_v_list = nn.ParameterList([nn.Parameter(torch.FloatTensor(patch_layer, 3, self.d_e)) for i in range(self.n_heads)])
        for param in self.w_v_list:
            nn.init.xavier_uniform_(param)
        for param in self.bias_v_list:
            nn.init.xavier_uniform_(param)

        self.layer_norm = nn.LayerNorm(d_input)


    def LearnableTE(self, tt):
        # tt: (N*M*B, L, 1)
        out1 = self.te_scale(tt)
        out2 = torch.sin(self.te_periodic(tt))
        return torch.cat([out1, out2], -1)
        # Normalization


    def forward(self, x, edge_index, edge_value, time_nodes, edge_same_time_diff_var, edge_diff_time_same_var, edge_diff_time_diff_var, n_layer):
        residual = x
        x = self.layer_norm(x)

        return self.propagate(edge_index, x=x, edges_temporal=edge_value,
                              edge_same_time_diff_var=edge_same_time_diff_var, edge_diff_time_same_var=edge_diff_time_same_var,
                              edge_diff_time_diff_var=edge_diff_time_diff_var,
                              n_layer=n_layer, residual=residual)

    def message(self, x_j, x_i, edge_index_i, edges_temporal, edge_same_time_diff_var, edge_diff_time_same_var, edge_diff_time_diff_var, n_layer):
        '''

           :param x_j: [num_edge, d] sender
           :param x_i: [num_edge,d]  receiver
           :param edge_index_i:  receiver node list [num_edge]
           :param edges_temporal: [num_edge,d]
           :return:
        '''
        messages = []
        for i in range(self.n_heads):
            w_k = self.w_k_list[i][n_layer]
            bias_k = self.bias_k_list[i][n_layer]
            # k_linear_diff = self.w_k_list_diff[i]
            w_q = self.w_q_list[i][n_layer]
            bias_q = self.bias_q_list[i][n_layer]

            w_v = self.w_v_list[i][n_layer]
            bias_v = self.bias_v_list[i][n_layer]


            x_j_transfer = x_j

            attention = self.each_head_attention(x_j_transfer, w_k, bias_k, w_q, bias_q, x_i,
                                                 edge_same_time_diff_var, edge_diff_time_same_var, edge_diff_time_diff_var)  # [4,1]
            attention = torch.div(attention, self.d_sqrt)
            attention = torch.pow(self.alpha, torch.abs(edges_temporal.squeeze())).unsqueeze(-1) * attention
            # attention = attention * edge_same_time_diff_var + attention * edge_diff_time_same_var + attention * edge_diff_time_diff_var * 0.1
            attention_norm = softmax(attention, edge_index_i)

            sender_stdv = edge_same_time_diff_var * (torch.matmul(x_j_transfer, w_v[0]) + bias_v[0])
            sender_dtsv = edge_diff_time_same_var * (torch.matmul(x_j_transfer, w_v[1]) + bias_v[1])
            sender_dtdv = edge_diff_time_diff_var * (torch.matmul(x_j_transfer, w_v[2]) + bias_v[2])
            sender = sender_stdv + sender_dtsv + sender_dtdv
            # sender = x_j_transfer
            # sender_diff = (1 - edge) * v_linear_diff(x_j_transfer)
            # sender = sender

            message = attention_norm * sender  # [4,3]
            messages.append(message)

        message_all_head = torch.cat(messages, 1)

        return message_all_head

    def each_head_attention(self, x_j_transfer, w_k, bias_k, w_q, bias_q, x_i,
                            edge_same_time_diff_var, edge_diff_time_same_var, edge_diff_time_diff_var):
        x_i_0 = edge_same_time_diff_var * (torch.matmul(x_i, w_q[0]) + bias_q[0]) # receiver #[num_edge,d*heads]
        x_i_1 = edge_diff_time_same_var * (torch.matmul(x_i, w_q[1]) + bias_q[1]) # receiver #[num_edge,d*heads]
        x_i_2 = edge_diff_time_diff_var * (torch.matmul(x_i, w_q[2]) + bias_q[2]) # receiver #[num_edge,d*heads]
        x_i = x_i_0 + x_i_1 + x_i_2
        # wraping k

        sender_0 = edge_same_time_diff_var * (torch.matmul(x_j_transfer, w_k[0]) + bias_k[0])
        sender_1 = edge_diff_time_same_var * (torch.matmul(x_j_transfer, w_k[1]) + bias_k[1])
        sender_2 = edge_diff_time_diff_var * (torch.matmul(x_j_transfer, w_k[2]) + bias_k[2])
        sender = sender_0 + sender_1 + sender_2
        # sender_diff = (1 - edge_same) * w_k_diff(x_j_transfer)
        # sender = sender_same + sender_diff  # [num_edge,d]

        # Calculate attention score
        attention = torch.bmm(torch.unsqueeze(sender, 1), torch.unsqueeze(x_i, 2))

        return torch.squeeze(attention, 1)

    def update(self, aggr_out, residual):
        x_new = self.res * residual + F.gelu(aggr_out)
        return x_new
        # return self.dropout(x_new)

    def __repr__(self):
        return '{}'.format(self.__class__.__name__)

class Hi_Patch(nn.Module):
    def __init__(self, args, supports=None):
        super(Hi_Patch, self).__init__()
        d_model = args.hid_dim
        self.device = args.device
        self.hid_dim = args.hid_dim
        self.N = args.ndim
        self.batch_size = None
        self.supports = supports
        self.n_layer = args.nlayer
        self.gcs = nn.ModuleList()
        self.alpha = args.alpha
        self.res = args.res
        self.te_scale = nn.Linear(1, 1)
        self.te_periodic = nn.Linear(1, args.hid_dim - 1)
        self.patch_layer = args.patch_layer
        self.obs_enc = nn.Linear(1, args.hid_dim)
        self.nodevec = nn.Embedding(self.N, d_model)
        self.relu = nn.ReLU()

        for l in range(self.n_layer):
            self.gcs.append(Intra_Inter_Graph_Layer(args.nhead, d_model, d_model, self.alpha, args.patch_layer, self.res))

        self.w_q = nn.Parameter(torch.FloatTensor(d_model, d_model))
        self.w_k = nn.Parameter(torch.FloatTensor(d_model, d_model))
        self.w_v = nn.Parameter(torch.FloatTensor(d_model, d_model))

        nn.init.xavier_uniform_(self.w_q)
        nn.init.xavier_uniform_(self.w_k)
        nn.init.xavier_uniform_(self.w_v)

        if args.task == 'forecasting':
            self.decoder = nn.Sequential(
                nn.Linear(d_model * 2, d_model),
                nn.ReLU(inplace=True),
                nn.Linear(d_model, d_model),
                nn.ReLU(inplace=True),
                nn.Linear(d_model, 1)
            )
        else:
            d_static = args.d_static
            if d_static != 0:
                self.emb = nn.Linear(d_static, args.ndim)
                self.classifier = nn.Sequential(
                    nn.Linear(args.ndim * 2, 200),
                    nn.ReLU(),
                    nn.Linear(200, args.n_class))
            else:
                self.classifier = nn.Sequential(
                    nn.Linear(args.ndim, 200),
                    nn.ReLU(),
                    nn.Linear(200, args.n_class))

    def LearnableTE(self, tt):
        out1 = self.te_scale(tt)
        out2 = torch.sin(self.te_periodic(tt))
        return torch.cat([out1, out2], -1)

    def IMTS_Model(self, x, mask_X, x_time, x_uncertainty):
        B, N, M, L, D = x.shape

        # 创建一个形状为 [N] 的张量，包含变量下标
        variable_indices = torch.arange(N).to(x.device)

        B, N, M, L, D = x.shape

        # 将其扩展成形状为 [1, N, 1, 1, 1]
        cur_variable_indices = variable_indices.view(1, N, 1, 1, 1)

        # 利用广播机制，将其扩展成形状为 [B, N, M, L, D]
        cur_variable_indices = cur_variable_indices.expand(B, N, M, L, 1)

        # 并行式
        cur_x = rearrange(x, 'b n m l c -> (b m n l) c')
        cur_variable_indices = rearrange(cur_variable_indices, 'b n m l c -> (b m n l) c')
        cur_x_time = rearrange(x_time, 'b n m l c -> (b m n l) c')

        # 生成图结构
        cur_mask = rearrange(mask_X, 'b n m l c -> b m (n l) c')
        cur_adj = torch.matmul(cur_mask, cur_mask.permute(0, 1, 3, 2))
        int_max = torch.iinfo(torch.int32).max
        element_count = cur_adj.shape[0] * cur_adj.shape[1] * cur_adj.shape[2] * cur_adj.shape[3]

        if element_count > int_max:
            once_num = int_max // (cur_adj.shape[1] * cur_adj.shape[2] * cur_adj.shape[3])
            sd = 0
            ed = once_num
            total_num = math.ceil(B / once_num)
            for k in range(total_num):
                if k == 0:
                    edge_ind = torch.where(cur_adj[sd:ed] == 1)
                    edge_ind_0 = edge_ind[0]
                    edge_ind_1 = edge_ind[1]
                    edge_ind_2 = edge_ind[2]
                    edge_ind_3 = edge_ind[3]
                elif k == total_num - 1:
                    cur_edge_ind = torch.where(cur_adj[sd:] == 1)
                    edge_ind_0 = torch.cat([edge_ind_0, cur_edge_ind[0] + k * once_num])
                    edge_ind_1 = torch.cat([edge_ind_1, cur_edge_ind[1]])
                    edge_ind_2 = torch.cat([edge_ind_2, cur_edge_ind[2]])
                    edge_ind_3 = torch.cat([edge_ind_3, cur_edge_ind[3]])
                    edge_ind = (edge_ind_0, edge_ind_1, edge_ind_2, edge_ind_3)
                else:
                    cur_edge_ind = torch.where(cur_adj[sd:ed] == 1)
                    edge_ind_0 = torch.cat([edge_ind_0, cur_edge_ind[0] + k * once_num])
                    edge_ind_1 = torch.cat([edge_ind_1, cur_edge_ind[1]])
                    edge_ind_2 = torch.cat([edge_ind_2, cur_edge_ind[2]])
                    edge_ind_3 = torch.cat([edge_ind_3, cur_edge_ind[3]])
                sd += once_num
                ed += once_num

        else:
            edge_ind = torch.where(cur_adj == 1)

        source_nodes = (N * M * L * edge_ind[0] + N * L * edge_ind[1] + edge_ind[2])
        target_nodes = (N * M * L * edge_ind[0] + N * L * edge_ind[1] + edge_ind[3])
        edge_index = torch.cat([source_nodes.unsqueeze(0), target_nodes.unsqueeze(0)])

        edge_time = torch.squeeze(cur_x_time[source_nodes] - cur_x_time[target_nodes])

        edge_diff_time_same_var = ((cur_variable_indices[source_nodes] - cur_variable_indices[target_nodes]) == 0).float()
        edge_same_time_diff_var= ((cur_x_time[source_nodes] - cur_x_time[target_nodes]) == 0).float()
        edge_diff_time_diff_var = ((edge_same_time_diff_var + edge_diff_time_same_var) == 0).float()
        edge_self = torch.where((edge_same_time_diff_var + edge_diff_time_same_var) == 2)
        edge_same_time_diff_var[edge_self] = 0.0

        # 图神经网络传播节点状态
        for gc in self.gcs:
            cur_x = gc(cur_x, edge_index, edge_time, cur_x_time, edge_same_time_diff_var, edge_diff_time_same_var, edge_diff_time_diff_var, 0)
        x = rearrange(cur_x, '(b m n l) c -> b n m l c', b=B, n=N, m=M, l=L)

        # 池化聚合同一Patch 同一变量的隐藏状态
        # 若Patch为奇数个，创建一个虚拟节点
        if M > 1 and M % 2 != 0:
            x = torch.cat([x, x[:, :, -1, :].unsqueeze(2)], dim=2)
            mask_X = torch.cat([mask_X, torch.zeros(size=[B, N, 1, L, 1]).to(x.device)], dim=2)
            x_time = torch.cat([x_time, torch.zeros(size=[B, N, 1, L, 1]).to(x.device)], dim=2)
            M = M + 1

        obs_num_per_patch = torch.sum(mask_X, dim=3)  # mask_X.shape[B, N, M, L, 1]
        x_time_per_patch = torch.sum(x_time, dim=3)  # x_time.shape[B, N, M, L, 1]
        avg_x_time = x_time_per_patch / torch.where(obs_num_per_patch == 0, torch.tensor(1, dtype=x.dtype),
                                                obs_num_per_patch)
        avg_te = self.LearnableTE(avg_x_time).unsqueeze(-2)  # (B, N, M, L, F_te)
        time_te = self.LearnableTE(x_time)  # (B, N, M, L, F_te)
        Q = torch.matmul(avg_te, self.w_q)
        K = torch.matmul(time_te, self.w_k)
        V = torch.matmul(x, self.w_v)
        attention = torch.matmul(Q, K.permute(0, 1, 2, 4, 3)).permute(0, 1, 2, 4, 3)
        attention = torch.div(attention, Q.shape[-1] ** 0.5)
        attention[torch.where(mask_X == 0)] = -1e10
        scale_attention = torch.softmax(attention, dim=-2)
        mask_X = (obs_num_per_patch > 0).float()
        x = torch.sum((V * scale_attention), dim=-2)
        x_time = avg_x_time

        for n_layer in range(1, self.patch_layer):
            B, N, T, D = x.shape

            cur_x = x.reshape(-1, D)
            cur_x_time = x_time.reshape(-1, 1)

            cur_variable_indices = variable_indices.view(1, N, 1, 1)

            # 利用广播机制，将其扩展成形状为 [B, N, M, L, D]
            cur_variable_indices = cur_variable_indices.expand(B, N, T, 1).reshape(-1, 1)

            patch_indices = torch.arange(T).float().to(x.device)

            cur_patch_indices = patch_indices.view(1, 1, T)
            missing_indices = torch.where(mask_X.reshape(B, -1) == 0)

            cur_patch_indices = cur_patch_indices.expand(B, N, T).reshape(B, -1)

            patch_indices_matrix_1 = cur_patch_indices.unsqueeze(1).expand(B, N * T, N * T)
            patch_indices_matrix_2 = cur_patch_indices.unsqueeze(-1).expand(B, N * T, N * T)

            patch_interval = patch_indices_matrix_1 - patch_indices_matrix_2
            patch_interval[missing_indices[0], missing_indices[1]] = torch.zeros(len(missing_indices[0]), N * T).to(x.device)
            patch_interval[missing_indices[0], :, missing_indices[1]] = torch.zeros(len(missing_indices[0]), N * T).to(x.device)

            # cur_adj = patch_interval == 1 and patch_interval == -1

            edge_ind = torch.where(torch.abs(patch_interval) == 1)

            source_nodes = (N * T * edge_ind[0] + edge_ind[1])
            target_nodes = (N * T * edge_ind[0] + edge_ind[2])
            edge_index = torch.cat([source_nodes.unsqueeze(0), target_nodes.unsqueeze(0)])

            edge_time = torch.squeeze(cur_x_time[source_nodes] - cur_x_time[target_nodes])

            edge_diff_time_same_var = (
                        (cur_variable_indices[source_nodes] - cur_variable_indices[target_nodes]) == 0).float()
            edge_same_time_diff_var = ((cur_x_time[source_nodes] - cur_x_time[target_nodes]) == 0).float()
            edge_diff_time_diff_var = ((edge_same_time_diff_var + edge_diff_time_same_var) == 0).float()
            edge_self = torch.where((edge_same_time_diff_var + edge_diff_time_same_var) == 2)
            edge_same_time_diff_var[edge_self] = 0.0

            if edge_index.shape[1] > 0:
            # 图神经网络传播节点状态
                for gc in self.gcs:
                    cur_x = gc(cur_x, edge_index, edge_time, cur_x_time, edge_same_time_diff_var, edge_diff_time_same_var,
                               edge_diff_time_diff_var, n_layer)
                x = rearrange(cur_x, '(b n t) c -> b n t c', b=B, n=N, t=T, c=D)

            # 池化聚合同一Patch 同一变量的隐藏状态
            # 若Patch为奇数个，创建一个虚拟节点
            if T > 1 and T % 2 != 0:
                x = torch.cat([x, x[:, :, -1, :].unsqueeze(-2)], dim=2)
                mask_X = torch.cat([mask_X, torch.zeros(size=[B, N, 1, 1]).to(x.device)], dim=2)
                x_time = torch.cat([x_time, torch.zeros(size=[B, N, 1, 1]).to(x.device)], dim=2)
                T = T + 1

            x = x.view(B, N, T // 2, 2, D)
            x_time = x_time.view(B, N, T // 2, 2, 1)
            mask_X = mask_X.view(B, N, T // 2, 2, 1)

            obs_num_per_patch = torch.sum(mask_X, dim=3)  # mask_X.shape[B, N, M, L, 1]
            x_time_per_patch = torch.sum(x_time, dim=3)  # x_time.shape[B, N, M, L, 1]
            avg_x_time = x_time_per_patch / torch.where(obs_num_per_patch == 0, torch.tensor(1, dtype=x.dtype),
                                                        obs_num_per_patch)
            avg_te = self.LearnableTE(avg_x_time).unsqueeze(-2)  # (B, N, M, L, F_te)
            time_te = self.LearnableTE(x_time)  # (B, N, M, L, F_te)
            Q = torch.matmul(avg_te, self.w_q)
            K = torch.matmul(time_te, self.w_k)
            V = torch.matmul(x, self.w_v)
            attention = torch.matmul(Q, K.permute(0, 1, 2, 4, 3)).permute(0, 1, 2, 4, 3)
            attention = torch.div(attention, Q.shape[-1] ** 0.5)
            attention[torch.where(mask_X == 0)] = -1e10
            scale_attention = torch.softmax(attention, dim=-2)

            mask_X = (obs_num_per_patch > 0).float()
            x = torch.sum((V * scale_attention), dim=-2)
            x_time = avg_x_time
        return torch.squeeze(x)

    def forecasting(self, time_steps_to_predict, X, truth_time_steps, mask=None):
        B, M, L_in, N = X.shape
        self.batch_size = B
        X = X.permute(0, 3, 1, 2).unsqueeze(-1)  # (B*N*M, L, 1)
        X = self.obs_enc(X)
        truth_time_steps = truth_time_steps.permute(0, 3, 1, 2).unsqueeze(-1)  # (B, N, M, L, 1)
        mask = mask.permute(0, 3, 1, 2).unsqueeze(-1)  # (B, N, M, L, 1)
        te_his = self.LearnableTE(truth_time_steps)  # (B, N, M, L, F_te)
        var_emb = self.nodevec.weight.view(1, N, 1, 1, self.hid_dim).repeat(B, 1, M, L_in, 1)
        X = self.relu(X + var_emb + te_his)  # (B*N*M, L, F)

        h = self.IMTS_Model(X, mask, truth_time_steps, None)  # (B, N, hid_dim)

        """ Decoder """
        L_pred = time_steps_to_predict.shape[-1]
        h = h.unsqueeze(dim=-2).repeat(1, 1, L_pred, 1)
        time_steps_to_predict = time_steps_to_predict.view(B, 1, L_pred, 1).repeat(1, N, 1, 1)
        te_pred = self.LearnableTE(time_steps_to_predict)

        h = torch.cat([h, te_pred], dim=-1)

        outputs = self.decoder(h).squeeze(dim=-1).permute(0, 2, 1).unsqueeze(dim=0)

        return outputs

    def classification(self, X, truth_time_steps, mask=None, P_static=None):
        B, M, L_in, N = X.shape
        self.batch_size = B
        X = X.permute(0, 3, 1, 2).unsqueeze(-1)  # (B*N*M, L, 1)
        X = self.obs_enc(X)
        truth_time_steps = truth_time_steps.permute(0, 3, 1, 2).unsqueeze(-1)  # (B, N, M, L, 1)
        mask = mask.permute(0, 3, 1, 2).unsqueeze(-1)  # (B, N, M, L, 1)
        te_his = self.LearnableTE(truth_time_steps)  # (B, N, M, L, F_te)
        # print(time.max(), time.mean(), time.min(), time.shape, te.shape)
        var_emb = self.nodevec.weight.view(1, N, 1, 1, self.hid_dim).repeat(B, 1, M, L_in, 1)
        # X = (X + var_emb + te_his) * mask  # (B*N*M, L, F)
        X = self.relu(X + var_emb + te_his)  # (B*N*M, L, F)
        ### *** a encoder to model irregular time series
        h = self.IMTS_Model(X, mask, truth_time_steps, None)  # (B, N, hid_dim)

        if P_static is not None:
            static_emb = self.emb(P_static)
            return self.classifier(torch.cat([torch.sum(h, dim=-1), static_emb], dim=-1))
        else:
            return self.classifier(torch.sum(h, dim=-1))