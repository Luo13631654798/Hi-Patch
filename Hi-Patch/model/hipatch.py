from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch_geometric.nn.conv import MessagePassing
from torch_scatter import scatter
from torch_geometric.utils.num_nodes import maybe_num_nodes

# Custom softmax function for handling node-wise softmax over a graph structure
def softmax(src, index):
    N = maybe_num_nodes(index)
    global_out = src - src.max()    # Global max normalization
    global_out = global_out.exp()
    global_out_sum = scatter(global_out, index, dim=0, dim_size=N, reduce='sum')[index]
    c =  global_out / (global_out_sum + 1e-16)
    return c

class Intra_Inter_Patch_Graph_Layer(MessagePassing):
    """
    Implementing of intra/inter patch graph layer.
    """
    def __init__(self, n_heads=2, d_input=6, d_k=6, alpha=0.9, patch_layer=1, res=1, **kwargs):
        super(Intra_Inter_Patch_Graph_Layer, self).__init__(aggr='add', **kwargs)
        self.n_heads = n_heads
        self.patch_layer = patch_layer
        self.res = res
        self.d_input = d_input
        self.d_k = d_k // n_heads
        self.d_q = d_k // n_heads
        self.d_e = d_input // n_heads
        self.d_sqrt = math.sqrt(d_k // n_heads)
        self.alpha = alpha

        # Define parameters for query, key, and value transformations
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

        # Layer normalization
        self.layer_norm = nn.LayerNorm(d_input)

    def forward(self, x, edge_index, edge_value, time_nodes, edge_same_time_diff_var, edge_diff_time_same_var, edge_diff_time_diff_var, n_layer):
        residual = x
        x = self.layer_norm(x)
        return self.propagate(edge_index, x=x, edges_temporal=edge_value,
                              edge_same_time_diff_var=edge_same_time_diff_var,
                              edge_diff_time_same_var=edge_diff_time_same_var,
                              edge_diff_time_diff_var=edge_diff_time_diff_var,
                              n_layer=n_layer, residual=residual)

    def message(self, x_j, x_i, edge_index_i, edges_temporal, edge_same_time_diff_var, edge_diff_time_same_var, edge_diff_time_diff_var, n_layer):
        # Attention and message calculation for each attention head
        messages = []
        for i in range(self.n_heads):
            w_k = self.w_k_list[i][n_layer]
            bias_k = self.bias_k_list[i][n_layer]

            w_q = self.w_q_list[i][n_layer]
            bias_q = self.bias_q_list[i][n_layer]

            w_v = self.w_v_list[i][n_layer]
            bias_v = self.bias_v_list[i][n_layer]

            attention = self.each_head_attention(x_j, w_k, bias_k, w_q, bias_q, x_i,
                                                 edge_same_time_diff_var, edge_diff_time_same_var, edge_diff_time_diff_var)
            attention = torch.div(attention, self.d_sqrt)
            attention = torch.pow(self.alpha, torch.abs(edges_temporal.squeeze())).unsqueeze(-1) * attention
            attention_norm = softmax(attention, edge_index_i)

            sender_stdv = edge_same_time_diff_var * (torch.matmul(x_j, w_v[0]) + bias_v[0])
            sender_dtsv = edge_diff_time_same_var * (torch.matmul(x_j, w_v[1]) + bias_v[1])
            sender_dtdv = edge_diff_time_diff_var * (torch.matmul(x_j, w_v[2]) + bias_v[2])
            sender = sender_stdv + sender_dtsv + sender_dtdv

            message = attention_norm * sender
            messages.append(message)

        # Concatenate messages from all heads
        message_all_head = torch.cat(messages, 1)
        return message_all_head

    def each_head_attention(self, x_j_transfer, w_k, bias_k, w_q, bias_q, x_i, edge_same_time_diff_var, edge_diff_time_same_var, edge_diff_time_diff_var):
        x_i_0 = edge_same_time_diff_var * (torch.matmul(x_i, w_q[0]) + bias_q[0])
        x_i_1 = edge_diff_time_same_var * (torch.matmul(x_i, w_q[1]) + bias_q[1])
        x_i_2 = edge_diff_time_diff_var * (torch.matmul(x_i, w_q[2]) + bias_q[2])
        x_i = x_i_0 + x_i_1 + x_i_2

        sender_0 = edge_same_time_diff_var * (torch.matmul(x_j_transfer, w_k[0]) + bias_k[0])
        sender_1 = edge_diff_time_same_var * (torch.matmul(x_j_transfer, w_k[1]) + bias_k[1])
        sender_2 = edge_diff_time_diff_var * (torch.matmul(x_j_transfer, w_k[2]) + bias_k[2])
        sender = sender_0 + sender_1 + sender_2

        attention = torch.bmm(torch.unsqueeze(sender, 1), torch.unsqueeze(x_i, 2))
        return torch.squeeze(attention, 1)

    def update(self, aggr_out, residual):
        # Apply residual connection and non-linearity
        return self.res * residual + F.gelu(aggr_out)

    def __repr__(self):
        return '{}'.format(self.__class__.__name__)

class Hi_Patch(nn.Module):
    """
    Hi-Patch model for ISMTS forecasting and classification.
    """
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

        # intra/inter patch graph layer for update
        for l in range(self.n_layer):
            self.gcs.append(Intra_Inter_Patch_Graph_Layer(args.nhead, d_model, d_model, self.alpha, args.patch_layer, self.res))

        # Query, key, value matrices for aggregation
        self.w_q = nn.Parameter(torch.FloatTensor(d_model, d_model))
        self.w_k = nn.Parameter(torch.FloatTensor(d_model, d_model))
        self.w_v = nn.Parameter(torch.FloatTensor(d_model, d_model))

        nn.init.xavier_uniform_(self.w_q)
        nn.init.xavier_uniform_(self.w_k)
        nn.init.xavier_uniform_(self.w_v)

        # Decoder initialization
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
        # learnable continuous time embeddings
        out1 = self.te_scale(tt)
        out2 = torch.sin(self.te_periodic(tt))
        return torch.cat([out1, out2], -1)

    def IMTS_Model(self, x, mask_X, x_time):
        """
        This function handles the irregular time series (IMTS) data modeling.
        x: Input time series data.
        mask_X: Mask for missing values.
        x_time: Time values corresponding to the series.
        """
        B, N, M, L, D = x.shape

        # Create a tensor of shape [N] containing variable indices
        variable_indices = torch.arange(N).to(x.device)

        B, N, M, L, D = x.shape

        # Expand it to shape [1, N, 1, 1, 1]
        cur_variable_indices = variable_indices.view(1, N, 1, 1, 1)

        # Use broadcasting to expand it to shape [B, N, M, L, D]
        cur_variable_indices = cur_variable_indices.expand(B, N, M, L, 1)

        cur_x = rearrange(x, 'b n m l c -> (b m n l) c')
        cur_variable_indices = rearrange(cur_variable_indices, 'b n m l c -> (b m n l) c')
        cur_x_time = rearrange(x_time, 'b n m l c -> (b m n l) c')

        # Generate graph structure
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

        # Intra Patch Graph Layer
        # Update node states through the graph using GAT
        for gc in self.gcs:
            cur_x = gc(cur_x, edge_index, edge_time, cur_x_time, edge_same_time_diff_var, edge_diff_time_same_var, edge_diff_time_diff_var, 0)
        x = rearrange(cur_x, '(b m n l) c -> b n m l c', b=B, n=N, m=M, l=L)

        # Aggregate node states of the same variable in the same patch
        # If the number of patches is odd, create a virtual node
        if M > 1 and M % 2 != 0:
            x = torch.cat([x, x[:, :, -1, :].unsqueeze(2)], dim=2)
            mask_X = torch.cat([mask_X, torch.zeros(size=[B, N, 1, L, 1]).to(x.device)], dim=2)
            x_time = torch.cat([x_time, torch.zeros(size=[B, N, 1, L, 1]).to(x.device)], dim=2)
            M = M + 1

        obs_num_per_patch = torch.sum(mask_X, dim=3)
        x_time_per_patch = torch.sum(x_time, dim=3)
        avg_x_time = x_time_per_patch / torch.where(obs_num_per_patch == 0, torch.tensor(1, dtype=x.dtype),
                                                obs_num_per_patch)
        avg_te = self.LearnableTE(avg_x_time).unsqueeze(-2)
        time_te = self.LearnableTE(x_time)
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

        # Inter Patch Graph Layers
        for n_layer in range(1, self.patch_layer):
            B, N, T, D = x.shape

            cur_x = x.reshape(-1, D)
            cur_x_time = x_time.reshape(-1, 1)

            cur_variable_indices = variable_indices.view(1, N, 1, 1)

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
                # Propagate node states through the graph using GAT
                for gc in self.gcs:
                    cur_x = gc(cur_x, edge_index, edge_time, cur_x_time, edge_same_time_diff_var, edge_diff_time_same_var,
                               edge_diff_time_diff_var, n_layer)
                x = rearrange(cur_x, '(b n t) c -> b n t c', b=B, n=N, t=T, c=D)

            # Aggregate node states of the same variable in the adjacent patch
            # If the number of patches is odd, create a virtual node
            if T > 1 and T % 2 != 0:
                x = torch.cat([x, x[:, :, -1, :].unsqueeze(-2)], dim=2)
                mask_X = torch.cat([mask_X, torch.zeros(size=[B, N, 1, 1]).to(x.device)], dim=2)
                x_time = torch.cat([x_time, torch.zeros(size=[B, N, 1, 1]).to(x.device)], dim=2)
                T = T + 1

            x = x.view(B, N, T // 2, 2, D)
            x_time = x_time.view(B, N, T // 2, 2, 1)
            mask_X = mask_X.view(B, N, T // 2, 2, 1)

            obs_num_per_patch = torch.sum(mask_X, dim=3)
            x_time_per_patch = torch.sum(x_time, dim=3)
            avg_x_time = x_time_per_patch / torch.where(obs_num_per_patch == 0, torch.tensor(1, dtype=x.dtype), obs_num_per_patch)
            avg_te = self.LearnableTE(avg_x_time).unsqueeze(-2)
            time_te = self.LearnableTE(x_time)
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
        X = X.permute(0, 3, 1, 2).unsqueeze(-1)
        X = self.obs_enc(X)
        truth_time_steps = truth_time_steps.permute(0, 3, 1, 2).unsqueeze(-1)
        mask = mask.permute(0, 3, 1, 2).unsqueeze(-1)
        te_his = self.LearnableTE(truth_time_steps)
        var_emb = self.nodevec.weight.view(1, N, 1, 1, self.hid_dim).repeat(B, 1, M, L_in, 1)
        X = self.relu(X + var_emb + te_his)

        h = self.IMTS_Model(X, mask, truth_time_steps)

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
        X = X.permute(0, 3, 1, 2).unsqueeze(-1)
        X = self.obs_enc(X)
        truth_time_steps = truth_time_steps.permute(0, 3, 1, 2).unsqueeze(-1)
        mask = mask.permute(0, 3, 1, 2).unsqueeze(-1)
        te_his = self.LearnableTE(truth_time_steps)
        var_emb = self.nodevec.weight.view(1, N, 1, 1, self.hid_dim).repeat(B, 1, M, L_in, 1)
        X = self.relu(X + var_emb + te_his)
        h = self.IMTS_Model(X, mask, truth_time_steps)

        if P_static is not None:
            static_emb = self.emb(P_static)
            return self.classifier(torch.cat([torch.sum(h, dim=-1), static_emb], dim=-1))
        else:
            return self.classifier(torch.sum(h, dim=-1))