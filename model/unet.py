import torch
import torch.nn as nn
import  cv2
import torch.nn.functional as F
from model.layer import ConvBnRelu,get_norm_layer,count_param,conv1x1,conv2x2,conv3x3
from model.transformer import TransformerModel
from collections import OrderedDict
import h5py,math
import os



class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, BatchNorm=nn.BatchNorm2d, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = BatchNorm(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.stdv = 1. / math.sqrt(in_channels)

    def reset_params(self):
        self.conv.weight.data.uniform_(-self.stdv, self.stdv)
        self.bn.weight.data.uniform_()
        self.bn.bias.data.zero_()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return self.relu(x)



def gating_signal(input, out_size, batch_norm=False):
    """
    Resize the down layer feature map into the same dimension as the up layer feature map
    using 1x1 conv
    :param input:   down-dim feature map
    :param out_size: output channel number
    :return: the gating feature map with the same dimension as the up layer feature map
    """
    device = input.device  

    
    conv_layer = nn.Conv2d(input.shape[1], out_size, kernel_size=1, stride=1, padding=0)
    conv_layer = conv_layer.to(device)

    if batch_norm:
        batch_norm_layer = nn.BatchNorm2d(out_size).to(device)
        x = batch_norm_layer(conv_layer(input))
    else:
        x = conv_layer(input)

    relu = nn.ReLU(inplace=True)
    relu =relu.to(device) 
    x = relu(x)

    return x



#CAG
class Attention_block(nn.Module):

    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g,
                      F_int,
                      kernel_size=1,
                      stride=1,
                      padding=0,
                      bias=True),
            nn.BatchNorm2d(F_int))

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l,
                      F_int,
                      kernel_size=1,
                      stride=1,
                      padding=0,
                      bias=True),
            nn.BatchNorm2d(F_int))

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1), nn.Sigmoid())

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):

        g_gray = torch.mean(g, dim=1, keepdim=True)
        x_gray = torch.mean(x, dim=1, keepdim=True)
        
        

        g_gray_np = g_gray.squeeze().detach().cpu().numpy()
    
        x_gray_np = x_gray.squeeze().detach().cpu().numpy()
                
        g_contours_np = cv2.Canny((g_gray_np * 255).astype('uint8'), 100, 200) / 255.0
  
        x_contours_np = cv2.Canny((x_gray_np * 255).astype('uint8'), 100, 200) / 255.0
    
            
        g1 = self.W_g(g)
        x1 = self.W_x(x)


        g_contours = torch.from_numpy(g_contours_np).unsqueeze(0).unsqueeze(0).float().to(g.device)
        g_contours = torch.from_numpy(g_contours_np).unsqueeze(1).unsqueeze(3).float().to(g.device)
        x_contours = torch.from_numpy(x_contours_np).unsqueeze(1).unsqueeze(3).float().to(x.device)



        g1 = g1 * (1 - g_contours)
        x1 = x1 * (1 - x_contours)


        
        x1 = F.interpolate(x1, size=g1.size()[2:], mode='bilinear', align_corners=False)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        
        psi = F.interpolate(psi, size=x.size()[2:], mode='bilinear', align_corners=False)

        return x * psi


class UpSamplingBlock(nn.Module):
    def __init__(self,in_channels, out_channels):
        super(UpSamplingBlock, self).__init__()
        self.up_sampling = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv =  conv2x2(in_channels, out_channels, dilation=1) 
        self.conv2 =  conv3x3(in_channels, out_channels, dilation=1) 
        self.concat = lambda x, y: torch.cat((x, y), dim=1)

    def forward(self, x, cag):
        x_conv = self.conv(x)
        x_upsampled = self.up_sampling(x_conv)

        x_concat = self.concat(x_upsampled, cag)
        x=self.conv2(x_concat)
        return x

class EnhanceModule(nn.Module):  #Feature Enhancement
    def __init__(self, num_in, plane_mid, mids, abn=nn.BatchNorm2d, normalize=False):
        super(EnhanceModule, self).__init__()
        
        
        self.normalize = normalize
        self.num_s = int(plane_mid)
        self.num_n = mids
        self.priors = nn.AdaptiveAvgPool2d(output_size=(mids + 2, mids + 2))

        self.conv_state = nn.Conv2d(num_in, self.num_s, kernel_size=1)
        self.conv_proj = nn.Conv2d(num_in, self.num_s, kernel_size=1)
        self.conv_extend = nn.Conv2d(self.num_s, num_in, kernel_size=1, bias=False)

        self.blocker = abn(num_in)

    def forward(self, x, contour):
        # contour = F.upsample(contour, (x.size()[-2], x.size()[-1]))
        contour = nn.functional.interpolate(contour, (x.size()[-2], x.size()[-1]))
        n, c, h, w = x.size()
        contour = torch.nn.functional.softmax(contour, dim=1)[:, 1, :, :].unsqueeze(1)

        
        # Construct projection matrix
        x_state_reshaped = self.conv_state(x).view(n, self.num_s, -1)       
        x_proj = self.conv_proj(x)
  

        x_mask = x_proj * contour
        x_anchor = self.priors(x_mask)[:, :, 1:-1, 1:-1].reshape(n, self.num_s, -1)
  

        x_proj_reshaped = torch.matmul(x_anchor.permute(0, 2, 1), x_proj.reshape(n, self.num_s, -1))
        x_proj_reshaped = torch.nn.functional.softmax(x_proj_reshaped, dim=1)
     

        return x_proj_reshaped
    

class CrossModalityBranchAttention(nn.Module):
    def __init__(self, num_channels, reduction_ratio=2):

        super(CrossModalityBranchAttention, self).__init__()
        num_channels_reduced = num_channels // reduction_ratio
        self.reduction_ratio = reduction_ratio

        self.global_avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.global_max_pool = nn.AdaptiveMaxPool2d(output_size=(1, 1))

        self.fc1 = nn.Linear(num_channels, num_channels_reduced, bias=True)
        self.fc2_branch_1 = nn.Linear(num_channels_reduced, num_channels, bias=True)
        self.fc2_branch_2 = nn.Linear(num_channels_reduced, num_channels, bias=True)

    def forward(self,input_tensor):
        batch_size, num_channels, H, W = input_tensor.size()
        avg_squeeze_tensor = self.global_avg_pool(input_tensor)
        max_squeeze_tensor = self.global_max_pool(input_tensor)

        #avg
        avg_fc_out_1 = self.fc1(avg_squeeze_tensor.view(batch_size, num_channels))
        max_fc_out_1 = self.fc1(max_squeeze_tensor.view(batch_size, num_channels))\

        shared_fc_out_1 = avg_fc_out_1 + max_fc_out_1

        branch_1_attention_weight = self.fc2_branch_1(shared_fc_out_1)
        branch_2_attention_weight = self.fc2_branch_2(shared_fc_out_1)

        branch_1_attention_weight = F.softmax(branch_1_attention_weight)
        branch_2_attention_weight = F.softmax(branch_2_attention_weight)

        branch_1_ca_weight = branch_1_attention_weight.view(batch_size, num_channels, 1, 1)
        branch_2_ca_weight = branch_2_attention_weight.view(batch_size, num_channels, 1, 1)

        return branch_1_ca_weight, branch_2_ca_weight

class CrossModalityInterConvModule(nn.Module):

    def __init__(self, num_channels, norm_layer, dropout, dilation=(1, 1), reduction_ratio=2):
        super(CrossModalityInterConvModule, self).__init__()
        self.modality_branch_channels = num_channels // 2
        self.branch_att_weight_func = CrossModalityBranchAttention(num_channels=num_channels,
                                                                reduction_ratio=reduction_ratio)

        self.modality_branch_1_conv = ConvBnRelu(self.modality_branch_channels, self.modality_branch_channels,
                                                 norm_layer, dilation=dilation[0], dropout=dropout)
        self.modality_branch_2_conv = ConvBnRelu(self.modality_branch_channels, self.modality_branch_channels,
                                                 norm_layer, dilation=dilation[0], dropout=dropout)
        self.modality_fusion_conv = ConvBnRelu(3 * num_channels, num_channels, norm_layer, dilation=dilation[1],
                                               dropout=dropout)

    def forward(self,input_tensor):
        # pairing
        x=input_tensor
        input_modality_branch_1_tensor, input_modality_branch_2_tensor = torch.chunk(input_tensor, 2, dim=1)
        # # # # # # # # # # # # # # # # # # #
        # Interaction
        # # # # # # # # # # # # # # # # # # #
        # attention weight

        branch_1_channel_weight, branch_2_channel_weight = self.branch_att_weight_func(input_tensor)
        conv_modality_branch_1_tensor = self.modality_branch_1_conv(input_modality_branch_1_tensor)
        conv_modality_branch_2_tensor = self.modality_branch_2_conv(input_modality_branch_2_tensor)

        concat_modality_branch_1_tensor = torch.cat([conv_modality_branch_1_tensor, input_modality_branch_2_tensor],
                                                    dim=1)
        concat_modality_branch_2_tensor = torch.cat([conv_modality_branch_2_tensor, input_modality_branch_1_tensor],
                                                    dim=1)

        recalibration_modality_branch_1_feature = torch.mul(concat_modality_branch_1_tensor, branch_1_channel_weight)
        recalibration_modality_branch_2_feature = torch.mul(concat_modality_branch_2_tensor, branch_2_channel_weight)

        # # # # # # # # # # # # # # # # # # #
        # Fusion
        # # # # # # # # # # # # # # # # # # #
        concat_tensor = torch.cat([recalibration_modality_branch_1_feature, recalibration_modality_branch_2_feature, x],
                                  dim=1)

        output_tensor = self.modality_fusion_conv(concat_tensor)

        return output_tensor



class CrossModalityInterConv(nn.Sequential):
    """Unet mainstream downblock.
    """

    def __init__(self, inplanes, outplanes, norm_layer, dilation=(1, 1), dropout=0):
        super(CrossModalityInterConv, self).__init__(
            OrderedDict(
                [
                    ('ConvBnRelu1',
                     CrossModalityInterConvModule(num_channels=inplanes, norm_layer=norm_layer, dilation=dilation,
                                                  dropout=dropout, reduction_ratio=2)),
                    ('ConvBnRelu2', ConvBnRelu(inplanes, outplanes, norm_layer, dilation=dilation[1], dropout=dropout)),
                ]
            )

        )


class MultiModalFeatureExtraction(nn.Module):
    def __init__(self, modal_num, base_n_filter):
        super(MultiModalFeatureExtraction, self).__init__()
        self.modal_num = modal_num
        self.base_n_filter = base_n_filter
        self.conv_filter = base_n_filter // modal_num
        self.conv_list = nn.ModuleList()

        for i in range(modal_num):
            self.conv_list.append(nn.Sequential(
                nn.Conv2d(1, self.base_n_filter, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
                nn.Conv2d(self.base_n_filter, self.conv_filter, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0),
                          bias=False),
                # nn.InstanceNorm2d(self.conv_filter),
                # nn.ReLU(inplace=True)
            ))

    def forward(self, input_tensor):
        x = input_tensor
        modal_input_list = torch.chunk(input_tensor, self.modal_num, dim=1)
        modal_feature_list = []

        for i, (current_conv, current_input) in enumerate(zip(self.conv_list, modal_input_list)):
            modal_feature_list.append(current_conv(current_input))

        output_tensor = torch.cat(modal_feature_list, dim=1)
        return output_tensor

class GraphNet(nn.Module):
    def __init__(self, node_num, dim, normalize_input=False):
        super(GraphNet, self).__init__()
        self.node_num = node_num
        self.dim = dim
        self.normalize_input = normalize_input

        self.anchor = nn.Parameter(torch.rand(node_num, dim))
        self.sigma = nn.Parameter(torch.rand(node_num, dim))

    def init(self, initcache):
        if not os.path.exists(initcache):
            print(initcache + ' not exist!!!\n')
        else:
            with h5py.File(initcache, mode='r') as h5:
                clsts = h5.get("centroids")[...]
                traindescs = h5.get("descriptors")[...]
                self.init_params(clsts, traindescs)
                del clsts, traindescs

    def init_params(self, clsts, traindescs=None):
        self.anchor = nn.Parameter(torch.from_numpy(clsts))

    def gen_soft_assign(self, x, sigma):
        B, C, H, W = x.size()
        N = H * W
        soft_assign = torch.zeros([B, self.node_num, N], device=x.device, dtype=x.dtype, layout=x.layout)
        for node_id in range(self.node_num):
            residual = (x.view(B, C, -1).permute(0, 2, 1).contiguous() - self.anchor[node_id, :]).div(
                sigma[node_id, :])  # + eps)
            soft_assign[:, node_id, :] = -torch.pow(torch.norm(residual, dim=2), 2) / 2

        soft_assign = F.softmax(soft_assign, dim=1)

        return soft_assign

    def forward(self, x):
        B, C, H, W = x.size()
        if self.normalize_input:
            x = F.normalize(x, p=2, dim=1)  # across descriptor dim

        sigma = torch.sigmoid(self.sigma)
        soft_assign = self.gen_soft_assign(x, sigma)  # B x C x N(N=HxW)
        #
        eps = 1e-9
        nodes = torch.zeros([B, self.node_num, C], dtype=x.dtype, layout=x.layout, device=x.device)
        for node_id in range(self.node_num):
            residual = (x.view(B, C, -1).permute(0, 2, 1).contiguous() - self.anchor[node_id, :]).div(
                sigma[node_id, :])  # + eps)
            nodes[:, node_id, :] = residual.mul(soft_assign[:, node_id, :].unsqueeze(2)).sum(dim=1) / (
                        soft_assign[:, node_id, :].sum(dim=1).unsqueeze(1) + eps)

        nodes = F.normalize(nodes, p=2, dim=2)  # intra-normalization
        nodes = nodes.view(B, -1).contiguous()
        nodes = F.normalize(nodes, p=2, dim=1)  # l2 normalize

        return nodes.view(B, C, self.node_num).contiguous(), soft_assign

class GraphConvNet(nn.Module):
    '''
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    '''

    def __init__(self, in_features, out_features, bias=False):
        super(GraphConvNet, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(1, 1, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x, adj):
        x_t = x.permute(0, 2, 1).contiguous()  # b x k x c
        support = torch.matmul(x_t, self.weight)  # b x k x c

        adj = torch.softmax(adj, dim=2)
        output = (torch.matmul(adj, support)).permute(0, 2, 1).contiguous()  # b x c x k

        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.in_features) + ' -> ' \
            + str(self.out_features) + ')'

class CascadeGCNet(nn.Module):
    def __init__(self, dim, loop):
        super(CascadeGCNet, self).__init__()
        self.gcn1 = GraphConvNet(dim, dim)
        self.gcn2 = GraphConvNet(dim, dim)
        self.gcn3 = GraphConvNet(dim, dim)
        self.gcns = [self.gcn1, self.gcn2, self.gcn3]
        assert (loop == 1 or loop == 2 or loop == 3)
        self.gcns = self.gcns[0:loop]
        self.relu = nn.ReLU()

    def forward(self, x):
        for gcn in self.gcns:
            x_t = x.permute(0, 2, 1).contiguous()  # b x k x c
            x = gcn(x, adj=torch.matmul(x_t, x))  # b x c x k
        x = self.relu(x)
        return x

class MutualModule0(nn.Module):
    def __init__(self, dim, BatchNorm=nn.BatchNorm2d, dropout=0.1):
        super(MutualModule0, self).__init__()
        self.gcn = CascadeGCNet(dim, loop=2)
        self.conv = nn.Sequential(BasicConv2d(dim, dim, BatchNorm, kernel_size=1, padding=0))

    # graph0: contour, graph1/2: region, assign:contour
    def forward(self, contour_graph, region_graph1, region_graph2, assign):
        m = self.corr_matrix(contour_graph, region_graph1, region_graph2)
        contour_graph = contour_graph + m

        contour_graph = self.gcn(contour_graph)
        contour_x = contour_graph.bmm(assign)  # reprojection
        contour_x = self.conv(contour_x.unsqueeze(3)).squeeze(3)
        return contour_x

    def corr_matrix(self, contour, region1, region2):
        assign = contour.permute(0, 2, 1).contiguous().bmm(region1)
        assign = F.softmax(assign, dim=-1)  # normalize region-node
        m = assign.bmm(region2.permute(0, 2, 1).contiguous())
        m = m.permute(0, 2, 1).contiguous()
        return m
    


class MutualModule1(nn.Module):
    def __init__(self, dim, BatchNorm=nn.BatchNorm2d, dropout=0.1):
        super(MutualModule1, self).__init__()
        self.dim = dim

        self.gcn = CascadeGCNet(dim, loop=3)

        self.pred0 = nn.Conv2d(self.dim, 3, kernel_size=1)  # predicted contour is used for contour-region mutual sub-module

        self.pred1_ = nn.Conv2d(self.dim, 3, kernel_size=1)  # region prediction

        # conv region feature afger reproj
        self.conv0 = nn.Sequential(BasicConv2d(self.dim, self.dim, BatchNorm, kernel_size=1, padding=0))
        self.conv1 = nn.Sequential(BasicConv2d(self.dim, self.dim, BatchNorm, kernel_size=1, padding=0))

        # self.ecg = ECGraphNet(self.dim, BatchNorm, dropout)

    def forward(self, region_x, region_graph, assign, contour_x):
        b, c, h, w = contour_x.shape

        contour = self.pred0(contour_x)

        region_graph = self.gcn(region_graph)
        n_region_x = region_graph.bmm(assign)
        n_region_x = self.conv0(n_region_x.view(region_x.size()))

        region_x = region_x + n_region_x  # raw-feature with residual

        region_x = region_x + contour_x
        region_x = self.conv1(region_x)


        region = self.pred1_(region_x)

        return  contour, region
    


class MutualNet(nn.Module):
    def __init__(self, BatchNorm=nn.BatchNorm2d, dim=32, num_clusters=8, dropout=0.1):
        super(MutualNet, self).__init__()

        self.dim = dim

        self.contour_proj0 = GraphNet(node_num=num_clusters, dim=self.dim, normalize_input=False)
        self.region_proj0 = GraphNet(node_num=num_clusters, dim=self.dim, normalize_input=False)

        self.contour_conv = nn.Sequential(BasicConv2d(self.dim, self.dim, BatchNorm, kernel_size=1, padding=0))
        self.contour_conv[0].reset_params()

        self.region_conv1 = nn.Sequential(BasicConv2d(self.dim, self.dim, BatchNorm, kernel_size=1, padding=0))
        self.region_conv1[0].reset_params()

        self.region_conv2 = nn.Sequential(BasicConv2d(self.dim, self.dim, BatchNorm, kernel_size=1, padding=0))
        self.region_conv2[0].reset_params()
        
        self.gcn = GraphConvNet(dim, dim)
        
        self.r2e = EnhanceModule(32, 32, 8, BatchNorm)

        self.e2r = MutualModule1(self.dim, BatchNorm, dropout)
        
        self.mdf = ModifiedMutualModule(self.dim,num_heads = 1)

    def forward(self, contour_x, region_x):

        region_graph, region_assign = self.region_proj0(region_x)
        contour_graph, contour_assign = self.contour_proj0(contour_x)

        contour_graph = self.contour_conv(contour_graph.unsqueeze(3)).squeeze(3)

        
        region_graph_T = region_graph.permute(0, 2, 1)  # (B, D, N)
        E = region_graph_T.bmm(contour_graph)         # (B, N, M)
        
        region_graph = self.gcn(region_graph,E)
        contour_graph = self.gcn(contour_graph,E)
        
#         n_region_x = self.r2e(region_x, contour_x)
#         # print("region_x")
#         # print(region_x.size())
        
#         region_x = region_x + n_region_x.view(region_x.size()).contiguous()
        
#         n_contour_x = self.r2e(contour_x,region_x)
#         contour_x = contour_x + n_contour_x.view(contour_x.size()).contiguous()

   
        region = self.mdf(region_graph, contour_graph, region_assign, contour_assign,region_x,contour_x)

        return   region

class GATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, dropout=0.1):
        super(GATLayer, self).__init__()
        self.num_heads = num_heads
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.attn_layers = nn.ModuleList([
            nn.Linear(self.in_dim, self.out_dim ) 
        ])
        self.dropout = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.output_proj = nn.Linear(out_dim, out_dim)  

    def forward(self, x, adj):
        x = x.permute(0, 2, 1) 
        batch_size, num_nodes, in_dim = x.shape
        attn_heads = []
        for attn_layer in self.attn_layers:
            x_flat = x.reshape(-1, self.in_dim)
            q = k = attn_layer(x_flat)  
            q = q.view(batch_size, num_nodes, -1) 
            k = k.view(batch_size, num_nodes, -1)
            scores = torch.bmm(q, k.transpose(1, 2)) 
            scores = scores / torch.sqrt(torch.tensor(scores.size(-1), dtype=torch.float32))
            scores = F.softmax(scores, dim=-1)
            attn_heads.append(torch.bmm(scores, q))  


        x = torch.cat(attn_heads, dim=-1)  
 
        x = self.output_proj(x)
        return x.permute(0, 2, 1)
    
    
class ModifiedMutualModule(nn.Module):
    def __init__(self, dim, BatchNorm=nn.BatchNorm2d,num_heads=4, dropout=0.1):
        super(ModifiedMutualModule, self).__init__()
        self.dim = dim
        
        self.gat = GATLayer(dim, dim, num_heads=num_heads, dropout=dropout)
        self.conv = nn.Sequential(nn.Conv2d(dim, dim, kernel_size=1, padding=0), nn.BatchNorm2d(dim), nn.ReLU())
        self.conv0 = nn.Sequential(BasicConv2d(self.dim, self.dim, BatchNorm, kernel_size=1, padding=0))
        self.conv1 = nn.Sequential(BasicConv2d(self.dim, self.dim, BatchNorm, kernel_size=1, padding=0))
        self.pred = nn.Conv2d(self.dim, 2, kernel_size=1)
        self.gcn = CascadeGCNet(dim, loop=3)

    def compute_similarity(self, contour_graph, region_graph):
        # 余弦相似性计算
        contour_graph = contour_graph.permute(0, 2, 1)  # 转换为 [batch, nodes, in_dim]
        region_graph = region_graph.permute(0, 2, 1)
        contour_norm = F.normalize(contour_graph, dim=-1)
        region_norm = F.normalize(region_graph, dim=-1)
        similarity = torch.bmm(contour_norm, region_norm.transpose(1, 2))
        return similarity

 
    def generate_adjacency(self, similarity, threshold=0.8):
        # 基于相似性添加跨图边
        adj = (similarity > threshold).float()
        adj = (adj + adj.transpose(1, 2)) / 2  # 确保对称性
        
        return adj   
    def generate_unified_adjacency(self, adj, cross_similarity, threshold=0.8):

     
        B, N, _ = adj.shape
        unified_adj = torch.zeros(B, 2 * N, 2 * N, device=adj.device)

        # 
        unified_adj[:, :N, :N] = adj  # contour_graph 
        unified_adj[:, N:, N:] = adj  # region_graph 

        cross_adj = (cross_similarity > threshold).float()

        unified_adj[:, :N, N:] = cross_adj
        unified_adj[:, N:, :N] = cross_adj.transpose(1, 2)

        unified_adj = (unified_adj + unified_adj.transpose(1, 2)) / 2
        return unified_adj
        
    
    def forward(self, region_graph, contour_graph, region_assign, contour_assign,region_x, contour_x):

        similarity = self.compute_similarity(contour_graph, region_graph)
        adj = self.generate_adjacency(similarity)

        # region_graph = self.gcn(region_graph)
        # contour_graph = self.gcn(contour_graph)
        
        contour_graph = self.gat(contour_graph, adj)  
        region_graph = self.gat(region_graph, adj)
        

        unified_graph = torch.cat([contour_graph, region_graph], dim=-1)
        unified_adj = self.generate_unified_adjacency(adj, similarity)
        unified_graph = self.gat(unified_graph, unified_adj)
 

        
        region_part = unified_graph[:, :, 8:]
        weights = torch.softmax(torch.cat((contour_assign, region_assign), dim=1), dim=1)
        contour_weight = weights[:, :8, :]  
        region_weight = weights[:, 8:, :]  
        fused_assign = contour_weight * contour_assign + region_weight * region_assign
    
        n_region_x = region_part.bmm(fused_assign)
        n_region_x = self.conv0(n_region_x.view(region_x.size()))
        
        region_x = region_x + n_region_x  # raw-feature with residual
        region_x = region_x + contour_x
        region_x = self.conv1(region_x)
     


        region = self.pred(region_x)

        
        return region



class model(nn.Module):
    name = "model"

    def __init__(self, inplanes, num_classes, width, norm_layer=None, deep_supervision=False, dropout=0, **kwargs):
        super(model, self).__init__()
        # width:64 -> [64, 128, 256，512]
        features = [width * 2 ** i for i in range(4)]
        print(features)

        self.deep_supervision = deep_supervision

        # self.multimodal_conv = nn.Sequential()
        # self.multimodal_conv.add_module('cross_modal_fusion_conv',
        #                                 MultiModalFeatureExtraction(modal_num=inplanes, base_n_filter=features[0]))
        # self.multimodal_conv.add_module('cross_modal_interaction_module1',
        #                                 CrossModalityInterConvModule(num_channels=features[0],
        #                                                              norm_layer=get_norm_layer(), dilation=(1, 1),
        #                                                              dropout=dropout, reduction_ratio=2))

        self.encoder1 = ConvBnRelu(inplanes, features[0], norm_layer, dropout=dropout)
        # self.encoder1 = ConvBnRelu(features[0], features[0], norm_layer, dropout=dropout)
        self.encoder2 = CrossModalityInterConv(features[0], features[1], norm_layer, dropout=dropout)
        self.encoder3 = CrossModalityInterConv(features[1], features[2], norm_layer, dropout=dropout)
        self.encoder4 = CrossModalityInterConv(features[2], features[3], norm_layer, dropout=dropout)

        self.downsample = nn.MaxPool2d(2, 2)



        self.cag3=  Attention_block(features[3],features[2],features[2])
        self.cag2 = Attention_block(features[2], features[1], features[1])
        self.cag1 = Attention_block(features[1], features[0], features[0])

        self.upsample3=UpSamplingBlock(features[3],features[2])
        self.upsample2=UpSamplingBlock(features[2],features[1])
        self.upsample1=UpSamplingBlock(features[1],features[0])

        self.decoder3 = ConvBnRelu(features[2], features[2], norm_layer, dropout=dropout)
        self.decoder2 = ConvBnRelu(features[1], features[1], norm_layer, dropout=dropout)
        self.decoder1 = ConvBnRelu(features[0], features[0], norm_layer, dropout=dropout)

        self.outconv = conv1x1(features[0], num_classes)
        self.outconv2 = conv1x1(num_classes*3,features[0] )

        self.MutualNet=MutualNet()

        self._init_weights()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)

        self.MutualNet = MutualNet().to(device)

        self.transformer = TransformerModel(
            embedding_dim=512,
            num_layer=4,
            num_head=8,
            hidden_dim=4096,
            dropout_rate=0.1,
            attn_dropout_rate=0.1,
        )
        self.pre_head_ln = nn.LayerNorm(512)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.Conv2d, nn.Conv1d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm3d, nn.GroupNorm, nn.InstanceNorm3d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _reshape_output(self, x):
        x = x.view(
            x.size(0),160,160,9)
        x = x.permute(0, 3, 1, 2).contiguous()

        return x




    def forward(self,x):
        # t1, t1ce, t2, flair
        # Cross-Modality Interaction Encoders
        # x = self.multimodal_conv(x)


        down1 = self.encoder1(x)

        down2 = self.downsample(down1)
        down2 = self.encoder2(down2)


        down3 = self.downsample(down2)
        down3 = self.encoder3(down3)


        down4 = self.downsample(down3)
        down4 = self.encoder4(down4)


        #contour
        cag3=self.cag3(down4,down3)

        up_concat3=self.upsample3(down4,cag3)

        up3=self.decoder3(up_concat3)


        # gating_2 = gating_signal(up3, 128, batch_norm=True)
        cag2 = self.cag2(up3, down2)
        up_concat3 = self.upsample2(up3,cag2)
        up2 = self.decoder2(up_concat3)


        # gating_1 = gating_signal(up2, 64, batch_norm=True)
        cag1 = self.cag1(up2, down1)
        up_concat2 = self.upsample1(up2,cag1)
        up1 = self.decoder1(up_concat2)


        #region
        up_skip3 = self.upsample3(down4, down3)
        up3_r = self.decoder3(up_skip3)

        up_skip2 = self.upsample2(up3_r , down2)
        up2_r = self.decoder2(up_skip2)

        up_skip1 = self.upsample1(up2_r, down1)
        up1_r = self.decoder1(up_skip1)

        out_contour = self.outconv(up1)
        out_region= self.outconv(up1_r)
        


   #mim
        softmaxed_out_region= F.softmax(out_region, dim=-1)
        probability_maps = torch.split(softmaxed_out_region, 1, dim=1)
        weighted_maps = [out_region * prob_map for prob_map in probability_maps]
        out_labeled = torch.cat(weighted_maps, dim=1)

        out_labeled= out_labeled .permute(0, 2, 3, 1).contiguous()
        out_labeled = out_labeled.view(out_labeled.size(0), -1, 512)

        out_trans= self.transformer(out_labeled)
        out_trans = self.pre_head_ln(out_trans)
        print("out_trans.shape())")
        print(out_trans.size())
        out_trans=self._reshape_output(out_trans)


        region=self.MutualNet(up1, up1_r)
    


        return   region,out_contour,out_region




