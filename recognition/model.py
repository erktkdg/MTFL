""" Reference source: https://github.com/tianyu0207/RTFM"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as torch_init
torch.set_default_tensor_type('torch.FloatTensor')


def weight_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        torch_init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)


class CVA(nn.Module):
    def __init__(self, input_dim=1024):
        """
        Cross-View Attention (CVA) module.

        Args:
            input_dim (int): Dimension of the input features.
        """
        super(CVA, self).__init__()
        drop_out_rate = 0.1
        num_heads = 4
        self.cross_attention = nn.MultiheadAttention(embed_dim=input_dim, num_heads=num_heads, dropout=drop_out_rate,
                                                     device='cuda')

    def forward(self, feature1, feature2):
        """
        Args:
            feature1 (torch.Tensor): one path features. Shape: B x T x C.
            feature2 (torch.Tensor): another path features. Shape: B x T x C.

        Returns:
            out1 (torch.Tensor): Processed features after cross-attention. Shape: B x T x C.
        """

        feature1 = F.layer_norm(feature1, [feature1.size(-1)])
        feature2 = F.layer_norm(feature2, [feature2.size(-1)])
        feature1 = feature1.permute(1, 0, 2)  # T B C
        feature2 = feature2.permute(1, 0, 2)

        out1, _ = self.cross_attention(query=feature1, key=feature2, value=feature2)  # T B C (For test:32 1 1024)
        out1 = out1 + feature1  # residual connection

        return out1  # B T C


class Aggregate(nn.Module):
    def __init__(self, input_dim):
        """
        An aggregate network including local temporal correlation learning, global temporal correlation learning,
            and feature fusion in MTFF.

        Args:
            input_dim (int): input features dim.
        """
        super(Aggregate, self).__init__()
        bn = nn.BatchNorm1d
        num_heads = 4
        self.input_dim = input_dim
        self.conv_1 = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=512, kernel_size=3,
                      stride=1,dilation=1, padding=1),
            nn.LeakyReLU(negative_slope=5e-2),
            bn(512)
        )
        self.conv_2 = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=512, kernel_size=3,
                      stride=1, dilation=2, padding=2),
            nn.LeakyReLU(negative_slope=5e-2),
            bn(512)
        )
        self.conv_3 = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=512, kernel_size=3,
                      stride=1, dilation=4, padding=4),
            nn.LeakyReLU(negative_slope=5e-2),
            bn(512)
        )
        self.conv_4 = nn.Sequential(
            nn.Conv1d(in_channels=input_dim*3, out_channels=512, kernel_size=1,
                      stride=1, padding=0, bias = False),
            nn.LeakyReLU(negative_slope=5e-2),
        )
        self.conv_5 = nn.Sequential(
            nn.Conv1d(in_channels=2048, out_channels=input_dim, kernel_size=3,
                      stride=1, padding=1, bias=False),
            nn.LeakyReLU(negative_slope=5e-2),
            nn.BatchNorm1d(input_dim),
        )
        self.self_attention = nn.MultiheadAttention(embed_dim=512, num_heads=num_heads,
                                                    dropout=0.1, device='cuda')

    def forward(self, input1, input2, input3):
        """
        Args:
            input1 (torch.Tensor): long-frame-length features. Shape: T x B x C.
            input2 (torch.Tensor): medium-frame-length features. Shape: T x B x C.
            input3 (torch.Tensor): short-frame-length features. Shape: T x B x C.

        Returns:
            torch.Tensor: Processed and fused output features. Shape: B x T x C.
        """
        x1 = input1.permute(1, 2, 0)  # B C T
        x2 = input2.permute(1, 2, 0)
        x3 = input3.permute(1, 2, 0)
        tensor_list = [x1, x2, x3]

        residual = torch.mean(torch.stack(tensor_list), dim=0)

        out1 = self.conv_1(x1)  # B C/2 T
        out2 = self.conv_2(x2)
        out3 = self.conv_3(x3)
        x = torch.cat([out1, out2, out3], dim=1)  # B 3C/2 T

        feature = torch.cat((x1, x2, x3), dim=1)
        out = self.conv_4(feature)
        out = out.permute(2, 0, 1)  # T B C/2
        out = F.layer_norm(out, normalized_shape=[out.size(-1)])
        out, _ = self.self_attention(out, out, out)  # T B C/2
        out = out.permute(1, 2, 0)  # B C/2 T
        out = torch.cat((x, out), dim=1)  # B 2C T
        out = self.conv_5(out)   # fuse all the features together
        out = out + residual
        out = out.permute(0, 2, 1)

        return out


class Encoder(nn.Module):
    def __init__(self, input_dim=1024, seg_num=32):
        """
        Multi-Temporal Feature Fusion (MTFF) module.

        Args:
            input_dim (int): Dimension of the input features.
            seg_num (int): Number of snippets in a video.
        """
        super(Encoder, self).__init__()
        self.drop_out_rate = 0.1
        self.input_dim = input_dim
        self.min_temporal_dim = seg_num
        self.CVA1 = CVA(input_dim=input_dim)
        self.CVA2 = CVA(input_dim=input_dim)
        self.CVA3 = CVA(input_dim=input_dim)

        self.aggregate = Aggregate(input_dim=input_dim)

    def forward(self, feature1, feature2, feature3):
        """
        Args:
            feature1 (torch.Tensor): long-frame-length features. Shape: B x T x C.
                (Batch size X The number of snippets x Input dimensions)
            feature2 (torch.Tensor): medium-frame-length features. Shape: B x T x C.
            feature3 (torch.Tensor): short-frame-length features. Shape: B x T x C.

        Returns:
            torch.Tensor: Fused and processed output features. Shape: B x T x C.
        """

        att1 = self.CVA1(feature1, feature2)
        att2 = self.CVA2(feature2, feature3)
        att3 = self.CVA3(feature3, feature1)

        out1 = self.aggregate(att1, att2, att3)  # B T C

        return out1


class Model(nn.Module):
    def __init__(self, feature_dim, batch_size, seg_num=32):
        """
        Multi-Temporal Feature Learning (MTFL) recognition model.

        Args:
            feature_dim (int): Dimension of the input features.
            batch_size (int): Batch size.
            seg_num (int): Number of snippets in a video.
        """
        super(Model, self).__init__()
        self.batch_size = batch_size
        self.num_segments = seg_num
        self.k_abn = self.num_segments // 10  # select 3 snippets
        self.k_nor = self.num_segments // 10

        self.Encoder = Encoder(input_dim=feature_dim, seg_num=seg_num)

        # Fully connected layers for classification
        self.fc1 = nn.Linear(feature_dim, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 18)  # class amount = 18

        self.drop_out = nn.Dropout(0.2)
        self.relu = nn.LeakyReLU(negative_slope=5e-2)
        self.sigmoid = nn.Sigmoid()
        self.apply(weight_init)

    def forward(self, input1, input2, input3):
        """
        Args:
            input1 (torch.Tensor): long-frame-length features. Shape: B x T x feature_dim.
            input2 (torch.Tensor): medium-frame-length features. Shape: B x T x feature_dim.
            input3 (torch.Tensor): short-frame-length features. Shape: B x T x feature_dim.

        Returns:
            score_abnormal (torch.Tensor): The mean scores for top-3 abnormal instances.
            score_normal (torch.Tensor): The mean scores for top-3 normal instances.
            feat_select_abn (torch.Tensor): Selected abnormal features.
            feat_select_normal (torch.Tensor): Selected normal features.
            scores (torch.Tensor): All computed scores. Shape: B x T x the number of classes (18)
        """
        k_abn = self.k_abn
        k_nor = self.k_nor
        ncrops = 1  # Reserving the parameter for spatial cropping, which is not used and defaults to 1

        # Multi-Temporal Feature Fusion
        out = self.Encoder(input1, input2, input3)
        bs, t, f = out.size()
        features = self.drop_out(out) # B T D

        # classification layers
        scores = self.relu(self.fc1(features))
        scores = self.drop_out(scores)
        scores = self.relu(self.fc2(scores))
        scores = self.drop_out(scores)
        scores = self.sigmoid(self.fc3(scores))
        scores = scores.view(bs, t, -1) # B T 18
        # B * t * f
        normal_features = features[0:self.batch_size]
        normal_scores = scores[0:self.batch_size]

        abnormal_features = features[self.batch_size:]
        abnormal_scores = scores[self.batch_size:]

        # Compute feature magnitudes
        feat_magnitudes = torch.norm(features, p=2, dim=2)
        feat_magnitudes = feat_magnitudes.view(bs, ncrops, -1).mean(1)
        nfea_magnitudes = feat_magnitudes[0:self.batch_size]  # normal feature magnitudes
        afea_magnitudes = feat_magnitudes[self.batch_size:]  # abnormal feature magnitudes
        n_size = nfea_magnitudes.shape[0]

        # Inference mode for batch size 1
        if nfea_magnitudes.shape[0] == 1:
            afea_magnitudes = nfea_magnitudes
            abnormal_scores = normal_scores
            abnormal_features = normal_features

        select_idx = torch.ones_like(nfea_magnitudes)
        select_idx = self.drop_out(select_idx)

        #######  process abnormal videos -> select top3 feature magnitude  #######
        afea_magnitudes_drop = afea_magnitudes * select_idx
        idx_abn = torch.topk(afea_magnitudes_drop, k_abn, dim=1)[1]
        idx_abn_feat = idx_abn.unsqueeze(2).expand([-1, -1, abnormal_features.shape[2]])

        abnormal_features = abnormal_features.view(n_size, ncrops, t, f) # B X N X T X F
        abnormal_features = abnormal_features.permute(1, 0, 2, 3)  # N X B X T X F

        total_select_abn_feature = torch.zeros(0, device=input1.device)
        for abnormal_feature in abnormal_features:
            feat_select_abn = torch.gather(abnormal_feature, 1, idx_abn_feat)   # top 3 features magnitude in abnormal bag
            total_select_abn_feature = torch.cat((total_select_abn_feature, feat_select_abn))

        idx_abn_score = idx_abn.unsqueeze(2).expand([-1, -1, abnormal_scores.shape[2]])
        # top 3 scores in abnormal bag based on the top-3 magnitude
        score_abnormal = torch.mean(torch.gather(abnormal_scores, 1, idx_abn_score), dim=1)


        ####### process normal videos -> select top3 feature magnitude #######

        select_idx_normal = torch.ones_like(nfea_magnitudes)
        select_idx_normal = self.drop_out(select_idx_normal)
        nfea_magnitudes_drop = nfea_magnitudes * select_idx_normal
        idx_normal = torch.topk(nfea_magnitudes_drop, k_nor, dim=1)[1]
        idx_normal_feat = idx_normal.unsqueeze(2).expand([-1, -1, normal_features.shape[2]])

        normal_features = normal_features.view(n_size, ncrops, t, f)
        normal_features = normal_features.permute(1, 0, 2, 3) # 1 B T D

        total_select_nor_feature = torch.zeros(0, device=input1.device)
        for nor_fea in normal_features:
            feat_select_normal = torch.gather(nor_fea, 1, idx_normal_feat)  # top 3 features magnitude in normal bag (hard negative)
            total_select_nor_feature = torch.cat((total_select_nor_feature, feat_select_normal))

        idx_normal_score = idx_normal.unsqueeze(2).expand([-1, -1, normal_scores.shape[2]])
        score_normal = torch.mean(torch.gather(normal_scores, 1, idx_normal_score), dim=1) # top 3 scores in normal bag

        feat_select_abn = total_select_abn_feature
        feat_select_normal = total_select_nor_feature

        return score_abnormal, score_normal, feat_select_abn, feat_select_normal, scores
