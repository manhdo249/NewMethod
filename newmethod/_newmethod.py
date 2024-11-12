# # Version 1
# import torch
# from torch import nn
# import torch.nn.functional as F

# from ._ETP import ETP
# from ._model_utils import pairwise_euclidean_distance


# class newmethod(nn.Module):
#     def __init__(self, num_topics: int, theta_temp: float=1.0, DT_alpha: float=3.0, TW_alpha: float=2.0,
#                  DC_alpha: float=2.5, TC_alpha: float=2.5):
#         super().__init__()

#         # Tham số cho các hàm mất mát
#         self.num_topics = num_topics
#         self.theta_temp = theta_temp
#         self.DT_alpha = DT_alpha  # doc-topic
#         self.TW_alpha = TW_alpha  # topic-word
#         self.DC_alpha = DC_alpha  # doc-cluster
#         self.TC_alpha = TC_alpha  # topic-cluster

#         self.epsilon = 1e-12

#     def init(self, vocab_size: int, embed_size: int, num_clusters: int):
#         # Tạo embedding cho từ, chủ đề, và cụm
#         self.word_embeddings = nn.init.trunc_normal_(torch.empty(vocab_size, embed_size))
#         self.word_embeddings = nn.Parameter(F.normalize(self.word_embeddings))

#         self.topic_embeddings = torch.empty((self.num_topics, embed_size))
#         nn.init.trunc_normal_(self.topic_embeddings, std=0.1)
#         self.topic_embeddings = nn.Parameter(F.normalize(self.topic_embeddings))

#         self.cluster_embeddings = nn.Parameter(torch.empty(num_clusters, embed_size))
#         nn.init.trunc_normal_(self.cluster_embeddings, std=0.1)
#         self.cluster_embeddings = nn.Parameter(F.normalize(self.cluster_embeddings))

#         # Các trọng số ban đầu
#         self.word_weights = nn.Parameter((torch.ones(vocab_size) / vocab_size).unsqueeze(1))
#         self.topic_weights = nn.Parameter((torch.ones(self.num_topics) / self.num_topics).unsqueeze(1))
#         self.cluster_weights = nn.Parameter((torch.ones(num_clusters) / num_clusters).unsqueeze(1))

#         # Khởi tạo các đối tượng ETP cho các hàm mất mát
#         self.DT_ETP = ETP(self.DT_alpha, init_b_dist=self.topic_weights)
#         self.TW_ETP = ETP(self.TW_alpha, init_b_dist=self.word_weights)
#         self.DC_ETP = ETP(self.DC_alpha, init_b_dist=self.cluster_weights)
#         self.TC_ETP = ETP(self.TC_alpha, init_b_dist=self.cluster_weights)

#     def get_transp_DT(self,
#                       doc_embeddings,
#                     ):

#         topic_embeddings = self.topic_embeddings.detach().to(doc_embeddings.device)
#         _, transp = self.DT_ETP(doc_embeddings, topic_embeddings)

#         return transp.detach().cpu().numpy()

#     # only for testing
#     def get_beta(self):
#         _, transp_TW = self.TW_ETP(self.topic_embeddings, self.word_embeddings)
#         # use transport plan as beta
#         beta = transp_TW * transp_TW.shape[0]

#         return beta

#     # only for testing
#     def get_theta(self,
#                   doc_embeddings,
#                   train_doc_embeddings
#                 ):
#         topic_embeddings = self.topic_embeddings.detach().to(doc_embeddings.device)
#         dist = pairwise_euclidean_distance(doc_embeddings, topic_embeddings)
#         train_dist = pairwise_euclidean_distance(train_doc_embeddings, topic_embeddings)

#         exp_dist = torch.exp(-dist / self.theta_temp)
#         exp_train_dist = torch.exp(-train_dist / self.theta_temp)

#         theta = exp_dist / (exp_train_dist.sum(0))
#         theta = theta / theta.sum(1, keepdim=True)

#         return theta

#     def forward(self, train_bow, doc_embeddings, cluster_labels):
#         # Tính `loss doc-topic`
#         loss_DT, transp_DT = self.DT_ETP(doc_embeddings, self.topic_embeddings)

#         # Tính `loss topic-word`
#         loss_TW, transp_TW = self.TW_ETP(self.topic_embeddings, self.word_embeddings)

#         # Tính `loss doc-cluster`
#         cluster_embeddings = self.calculate_cluster_centers(doc_embeddings, cluster_labels)
#         loss_DC, transp_DC = self.DC_ETP(doc_embeddings, cluster_embeddings)

#         # Tính `loss topic-cluster`
#         loss_TC, transp_TC = self.TC_ETP(self.topic_embeddings, cluster_embeddings)

#         # Tính `loss DSR`
#         theta = transp_DT * transp_DT.shape[0]
#         beta = transp_TW * transp_TW.shape[0]
#         recon = torch.matmul(theta, beta)
#         loss_DSR = -(train_bow * (recon + self.epsilon).log()).sum(axis=1).mean()

#         # Tổng hợp các loss
#         total_loss = loss_DSR + loss_DT + loss_TW + loss_DC + loss_TC

#         return {
#             'loss': total_loss,
#             'loss_DSR': loss_DSR,
#             'loss_DT': loss_DT,
#             'loss_TW': loss_TW,
#             'loss_DC': loss_DC,
#             'loss_TC': loss_TC
#         }
    
#     def calculate_cluster_centers(self, doc_embeddings, cluster_labels):
#         # Tính trung bình của các doc embeddings trong mỗi cụm dựa trên cluster_labels
#         unique_labels = torch.unique(cluster_labels)
#         cluster_centers = torch.stack([doc_embeddings[cluster_labels == label].mean(dim=0) for label in unique_labels])
#         return cluster_centers

# Version 2
import torch
from torch import nn
import torch.nn.functional as F

from ._ETP import ETP
from ._model_utils import pairwise_euclidean_distance


class newmethod(nn.Module):
    def __init__(self, num_topics: int, theta_temp: float=1.0, DT_alpha: float=3.0, TW_alpha: float=2.0,
                 DC_alpha: float=2.5, TC_alpha: float=2.5):
        super().__init__()

        # Tham số cho các hàm mất mát
        self.num_topics = num_topics
        self.theta_temp = theta_temp
        self.DT_alpha = DT_alpha  # doc-topic
        self.TW_alpha = TW_alpha  # topic-word
        self.DC_alpha = DC_alpha  # doc-cluster
        self.TC_alpha = TC_alpha  # topic-cluster

        self.epsilon = 1e-12

    def init(self, vocab_size: int, embed_size: int, num_clusters: int):
        # Tạo embedding cho từ, chủ đề, và cụm
        self.word_embeddings = nn.init.trunc_normal_(torch.empty(vocab_size, embed_size))
        self.word_embeddings = nn.Parameter(F.normalize(self.word_embeddings))

        self.topic_embeddings = torch.empty((self.num_topics, embed_size))
        nn.init.trunc_normal_(self.topic_embeddings, std=0.1)
        self.topic_embeddings = nn.Parameter(F.normalize(self.topic_embeddings))

        self.cluster_embeddings = nn.Parameter(torch.empty(num_clusters, embed_size))
        nn.init.trunc_normal_(self.cluster_embeddings, std=0.1)
        self.cluster_embeddings = nn.Parameter(F.normalize(self.cluster_embeddings))

        # Các trọng số ban đầu
        self.word_weights = nn.Parameter((torch.ones(vocab_size) / vocab_size).unsqueeze(1))
        self.topic_weights = nn.Parameter((torch.ones(self.num_topics) / self.num_topics).unsqueeze(1))
        self.cluster_weights = nn.Parameter((torch.ones(num_clusters) / num_clusters).unsqueeze(1))

        # Khởi tạo các đối tượng ETP cho các hàm mất mát
        self.DT_ETP = ETP(self.DT_alpha, init_b_dist=self.topic_weights)
        self.TW_ETP = ETP(self.TW_alpha, init_b_dist=self.word_weights)
        self.DC_ETP = ETP(self.DC_alpha, init_b_dist=self.cluster_weights)
        self.TC_ETP = ETP(self.TC_alpha, init_b_dist=self.cluster_weights)

    def get_transp_DT(self,
                      doc_embeddings,
                    ):

        topic_embeddings = self.topic_embeddings.detach().to(doc_embeddings.device)
        _, transp = self.DT_ETP(doc_embeddings, topic_embeddings)

        return transp.detach().cpu().numpy()

    # only for testing
    def get_beta(self):
        _, transp_TW = self.TW_ETP(self.topic_embeddings, self.word_embeddings)
        # use transport plan as beta
        beta = transp_TW * transp_TW.shape[0]

        return beta

    # only for testing
    def get_theta(self,
                  doc_embeddings,
                  train_doc_embeddings
                ):
        topic_embeddings = self.topic_embeddings.detach().to(doc_embeddings.device)
        dist = pairwise_euclidean_distance(doc_embeddings, topic_embeddings)
        train_dist = pairwise_euclidean_distance(train_doc_embeddings, topic_embeddings)

        exp_dist = torch.exp(-dist / self.theta_temp)
        exp_train_dist = torch.exp(-train_dist / self.theta_temp)

        theta = exp_dist / (exp_train_dist.sum(0))
        theta = theta / theta.sum(1, keepdim=True)

        return theta
    
    def global_encode(self, input):
        e1 = F.softplus(self.fc11(input))
        e1 = F.softplus(self.fc12(e1))
        e1 = self.fc1_dropout(e1)
        mu = self.mean_bn(self.fc21(e1))
        logvar = self.logvar_bn(self.fc22(e1))
        z = self.global_reparameterize(mu, logvar)
        theta = F.softmax(z, dim=1)

        loss_KL = self.compute_loss_global_KL(mu, logvar)

        return theta, loss_KL

    def global_reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + (eps * std)
        else:
            return mu
    
    def compute_loss_global_KL(self, mu, logvar):
        var = logvar.exp()
        var_division = var / self.var2
        diff = mu - self.mu2
        diff_term = diff * diff / self.var2
        logvar_division = self.var2.log() - logvar
        # KLD: N*K
        KLD = 0.5 * ((var_division + diff_term + logvar_division).sum(axis=1) - self.num_topics)
        KLD = KLD.mean()
        return KLD
    
    def noise_local_encode(self, input):
        e1 = F.softplus(self.fc11_noise(input))
        e1 = F.softplus(self.fc12_noise(e1))
        e1 = self.fc1_noise_dropout(e1)
        mu = self.noise_mean_bn(self.fc21_noise(e1))
        logvar = self.noise_logvar_bn(self.fc22_noise(e1))
        z = self.noise_local_reparameterize(mu, logvar)
        theta = F.softmax(z, dim=1)

        loss_KL = self.compute_loss_local_KL(mu, logvar)

        return theta, loss_KL

    def noise_local_reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + (eps * std)
        else:
            return mu 
        
    def compute_loss_local_KL(self, mu, logvar):
        var = logvar.exp()
        var_division = var / self.doc_noise_var
        diff = mu - self.doc_noise_mu
        diff_term = diff * diff / self.doc_noise_var
        logvar_division = self.doc_noise_var.log() - logvar
        # KLD: N*K
        KLD = 0.5 * ((var_division +  diff_term + logvar_division).sum(axis=1) - self.num_topics)
        KLD = KLD.mean()
        return KLD

    def forward(self, input, doc_embeddings, cluster_labels):
        local_x = input[:, :self.vocab_size]
        global_x = input[:, self.vocab_size:]
        local_noise_theta, local_noise_loss_KL = self.noise_local_encode(local_x)
        global_theta, global_loss_KL = self.global_encode(global_x)
        beta = self.get_beta()

        recon = F.softmax(self.decoder_bn(torch.matmul(global_theta * local_noise_theta, beta)), dim=-1)
        recon_loss = -((local_x + self.alpha_augment*global_x) * recon.log()).sum(axis=1).mean()

        loss_TM = recon_loss + global_loss_KL + local_noise_loss_KL

        # Tính `loss doc-topic`
        loss_DT, transp_DT = self.DT_ETP(doc_embeddings, self.topic_embeddings)

        # Tính `loss topic-word`
        loss_TW, transp_TW = self.TW_ETP(self.topic_embeddings, self.word_embeddings)

        # Tính `loss doc-cluster`
        cluster_embeddings = self.calculate_cluster_centers(doc_embeddings, cluster_labels)
        loss_DC, transp_DC = self.DC_ETP(doc_embeddings, cluster_embeddings)

        # Tính `loss topic-cluster`
        loss_TC, transp_TC = self.TC_ETP(self.topic_embeddings, cluster_embeddings)

        # Tính `loss DSR`
        theta = transp_DT * transp_DT.shape[0]
        beta = transp_TW * transp_TW.shape[0]
        recon = torch.matmul(theta, beta)
        # loss_DSR = -(train_bow * (recon + self.epsilon).log()).sum(axis=1).mean()

        # Tổng hợp các loss
        total_loss = loss_TM + loss_DT + loss_TW + loss_DC + loss_TC

        return {
            'loss': total_loss,
            'loss_DSR': loss_TM,
            'loss_DT': loss_DT,
            'loss_TW': loss_TW,
            'loss_DC': loss_DC,
            'loss_TC': loss_TC
        }
    
    def calculate_cluster_centers(self, doc_embeddings, cluster_labels):
        # Tính trung bình của các doc embeddings trong mỗi cụm dựa trên cluster_labels
        unique_labels = torch.unique(cluster_labels)
        cluster_centers = torch.stack([doc_embeddings[cluster_labels == label].mean(dim=0) for label in unique_labels])
        return cluster_centers
