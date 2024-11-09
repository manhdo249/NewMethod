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

    def forward(self, train_bow, doc_embeddings, cluster_labels):
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
        loss_DSR = -(train_bow * (recon + self.epsilon).log()).sum(axis=1).mean()

        # Tổng hợp các loss
        total_loss = loss_DSR + loss_DT + loss_TW + loss_DC + loss_TC

        return {
            'loss': total_loss,
            'loss_DSR': loss_DSR,
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



