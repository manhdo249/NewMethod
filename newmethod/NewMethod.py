import numpy as np
import pandas as pd
from collections import defaultdict
from pathlib import Path

import torch
from topmost.utils._utils import get_top_words
from topmost.data import RawDataset
from topmost.preprocessing import Preprocessing

from tqdm import tqdm

from . import _plot
from ._newmethod import newmethod
from ._utils import Logger, check_fitted, DocEmbedModel

from typing import List, Tuple, Union, Mapping, Any, Callable


logger = Logger("WARNING")


class NewMethod:
    def __init__(
        self,
        num_topics: int,
        num_clusters: int,
        preprocessing: Preprocessing = None,
        doc_embed_model: Union[str, Callable] = "all-MiniLM-L6-v2",
        num_top_words: int = 15,
        DT_alpha: float = 3.0,
        TW_alpha: float = 2.0,
        DC_alpha: float = 2.5,
        TC_alpha: float = 2.5,
        theta_temp: float = 1.0,
        epochs: int = 200,
        learning_rate: float = 0.002,
        device: str = None,
        normalize_embeddings: bool = False,
        save_memory: bool = False,
        batch_size: int = None,
        log_interval: int = 10,
        verbose: bool = False,
    ):
        """
        NewMethod initialization with new parameters for doc-cluster and topic-cluster losses.

        Args:
            num_topics: The number of topics.
            num_clusters: The number of clusters for doc-cluster and topic-cluster losses.
            preprocessing: Preprocessing class from topmost.preprocessing.Preprocessing
            doc_embed_model: The document embedding model or a callable with `.encode(docs)`.
            num_top_words: The number of top words to return per topic.
            DT_alpha: Sinkhorn alpha between doc and topic embeddings.
            TW_alpha: Sinkhorn alpha between topic and word embeddings.
            DC_alpha: Sinkhorn alpha between doc and cluster embeddings.
            TC_alpha: Sinkhorn alpha between topic and cluster embeddings.
            theta_temp: Softmax temperature for computing doc-topic distributions.
            epochs: Number of training epochs.
            learning_rate: Learning rate.
            device: Device to run the model on.
            normalize_embeddings: Whether to normalize document embeddings.
            save_memory: If True, uses batch training to save memory.
            batch_size: Batch size for memory saving.
            log_interval: Interval for logging training progress.
            verbose: Enables verbose logging if set to True.
        """
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        self.save_memory = save_memory
        self.batch_size = batch_size

        self.num_top_words = num_top_words
        self.epochs = epochs
        self.learning_rate = learning_rate

        self.preprocessing = preprocessing
        self.doc_embed_model = doc_embed_model
        self.num_clusters = num_clusters
        self.vocab = None
        self.doc_embedder = None
        self.train_doc_embeddings = None
        self.normalize_embeddings = normalize_embeddings

        self.beta = None
        self.train_theta = None

        # Initialize newmethod model with the additional cluster loss parameters
        self.model = newmethod(num_topics, theta_temp, DT_alpha, TW_alpha, DC_alpha, TC_alpha)

        self.log_interval = log_interval
        self.verbose = verbose
        if verbose:
            logger.set_level("DEBUG")
        else:
            logger.set_level("WARNING")

        logger.info(f'use device: {device}')

    def make_optimizer(self, learning_rate: float):
        return torch.optim.Adam(self.model.parameters(), lr=learning_rate)

    def fit(
        self,
        docs: List[str]
    ):
        self.fit_transform(docs)
        return self

    def fit_transform(self, docs: List[str]):
        # Preprocess documents and prepare embeddings
        data_size = len(docs)
        if self.save_memory:
            assert self.batch_size is not None
        else:
            self.batch_size = data_size

        dataset_device = 'cpu' if self.save_memory else self.device
        self.doc_embedder = DocEmbedModel(self.doc_embed_model, self.normalize_embeddings, self.device)
        dataset = RawDataset(
            docs,
            self.preprocessing,
            batch_size=self.batch_size,
            device=dataset_device,
            pretrained_WE=False,
            contextual_embed=True,
            doc_embed_model=self.doc_embedder,
            embed_model_device=self.device,
            verbose=self.verbose,
        )

        self.train_doc_embeddings = torch.as_tensor(dataset.train_contextual_embed)

        if not self.save_memory:
            self.train_doc_embeddings = self.train_doc_embeddings.to(self.device)

        self.vocab = dataset.vocab
        embed_size = dataset.contextual_embed_size
        vocab_size = dataset.vocab_size

        # Initialize the newmethod model with vocab and embed size, and the number of clusters
        self.model.init(vocab_size, embed_size, self.num_clusters)
        self.model = self.model.to(self.device)

        optimizer = self.make_optimizer(self.learning_rate)

        self.model.train()
        for epoch in tqdm(range(1, self.epochs + 1), desc="Training NewMethod"):
            loss_rst_dict = defaultdict(float)

            # Cluster the document embeddings
            cluster_labels = self.cluster_docs(self.train_doc_embeddings, self.num_clusters)

            for batch_data in dataset.train_dataloader:
                batch_bow = batch_data[:, :vocab_size]
                batch_doc_emb = batch_data[:, vocab_size:]

                if self.save_memory:
                    batch_doc_emb = batch_doc_emb.to(self.device)
                    batch_bow = batch_bow.to(self.device)

                # Forward pass with cluster labels
                rst_dict = self.model(batch_bow, batch_doc_emb, cluster_labels)
                batch_loss = rst_dict['loss']

                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()

                for key in rst_dict:
                    loss_rst_dict[key] += rst_dict[key].item() * batch_data.shape[0]

            if epoch % self.log_interval == 0:
                output_log = f'Epoch: {epoch:03d}'
                for key in loss_rst_dict:
                    output_log += f' {key}: {loss_rst_dict[key] / data_size :.3f}'
                logger.info(output_log)

        self.beta = self.get_beta()
        self.top_words = self.get_top_words(self.num_top_words)
        self.train_theta = self.transform(self, doc_embeddings=self.train_doc_embeddings)

        return self.top_words, self.train_theta

    def cluster_docs(self, doc_embeddings, num_clusters):
        # Use KMeans to cluster document embeddings
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=num_clusters).fit(doc_embeddings.cpu().numpy())
        return torch.tensor(kmeans.labels_).to(doc_embeddings.device)

    def transform(self, docs: List[str] = None, doc_embeddings: np.ndarray = None):
        if docs is None and doc_embeddings is None:
            raise ValueError("Must set either docs or doc_embeddings.")

        if doc_embeddings is None and self.doc_embedder is None:
            raise ValueError("Must set doc embeddings.")

        if doc_embeddings is None:
            doc_embeddings = torch.as_tensor(self.doc_embedder.encode(docs))
            if not self.save_memory:
                doc_embeddings = doc_embeddings.to(self.device)

        with torch.no_grad():
            self.model.eval()
            theta = self.model.get_theta(doc_embeddings, self.train_doc_embeddings)
            theta = theta.detach().cpu().numpy()

        return theta

    def get_beta(self):
        beta = self.model.get_beta().detach().cpu().numpy()
        return beta

    def get_top_words(self, num_top_words=15, verbose=None):
        if verbose is None:
            verbose = self.verbose
        beta = self.get_beta()
        top_words = get_top_words(beta, self.vocab, num_top_words, verbose)
        return top_words

    @property
    def topic_embeddings(self):
        return self.model.topic_embeddings.detach().cpu().numpy()

    @property
    def word_embeddings(self):
        return self.model.word_embeddings.detach().cpu().numpy()

    @property
    def transp_DT(self):
        return self.model.get_transp_DT(self.train_doc_embeddings)

    def save(self, path: str):
        check_fitted(self)

        path = Path(path)
        parent_dir = path.parent
        if not parent_dir.exists():
            parent_dir.mkdir(parents=True, exist_ok=True)

        instance_dict = {key: value for key, value in self.__dict__.items() if key not in ['doc_embedder']}
        state = {"instance_dict": instance_dict}
        torch.save(state, path)

    @classmethod
    def from_pretrained(cls, path: str, device: str = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        state = torch.load(path, map_location=device)
        instance_dict = state["instance_dict"]
        instance_dict["device"] = device

        instance = cls.__new__(cls)
        instance.__dict__.update(instance_dict)

        instance.doc_embedder = DocEmbedModel(
            instance_dict["doc_embed_model"],
            instance_dict["normalize_embeddings"],
            instance_dict["device"],
        )

        return instance

    def get_topic(self, topic_idx: int, num_top_words: int = 5):
        check_fitted(self)
        words = self.top_words[topic_idx].split()[:num_top_words]
        scores = np.sort(self.beta[topic_idx])[:-(num_top_words + 1):-1]
        return tuple(zip(words, scores))

    def get_topic_weights(self):
        check_fitted(self)
        topic_weights = self.transp_DT.sum(0)
        return topic_weights

    def visualize_topic(self, **args):
        check_fitted(self)
        return _plot.visualize_topic(self, **args)

    def visualize_topic_hierarchy(self, **args):
        check_fitted(self)
        return _plot.visualize_hierarchy(self, **args)

    def topic_activity_over_time(self, time_slices: List[int]):
        check_fitted(self)
        topic_activity = self.transp_DT * self.transp_DT.shape[0]
        assert len(time_slices) == topic_activity.shape[0]
        df = pd.DataFrame(topic_activity)
        df['time_slices'] = time_slices
        topic_activity = df.groupby('time_slices').mean().to_numpy().transpose()
        return topic_activity

    def visualize_topic_activity(self, **args):
        check_fitted(self)
        return _plot.visualize_activity(self, **args)

    def visualize_topic_weights(self, **args):
        check_fitted(self)
        return _plot.visualize_topic_weights(self, **args)

