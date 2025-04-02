"""Graph2Vec module."""

import os
import json
import glob
import hashlib
import pandas as pd
import networkx as nx
from tqdm import tqdm
from joblib import Parallel, delayed
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import logging

logger = logging.getLogger(__name__)

class WeisfeilerLehmanMachine:
    """
    Weisfeiler Lehman feature extractor class.
    """
    def __init__(self, graph, features, iterations):
        """
        Initialization method which also executes feature extraction.
        :param graph: The Nx graph object.
        :param features: Feature hash table.
        :param iterations: Number of WL iterations.
        """
        self.iterations = iterations
        self.graph = graph
        self.features = features
        self.nodes = self.graph.nodes()
        self.extracted_features = [str(v) for k, v in features.items()]
        self.do_recursions()

    def do_a_recursion(self):
        """
        The method does a single WL recursion.
        :return new_features: The hash table with extracted WL features.
        """
        new_features = {}
        for node in self.nodes:
            nebs = self.graph.neighbors(node)
            degs = [self.features[neb] for neb in nebs]
            features = [str(self.features[node])]+sorted([str(deg) for deg in degs])
            features = "_".join(features)
            hash_object = hashlib.md5(features.encode())
            hashing = hash_object.hexdigest()
            new_features[node] = hashing
        self.extracted_features = self.extracted_features + list(new_features.values())
        return new_features

    def do_recursions(self):
        """
        The method does a series of WL recursions.
        """
        for _ in range(self.iterations):
            self.features = self.do_a_recursion()

def path2name(path):
    base = os.path.basename(path)
    return os.path.splitext(base)[0]

def dataset_reader(path):
    """
    Function to read the graph and features from a json file.
    :param path: The path to the graph json.
    :return graph: The graph object.
    :return features: Features hash table.
    :return name: Name of the graph.
    """
    name = path2name(path)
    data = json.load(open(path))
    graph = nx.from_edgelist(data["edges"])

    if "features" in data.keys():
        features = data["features"]
        features = {int(k): v for k, v in features.items()}
    else:
        features = nx.degree(graph)
        features = {int(k): v for k, v in features}
       
    return graph, features, name

def feature_extractor(path, rounds):
    """
    Function to extract WL features from a graph.
    :param path: The path to the graph json.
    :param rounds: Number of WL iterations.
    :return doc: Document collection object.
    """
    graph, features, name = dataset_reader(path)
    machine = WeisfeilerLehmanMachine(graph, features, rounds)
    doc = TaggedDocument(words=machine.extracted_features, tags=["g_" + name])
    return doc

class Graph2Vec:
    def __init__(self, input_path, dimensions=128, min_count=2, workers=10):
        self.input_path = input_path
        self.dimensions = dimensions
        self.wl_iterations = 2
        self.min_count = min_count
        self.down_sampling = 0.0001
        self.workers = workers
        self.epochs = 10
        self.learning_rate = 0.025
        self.model = None

    def get_embeddings(self):
        graphs = glob.glob(os.path.join(self.input_path, "*.json"))
        graphs.sort(key=lambda x: int(path2name(x)))
        print("\nCFG feature extraction started.\n")
        document_collections = Parallel(n_jobs=self.workers)(delayed(feature_extractor)(g, self.wl_iterations) for g in tqdm(graphs))
        print("\nGenerating CFG vectors...\n")
        #print(document_collections)
        model = Doc2Vec(document_collections,
                        vector_size=self.dimensions,
                        window=0,
                        min_count=self.min_count,
                        dm=0,
                        sample=self.down_sampling,
                        workers=self.workers,
                        epochs=self.epochs,
                        alpha=self.learning_rate)
        embeddings = []
        for f in graphs:
            identifier = path2name(f)
            embeddings.append(list(model.dv["g_"+identifier]))
        
        return embeddings

        