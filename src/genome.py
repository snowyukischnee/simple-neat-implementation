from typing import Any
from genes import DefaultNodeGene, DefaultConnectionGene, NeuralNodeGene, NeuralConnectionGene
import activation_functions
import aggregation_functions
import random


class DefaultGenome(object):
    def __init__(self, key: Any):
        self.key = key
        self.nodes = {}
        self.connections = {}
        self.fitness = None

    def configure_new(self, config: object) -> None:
        for nk in getattr(config, 'output_keys'):
            self.nodes[nk] = self.create_node(config, nk)

    def configure_crossover(self, parent1: Any, parent2: Any, config: object) -> None:
        assert isinstance(parent1.fitness, float) and isinstance(parent2.fitness, float)
        if parent1.fitness < parent2.fitness:
            parent1, parent2 = parent2, parent1
        for nk1, ng1 in parent1.nodes.items():
            ng2 = parent2.nodes.get(nk1)
            if ng2 is None:
                self.nodes[nk1] = ng1.copy()
            else:
                self.nodes[nk1] = ng1.crossover(ng2)
        for ck1, cg1 in parent1.connections.items():
            cg2 = parent2.connections.get(ck1)
            if cg2 is None:
                self.connections[ck1] = cg1.copy()
            else:
                self.connections[ck1] = cg1.crossover(cg2)

    def distance(self, other: Any, config: object) -> Any:
        node_distance = 0.
        if (len(self.nodes) > 0) or (len(other.nodes) > 0):
            disjoint_nodes = 0
            for k2 in other.nodes:
                if k2 not in self.nodes:
                    disjoint_nodes += 1  # node in other genome but not this genome
            for k1, n1 in self.nodes.items():
                n2 = other.nodes.get(k1)
                if n2 is None:
                    disjoint_nodes += 1  # node in this genome but not other genome
                else:
                    node_distance += n1.distance(n2)
            node_distance = (node_distance + disjoint_nodes) / max(len(self.nodes), len(other.nodes))  # mean distance
        connection_distance = 0.
        if (len(self.connections) > 0) or (len(other.connections) > 0):
            disjoint_connections = 0
            for k2 in other.connections:
                if k2 not in self.connections:
                    disjoint_connections += 1  # connection in other genome but not this genome
            for k1, n1 in self.connections.items():
                n2 = other.connections.get(k1)
                if n2 is None:
                    disjoint_connections += 1  # connection in this genome but not other genome
                else:
                    connection_distance += n1.distance(n2)
            connection_distance = (connection_distance + disjoint_connections) / max(len(self.connections), len(other.connections))  # mean distance
        distance = node_distance + connection_distance
        return distance

    def mutate(self, config: object) -> None:
        add_node_mutation_prob = getattr(config, 'add_node_mutation_prob', 0.0)
        del_node_mutation_prob = getattr(config, 'del_node_mutation_prob', 0.0)
        add_connection_mutation_prob = getattr(config, 'add_connection_mutation_prob', 0.0)
        del_connection_mutation_prob = getattr(config, 'del_connection_mutation_prob', 0.0)
        if random.random() < add_node_mutation_prob:
            self.mutate_add_node(config)
        if random.random() < del_node_mutation_prob:
            self.mutate_del_node(config)
        if random.random() < add_connection_mutation_prob:
            self.mutate_add_connection(config)
        if random.random() < del_connection_mutation_prob:
            self.mutate_del_connection(config)
        for ng in self.nodes.values():
            ng.mutate(config)
        for cg in self.connections.values():
            cg.mutate(config)

    def mutate_add_node(self, config: object) -> None:
        if len(self.connections) == 0:
            return
        conn_to_split = random.choice(list(self.connections.values()))
        new_node_key = len(self.nodes)
        nng = self.create_node(config, new_node_key)
        self.nodes[new_node_key] = nng
        conn_to_split.enabled = False
        inode_key, onode_key = conn_to_split.key
        new_connection_1 = self.create_connection(config, inode_key, new_node_key)
        new_connection_1.weight = 1.0
        new_connection_1.enabled = True
        self.connections[new_connection_1.key] = new_connection_1
        new_connection_2 = self.create_connection(config, new_node_key, onode_key)
        new_connection_2.weight = conn_to_split.weight
        new_connection_1.enabled = True
        self.connections[new_connection_2.key] = new_connection_2

    def mutate_del_node(self, config: object) -> None:
        available_node_keys = [nk for nk in self.nodes.keys() if nk not in getattr(config, 'output_keys')]
        if len(available_node_keys) == 0:
            return
        del_node_key = random.choice(available_node_keys)
        # delete connection that connected to 'will be deleted' node
        for ck, cg in self.connections.items():
            if del_node_key in ck:
                del self.connections[ck]
        del self.nodes[del_node_key]

    def mutate_add_connection(self, config: object) -> None:
        available_onode_keys = list(self.nodes.keys())
        available_inode_keys = list(set(available_onode_keys + getattr(config, 'input_keys')))
        connection_inode = random.choice(available_inode_keys)
        connection_onode = random.choice(available_onode_keys)
        connection_key = (connection_inode, connection_onode)
        if connection_key in self.connections:
            return
        if (connection_inode in getattr(config, 'output_keys')) and (connection_onode in getattr(config, 'output_keys')):
            return
        # STILL NOT UNDERSTAND, COMMENTED
        # if config.feed_forward and creates_cycle(list(self.connections.keys()), connection_key):
        #     return
        ncg = self.create_connection(config, connection_inode, connection_onode)
        self.connections[ncg.key] = ncg

    def mutate_del_connection(self, config: object) -> None:
        if len(self.connections) > 0:
            del_connection_key = random.choice(list(self.connections.keys()))
            del self.connections[del_connection_key]

    @staticmethod
    def create_node(config: object, key: int) -> Any:
        new_node = getattr(config, 'node_gene_type')(key)
        new_node.init_attributes(config)
        return new_node

    @staticmethod
    def create_connection(config: object, inode_key: int, onode_key: int) -> Any:
        new_connection = getattr(config, 'connection_gene_type')((inode_key, onode_key))
        new_connection.init_attributes(config)
        return new_connection


if __name__ == '__main__':
    y = {
        'node_gene_type': NeuralNodeGene,
        'connection_gene_type': NeuralConnectionGene,
        'genome_type': DefaultGenome,
        'compatibility_weight_coefficient': 1.0,
        'compatibility_disjoint_coefficient': 1.0,
        'num_inputs': 2,
        'input_keys': [-1, -2],
        'output_keys': [0],
        'num_outputs': 1,
        'add_node_mutation_prob': 0.99,
        'del_node_mutation_prob': 0.1,
        'add_connection_mutation_prob': 0.99,
        'del_connection_mutation_prob': 0.1,

        'weight_init_type': 'normal',
        'weight_default_value': 0.0,
        'weight_mean': 0.0,
        'weight_stdev': 1.0,
        'weight_max_value': 2.0,
        'weight_min_value': -2.0,
        'weight_mutation_power': 1.0,
        'weight_mutation_rate': 0.6,
        'weight_replace_rate': 0.2,

        'response_init_type': 'normal',
        'response_default_value': 0.0,
        'response_mean': 0.0,
        'response_stdev': 1.0,
        'response_max_value': 2.0,
        'response_min_value': -2.0,
        'response_mutation_power': 1.0,
        'response_mutation_rate': 0.6,
        'response_replace_rate': 0.2,

        'bias_init_type': 'normal',
        'bias_default_value': 0.0,
        'bias_mean': 0.0,
        'bias_stdev': 1.0,
        'bias_max_value': 2.0,
        'bias_min_value': -2.0,
        'bias_mutation_power': 1.0,
        'bias_mutation_rate': 0.6,
        'bias_replace_rate': 0.2,

        'activation_function_def': {
            'sigmoid': activation_functions.SigmoidActivationFunction,
            'tanh': activation_functions.TanhActivationFunction,
            'relu': activation_functions.ReluActivationFunction,
            'gauss': activation_functions.GaussianActivationFunction
        },
        'aggregation_function_def': {
            'sum': aggregation_functions.SumAggregationFunction,
            'mean': aggregation_functions.MeanAggregationFunction,
            'product': aggregation_functions.ProductAggregationFunction
        }
    }
    from collections import namedtuple
    yp = namedtuple('config', y.keys())(*y.values())

    # class Config(object):
    #     node_gene_type = NeuralNodeGene
    #     connection_gene_type = NeuralConnectionGene
    #     genome_type = DefaultGenome
    #     compatibility_weight_coefficient = 1.0
    #     compatibility_disjoint_coefficient = 1.0
    #     num_inputs = 2
    #     input_keys = [-1, -2]
    #     output_keys = [0]
    #     num_outputs = 1
    #     add_node_mutation_prob = 0.99
    #     del_node_mutation_prob = 0.1
    #     add_connection_mutation_prob = 0.99
    #     del_connection_mutation_prob = 0.1
    #
    #     weight_init_type = 'normal'
    #     weight_default_value = 0.0
    #     weight_mean = 0.0
    #     weight_stdev = 1.0
    #     weight_max_value = 2.0
    #     weight_min_value = -2.0
    #     weight_mutation_power = 1.0
    #     weight_mutation_rate = 0.6
    #     weight_replace_rate = 0.2
    #
    #     response_init_type = 'normal'
    #     response_default_value = 0.0
    #     response_mean = 0.0
    #     response_stdev = 1.0
    #     response_max_value = 2.0
    #     response_min_value = -2.0
    #     response_mutation_power = 1.0
    #     response_mutation_rate = 0.6
    #     response_replace_rate = 0.2
    #
    #     bias_init_type = 'normal'
    #     bias_default_value = 0.0
    #     bias_mean = 0.0
    #     bias_stdev = 1.0
    #     bias_max_value = 2.0
    #     bias_min_value = -2.0
    #     bias_mutation_power = 1.0
    #     bias_mutation_rate = 0.6
    #     bias_replace_rate = 0.2
    #
    #     activation_function_def = {
    #         'sigmoid': activation_functions.SigmoidActivationFunction,
    #         'tanh': activation_functions.TanhActivationFunction,
    #         'relu': activation_functions.ReluActivationFunction,
    #         'gauss': activation_functions.GaussianActivationFunction
    #     }
    #     aggregation_function_def = {
    #         'sum': aggregation_functions.SumAggregationFunction,
    #         'mean': aggregation_functions.MeanAggregationFunction,
    #         'product': aggregation_functions.ProductAggregationFunction
    #     }
    # yp = Config()
    x = DefaultGenome('test_genome')
    x.configure_new(yp)
    x.mutate_add_connection(yp)
    x.mutate_add_connection(yp)
    x.mutate_add_connection(yp)
    x.mutate_add_node(yp)
    x.mutate_add_node(yp)
    ng = x.create_node(yp, 3)
    x.nodes[ng.key] = ng
    from utils import required_for_output
    print(x.connections.keys(), x.nodes.keys())
    print(required_for_output(yp.input_keys, yp.output_keys, list(x.connections.keys())))
    xx = x.nodes.get(2)
    print(xx.forward(yp, [1., 2., 4.]))