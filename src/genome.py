from typing import Any
from genes import DefaultNodeGene, DefaultConnectionGene, NeuralNodeGene, NeuralConnectionGene


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

    def mutate_add_node(self, config: object) -> None:
        pass

    def mutate_del_node(self, config: object) -> None:
        pass

    def mutate_add_connection(self, config: object) -> None:
        pass

    def mutate_del_connection(self, config: object) -> None:
        pass

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
        'output_keys': [0, 1],
        'num_outputs': 1,

        'weight_init_type': 'normal',
        'weight_default_value': 0.0,
        'weight_mean': 0.0,
        'weight_stdev': 1.0,
        'weight_max_value': 2.0,
        'weight_min_value': -2.0,
        'weight_mutation_power': 1.0,
        'weight_mutation_rate': 0.6,
        'weight_replace_rate': 0.2,

    }
    from collections import namedtuple
    yp = namedtuple('config', y.keys())(*y.values())