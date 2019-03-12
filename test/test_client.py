import unittest
import networkx as nx
from cdlib import NodeClustering
from cdlib_rest import CDlib_API


class Client(unittest.TestCase):

    def test_endpoints(self):

        with CDlib_API("http://0.0.0.0", 8081) as api:
            g = nx.karate_club_graph()
            api.load_network(g)

            coms = api.angel(threshold=0.75)
            self.assertIsInstance(coms, NodeClustering)

            coms2 = api.demon(epsilon=0.25)
            self.assertIsInstance(coms2, NodeClustering)

            coms3 = api.ego_networks()
            self.assertIsInstance(coms3, NodeClustering)

            coms4 = api.node_perception()
            self.assertIsInstance(coms4, NodeClustering)

            seed_nodes = [0, 2, 6]
            coms5 = api.overlapping_seed_set_expansion(seeds=seed_nodes)
            self.assertIsInstance(coms5, NodeClustering)

            com6 = api.kclique(k=3)
            self.assertIsInstance(com6, NodeClustering)

            com7 = api.lfm(alpha=0.8)
            self.assertIsInstance(com7, NodeClustering)

            com8 = api.lais2()
            self.assertIsInstance(com8, NodeClustering)

            com9 = api.congo(number_communities=2)
            self.assertIsInstance(com9, NodeClustering)

            com10 = api.conga(number_communities=2)
            self.assertIsInstance(com10, NodeClustering)

            com11 = api.lemon(seed_nodes, min_com_size=10, max_com_size=50)
            self.assertIsInstance(com11, NodeClustering)

            com12 = api.slpa()
            self.assertIsInstance(com12, NodeClustering)

            com13 = api.multicom(seed_node=0)
            self.assertIsInstance(com13, NodeClustering)

            com14 = api.big_clam()
            self.assertIsInstance(com14, NodeClustering)

            com15 = api.girvan_newman(level=3)
            self.assertIsInstance(com15, NodeClustering)

            com16 = api.em(k=3)
            self.assertIsInstance(com16, NodeClustering)

            com17 = api.scan(0.7, 3)
            self.assertIsInstance(com17, NodeClustering)

            com18 = api.gdmp2()
            self.assertIsInstance(com18, NodeClustering)

            com19 = api.spinglass()
            self.assertIsInstance(com19, NodeClustering)

            com20 = api.eigenvector()
            self.assertIsInstance(com20, NodeClustering)

            com21 = api.agdl(3, 2, 2, 0.5)
            self.assertIsInstance(com21, NodeClustering)

            com22 = api.louvain()
            self.assertIsInstance(com22, NodeClustering)

            com23 = api.leiden()
            self.assertIsInstance(com23, NodeClustering)

            com24 = api.rb_pots()
            self.assertIsInstance(com24, NodeClustering)

            com25 = api.rber_pots()
            self.assertIsInstance(com25, NodeClustering)

            com26 = api.cpm()
            self.assertIsInstance(com26, NodeClustering)

            com27 = api.significance_communities()
            self.assertIsInstance(com27, NodeClustering)

            com28 = api.surprise_communities()
            self.assertIsInstance(com28, NodeClustering)

            com29 = api.greedy_modularity()
            self.assertIsInstance(com29, NodeClustering)

            com30 = api.infomap()
            self.assertIsInstance(com30, NodeClustering)

            com31 = api.walktrap()
            self.assertIsInstance(com31, NodeClustering)

            com32 = api.label_propagation()
            self.assertIsInstance(com32, NodeClustering)

            com33 = api.async_fluid(k=3)
            self.assertIsInstance(com33, NodeClustering)

            com34 = api.der()
            self.assertIsInstance(com34, NodeClustering)

            com35 = api.frc_fgsn(1, 0.5, 3)
            self.assertIsInstance(com35, NodeClustering)

            stats = api.fitness_scores([coms, coms2, coms3, coms4, coms5, com6, com7, com8, com9, com10], summary=False)
            stats1 = api.fitness_scores([com11, com12, com13, com14, com15, com16, com17, com18, com19, com20],
                                        summary=True)
            stats2 = api.fitness_scores([com21, com22, com23, com24, com25, com26, com27, com28, com29, com30],
                                        summary=False)
            stats3 = api.fitness_scores([com31, com32, com33, com34, com35], summary=False)

            self.assertIsInstance(stats, dict)
            self.assertIsInstance(stats1, dict)
            self.assertIsInstance(stats2, dict)
            self.assertIsInstance(stats3, dict)

            comp = api.compare_communities([coms2, coms2])['data']
            self.assertIsInstance(comp, dict)
