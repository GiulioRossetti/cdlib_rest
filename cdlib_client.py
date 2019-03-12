import aiohttp
import asyncio
import json
import networkx as nx
from networkx.readwrite import json_graph
from cdlib import readwrite, NodeClustering


class CDlib_API(object):

    def __init__(self, server: str, port: int):
        self.server = "{}:{}".format(server, port)
        self.loop = asyncio.get_event_loop()
        self.token = None
        self.network = None

    def __enter__(self):
        self.__create_experiment()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.__destroy_experiment()
        self.loop.close()

    def __create_experiment(self):
        data = self.loop.run_until_complete(self.__experiment("create_experiment", True))
        self.token = json.loads(data)['data']['token']

    def __destroy_experiment(self):
        self.loop.run_until_complete(self.__experiment("destroy_experiment", False))

    def __rebuild_communities(self, res: str) -> NodeClustering:
        cms = json.loads(res)['data']
        communities = readwrite.read_community_from_json_string(json.dumps(cms))
        communities.graph = self.network
        return communities

    @staticmethod
    async def __get(session, url: str) -> str:
        async with session.get(url) as response:
            return await response.text()

    @staticmethod
    async def __post(session, url: str, parameters: dict) -> str:
        async with session.post(url, params=parameters) as response:
            return await response.text()

    async def __load_data(self, endpoint, parameters: dict) -> str:
        async with aiohttp.ClientSession(loop=self.loop) as session:
            res = await self.__post(session, "%s/api/%s" % (self.server, endpoint), parameters)
            return res

    async def __experiment(self, endpoint, create: bool) -> str:
        async with aiohttp.ClientSession(loop=self.loop) as session:
            if create:
                res = await self.__get(session, "%s/api/%s" % (self.server, endpoint))
            else:
                res = await self.__post(session, "%s/api/%s" % (self.server, endpoint), {'token': self.token})
            return res

    def load_network(self, network: nx.Graph):
        self.network = network
        self.loop.run_until_complete(self.__load_data("upload/network", {'token': self.token,
                                                                         'network':
                                                                             json.dumps(
                                                                                 json_graph.node_link_data(network))}))

    def fitness_scores(self, communities: list, summary: bool = False) -> dict:
        community_names = ["%s_%s" % (community.method_name, community.method_parameters) for community in communities]

        res = self.loop.run_until_complete(self.__load_data("evaluation/fitness_scores",
                                                            {'token': self.token,
                                                             'community_names': json.dumps(community_names),
                                                             'summary': str(summary)}))
        return json.loads(res)

    def compare_communities(self, communities: list) -> dict:
        community_names = ["%s_%s" % (community.method_name, community.method_parameters) for community in communities]

        res = self.loop.run_until_complete(self.__load_data("evaluation/community_comparison",
                                                            {'token': self.token,
                                                             'community_names': json.dumps(community_names)}))
        return json.loads(res)

    def angel(self, threshold: float = 0.25, min_com_size: int = 3) -> NodeClustering:
        res = self.loop.run_until_complete(self.__load_data("cd/angel", {'token': self.token,
                                                                         'threshold': str(threshold),
                                                                         'min_com_size': str(min_com_size)}))
        communities = self.__rebuild_communities(res)

        return communities

    def demon(self, epsilon: float = 0.25, min_com_size: int = 3) -> NodeClustering:
        res = self.loop.run_until_complete(self.__load_data("cd/demon", {'token': self.token,
                                                                         'epsilon': str(epsilon),
                                                                         'min_com_size': str(min_com_size)}))
        communities = self.__rebuild_communities(res)

        return communities


    def ego_networks(self, level: int = 1) -> NodeClustering:
        res = self.loop.run_until_complete(self.__load_data("cd/ego_networks", {'token': self.token,
                                                                                'level': str(level)}))
        communities = self.__rebuild_communities(res)

        return communities

    def kclique(self, k) -> NodeClustering:
        res = self.loop.run_until_complete(self.__load_data("cd/kclique", {'token': self.token,
                                                                                'k': str(k)}))
        communities = self.__rebuild_communities(res)

        return communities



    def lfm(self, alpha) -> NodeClustering:
        res = self.loop.run_until_complete(self.__load_data("cd/lfm", {'token': self.token,
                                                                                'alpha': str(alpha)}))
        communities = self.__rebuild_communities(res)

        return communities


    def lais2(self) -> NodeClustering:
        res = self.loop.run_until_complete(self.__load_data("cd/lais2", {'token': self.token}))
        communities = self.__rebuild_communities(res)

        return communities


    def congo(self, number_communities, height: int = 2) -> NodeClustering:
        res = self.loop.run_until_complete(self.__load_data("cd/congo", {'token': self.token,
                                                                         'number_communities': str(number_communities),
                                                                         'height': str(height)}))
        communities = self.__rebuild_communities(res)

        return communities


    def conga(self, number_communities) -> NodeClustering:
        res = self.loop.run_until_complete(self.__load_data("cd/conga", {'token': self.token,
                                                                         'number_communities': str(number_communities)}))
        communities = self.__rebuild_communities(res)

        return communities



    def overlapping_seed_set_expansion(self, seeds, ninf: bool = False, expansion: str = 'ppr', stopping: str = 'cond',
                                nworkers: int = 1, nruns: int = 13, alpha: float = 0.99, delta: float = 0.2) -> NodeClustering:
        res = self.loop.run_until_complete(self.__load_data("cd/overlapping_seed_set_expansion", {'token': self.token,
                                                                    'seeds': json.dumps(seeds), 'ninf': str(ninf),
                                                                    'expansion': str(expansion), 'stopping': str(stopping),
                                                                    'nworkers': str(nworkers), 'nruns': str(nruns),
                                                                    'alpha': str(alpha), 'delta': str(delta)}))
        communities = self.__rebuild_communities(res)

        return communities


    def lemon(self, seeds, min_com_size: int = 20, max_com_size: int = 50, expand_step: int = 6,
                                       subspace_dim: int = 3, walk_steps: int = 3, biased: bool = False) -> NodeClustering:
        res = self.loop.run_until_complete(self.__load_data("cd/lemon", {'token': self.token, 'seeds': json.dumps(seeds),
                                                                            'min_com_size': str(min_com_size),
                                                                            'max_com_size': str( max_com_size),
                                                                            'expand_step': str(expand_step),
                                                                            'subspace_dim': str( subspace_dim),
                                                                            'walk_steps': str(walk_steps), 'biased': str(biased)}))
        communities = self.__rebuild_communities(res)

        return communities


    def slpa(self, t: int=21, r: float=0.1) -> NodeClustering:
        res = self.loop.run_until_complete(self.__load_data("cd/slpa", {'token': self.token,
                                                                         't': str(t), 'r': str(r)}))
        communities = self.__rebuild_communities(res)

        return communities


    def multicom(self, seed_node) -> NodeClustering:
        res = self.loop.run_until_complete(self.__load_data("cd/multicom", {'token': self.token,
                                                                        'seed_node': json.dumps(seed_node)}))
        communities = self.__rebuild_communities(res)

        return communities


    def big_clam(self, number_communities: int = 5) -> NodeClustering:
        res = self.loop.run_until_complete(self.__load_data("cd/big_clam", {'token': self.token,
                                                                        'number_communities': str(number_communities)}))
        communities = self.__rebuild_communities(res)

        return communities



    def node_perception(self, threshold: float = 0.2, overlap_threshold: float = 1.0, min_comm_size: int = 2) -> NodeClustering:
        res = self.loop.run_until_complete(self.__load_data("cd/node_perception", {'token': self.token,
                                                                                'threshold': str(threshold),
                                                                                'overlap_threshold': str(overlap_threshold),
                                                                                'min_comm_size': str(min_comm_size)}))
        communities = self.__rebuild_communities(res)

        return communities


# crisp partition
    def girvan_newman(self, level) -> NodeClustering:
        res = self.loop.run_until_complete(self.__load_data("cd/girvan_newman", {'token': self.token,
                                                                        'level': str(level)}))
        communities = self.__rebuild_communities(res)

        return communities


    def em(self, k) -> NodeClustering:
        res = self.loop.run_until_complete(self.__load_data("cd/em", {'token': self.token,
                                                                        'k': str(k)}))
        print(res)
        communities = self.__rebuild_communities(res)

        return communities

    def scan(self, epsilon, mu) -> NodeClustering:
        res = self.loop.run_until_complete(self.__load_data("cd/scan", {'token': self.token,
                                                                        'epsilon': str(epsilon), 'mu': str(mu)}))
        communities = self.__rebuild_communities(res)

        return communities


    def gdmp2(self, min_threshold: float = 0.75) -> NodeClustering:
        res = self.loop.run_until_complete(self.__load_data("cd/gdmp2", {'token': self.token,
                                                                        'min_threshold': str(min_threshold)}))
        communities = self.__rebuild_communities(res)

        return communities

    def spinglass(self) -> NodeClustering:
        res = self.loop.run_until_complete(self.__load_data("cd/spinglass", {'token': self.token}))
        communities = self.__rebuild_communities(res)

        return communities

    def eigenvector(self) -> NodeClustering:
        res = self.loop.run_until_complete(self.__load_data("cd/eigenvector", {'token': self.token}))
        communities = self.__rebuild_communities(res)

        return communities

    def agdl(self, number_communities, number_neighbors, kc, a) -> NodeClustering:
        res = self.loop.run_until_complete(self.__load_data("cd/agdl", {'token': self.token,
                                                                        'number_communities': str(number_communities),
                                                                        'number_neighbors': str(number_neighbors),
                                                                        'kc': str(kc), 'a': str(a)}))
        communities = self.__rebuild_communities(res)

        return communities

    def louvain(self, weight: str = 'weight', resolution: int = 1., randomize: bool = False) -> NodeClustering:
        res = self.loop.run_until_complete(self.__load_data("cd/louvain", {'token': self.token,
                                                                            'weight': str(weight),
                                                                               'resolution': str(resolution),
                                                                               'randomize': str(randomize)}))
        communities = self.__rebuild_communities(res)

        return communities


    def leiden(self, initial_membership: list = None, weights: list = None) -> NodeClustering:
        res = self.loop.run_until_complete(self.__load_data("cd/leiden", {'token': self.token,
                                                                            'initial_membership': json.dumps(initial_membership),
                                                                               'weights': json.dumps(weights)}))
        communities = self.__rebuild_communities(res)

        return communities


    def rb_pots(self, initial_membership: list = None, weights: list = None, resolution_parameter: float = 1.0) -> NodeClustering:
        res = self.loop.run_until_complete(self.__load_data("cd/rb_pots", {'token': self.token,
                                                                            'initial_membership': json.dumps(initial_membership),
                                                                               'weights': json.dumps(weights),
                                                                          'resolution_parameter': str(resolution_parameter)}))
        communities = self.__rebuild_communities(res)

        return communities


    def rber_pots(self, initial_membership: list = None, weights: list = None, node_sizes: list = None,
                  resolution_parameter: float = 1.0) -> NodeClustering:
        res = self.loop.run_until_complete(self.__load_data("cd/rber_pots", {'token': self.token,
                                                                            'initial_membership': json.dumps(initial_membership),
                                                                               'weights': json.dumps(weights),
                                                                           'node_sizes': json.dumps(node_sizes),
                                                                          'resolution_parameter': str(resolution_parameter)}))
        communities = self.__rebuild_communities(res)

        return communities


    def cpm(self, initial_membership: list = None, weights: list = None, node_sizes: list = None,
                  resolution_parameter: float = 1.0) -> NodeClustering:
        res = self.loop.run_until_complete(self.__load_data("cd/cpm", {'token': self.token,
                                                                            'initial_membership': json.dumps(initial_membership),
                                                                               'weights': json.dumps(weights),
                                                                           'node_sizes': json.dumps(node_sizes),
                                                                          'resolution_parameter': str(resolution_parameter)}))
        communities = self.__rebuild_communities(res)

        return communities

    def significance_communities(self, initial_membership: list = None,node_sizes: list = None) -> NodeClustering:
        res = self.loop.run_until_complete(self.__load_data("cd/significance_communities", {'token': self.token,
                                                                       'initial_membership': json.dumps(
                                                                           initial_membership),
                                                                       'node_sizes': json.dumps(node_sizes)}))
        communities = self.__rebuild_communities(res)

        return communities


    def surprise_communities(self, initial_membership: list = None, weights: list = None, node_sizes: list = None) -> NodeClustering:
        res = self.loop.run_until_complete(self.__load_data("cd/surprise_communities", {'token': self.token,
                                                                            'initial_membership': json.dumps(initial_membership),
                                                                               'weights': json.dumps(weights),
                                                                           'node_sizes': json.dumps(node_sizes)}))
        communities = self.__rebuild_communities(res)

        return communities


    def greedy_modularity(self, weights: list = None) -> NodeClustering:
        res = self.loop.run_until_complete(self.__load_data("cd/greedy_modularity", {'token': self.token,
                                                                             'weights': json.dumps(weights)}))
        communities = self.__rebuild_communities(res)

        return communities

    def infomap(self) -> NodeClustering:
        res = self.loop.run_until_complete(self.__load_data("cd/infomap", {'token': self.token}))
        communities = self.__rebuild_communities(res)

        return communities

    def walktrap(self) -> NodeClustering:
        res = self.loop.run_until_complete(self.__load_data("cd/walktrap", {'token': self.token}))
        communities = self.__rebuild_communities(res)

        return communities

    def label_propagation(self) -> NodeClustering:
        res = self.loop.run_until_complete(self.__load_data("cd/label_propagation", {'token': self.token}))
        communities = self.__rebuild_communities(res)

        return communities


    def async_fluid(self, k) -> NodeClustering:
        res = self.loop.run_until_complete(self.__load_data("cd/async_fluid", {'token': self.token,
                                                                             'k': str(k)}))
        communities = self.__rebuild_communities(res)

        return communities

    def der(self, walk_len: int = 3, threshold: float = .00001, iter_bound: int = 50) -> NodeClustering:
        res = self.loop.run_until_complete(self.__load_data("cd/der", {'token': self.token,'walk_len': str(walk_len),
                                                                       'threshold': str(threshold),
                                                                       'iter_bound': str(iter_bound)}))
        communities = self.__rebuild_communities(res)

        return communities

    def frc_fgsn(self, theta, eps, r) -> NodeClustering:
        res = self.loop.run_until_complete(self.__load_data("cd/frc_fgsn", {'token': self.token,'theta': str(theta),
                                                                       'eps': str(eps),'r': str(r)}))
        communities = self.__rebuild_communities(res)

        return communities



if __name__ == '__main__':

    with CDlib_API("http://0.0.0.0", 8081) as api:

        g = nx.karate_club_graph()
        api.load_network(g)
        coms = api.angel(threshold=0.75)
        print("angel")
        print(coms.communities)
        coms2 = api.demon(epsilon=0.25)
        print("demon")
        print(coms2.communities)
        coms3 = api.ego_networks()
        print("ego_networks")
        print(coms3.communities)
        coms4 = api.node_perception() #-> da errore la fitness -> dice che divide per 0
        print("node_perception")
        print(coms4.communities)
        seeds = [0,2,6]
        coms5 = api.overlapping_seed_set_expansion(seeds=seeds)
        print("overlapping_seed_set_expansion")
        print(coms5.communities)
        com6 = api.kclique(k=3)
        print("kclique")
        print(com6.communities)
        com7 = api.lfm(alpha=0.8)
        print("lfm")
        print(com7.communities)
        com8 = api.lais2()
        print("lais2")
        print(com8.communities)
        com9 = api.congo(number_communities=2)
        print("congo")
        print(com9.communities)
        com10 = api.conga(number_communities=2)
        print("conga")
        print(com10.communities)
        #com11 = api.lemon(seeds, min_com_size=10, max_com_size=50)
        com12 = api.slpa()
        print("slpa")
        print(com12.communities)
        #com13 = api.multicom(seed_node=0)# dice che non e serializzabile
        com14 = api.big_clam() #-> da errore la fitness -> dice che divide per 0
        print("big_clam")
        print(com14.communities)
        com15 = api.girvan_newman(level=3) #-> da errore la fitness
        print("girvan_newman")
        print(com15.communities)
       # print("16\n")
        #print(com16.communities)
        #com16 = api.em(k=3) #dice che non e serializzabile
        com17 = api.scan(0.7, 3)  #-> da errore la fitness
        print("scan")
        print(com17.communities)
        com18 = api.gdmp2()
        print("gdmp2")
        print(com18.communities)
        com19 = api.spinglass() #-> da errore la fitness
        print("spinglass")
        print(com19.communities)
        com20 = api.eigenvector()
        print("eigenvector")
        print(com20.communities)
        com21 = api.agdl(3, 2, 2, 0.5)
        print("agdl")
        print(com21.communities)
        com22 = api.louvain()
        print("louvain")
        print(com22.communities)
        com23 = api.leiden()
        print("leiden")
        print(com23.communities)
        com24 = api.rb_pots()
        print("rb_pots")
        print(com24.communities)
        com25 = api.rber_pots()
        print("rber_pots")
        print(com25.communities)
        com26 = api.cpm()
        print("cpm")
        print(com26.communities)
        com27 = api.significance_communities()
        print("significance_communities")
        print(com27.communities)
        com28 = api.surprise_communities()
        print("surprise_communities")
        print(com28.communities)
        com29 = api.greedy_modularity()
        print("greedy_modularity")
        print(com29.communities)
        com30 = api.infomap()
        print("infomap")
        print(com30.communities)
        com31 = api.walktrap()
        print("walktrap")
        print(com31.communities)
        com32 = api.label_propagation()
        print("label_propagation")
        print(com32.communities)
        com33 = api.async_fluid(k=3)
        print("async_fluid")
        print(com33.communities)
        com34 = api.der()
        print("der")
        print(com34.communities)
        com35 = api.frc_fgsn(1, 0.5, 3)
        print("frc_fgsn")
        print(com35.communities)




        stats = api.fitness_scores([coms, coms2,coms3,coms5,com6,com7,com8,com9,com10,com12,com18,com20,com21,com22, com23,com24], summary=False)['data']
        #for c, v in stats.items():
        #    print(c, v)
        stats = api.fitness_scores([coms, coms2,coms3,coms5,com6,com7,com8,com9,com10,com12,com18,com20,com21, com22,
                                    com23,com24], summary=True)['data']
        #for c, v in stats.items():
        #    print(c, v)

        comp = api.compare_communities([coms2, coms2])['data']
        #print(comp)
