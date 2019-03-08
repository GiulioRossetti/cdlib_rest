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

    def fitness_scores(self, community: NodeClustering, summary: bool = False) -> dict:
        community_name = "%s_%s" % (community.method_name, community.method_parameters)
        res = self.loop.run_until_complete(self.__load_data("evaluation/fitness_scores",
                                                            {'token': self.token, 'community_name': community_name,
                                                             'summary': str(summary)}))
        return json.loads(res)

    def angel(self, threshold: float = 0.25) -> NodeClustering:
        res = self.loop.run_until_complete(self.__load_data("cd/angel", {'token': self.token,
                                                                         'threshold': str(threshold)}))
        communities = self.__rebuild_communities(res)

        return communities


if __name__ == '__main__':

    with CDlib_API("http://0.0.0.0", 8081) as api:

        g = nx.karate_club_graph()
        api.load_network(g)
        coms = api.angel(threshold=0.75)
        print(coms.communities)
        stats = api.fitness_scores(coms, summary=False)
        print(stats)
        stats = api.fitness_scores(coms, summary=True)
        print(stats)

