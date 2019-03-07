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

    def create_experiment(self):
        data = self.loop.run_until_complete(self.__experiment("create_experiment", True))
        self.token = json.loads(data)['data']['token']

    def destroy_experiment(self):
        data = self.loop.run_until_complete(self.__experiment("destroy_experiment", False))
        return json.loads(data)

    def load_network(self, network: nx.Graph):
        self.network = network
        self.loop.run_until_complete(self.__load_data("upload/network", {'token': self.token,
                                                                         'network':
                                                                             json.dumps(
                                                                                 json_graph.node_link_data(network))}))

    def angel(self, threshold: float = 0.25) -> NodeClustering:
        res = self.loop.run_until_complete(self.__load_data("cd/angel", {'token': self.token,
                                                                         'threshold': str(threshold)}))
        communities = self.__rebuild_communities(res)

        return communities


if __name__ == '__main__':

    api = CDlib_API("http://0.0.0.0", 8080)
    g = nx.karate_club_graph()

    api.create_experiment()
    api.load_network(g)

    coms = api.angel(threshold=0.75)
    api.destroy_experiment()
