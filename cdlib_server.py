from aiohttp import web
from aiohttp_swagger import *
import json
from cdlib import algorithms, readwrite
from networkx.readwrite import json_graph
import uuid
import os, shutil


def __unpack_stats(stats):
    return {"min": stats[0], "max": stats[1], "mean": stats[2], "std": stats[3]}


def __check_token(request):
    token = request.query['token']
    if not os.path.exists("data/db/%s" % token):
        response_obj = {'status': 'failure', 'description': "token not valid"}
        return 500, response_obj
    else:
        return 200, None


def create_experiment(request):
    """
        ---
        description: This end-point allows to create a new experiment.
        tags:
        - Create Experiment
        produces:
        - text/plain
        responses:
            "200":
                description: successful operation. Return experiment "token"

    """
    token = str(uuid.uuid4())
    directory = "data/db/%s" % token
    if not os.path.exists(directory):
        os.makedirs(directory)
    response_obj = {"status": "success", "data": {"token": token}}
    return web.Response(text=json.dumps(response_obj), status=200)


def destroy_experiment(request):
    """
        ---
        description: This end-point allows to destroy an existing experiment.
        tags:
            - Destroy Experiment
        produces:
            - text/plain
        responses:
            "200":
                description: successful operation.
        parameters:
        - in: query
          name: token
          schema:
            type: string
          required: true
          description: Experiment token
    """
    token = request.query['token']
    if not os.path.exists("data/db/%s" % token):
        response_obj = {"status": "failure", "description": "token not valid"}
        return web.Response(text=json.dumps(response_obj), status=500)
    else:
        shutil.rmtree("data/db/%s" % token)
        response_obj = {"status": "success"}
        return web.Response(text=json.dumps(response_obj), status=200)


async def __save_communities(communities, request):
    token = request.query['token']
    readwrite.write_community_json(communities,
                                   "data/db/%s/%s_%s" % (token, communities.method_name,
                                                         communities.method_parameters))


async def __save_network(request):
    token = request.query['token']
    network = request.query['network']
    with open("data/db/%s/network.json" % token, "w") as f:
        f.write(network)


async def __load_network(request):
    token = request.query['token']
    data = json.load(open("data/db/%s/network.json" % token))
    network = json_graph.node_link_graph(data)
    return network


async def __load_communities(request):
    token = request.query['token']
    community_name = request.query['community_name']
    community = readwrite.read_community_json("data/db/%s/%s" % (token, community_name))
    return community


async def upload_network(request):
    """
        ---
        description: This end-point allows to upload a network dataset.
        tags:
            - Upload Network
        produces:
            - text/plain
        responses:
            "200":
                description: successful operation.
            "500":
                description: operation failed.
        parameters:
        - in: query
          name: token
          schema:
            type: string
          required: true
          description: Experiment token
        - in: query
          name: network
          schema:
            type: string
          required: true
          description: JSON string representing a networkx.Graph object

    """
    code, resp = __check_token(request)
    if code == 500:
        return web.Response(text=json.dumps(resp), status=500)

    await __save_network(request)
    response_obj = {"status": "success"}
    return web.Response(text=json.dumps(response_obj), status=200)


async def angel(request):
    """
        ---
        description: This end-point allows to compute the Angel Community Discovery algorithm to a network dataset.
        tags:
            - Angel
        produces:
            - application/json
        responses:
            "200":
                description: successful operation.
            "500":
                description: operation failed

        parameters:
        - in: query
          name: token
          schema:
            type: string
          required: true
          description: Experiment token
        - in: query
          name: threshold
          schema:
            type: string
          required: false
          description: merging threshold
        """
    code, resp = __check_token(request)
    if code == 500:
        return web.Response(text=json.dumps(resp), status=500)

    g = await __load_network(request)

    try:
        th = float(request.query['threshold'])
        communities = algorithms.angel(g, th)
        resp = json.loads(communities.to_json())
        response_obj = {'status': 'success', "data": resp}
        await __save_communities(communities, request)
        return web.Response(text=json.dumps(response_obj), status=200)

    except Exception as e:
        response_obj = {'status': 'failure', 'description': str(e)}
        return web.Response(text=json.dumps(response_obj), status=500)


async def community_statistics(request):
    """
        ---
        description: This end-point allows to compute aggregate statistics for the computed partition.
        tags:
            - Statistics
        produces:
            - application/json
        responses:
            "200":
                description: successful operation.
            "500":
                description: operation failed

        parameters:
        - in: query
          name: token
          schema:
            type: string
          required: true
          description: Experiment token
        - in: query
          name: partition_name
          schema:
            type: string
          required: false
          description: Name of the partition
        - in: query
          name: summary
          schema:
            type: string
          required: false
          description: Whether or not to return an aggregated view of community-wise statistics
        """
    code, resp = __check_token(request)
    if code == 500:
        return web.Response(text=json.dumps(resp), status=500)

    g = await __load_network(request)
    coms = await __load_communities(request)
    coms.graph = g

    try:
        simple = dict(er_modularity=coms.erdos_renyi_modularity(), modularity_density=coms.modularity_density(),
                      gn_modularity=coms.newman_girvan_modularity(), z_modularity=coms.z_modularity(),
                      link_modularity=coms.link_modularity(), surprise=coms.surprise(),
                      significance=coms.significance())

        summary = request.query['summary']

        if summary == "True":
            composed = dict(size=__unpack_stats(coms.size()), conductance=__unpack_stats(coms.conductance()),
                            normalized_cut=__unpack_stats(coms.normalized_cut()),
                            triangle_participation_ratio=__unpack_stats(coms.triangle_participation_ratio()),
                            max_odf=__unpack_stats(coms.max_odf()), avg_odf=__unpack_stats(coms.avg_odf()),
                            flake_odf=__unpack_stats(coms.flake_odf()),
                            edges_inside=__unpack_stats(coms.edges_inside()),
                            fraction_over_median_degree=__unpack_stats(coms.fraction_over_median_degree()),
                            expansion=__unpack_stats(coms.expansion()), cut_ratio=__unpack_stats(coms.cut_ratio()),
                            internal_edge_density=__unpack_stats(coms.internal_edge_density()),
                            average_internal_degree=__unpack_stats(coms.average_internal_degree()))
        else:
            composed = dict(size=coms.size(summary=False), conductance=coms.conductance(summary=False),
                            normalized_cut=coms.normalized_cut(summary=False),
                            triangle_participation_ratio=coms.triangle_participation_ratio(summary=False),
                            max_odf=coms.max_odf(summary=False), avg_odf=coms.avg_odf(summary=False),
                            flake_odf=coms.flake_odf(summary=False), edges_inside=coms.edges_inside(summary=False),
                            fraction_over_median_degree=coms.fraction_over_median_degree(summary=False),
                            expansion=coms.expansion(summary=False), cut_ratio=coms.cut_ratio(summary=False),
                            internal_edge_density=coms.internal_edge_density(summary=False),
                            average_internal_degree=coms.average_internal_degree(summary=False))

        response_obj = {'status': 'success', "data": {**simple, **composed}}
        return web.Response(text=json.dumps(response_obj), status=200)

    except Exception as e:
        response_obj = {'status': 'failure', 'description': str(e)}
        return web.Response(text=json.dumps(response_obj), status=500)


async def make_app():
    app = web.Application()

    # add the routes
    app.add_routes([
        web.get('/api/create_experiment', create_experiment),
        web.post('/api/destroy_experiment', destroy_experiment),
        web.post('/api/upload/network', upload_network),
        web.post('/api/evaluation/fitness_scores', community_statistics),
        web.post('/api/cd/angel', angel),
    ])

    setup_swagger(app, swagger_url="/api/v1/doc", description="",
                  title="CDlib Server API",
                  api_version="0.1.3",
                  contact="giulio.rossetti@gmail.com")

    return app


if __name__ == '__main__':
    web.run_app(make_app(), port=8081)
