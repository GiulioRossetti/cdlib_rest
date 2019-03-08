import json
import os
import shutil
import uuid

from aiohttp import web
from aiohttp_swagger import *
from cdlib import algorithms, readwrite, NodeClustering
from networkx.readwrite import json_graph


def __unpack_stats(stats):
    return {"min": stats[0], "max": stats[1], "mean": stats[2], "std": stats[3]}


def __check_token(request):
    token = request.query['token']
    if not os.path.exists("data/db/%s" % token):
        response_obj = {'status': 'failure', 'description': "token not valid"}
        return 500, response_obj
    else:
        return 200, None


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


async def __load_communities(request) -> list:
    token = request.query['token']
    community_names = json.loads(request.query['community_names'])
    community = [readwrite.read_community_json("data/db/%s/%s" % (token, name)) for name in community_names]
    return community


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


async def community_comparison(request):
    """
            ---
            description: This end-point allows to compare two clusterings applying several state of art scores.
            tags:
                - Community Comparisons
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
              name: partition_names
              schema:
                type: string
              required: false
              description: Name of the partitions

        """
    code, resp = __check_token(request)
    if code == 500:
        return web.Response(text=json.dumps(resp), status=500)

    g = await __load_network(request)
    coms = await __load_communities(request)

    if len(coms) < 2 or len(coms) > 2:
        response_obj = dict(status='failure',
                            description='to perform the comparison exactly two clusterings are required')
        return web.Response(text=json.dumps(response_obj), status=500)

    com1, com2 = coms
    com1.graph = g
    com2.graph = g

    try:
        f1 = com1.f1(com2)
        data = dict(onmi=com1.overlapping_normalized_mutual_information(com2), omega=com1.omega(com2),
                    f1={"mean": f1[0], "std": f1[1]}, nf1=com1.nf1(com2))

        if not com1.overlap and not com2.overlap:
            crisp = dict(nmi=com1.normalized_mutual_information(com2),
                         ari=com1.adjusted_rand_index(com2), ami=com1.adjusted_mutual_information(com2),
                         vi=com1.variation_of_information(com2))
            data = {**data, **crisp}

        response_obj = {'status': 'success', "data": data}
        return web.Response(text=json.dumps(response_obj), status=200)

    except Exception as e:
        response_obj = {'status': 'failure', 'description': str(e)}
        return web.Response(text=json.dumps(response_obj), status=500)


async def community_statistics(request):
    """
        ---
        description: This end-point allows to compute aggregate statistics for the computed partition.
        tags:
            - Fitness Scores
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
          name: partition_names
          schema:
            type: string
          required: false
          description: Name of the partitions
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

    data = {}

    try:
        for com in coms:
            com.graph = g

            simple = dict(er_modularity=com.erdos_renyi_modularity(), modularity_density=com.modularity_density(),
                          gn_modularity=com.newman_girvan_modularity(), z_modularity=com.z_modularity(),
                          link_modularity=com.link_modularity(), surprise=com.surprise(),
                          significance=com.significance())

            summary = request.query['summary']

            if summary == "True":
                composed = dict(size=__unpack_stats(com.size()), conductance=__unpack_stats(com.conductance()),
                                normalized_cut=__unpack_stats(com.normalized_cut()),
                                triangle_participation_ratio=__unpack_stats(com.triangle_participation_ratio()),
                                max_odf=__unpack_stats(com.max_odf()), avg_odf=__unpack_stats(com.avg_odf()),
                                flake_odf=__unpack_stats(com.flake_odf()),
                                edges_inside=__unpack_stats(com.edges_inside()),
                                fraction_over_median_degree=__unpack_stats(com.fraction_over_median_degree()),
                                expansion=__unpack_stats(com.expansion()), cut_ratio=__unpack_stats(com.cut_ratio()),
                                internal_edge_density=__unpack_stats(com.internal_edge_density()),
                                average_internal_degree=__unpack_stats(com.average_internal_degree()))
            else:
                composed = dict(size=com.size(summary=False), conductance=com.conductance(summary=False),
                                normalized_cut=com.normalized_cut(summary=False),
                                triangle_participation_ratio=com.triangle_participation_ratio(summary=False),
                                max_odf=com.max_odf(summary=False), avg_odf=com.avg_odf(summary=False),
                                flake_odf=com.flake_odf(summary=False), edges_inside=com.edges_inside(summary=False),
                                fraction_over_median_degree=com.fraction_over_median_degree(summary=False),
                                expansion=com.expansion(summary=False), cut_ratio=com.cut_ratio(summary=False),
                                internal_edge_density=com.internal_edge_density(summary=False),
                                average_internal_degree=com.average_internal_degree(summary=False))

            data["%s_%s" % (com.method_name, com.method_parameters)] = {**simple, **composed}

        response_obj = {'status': 'success', "data": data}
        return web.Response(text=json.dumps(response_obj), status=200)

    except Exception as e:
        response_obj = {'status': 'failure', 'description': str(e)}
        return web.Response(text=json.dumps(response_obj), status=500)


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
          required: true
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


async def demon(request):
    """
        ---
        description: This end-point allows to compute the Demon Community Discovery algorithm to a network dataset.
        tags:
            - Demon
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
          name: epsilon
          schema:
            type: string
          required: true
          description: merging threshold
        """
    code, resp = __check_token(request)
    if code == 500:
        return web.Response(text=json.dumps(resp), status=500)

    g = await __load_network(request)

    try:
        th = float(request.query['epsilon'])
        communities = algorithms.demon(g, th)
        resp = json.loads(communities.to_json())
        response_obj = {'status': 'success', "data": resp}
        await __save_communities(communities, request)
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
        web.post('/api/evaluation/community_comparison', community_comparison),
        web.post('/api/cd/angel', angel),
        web.post('/api/cd/demon', demon),
    ])

    setup_swagger(app, swagger_url="/api/v1/doc", description="",
                  title="CDlib Server API",
                  api_version="0.1.3",
                  contact="giulio.rossetti@gmail.com")

    return app


if __name__ == '__main__':
    web.run_app(make_app(), port=8081, host="0.0.0.0")
