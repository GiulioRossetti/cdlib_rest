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
            type: float
          required: true
          description: merging threshold
        - in: query
          name: min_com_size
          schema:
            type: integer
          required: true
          description: minimum community size
        """
    code, resp = __check_token(request)
    if code == 500:
        return web.Response(text=json.dumps(resp), status=500)

    g = await __load_network(request)

    try:
        th = float(request.query['threshold'])
        com_size = int(request.query['min_com_size'])
        communities = algorithms.angel(g, th, com_size)
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
            type: float
          required: true
          description: merging threshold
        - in: query
          name: min_com_size
          schema:
            type: integer
          required: true
          description: minimum community size
        """
    code, resp = __check_token(request)
    if code == 500:
        return web.Response(text=json.dumps(resp), status=500)

    g = await __load_network(request)

    try:
        th = float(request.query['epsilon'])
        com_size = int(request.query['min_com_size'])
        communities = algorithms.demon(g, th, com_size)
        resp = json.loads(communities.to_json())
        response_obj = {'status': 'success', "data": resp}
        await __save_communities(communities, request)
        return web.Response(text=json.dumps(response_obj), status=200)

    except Exception as e:
        response_obj = {'status': 'failure', 'description': str(e)}
        return web.Response(text=json.dumps(response_obj), status=500)


async def kclique(request):
    """
        ---
        description: This end-point allows to compute kclique Community Discovery algorithm to a network dataset.
        tags:
            - Kclique
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
          name: k
          schema:
            type: integer
          required: true
          description: Size of smallest clique
        """
    code, resp = __check_token(request)
    if code == 500:
        return web.Response(text=json.dumps(resp), status=500)

    g = await __load_network(request)

    try:
        k = int(request.query['k'])
        communities = algorithms.kclique(g, k)
        resp = json.loads(communities.to_json())
        response_obj = {'status': 'success', "data": resp}
        await __save_communities(communities, request)
        return web.Response(text=json.dumps(response_obj), status=200)

    except Exception as e:
        response_obj = {'status': 'failure', 'description': str(e)}
        return web.Response(text=json.dumps(response_obj), status=500)


async def lfm(request):
    """
        ---
        description: This end-point allows to compute Lfm Community Discovery algorithm to a network dataset.
        tags:
            - LFM
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
          name: alpha
          schema:
            type: float
          required: true
          description: controll the size of the communities:  Large values of alpha yield very small communities, small values instead deliver large modules. If alpha is small enough, all nodes end up in the same cluster, the network itself. In most cases, for alpha < 0.5 there is only one community, for alpha > 2 one recovers the smallest communities. A natural choise is alpha =1.
        """
    code, resp = __check_token(request)
    if code == 500:
        return web.Response(text=json.dumps(resp), status=500)

    g = await __load_network(request)

    try:
        alpha = float(request.query['alpha'])
        communities = algorithms.lfm(g, alpha)
        resp = json.loads(communities.to_json())
        response_obj = {'status': 'success', "data": resp}
        await __save_communities(communities, request)
        return web.Response(text=json.dumps(response_obj), status=200)

    except Exception as e:
        response_obj = {'status': 'failure', 'description': str(e)}
        return web.Response(text=json.dumps(response_obj), status=500)


async def ego_networks(request):
    """
        ---
        description: This end-point allows to compute the Ego-networks Community Discovery algorithm to a network dataset.
        tags:
            - Ego-networks
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
          name: level
          schema:
            type: integer
          required: true
          description: extrac communities with all neighbors of distance<=level from a node.
        """
    code, resp = __check_token(request)
    if code == 500:
        return web.Response(text=json.dumps(resp), status=500)

    g = await __load_network(request)

    try:
        com_level = int(request.query['level'])
        communities = algorithms.ego_networks(g, com_level)
        resp = json.loads(communities.to_json())
        response_obj = {'status': 'success', "data": resp}
        await __save_communities(communities, request)
        return web.Response(text=json.dumps(response_obj), status=200)

    except Exception as e:
        response_obj = {'status': 'failure', 'description': str(e)}
        return web.Response(text=json.dumps(response_obj), status=500)


async def overlapping_seed_set_expansion(request):
    """
        ---
        description: This end-point allows to compute the OSSE Community Discovery algorithm to a network dataset.
        tags:
            - Overlapping seed set expansion
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
          name: seeds
          schema:
            type: list
          required: true
          description: Node list.
        - in: query
          name: ninf
          schema:
            type: boolean
          required: true
          description: Neighbourhood Inflation parameter.
        - in: query
          name: expansion
          schema:
            type: string
          required: true
          description: Seed expansion: ppr or vppr.
        - in: query
          name: stopping
          schema:
            type: sting
          required: true
          description: Stopping criteria: cond.
        - in: query
          name: nworkers
          schema:
            type: integer
          required: true
          description: Number of Workers.
        - in: query
          name: nruns
          schema:
            type: integer
          required: true
          description: Number of runs.
        - in: query
          name: alpha
          schema:
            type: float
          required: true
          description: alpha value for Personalized PageRank expansion.
        - in: query
          name: maxexpand
          schema:
            type: float
          required: true
          description: Maximum expansion allowed for approximate ppr.
        - in: query
          name: delta
          schema:
            type: float
          required: true
          description: Minimum distance parameter for near duplicate communities.
        """
    code, resp = __check_token(request)
    if code == 500:
        return web.Response(text=json.dumps(resp), status=500)

    g = await __load_network(request)

    try:
        seeds = json.loads(request.query['seeds'])
        tmp_ninf = str(request.query['ninf'])
        if tmp_ninf == "False":
            ninf = False
        else:
            ninf = True
        expansion = str(request.query['expansion'])
        nworkers = int(request.query['nworkers'])
        stopping = str(request.query['stopping'])
        nruns = int(request.query['nruns'])
        alpha = float(request.query['alpha'])
        delta = float(request.query['delta'])
        communities = algorithms.overlapping_seed_set_expansion(g, seeds,ninf,expansion,stopping,nworkers,nruns,alpha,float('INF'),delta)
        resp = json.loads(communities.to_json())
        response_obj = {'status': 'success', "data": resp}
        await __save_communities(communities, request)
        return web.Response(text=json.dumps(response_obj), status=200)

    except Exception as e:
        response_obj = {'status': 'failure', 'description': str(e)}
        return web.Response(text=json.dumps(response_obj), status=500)


async def lais2(request):
    """
        ---
        description: This end-point allows to compute Lais2 Community Discovery algorithm to a network dataset.
        tags:
            - Lais2
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
        """
    code, resp = __check_token(request)
    if code == 500:
        return web.Response(text=json.dumps(resp), status=500)

    g = await __load_network(request)

    try:
        communities = algorithms.lais2(g)
        resp = json.loads(communities.to_json())
        response_obj = {'status': 'success', "data": resp}
        await __save_communities(communities, request)
        return web.Response(text=json.dumps(response_obj), status=200)

    except Exception as e:
        response_obj = {'status': 'failure', 'description': str(e)}
        return web.Response(text=json.dumps(response_obj), status=500)


async def congo(request):
    """
        ---
        description: This end-point allows to compute Congo Community Discovery algorithm to a network dataset.
        tags:
            - Congo
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
          name: number_communities
          schema:
            type: integer
          required: true
          description: the number of communities desired
         - in: query
          name: height
          schema:
            type: integer
          required: true
          description: The lengh of the longest shortest paths that CONGO considers
        """
    code, resp = __check_token(request)
    if code == 500:
        return web.Response(text=json.dumps(resp), status=500)

    g = await __load_network(request)

    try:
        number_communities = int(request.query['number_communities'])
        height = int(request.query['height'])
        communities = algorithms.congo(g,number_communities,height)
        resp = json.loads(communities.to_json())
        response_obj = {'status': 'success', "data": resp}
        await __save_communities(communities, request)
        return web.Response(text=json.dumps(response_obj), status=200)

    except Exception as e:
        response_obj = {'status': 'failure', 'description': str(e)}
        return web.Response(text=json.dumps(response_obj), status=500)


async def conga(request):
    """
        ---
        description: This end-point allows to compute Conga Community Discovery algorithm to a network dataset.
        tags:
            - Conga
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
          name: number_communities
          schema:
            type: integer
          required: true
          description: the number of communities desired
        """
    code, resp = __check_token(request)
    if code == 500:
        return web.Response(text=json.dumps(resp), status=500)

    g = await __load_network(request)

    try:
        number_communities = int(request.query['number_communities'])
        communities = algorithms.conga(g,number_communities)
        resp = json.loads(communities.to_json())
        response_obj = {'status': 'success', "data": resp}
        await __save_communities(communities, request)
        return web.Response(text=json.dumps(response_obj), status=200)

    except Exception as e:
        response_obj = {'status': 'failure', 'description': str(e)}
        return web.Response(text=json.dumps(response_obj), status=500)


async def lemon(request):
    """
        ---
        description: This end-point allows to compute Lemon Community Discovery algorithm to a network dataset.
        tags:
            - Lemon
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
          name: seeds
          schema:
            type: list
          required: true
          description: Node list
        - in: query
          name: min_com_size
          schema:
            type: integer
          required: true
          description: the minimum size of a single community in the network
         - in: query
          name: max_com_size
          schema:
            type: integer
          required: true
          description: the maximum size of a single community in the network
        - in: query
          name: expand_step
          schema:
            type: integer
          required: true
          description: the step of seed set increasement during expansion process
        - in: query
          name: subspace_dim
          schema:
            type: integer
          required: true
          description: dimension of the subspace; choosing a large dimension is undesirable because it would increase the computation cost of generating local spectra
        - in: query
          name: walk_steps
          schema:
            type: integer
          required: true
          description: the number of step for the random walk
         - in: query
          name: biased
          schema:
            type: boolean
          required: true
          description: set if the random walk starting from seed nodes
        """
    code, resp = __check_token(request)
    if code == 500:
        return web.Response(text=json.dumps(resp), status=500)

    g = await __load_network(request)

    try:
        seeds = json.loads(request.query['seeds'])
        min_com_size = int(request.query['min_com_size'])
        max_com_size = int(request.query['max_com_size'])
        expand_step = int(request.query['expand_step'])
        walk_steps = int(request.query['walk_steps'])
        tmp_biased = str(request.query['biased'])
        if tmp_biased == "False":
            biased = False
        else:
            biased = True
        subspace_dim = int(request.query['subspace_dim'])
        print("ok")
        communities = algorithms.lemon(g,seeds,min_com_size,max_com_size,expand_step,subspace_dim,walk_steps,biased)
        print(communities.communities)
        resp = json.loads(communities.to_json())
        response_obj = {'status': 'success', "data": resp}
        await __save_communities(communities, request)
        return web.Response(text=json.dumps(response_obj), status=200)

    except Exception as e:
        response_obj = {'status': 'failure', 'description': str(e)}
        return web.Response(text=json.dumps(response_obj), status=500)


async def slpa(request):
    """
        ---
        description: This end-point allows to compute slpa Community Discovery algorithm to a network dataset.
        tags:
            - slpa
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
          name: t
          schema:
            type: integer
          required: true
          description:  maximum number of iterations
         - in: query
          name: r
          schema:
            type: float
          required: true
          description:  threshold  ∈ [0, 1]. It is used in the post-processing stage: if the probability of seeing a particular label during the whole process is less than r, this label is deleted from a node’s memory.
        """
    code, resp = __check_token(request)
    if code == 500:
        return web.Response(text=json.dumps(resp), status=500)

    g = await __load_network(request)

    try:
        t = int(request.query['t'])
        r = float(request.query['r'])
        communities = algorithms.slpa(g,t,r)
        resp = json.loads(communities.to_json())
        response_obj = {'status': 'success', "data": resp}
        await __save_communities(communities, request)
        return web.Response(text=json.dumps(response_obj), status=200)

    except Exception as e:
        response_obj = {'status': 'failure', 'description': str(e)}
        return web.Response(text=json.dumps(response_obj), status=500)


async def multicom(request):
    """
        ---
        description: This end-point allows to compute multicom Community Discovery algorithm to a network dataset.
        tags:
            - multicom
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
          name: seed_node
          schema:
            type: integer
          required: true
          description:  Id of the seed node around which we want to detect communities
        """
    code, resp = __check_token(request)
    if code == 500:
        return web.Response(text=json.dumps(resp), status=500)

    g = await __load_network(request)

    try:
        seed_node = json.loads(request.query['seed_node'])
        communities = algorithms.multicom(g, seed_node)
        resp = json.loads(communities.to_json())
        response_obj = {'status': 'success', "data": resp}
        await __save_communities(communities, request)
        return web.Response(text=json.dumps(response_obj), status=200)

    except Exception as e:
        response_obj = {'status': 'failure', 'description': str(e)}
        return web.Response(text=json.dumps(response_obj), status=500)


async def big_clam(request):
    """
        ---
        description: This end-point allows to compute big_clam Community Discovery algorithm to a network dataset.
        tags:
            - Big_clam
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
          name: number_communities
          schema:
            type: integer
          required: true
          description:  number communities desired
        """
    code, resp = __check_token(request)
    if code == 500:
        return web.Response(text=json.dumps(resp), status=500)

    g = await __load_network(request)

    try:
        number_communities = int(request.query['number_communities'])
        communities = algorithms.big_clam(g,number_communities)
        resp = json.loads(communities.to_json())
        response_obj = {'status': 'success', "data": resp}
        await __save_communities(communities, request)
        return web.Response(text=json.dumps(response_obj), status=200)

    except Exception as e:
        response_obj = {'status': 'failure', 'description': str(e)}
        return web.Response(text=json.dumps(response_obj), status=500)


async def node_perception(request):
    """
        ---
        description: This end-point allows to compute the Node Perception Community Discovery algorithm to a network dataset.
        tags:
            - Node Perception
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
            type: float
          required: true
          description: the tolerance required in order to merge communities.
        - in: query
          name: overlap_threshold
          schema:
            type: float
          required: true
          description: the overlap tolerance.
        - in: query
          name: min_comm_size
          schema:
            type: integer
          required: true
          description: minimum community size.
        """
    code, resp = __check_token(request)
    if code == 500:
        return web.Response(text=json.dumps(resp), status=500)

    g = await __load_network(request)

    try:
        th = float(request.query['threshold'])
        ov_th = float(request.query['overlap_threshold'])
        com_size = int(request.query['min_comm_size'])
        communities = algorithms.node_perception(g, th, ov_th, com_size)
        resp = json.loads(communities.to_json())
        response_obj = {'status': 'success', "data": resp}
        await __save_communities(communities, request)
        return web.Response(text=json.dumps(response_obj), status=200)

    except Exception as e:
        response_obj = {'status': 'failure', 'description': str(e)}
        return web.Response(text=json.dumps(response_obj), status=500)



# crisp partition
async def girvan_newman(request):
    """
        ---
        description: This end-point allows to compute girvan_newman Community Discovery algorithm to a network dataset.
        tags:
            - girvan_newman
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
          name: level
          schema:
            type: integer
          required: true
          description:  the level where to cut the dendrogram
        """
    code, resp = __check_token(request)
    if code == 500:
        return web.Response(text=json.dumps(resp), status=500)

    g = await __load_network(request)

    try:
        level = int(request.query['level'])
        communities = algorithms.girvan_newman(g,level)
        resp = json.loads(communities.to_json())
        response_obj = {'status': 'success', "data": resp}
        await __save_communities(communities, request)
        return web.Response(text=json.dumps(response_obj), status=200)

    except Exception as e:
        response_obj = {'status': 'failure', 'description': str(e)}
        return web.Response(text=json.dumps(response_obj), status=500)


async def em(request):
    """
        ---
        description: This end-point allows to compute EM Community Discovery algorithm to a network dataset.
        tags:
            - EM
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
          name: k
          schema:
            type: integer
          required: true
          description:  tthe number of desired communities
        """
    code, resp = __check_token(request)
    if code == 500:
        return web.Response(text=json.dumps(resp), status=500)

    g = await __load_network(request)

    try:
        k = int(request.query['k'])
        communities = algorithms.em(g,k)
        resp = json.loads(communities.to_json())
        response_obj = {'status': 'success', "data": resp}
        await __save_communities(communities, request)
        return web.Response(text=json.dumps(response_obj), status=200)

    except Exception as e:
        response_obj = {'status': 'failure', 'description': str(e)}
        return web.Response(text=json.dumps(response_obj), status=500)


async def scan(request):
    """
        ---
        description: This end-point allows to compute Scan Community Discovery algorithm to a network dataset.
        tags:
            - Scan
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
            type: float
          required: true
          description: the minimum threshold to assigning cluster membership
         - in: query
          name: mu
          schema:
            type: integer
          required: true
          description:  minimum number of neineighbors with a structural similarity that exceeds the threshold epsilon
        """
    code, resp = __check_token(request)
    if code == 500:
        return web.Response(text=json.dumps(resp), status=500)

    g = await __load_network(request)

    try:
        epsilon = float(request.query['epsilon'])
        mu = int(request.query['mu'])
        communities = algorithms.scan(g,epsilon,mu)
        resp = json.loads(communities.to_json())
        response_obj = {'status': 'success', "data": resp}
        await __save_communities(communities, request)
        return web.Response(text=json.dumps(response_obj), status=200)

    except Exception as e:
        response_obj = {'status': 'failure', 'description': str(e)}
        return web.Response(text=json.dumps(response_obj), status=500)


async def gdmp2(request):
    """
        ---
        description: This end-point allows to compute gdmp2 Community Discovery algorithm to a network dataset.
        tags:
            - gdmp2
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
          name: min_threshold
          schema:
            type: float
          required: true
          description: the minimum density threshold parameter to control the density of the output subgraphs
        """
    code, resp = __check_token(request)
    if code == 500:
        return web.Response(text=json.dumps(resp), status=500)

    g = await __load_network(request)

    try:
        min_threshold = float(request.query['min_threshold'])
        communities = algorithms.gdmp2(g,min_threshold)
        resp = json.loads(communities.to_json())
        response_obj = {'status': 'success', "data": resp}
        await __save_communities(communities, request)
        return web.Response(text=json.dumps(response_obj), status=200)

    except Exception as e:
        response_obj = {'status': 'failure', 'description': str(e)}
        return web.Response(text=json.dumps(response_obj), status=500)


async def spinglass(request):
    """
        ---
        description: This end-point allows to compute spinglass Community Discovery algorithm to a network dataset.
        tags:
            - spinglass
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
        """
    code, resp = __check_token(request)
    if code == 500:
        return web.Response(text=json.dumps(resp), status=500)

    g = await __load_network(request)

    try:
        communities = algorithms.spinglass(g)
        resp = json.loads(communities.to_json())
        response_obj = {'status': 'success', "data": resp}
        await __save_communities(communities, request)
        return web.Response(text=json.dumps(response_obj), status=200)

    except Exception as e:
        response_obj = {'status': 'failure', 'description': str(e)}
        return web.Response(text=json.dumps(response_obj), status=500)


async def eigenvector(request):
    """
        ---
        description: This end-point allows to compute eigenvector Community Discovery algorithm to a network dataset.
        tags:
            - eigenvector
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
        """
    code, resp = __check_token(request)
    if code == 500:
        return web.Response(text=json.dumps(resp), status=500)

    g = await __load_network(request)

    try:
        communities = algorithms.eigenvector(g)
        resp = json.loads(communities.to_json())
        response_obj = {'status': 'success', "data": resp}
        await __save_communities(communities, request)
        return web.Response(text=json.dumps(response_obj), status=200)

    except Exception as e:
        response_obj = {'status': 'failure', 'description': str(e)}
        return web.Response(text=json.dumps(response_obj), status=500)


async def agdl(request):
    """
        ---
        description: This end-point allows to compute agdl Community Discovery algorithm to a network dataset.
        tags:
            - agdl
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
          name: number_communities
          schema:
            type: integer
          required: true
          description: number of communities
        - in: query
          name: number_neighbors
          schema:
            type: integer
          required: true
          description: Number of neighbors to use for KNN
        - in: query
          name: kc
          schema:
            type: integer
          required: true
          description: size of the neighbor set for each cluster
        - in: query
          name: a
          schema:
            type: float
          required: true
          description: range(-infinity;+infinty)
        """
    code, resp = __check_token(request)
    if code == 500:
        return web.Response(text=json.dumps(resp), status=500)

    g = await __load_network(request)

    try:
        number_communities = int(request.query['number_communities'])
        number_neighbors = int(request.query['number_neighbors'])
        kc = int(request.query['kc'])
        a = float(request.query['a'])
        communities = algorithms.agdl(g,number_communities,number_neighbors,kc,a)
        resp = json.loads(communities.to_json())
        response_obj = {'status': 'success', "data": resp}
        await __save_communities(communities, request)
        return web.Response(text=json.dumps(response_obj), status=200)

    except Exception as e:
        response_obj = {'status': 'failure', 'description': str(e)}
        return web.Response(text=json.dumps(response_obj), status=500)


async def louvain(request):
    """
        ---
        description: This end-point allows to compute louvain Community Discovery algorithm to a network dataset.
        tags:
            - louvain
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
          name: weight
          schema:
            type: string
          required: true
          description: optional the key in graph to use as weight
        - in: query
          name: resolution
          schema:
            type: string
          required: float
          description: Will change the size of the communities, default to 1.
        - in: query
          name: randomize
          schema:
            type: boolean
          required: true
          description: Will randomize the node evaluation order and the community evaluation  order to get different partitions at each call, default False
        """
    code, resp = __check_token(request)
    if code == 500:
        return web.Response(text=json.dumps(resp), status=500)

    g = await __load_network(request)

    try:
        weight = str(request.query['weight'])
        resolution = float(request.query['resolution'])
        tmp_randomize = str(request.query['randomize'])
        if tmp_randomize == "False":
            randomize = False
        else:
            randomize = True
        communities = algorithms.louvain(g,weight,resolution,randomize)
        resp = json.loads(communities.to_json())
        response_obj = {'status': 'success', "data": resp}
        await __save_communities(communities, request)
        return web.Response(text=json.dumps(response_obj), status=200)

    except Exception as e:
        response_obj = {'status': 'failure', 'description': str(e)}
        return web.Response(text=json.dumps(response_obj), status=500)


async def leiden(request):
    """
        ---
        description: This end-point allows to compute leiden Community Discovery algorithm to a network dataset.
        tags:
            - leiden
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
          name: initial_membership
          schema:
            type: list
          required: true
          description: list of int Initial membership for the partition. If :obj:`None` then defaults to a singleton partition. Deafault None
        - in: query
          name: weights
          schema:
            type: list of double
          required: float
          description: or edge attribute Weights of edges. Can be either an iterable or an edge attribute. Deafault None
        """
    code, resp = __check_token(request)
    if code == 500:
        return web.Response(text=json.dumps(resp), status=500)

    g = await __load_network(request)

    try:
        initial_membership = json.loads(request.query['initial_membership'])
        weights = json.loads(request.query['weights'])
        communities = algorithms.leiden(g,initial_membership,weights)
        resp = json.loads(communities.to_json())
        response_obj = {'status': 'success', "data": resp}
        await __save_communities(communities, request)
        return web.Response(text=json.dumps(response_obj), status=200)

    except Exception as e:
        response_obj = {'status': 'failure', 'description': str(e)}
        return web.Response(text=json.dumps(response_obj), status=500)


async def rb_pots(request):
    """
        ---
        description: This end-point allows to compute rb_pots Community Discovery algorithm to a network dataset.
        tags:
            - rb_pots
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
          name: initial_membership
          schema:
            type: list
          required: true
          description: list of int Initial membership for the partition. If :obj:`None` then defaults to a singleton partition. Deafault None
        - in: query
          name: weights
          schema:
            type: list of double
          required: float
          description: or edge attribute Weights of edges. Can be either an iterable or an edge attribute. Deafault None
        - in: query
          name: resolution_parameter
          schema:
            type: double
          required: float
          description: double >0 A parameter value controlling the coarseness of the clustering. Higher resolutions lead to more communities, while lower resolutions lead to fewer communities. Default 1
        """
    code, resp = __check_token(request)
    if code == 500:
        return web.Response(text=json.dumps(resp), status=500)

    g = await __load_network(request)

    try:
        initial_membership = json.loads(request.query['initial_membership'])
        weights = json.loads(request.query['weights'])
        resolution_parameter= float(request.query['resolution_parameter'])
        communities = algorithms.rb_pots(g,initial_membership,weights,resolution_parameter)
        resp = json.loads(communities.to_json())
        response_obj = {'status': 'success', "data": resp}
        await __save_communities(communities, request)
        return web.Response(text=json.dumps(response_obj), status=200)

    except Exception as e:
        response_obj = {'status': 'failure', 'description': str(e)}
        return web.Response(text=json.dumps(response_obj), status=500)


async def rber_pots(request):
    """
        ---
        description: This end-point allows to compute rber_pots Community Discovery algorithm to a network dataset.
        tags:
            - rber_pots
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
          name: initial_membership
          schema:
            type: list
          required: true
          description: list of int Initial membership for the partition. If :obj:`None` then defaults to a singleton partition. Deafault None
        - in: query
          name: weights
          schema:
            type: list of double
          required: float
          description: or edge attribute Weights of edges. Can be either an iterable or an edge attribute. Deafault None
        - in: query
          name: node_sizes
          schema:
            type: list of integer
          required: float
          description:  list of int, or vertex attribute Sizes of nodes are necessary to know the size of communities in aggregate graphs. Usually this is set to 1 for all nodes, but in specific cases  this could be changed. Deafault None
        - in: query
          name: resolution_parameter
          schema:
            type: double
          required: float
          description: double >0 A parameter value controlling the coarseness of the clustering. Higher resolutions lead to more communities, while lower resolutions lead to fewer communities. Default 1
        """
    code, resp = __check_token(request)
    if code == 500:
        return web.Response(text=json.dumps(resp), status=500)

    g = await __load_network(request)

    try:
        initial_membership = json.loads(request.query['initial_membership'])
        weights = json.loads(request.query['weights'])
        node_sizes = json.loads(request.query['node_sizes'])
        resolution_parameter= float(request.query['resolution_parameter'])
        communities = algorithms.rber_pots(g,initial_membership,weights,node_sizes,resolution_parameter)
        resp = json.loads(communities.to_json())
        response_obj = {'status': 'success', "data": resp}
        await __save_communities(communities, request)
        return web.Response(text=json.dumps(response_obj), status=200)

    except Exception as e:
        response_obj = {'status': 'failure', 'description': str(e)}
        return web.Response(text=json.dumps(response_obj), status=500)


async def cpm(request):
    """
        ---
        description: This end-point allows to compute cpm Community Discovery algorithm to a network dataset.
        tags:
            - cpm
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
          name: initial_membership
          schema:
            type: list
          required: true
          description: list of int Initial membership for the partition. If :obj:`None` then defaults to a singleton partition. Deafault None
        - in: query
          name: weights
          schema:
            type: list of double
          required: float
          description: or edge attribute Weights of edges. Can be either an iterable or an edge attribute. Deafault None
        - in: query
          name: node_sizes
          schema:
            type: list of integer
          required: float
          description:  list of int, or vertex attribute Sizes of nodes are necessary to know the size of communities in aggregate graphs. Usually this is set to 1 for all nodes, but in specific cases  this could be changed. Deafault None
        - in: query
          name: resolution_parameter
          schema:
            type: double
          required: float
          description: double >0 A parameter value controlling the coarseness of the clustering. Higher resolutions lead to more communities, while lower resolutions lead to fewer communities. Default 1
        """
    code, resp = __check_token(request)
    if code == 500:
        return web.Response(text=json.dumps(resp), status=500)

    g = await __load_network(request)

    try:
        initial_membership = json.loads(request.query['initial_membership'])
        weights = json.loads(request.query['weights'])
        node_sizes = json.loads(request.query['node_sizes'])
        resolution_parameter= float(request.query['resolution_parameter'])
        communities = algorithms.cpm(g,initial_membership,weights,node_sizes,resolution_parameter)
        resp = json.loads(communities.to_json())
        response_obj = {'status': 'success', "data": resp}
        await __save_communities(communities, request)
        return web.Response(text=json.dumps(response_obj), status=200)

    except Exception as e:
        response_obj = {'status': 'failure', 'description': str(e)}
        return web.Response(text=json.dumps(response_obj), status=500)


async def significance_communities(request):
    """
        ---
        description: This end-point allows to compute significance_communities Community Discovery algorithm to a network dataset.
        tags:
            - significance_communities
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
          name: initial_membership
          schema:
            type: list
          required: true
          description: list of int Initial membership for the partition. If :obj:`None` then defaults to a singleton partition. Deafault None
        - in: query
          name: node_sizes
          schema:
            type: list of integer
          required: float
          description:  list of int, or vertex attribute Sizes of nodes are necessary to know the size of communities in aggregate graphs. Usually this is set to 1 for all nodes, but in specific cases  this could be changed. Deafault None
        """
    code, resp = __check_token(request)
    if code == 500:
        return web.Response(text=json.dumps(resp), status=500)

    g = await __load_network(request)

    try:
        initial_membership = json.loads(request.query['initial_membership'])
        node_sizes = json.loads(request.query['node_sizes'])
        communities = algorithms.significance_communities(g,initial_membership,node_sizes)
        resp = json.loads(communities.to_json())
        response_obj = {'status': 'success', "data": resp}
        await __save_communities(communities, request)
        return web.Response(text=json.dumps(response_obj), status=200)

    except Exception as e:
        response_obj = {'status': 'failure', 'description': str(e)}
        return web.Response(text=json.dumps(response_obj), status=500)


async def surprise_communities(request):
    """
        ---
        description: This end-point allows to compute surprise_communities Community Discovery algorithm to a network dataset.
        tags:
            - surprise_communities
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
          name: initial_membership
          schema:
            type: list
          required: true
          description: list of int Initial membership for the partition. If :obj:`None` then defaults to a singleton partition. Deafault None
        - in: query
          name: weights
          schema:
            type: list of double
          required: float
          description: or edge attribute Weights of edges. Can be either an iterable or an edge attribute. Deafault None
        - in: query
          name: node_sizes
          schema:
            type: list of integer
          required: float
          description:  list of int, or vertex attribute Sizes of nodes are necessary to know the size of communities in aggregate graphs. Usually this is set to 1 for all nodes, but in specific cases  this could be changed. Deafault None
        """
    code, resp = __check_token(request)
    if code == 500:
        return web.Response(text=json.dumps(resp), status=500)

    g = await __load_network(request)

    try:
        initial_membership = json.loads(request.query['initial_membership'])
        weights = json.loads(request.query['weights'])
        node_sizes = json.loads(request.query['node_sizes'])
        communities = algorithms.surprise_communities(g,initial_membership,weights,node_sizes)
        resp = json.loads(communities.to_json())
        response_obj = {'status': 'success', "data": resp}
        await __save_communities(communities, request)
        return web.Response(text=json.dumps(response_obj), status=200)

    except Exception as e:
        response_obj = {'status': 'failure', 'description': str(e)}
        return web.Response(text=json.dumps(response_obj), status=500)


async def greedy_modularity(request):
    """
        ---
        description: This end-point allows to compute greedy_modularity Community Discovery algorithm to a network dataset.
        tags:
            - greedy_modularity
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
          name: weights
          schema:
            type: list of double
          required: float
          description: or edge attribute Weights of edges. Can be either an iterable or an edge attribute. Deafault None
        """
    code, resp = __check_token(request)
    if code == 500:
        return web.Response(text=json.dumps(resp), status=500)

    g = await __load_network(request)

    try:
        weights = json.loads(request.query['weights'])
        communities = algorithms.greedy_modularity(g,weights)
        resp = json.loads(communities.to_json())
        response_obj = {'status': 'success', "data": resp}
        await __save_communities(communities, request)
        return web.Response(text=json.dumps(response_obj), status=200)

    except Exception as e:
        response_obj = {'status': 'failure', 'description': str(e)}
        return web.Response(text=json.dumps(response_obj), status=500)


async def infomap(request):
    """
        ---
        description: This end-point allows to compute infomap Community Discovery algorithm to a network dataset.
        tags:
            - infomap
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
        """
    code, resp = __check_token(request)
    if code == 500:
        return web.Response(text=json.dumps(resp), status=500)

    g = await __load_network(request)

    try:
        communities = algorithms.infomap(g)
        resp = json.loads(communities.to_json())
        response_obj = {'status': 'success', "data": resp}
        await __save_communities(communities, request)
        return web.Response(text=json.dumps(response_obj), status=200)

    except Exception as e:
        response_obj = {'status': 'failure', 'description': str(e)}
        return web.Response(text=json.dumps(response_obj), status=500)


async def walktrap(request):
    """
        ---
        description: This end-point allows to compute walktrap Community Discovery algorithm to a network dataset.
        tags:
            - walktrap
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
        """
    code, resp = __check_token(request)
    if code == 500:
        return web.Response(text=json.dumps(resp), status=500)

    g = await __load_network(request)

    try:
        communities = algorithms.walktrap(g)
        resp = json.loads(communities.to_json())
        response_obj = {'status': 'success', "data": resp}
        await __save_communities(communities, request)
        return web.Response(text=json.dumps(response_obj), status=200)

    except Exception as e:
        response_obj = {'status': 'failure', 'description': str(e)}
        return web.Response(text=json.dumps(response_obj), status=500)


async def label_propagation(request):
    """
        ---
        description: This end-point allows to compute label_propagation Community Discovery algorithm to a network dataset.
        tags:
            - label_propagation
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
        """
    code, resp = __check_token(request)
    if code == 500:
        return web.Response(text=json.dumps(resp), status=500)

    g = await __load_network(request)

    try:
        communities = algorithms.label_propagation(g)
        resp = json.loads(communities.to_json())
        response_obj = {'status': 'success', "data": resp}
        await __save_communities(communities, request)
        return web.Response(text=json.dumps(response_obj), status=200)

    except Exception as e:
        response_obj = {'status': 'failure', 'description': str(e)}
        return web.Response(text=json.dumps(response_obj), status=500)


async def async_fluid(request):
    """
        ---
        description: This end-point allows to compute async_fluid Community Discovery algorithm to a network dataset.
        tags:
            - async_fluid
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
          name: k
          schema:
            type: integer
          required: true
          description: Number of communities to search
        """
    code, resp = __check_token(request)
    if code == 500:
        return web.Response(text=json.dumps(resp), status=500)

    g = await __load_network(request)

    try:
        k = int(request.query['k'])
        communities = algorithms.async_fluid(g,k)
        resp = json.loads(communities.to_json())
        response_obj = {'status': 'success', "data": resp}
        await __save_communities(communities, request)
        return web.Response(text=json.dumps(response_obj), status=200)

    except Exception as e:
        response_obj = {'status': 'failure', 'description': str(e)}
        return web.Response(text=json.dumps(response_obj), status=500)


async def der(request):
    """
        ---
        description: This end-point allows to compute der Community Discovery algorithm to a network dataset.
        tags:
            - der
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
          name: walk_len
          schema:
            type: integer
          required: true
          description: length of the random walk, default 3
        - in: query
          name: threshold
          schema:
            type: float
          required: true
          description: threshold for stop criteria; if the likelihood_diff is less than threshold tha algorithm stops, default 0.00001
        - in: query
          name: iter_bound
          schema:
            type: integer
          required: true
          description: maximum number of iteration, default 50
        """
    code, resp = __check_token(request)
    if code == 500:
        return web.Response(text=json.dumps(resp), status=500)

    g = await __load_network(request)

    try:
        walk_len = int(request.query['walk_len'])
        threshold = float(request.query['threshold'])
        iter_bound = int(request.query['iter_bound'])
        communities = algorithms.der(g,walk_len,threshold,iter_bound)
        resp = json.loads(communities.to_json())
        response_obj = {'status': 'success', "data": resp}
        await __save_communities(communities, request)
        return web.Response(text=json.dumps(response_obj), status=200)

    except Exception as e:
        response_obj = {'status': 'failure', 'description': str(e)}
        return web.Response(text=json.dumps(response_obj), status=500)



async def frc_fgsn(request):
    """
        ---
        description: This end-point allows to compute frc_fgsn Community Discovery algorithm to a network dataset.
        tags:
            - frc_fgsn
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
          name: theta
          schema:
            type: float
          required: true
          description: community density coefficient
        - in: query
          name: eps
          schema:
            type: float
          required: true
          description: coupling coefficient of the community. Ranges in [0, 1], small values ensure that only strongly connected node granules are merged togheter.
        - in: query
          name: r
          schema:
            type: integer
          required: true
          description: radius of the granule
        """
    code, resp = __check_token(request)
    if code == 500:
        return web.Response(text=json.dumps(resp), status=500)

    g = await __load_network(request)

    try:
        theta = float(request.query['theta'])
        eps = float(request.query['eps'])
        r = int(request.query['r'])
        communities = algorithms.frc_fgsn(g,theta,eps,r)
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
        web.post('/api/cd/ego_networks', ego_networks),
        web.post('/api/cd/node_perception', node_perception),
        web.post('/api/cd/overlapping_seed_set_expansion', overlapping_seed_set_expansion),
        web.post('/api/cd/kclique', kclique),
        web.post('/api/cd/lfm', lfm),
        web.post('/api/cd/lais2', lais2),
        web.post('/api/cd/congo', congo),
        web.post('/api/cd/conga', conga),
        web.post('/api/cd/lemon', lemon),
        web.post('/api/cd/slpa', slpa),
        web.post('/api/cd/multicom', multicom),
        web.post('/api/cd/big_clam', big_clam),
        web.post('/api/cd/girvan_newman', girvan_newman),
        web.post('/api/cd/em', em),
        web.post('/api/cd/scan', scan),
        web.post('/api/cd/gdmp2', gdmp2),
        web.post('/api/cd/spinglass', spinglass),
        web.post('/api/cd/eigenvector', eigenvector),
        web.post('/api/cd/agdl', agdl),
        web.post('/api/cd/louvain', louvain),
        web.post('/api/cd/leiden', leiden),
        web.post('/api/cd/rb_pots', rb_pots),
        web.post('/api/cd/rber_pots', rber_pots),
        web.post('/api/cd/cpm', cpm),
        web.post('/api/cd/significance_communities', significance_communities),
        web.post('/api/cd/surprise_communities', surprise_communities),
        web.post('/api/cd/greedy_modularity', greedy_modularity),
        web.post('/api/cd/infomap', infomap),
        web.post('/api/cd/walktrap', walktrap),
        web.post('/api/cd/label_propagation', label_propagation),
        web.post('/api/cd/async_fluid', async_fluid),
        web.post('/api/cd/der', der),
        web.post('/api/cd/frc_fgsn', frc_fgsn)

    ])

    setup_swagger(app, swagger_url="/api/v1/doc", description="",
                  title="CDlib Server API",
                  api_version="0.1.3",
                  contact="giulio.rossetti@gmail.com")

    return app


if __name__ == '__main__':
    web.run_app(make_app(), port=8081, host="0.0.0.0")
