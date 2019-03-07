from aiohttp import web
from aiohttp_swagger import *
import json
from cdlib import algorithms, readwrite
from networkx.readwrite import json_graph
import uuid
import os, shutil


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
        - Health check
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


async def upload_network(request):
    code, resp = __check_token(request)
    if code == 500:
        return web.Response(text=json.dumps(resp), status=500)

    await __save_network(request)
    response_obj = {"status": "success"}
    return web.Response(text=json.dumps(response_obj), status=200)


async def angel(request):
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


async def make_app():
    app = web.Application()

    # add the routes
    app.add_routes([
        web.get('/api/create_experiment', create_experiment),
        web.post('/api/destroy_experiment', destroy_experiment),
        web.post('/api/upload/network', upload_network),
        web.post('/api/cd/angel', angel),
    ])

    setup_swagger(app, swagger_url="/api/v1/doc", description="",
                  title="CDlib Server API",
                  api_version="0.1.3",
                  contact="giulio.rossetti@gmail.com")

    return app


if __name__ == '__main__':
    web.run_app(make_app(), port=8080)
