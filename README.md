# CDlib-Rest - Community Discovery Library REST Service.

This project offers a REST interface for the [cdlib](https://github.com/GiulioRossetti/cdlib) Python library.


#### Tools
* REST service: cdlib_rest/cdlib_server.py
  * Web API docs: http://0.0.0.0:8081/api/v1/doc/
  * Unittest: test/test_client.py
* Python REST client: cdlib_rest/cdlib_client.py



#### REST service setup
Local testing
```python
python cdlib_rest/cdlib_server.py
```

In order to change the binding IP/port modify the cdlib_server.py file.

#### Client API
To query cdlib_rest import the API in your application and use the following template:

```python
import networkx as nx
from cdlib_rest import CDlib_API

with CDlib_API("http://0.0.0.0", 8081) as api:
    
    g = nx.karate_club_graph()
    api.load_network(g)

    coms = api.demon(epsilon=0.25)
    stats = api.fitness_scores([coms], summary=False)
           
```
