from flask import Flask, send_from_directory, request, jsonify
from rdflib import Graph, Namespace, URIRef
from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch
from fuzzywuzzy import fuzz
import json

app = Flask(__name__)
app.secret_key = "Ontology"

DB_PATH = 'ontologyx.owl'
BASE_URL = 'http://www.sea-cucumber.org/'

ont = Namespace(BASE_URL)
rdf = Namespace("http://www.w3.org/1999/02/22-rdf-syntax-ns#")
rdfs = Namespace("http://www.w3.org/2000/01/rdf-schema#")
owl = Namespace("http://www.w3.org/2002/07/owl#")

# Load the ontology graph
g = Graph()
g.parse(DB_PATH, format='xml')

# Initialize the LLM model and tokenizer
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForMaskedLM.from_pretrained(model_name)

# Extract all class names from the ontology
class_names = set()
for s, p, o in g:
    if str(p) == "http://www.w3.org/1999/02/22-rdf-syntax-ns#type" and str(o) == "http://www.w3.org/2002/07/owl#Class":
        class_name = str(s).split('/')[-1]
        class_names.add(class_name)

# Fine-tune the model on the class names
fine_tune_data = [f"The sea cucumber species {name} is important." for name in class_names]
inputs = tokenizer(fine_tune_data, return_tensors="pt", padding=True, truncation=True)
outputs = model(**inputs, labels=inputs["input_ids"])
loss = outputs.loss
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
loss.backward()
optimizer.step()

def autocomplete(query):
    input_ids = tokenizer.encode(query + " [MASK]", return_tensors="pt")
    with torch.no_grad():
        outputs = model(input_ids)
    predictions = outputs[0][0, -1].topk(5)
    return [tokenizer.decode([pred.item()]).strip() for pred in predictions.indices]

def fuzzy_search(query):
    return sorted(class_names, key=lambda x: fuzz.ratio(query.lower(), x.lower()), reverse=True)[:5]

def readable_uri_component(text):
    return text.replace(' ', "_")

def get_full_url(path=[]):
    return f"classes/{'/'.join(path)}"

def graph_json(g):
    json_data = []

    def find_node(path, data=[], current_index=0):
        for datum in data:
            if datum['name'] == path[current_index]:
                if len(path) - 1 == current_index:
                    return datum
                else:
                    return find_node(path, data=datum['data'], current_index=current_index + 1)
        else:
            x = {
                'name': path[current_index],
                'path': "/".join(path[0:current_index + 1]),
                'relationship': None,
                'data': []
            }
            data.append(x)

            if len(path) - 1 == current_index:
                return x
            else:
                return find_node(path, data=x['data'], current_index=current_index + 1)

    for child, relationship, parent in list(g.triples((None, None, None))):
        parent = str(parent).removeprefix(BASE_URL).removeprefix("classes/")
        child = str(child).removeprefix(BASE_URL).removeprefix("classes/")
        relationship = str(relationship).removeprefix(BASE_URL).removeprefix("relationship/")

        parent_name = parent.split("/")[-1]
        child_name = child.split("/")[-1]
        relationship_name = relationship.split("/")[-1]

        parent_path = parent.split("/")
        child_path = child.split("/")
        relationship_path = relationship.split("/")

        if parent_name in ["owl#Class", "owl#Ontology", "", ""]:
            continue

        y = find_node(parent_path, json_data, 0)
        if relationship_name in ["rdf-schema#subClassOf"]:
            z = y['data']
            for entry in z:
                if entry['name'] == child_name:
                    break
            else:
                y['data'].append({
                    'name': child_name,
                    'path': child,
                    'data': []
                })

        else:
            y['relationship'] = relationship_name

    return json_data

def search_graph(g, query):
    # First, try to autocomplete the query
    autocompleted = autocomplete(query)

    # If autocomplete doesn't work well, use fuzzy search
    if not any(query.lower() in result.lower() for result in autocompleted):
        autocompleted = fuzzy_search(query)

    # Use the best match (first result) for the actual search
    best_match = autocompleted[0]

    json_data = []
    found_nodes = []
    found_full_nodes = []
    out_data = {}

    def find_node(path, data=[], current_index=0):
        for datum in data:
            if datum['name'] == path[current_index]:
                if len(path) - 1 == current_index:
                    return datum
                else:
                    return find_node(path, data=datum['data'], current_index=current_index + 1)
        else:
            x = {
                'name': path[current_index],
                'path': "/".join(path[0:current_index + 1]),
                'relationship': None,
                'data': []
            }
            data.append(x)

            if len(path) - 1 == current_index:
                return x
            else:
                return find_node(path, data=x['data'], current_index=current_index + 1)

    for child, relationship, parent in list(g.triples((None, None, None))):
        parent = str(parent).removeprefix(BASE_URL).removeprefix("classes/")
        child = str(child).removeprefix(BASE_URL).removeprefix("classes/")
        relationship = str(relationship).removeprefix(BASE_URL).removeprefix("relationship/")

        parent_name = parent.split("/")[-1]
        child_name = child.split("/")[-1]
        relationship_name = relationship.split("/")[-1]

        parent_path = parent.split("/")
        child_path = child.split("/")
        relationship_path = relationship.split("/")

        if best_match.lower() in child_name.lower():
            if child_path not in found_nodes:
                found_nodes.append(child_path)

        if parent_name in ["owl#Class", "owl#Ontology", "", ""]:
            continue

        y = find_node(parent_path, json_data, 0)
        if relationship_name in ["rdf-schema#subClassOf"]:
            z = y['data']
            for entry in z:
                if entry['name'] == child_name:
                    break
            else:
                y['data'].append({
                    'name': child_name,
                    'path': child,
                    'data': []
                })

        else:
            y['relationship'] = relationship_name

    for child, relationship, parent in list(g.triples((None, None, None))):
        parent = str(parent).removeprefix(BASE_URL).removeprefix("classes/")
        child = str(child).removeprefix(BASE_URL).removeprefix("classes/")
        relationship = str(relationship).removeprefix(BASE_URL).removeprefix("relationship/")

        parent_name = parent.split("/")[-1]
        child_name = child.split("/")[-1]
        relationship_name = relationship.split("/")[-1]

        parent_path = parent.split("/")
        child_path = child.split("/")
        relationship_path = relationship.split("/")

        for node in found_nodes:
            x = "/".join(node)
            if child.startswith(x):
                found_full_nodes.append(child_path)

    found_nodes = []

    max_len = 0

    for node in found_full_nodes:
        if len(node) > max_len:
            max_len = len(node)

    for node in found_full_nodes:
        if len(node) == max_len:
            found_nodes.append(node)
            out_data["/".join(node)] = {}

    for child, relationship, parent in list(g.triples((None, None, None))):
        parent = str(parent).removeprefix(BASE_URL).removeprefix("classes/")
        child = str(child).removeprefix(BASE_URL).removeprefix("classes/")
        relationship = str(relationship).removeprefix(BASE_URL).removeprefix("relationship/")

        parent_name = parent.split("/")[-1]
        child_name = child.split("/")[-1]
        relationship_name = relationship.split("/")[-1]

        for node in found_nodes:

            for i in range(len(node) - 1):
                parent = node[i]
                child = node[i + 1]

                if parent_name != parent: continue
                if child_name != child: continue
                if relationship_name == "rdf-schema#subClassOf": continue

                out_data["/".join(node)][i] = relationship_name

    out = []

    for node in out_data:
        relationship = out_data[node]
        node = str(node).split("/")
        text = ""
        for i in range(len(node) - 1):
            text += f"{node[i]} {relationship[i]} "
        text += f"{node[len(node) - 1]}"

        out.append(text)

    return out, autocompleted

def add_node(graph, parent, node, path=[]):
    parent_node_path = readable_uri_component(get_full_url(path=path))
    current_node_path = parent_node_path + "/" + readable_uri_component(node['name'])

    relationship_node_path = URIRef(BASE_URL + "relationship/" + readable_uri_component(parent['relationship']))

    graph.add((ont[current_node_path], rdf.type, owl.Class))
    graph.add((ont[current_node_path], rdfs.subClassOf, ont[parent_node_path]))
    graph.add((ont[current_node_path], relationship_node_path, ont[parent_node_path]))

    try:
        for child in node['data']:
            x = path.copy()
            x.append(node['name'])
            add_node(graph=graph, parent=node, node=child, path=x)
    except Exception as e:
        pass

def get_new_graph(template=False):
    g = Graph()
    g.bind("ont", ont)
    g.bind("rdf", rdf)
    g.bind("rdfs", rdfs)
    g.bind("owl", owl)

    g.add((ont.Ontology, rdf.type, owl.Ontology))
    if template:
        g.add((ont["remove"], rdf.type, owl.Class))
        g.add((ont["remove"], rdfs.subClassOf, ont["seacucumber"]))
        g.add((ont["remove"], URIRef(BASE_URL + "relationship/" + readable_uri_component('test')), ont["seacucumber"]))

    return g

def load_graph():
    try:
        g = Graph()
        g.parse(DB_PATH, format='xml')
        return g
    except Exception as e:
        print(e)
        return get_new_graph(template=True)

def save_graph(g):
    g.serialize(DB_PATH, format='xml')


@app.route('/<path:path>')
def send_static_file(path):
    return send_from_directory('static', path)


@app.route('/')
def send_index():
    return send_from_directory('static', "index.html")


@app.get('/api/graph/json')
def get_json():
    graph = load_graph()
    return json.dumps(graph_json(graph), indent=2)


@app.post('/api/graph/update')
def post_type():
    data = request.get_json()

    print(json.dumps(data, indent=2))

    g = get_new_graph()

    parent = data[0]
    data = data[0]['data']

    for node in data:
        add_node(g, parent, node, path=['seacucumber'])

    save_graph(g)
    return graph_json(g)


@app.post('/api/query')
def post_query():
    try:
        data = request.get_json()
        query = data['query']

        graph = load_graph()

        # Execute the SPARQL query
        query_results = graph.query(query)

        # Process the results
        results = []
        for row in query_results:
            result = {}
            for var in query_results.vars:
                result[var] = str(row[var])
            results.append(result)

        return jsonify(results)
    except Exception as e:
        print(f"Error executing query: {e}")
        return jsonify({"error": str(e)}), 400

@app.get('/api/search')
def search():
    try:
        data = request.args
        query = data['query']

        graph = load_graph()

        results, autocomplete_suggestions = search_graph(graph, query)

        # Return only the results, keeping the original format
        return jsonify(results)
    except Exception as e:
        print(e)

        return jsonify(["Unable to find any results"])


if __name__ == '__main__':
    app.run(debug=True)