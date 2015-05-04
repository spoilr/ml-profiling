#!flask/bin/python
from flask import Flask
from flask import jsonify
from flask import request
from flask import render_template
from flask import url_for
from create_struct import *
from urllib import urlopen
from urllib import urlencode
import json
from copy import deepcopy

import sys
sys.path.insert(0, 'utils/')

from explanatory_data import *

app = Flask(__name__)
	
def graph(nodes, edges, cpts):
	graph = dict()
	graph['nodes'] = nodes
	graph['links'] = edges
	graph['cpts'] = cpts
	return json.dumps(graph, indent=2)

@app.route('/', methods=['GET'])
def show():
	return render_template('index.html')

@app.route('/net', methods=['GET'])
def show_net():
	return render_template('show_net.html')

@app.route('/networks/net.json', methods=['GET', 'POST'])
def net_graph():
	
	if request.method == 'POST':
		nodes = request.get_json()["nodes"]
		edges = request.get_json()["links"]
		cpts = request.get_json()["cpts"]
		bn_net.V = nodes
		bn_net.E = edges
		bn_net.Vdata = update_bayesian_network_structure("net", nodes, edges, cpts)
		return render_template('show_net.html')

	if request.method == 'GET':	
		nodes = bn_net.V
		edges = bn_net.E
		cpts = bn_net.Vdata
		return graph(nodes, edges, cpts)

@app.route('/ill', methods=['GET'])
def show_ill():
	return render_template('show_ill.html')

@app.route('/networks/ill.json', methods=['GET', 'POST'])
def ill_graph():
	if request.method == 'POST':
		nodes = request.get_json()["nodes"]
		edges = request.get_json()["links"]
		cpts = request.get_json()["cpts"]
		bn_ill.V = nodes
		bn_ill.E = edges
		bn_ill.Vdata = update_bayesian_network_structure("ill", nodes, edges, cpts)
		return render_template('show_ill.html')

	if request.method == 'GET':		
		nodes = bn_ill.V
		edges = bn_ill.E
		cpts = bn_ill.Vdata
		return graph(nodes, edges, cpts)

@app.route('/ideo', methods=['GET'])
def show_ideo():
	return render_template('show_ideo.html')

@app.route('/networks/ideo.json', methods=['GET', 'POST'])
def ideo_graph():
	if request.method == 'POST':
		nodes = request.get_json()["nodes"]
		edges = request.get_json()["links"]
		cpts = request.get_json()["cpts"]
		bn_ideo.V = nodes
		bn_ideo.E = edges
		bn_ideo.Vdata = update_bayesian_network_structure("ideo", nodes, edges, cpts)
		return render_template('show_ideo.html')

	if request.method == 'GET':		
		nodes = bn_ideo.V
		edges = bn_ideo.E
		cpts = bn_ideo.Vdata
		return graph(nodes, edges, cpts)

def create_evidence_and_inference(categs, theme):
	if request.method == 'POST':
		evidence = dict()	
		encoded_evidence = dict()
		submit = request.form
		
		for key, value in submit.iteritems():
			if value != "None":
				if key in explanatory_data:
					evidence[key] = explanatory_data[key][value]
					encoded_evidence[key] = value
				else:
					if value == "no":
						evidence[key] = 0
						encoded_evidence[key] = "no"
					elif value == "yes":
						evidence[key] = 1
						encoded_evidence[key] = "yes"
			else:
				encoded_evidence[key] = None
		
		url = "http://127.0.0.1:5000/inference/" + theme + "?" + urlencode(evidence)
		
		response = urlopen(url)
		json_inf = json.loads(response.read())

		inference = dict()
		for key, values in json_inf.iteritems():
			if key in inv_explanatory_data:
				temp = dict()
				for k, v in values.iteritems():
					temp[inv_explanatory_data[key][int(k)]] = v
				inference[key] = temp
			else:
				temp = dict()
				for k, v in values.iteritems():
					if k == "0":
						temp["no"] = v
					elif k == "1":
						temp["yes"] = v
				inference[key] = temp

		return inference, encoded_evidence
	return None, None

def get_categories(nodes):
	categs = dict()
	for n in nodes:
		if n in inv_explanatory_data:
			categs[n] = inv_explanatory_data[n]
		else:
			categs[n] = {0:"no", 1:"yes"}
	return categs		

@app.route('/bayes/net', methods=['GET', 'POST'])
def bayes_net():
	nodes = bn_net.V
	categs = get_categories(nodes)
	inference, evidence = create_evidence_and_inference(categs, "net")
	return render_template('net.html', categs=categs, evidence=evidence, inference=inference)

@app.route('/bayes/ill', methods=['GET', 'POST'])
def bayes_ill():
	nodes = bn_ill.V
	categs = get_categories(nodes)
	inference, evidence = create_evidence_and_inference(categs, "ill")
	return render_template('ill.html', categs=categs, evidence=evidence, inference=inference)

@app.route('/bayes/ideo', methods=['GET', 'POST'])
def bayes_ideo():
	nodes = bn_ideo.V
	categs = get_categories(nodes)
	inference,evidence = create_evidence_and_inference(categs, "ideo")
	return render_template('ideo.html', categs=categs, evidence=evidence, inference=inference)

@app.route('/inference/net', methods=['GET'])
def net_inferences():
	evidence = request.args.to_dict()
	evidence = dict((k,int(v)) for k,v in evidence.iteritems())
	fn = TableCPDFactorization(bn_net)
	inf = inference(bn_net, evidence)
	return inf

@app.route('/inference/ill', methods=['GET'])
def ill_inferences():
	evidence = request.args.to_dict()
	evidence = dict((k,int(v)) for k,v in evidence.iteritems())
	fn = TableCPDFactorization(bn_ill)
	inf = inference(bn_ill, evidence)
	return inf

@app.route('/inference/ideo', methods=['GET'])
def ideo_inferences():
	evidence = request.args.to_dict()
	evidence = dict((k,int(v)) for k,v in evidence.iteritems())
	fn = TableCPDFactorization(bn_ideo)
	inf = inference(bn_ideo, evidence)
	return inf	

def shutdown_server():
    func = request.environ.get('werkzeug.server.shutdown')
    if func is None:
        raise RuntimeError('Not running with the Werkzeug Server')
    func()

@app.route('/shutdown', methods=['POST'])
def shutdown():
    shutdown_server()
    return 'Server shutting down...'

@app.route('/originals/networks/net.json', methods=['POST'])
def original_net_graph():
	bn_net.V = original_bn_net.V
	bn_net.E = original_bn_net.E
	bn_net.Vdata = original_bn_net.Vdata

@app.route('/originals/networks/ill.json', methods=['POST'])
def original_ill_graph():
	bn_ill.V = original_bn_ill.V
	bn_ill.E = original_bn_ill.E
	bn_ill.Vdata = original_bn_ill.Vdata

@app.route('/originals/networks/ideo.json', methods=['POST'])
def original_ideo_graph():
	bn_ideo.V = original_bn_ideo.V
	bn_ideo.E = original_bn_ideo.E
	bn_ideo.Vdata = original_bn_ideo.Vdata
		

if __name__ == '__main__':
	global bn_net
	global bn_ill
	global bn_ideo
	bn_net = create_bayesian_network_structure('net')
	bn_ill = create_bayesian_network_structure('ill')
	bn_ideo = create_bayesian_network_structure('ideo')
	original_bn_net = deepcopy(bn_net)
	original_bn_ill = deepcopy(bn_ill)
	original_bn_ideo = deepcopy(bn_ideo)
	app.run(debug=True, threaded=True)