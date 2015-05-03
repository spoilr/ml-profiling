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

import sys
sys.path.insert(0, 'utils/')

from explanatory_data import *

app = Flask(__name__)
	
def graph(nodes, edges):
	graph = dict()
	graph['nodes'] = nodes
	graph['links'] = edges
	return json.dumps(graph, indent=2)

@app.route('/', methods=['GET'])
def show():
	return render_template('index.html')

@app.route('/networks/test.json', methods=['GET'])
def home():
	bn_test = create_bayesian_network_structure('test')
	nodes = bn_test.V
	edges = bn_test.E
	return graph(nodes, edges)

@app.route('/net', methods=['GET'])
def show_net():
	return render_template('show_net.html')

@app.route('/networks/net.json', methods=['GET'])
def net_graph():
	nodes = bn_net.V
	edges = bn_net.E
	
	return graph(nodes, edges)

@app.route('/ill', methods=['GET'])
def show_ill():
	return render_template('show_ill.html')

@app.route('/networks/ill.json', methods=['GET'])
def ill_graph():
	nodes = bn_ill.V
	edges = bn_ill.E
	return graph(nodes, edges)

@app.route('/ideo', methods=['GET'])
def show_ideo():
	return render_template('show_ideo.html')

@app.route('/networks/ideo.json', methods=['GET'])
def ideo_graph():
	nodes = bn_ideo.V
	edges = bn_ideo.E
	return graph(nodes, edges)

def create_evidence_and_inference(categs, theme):
	if request.method == 'POST':
		evidence = dict()	
		submit = request.form
		
		for key, value in submit.iteritems():
			if value != "None":
				if key in explanatory_data:
					evidence[key] = explanatory_data[key][value]
				else:
					if value == "no":
						evidence[key] = 0
					elif value == "yes":
						evidence[key] = 1
		
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

		return inference
	return None		

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
	inference = create_evidence_and_inference(categs, "net")
	return render_template('net.html', categs=categs, inference=inference)

@app.route('/bayes/ill', methods=['GET', 'POST'])
def bayes_ill():
	nodes = bn_ill.V
	categs = get_categories(nodes)
	inference = create_evidence_and_inference(categs, "ill")
	return render_template('ill.html', categs=categs, inference=inference)

@app.route('/bayes/ideo', methods=['GET', 'POST'])
def bayes_ideo():
	nodes = bn_ideo.V
	categs = get_categories(nodes)
	inference = create_evidence_and_inference(categs, "ideo")
	return render_template('ideo.html', categs=categs, inference=inference)

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

if __name__ == '__main__':
	bn_net = create_bayesian_network_structure('net')
	bn_ill = create_bayesian_network_structure('ill')
	bn_ideo = create_bayesian_network_structure('ideo')
	app.run(debug=True, threaded=True)