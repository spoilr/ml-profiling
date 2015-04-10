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

@app.route('/')
def index():
	return "Hello World"
	
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
		print 'Evidence ' + str(evidence)

		url = "http://127.0.0.1:5000/" + theme + "?" + urlencode(evidence)
		print url
		response = urlopen(url)
		json_inf = json.loads(response.read())
		print 'JSON inference ' + str(json_inf)

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

@app.route('/net', methods=['GET'])
def net_inferences():
	evidence = request.args.to_dict()
	evidence = dict((k,int(v)) for k,v in evidence.iteritems())
	fn = TableCPDFactorization(bn_net)
	inf = inference(bn_net, evidence)
	return inf

@app.route('/ill', methods=['GET'])
def ill_inferences():
	evidence = request.args.to_dict()
	evidence = dict((k,int(v)) for k,v in evidence.iteritems())
	fn = TableCPDFactorization(bn_ill)
	inf = inference(bn_ill, evidence)
	return inf

@app.route('/ideo', methods=['GET'])
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