#!flask/bin/python
from flask import Flask
from flask import jsonify
from flask import request
from flask import render_template
from flask import url_for
from create_struct import *

import sys
sys.path.insert(0, 'utils/')

from explanatory_data import *

app = Flask(__name__)

@app.route('/')
def index():
	return "Hello World"
	
@app.route('/bayes/net')
def bayes_net():
	nodes = bn_net.V
	categs = dict()
	for n in nodes:
		if n in explanatory_data:
			categs[n] = explanatory_data[n]
		else:
			categs[n] = {0:"no", 1:"yes"}
	print categs		
	return render_template('net.html', categs=categs)	

@app.route('/bayes/ill')
def bayes_ill():
	nodes = bn_ill.V
	categs = dict()
	for n in nodes:
		if n in explanatory_data:
			categs[n] = explanatory_data[n]
		else:
			categs[n] = {0:"no", 1:"yes"}
	print categs		
	return render_template('net.html', categs=categs)	

@app.route('/bayes/ideo')
def bayes_ideo():
	nodes = bn_ideo.V
	categs = dict()
	for n in nodes:
		if n in explanatory_data:
			categs[n] = explanatory_data[n]
		else:
			categs[n] = {0:"no", 1:"yes"}
	print categs		
	return render_template('net.html', categs=categs)

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