#!flask/bin/python
from flask import Flask
from flask import jsonify
from flask import request
from create_struct import *

app = Flask(__name__)

@app.route('/')
def index():
	return "Hello World"

@app.route('/net', methods=['GET'])
def net_inferences():
	evidence = request.args.to_dict()
	evidence = dict((k,int(v)) for k,v in evidence.iteritems())
	bn = create_bayesian_network_structure('net')
	fn = TableCPDFactorization(bn)
	inf = inference(bn, evidence)
	return inf

@app.route('/ill', methods=['GET'])
def ill_inferences():
	evidence = request.args.to_dict()
	evidence = dict((k,int(v)) for k,v in evidence.iteritems())
	bn = create_bayesian_network_structure('ill')
	fn = TableCPDFactorization(bn)
	inf = inference(bn, evidence)
	return inf

@app.route('/ideo', methods=['GET'])
def ideo_inferences():
	evidence = request.args.to_dict()
	evidence = dict((k,int(v)) for k,v in evidence.iteritems())
	bn = create_bayesian_network_structure('ideo')
	fn = TableCPDFactorization(bn)
	inf = inference(bn, evidence)
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
	app.run(debug=True)