import json
import sys
import os.path

# from libpgm.nodedata import NodeData
from noded import NodeData
# from libpgm.graphskeleton import GraphSkeleton
from graphskel import GraphSkeleton
# from libpgm.discretebayesiannetwork import DiscreteBayesianNetwork
from dbn import DiscreteBayesianNetwork
from libpgm.tablecpdfactorization import TableCPDFactorization
from libpgm.sampleaggregator import SampleAggregator
# from libpgm.pgmlearner import PGMLearner
from pgml import PGMLearner
from learning_data_ideo import data_ideo
from learning_data_ill import data_ill
from learning_data_net import data_net

GIBBS_ITERATIONS = 300

def save_network(file_name, network, theme):
  open(file_name, "w").close() # erase prev contents
  myfile = open(file_name, "w")

  # create json file - libpgm works with files
  myfile.write("{\n")
  myfile.write("\"V\": ")
  json.dump(network.V, myfile)
  myfile.write(",\n\"E\": ")
  json.dump(network.E, myfile)
  myfile.write(",\n\"Vdata\": ")
  json.dump(network.Vdata, myfile)
  myfile.write("\n}")
  myfile.close()

def select_theme_data(theme):
  if theme == 'net':
    return data_net
  elif theme == 'ill':
    return data_ill
  elif theme == 'ideo':
    return data_ideo
  else:
    print 'ERROR! Unknown theme.'      

def create_bayesian_network_structure(theme):
  file_name = "bayes/bayes" + str(theme) + ".txt"  
  data = select_theme_data(theme)

  if not os.path.exists(file_name):
    learner = PGMLearner()
    result = learner.discrete_estimatebn(data, pvalparam=0.05, indegree=1)

    assert len(data[0]) == len(result.V)  # check number of nodes
    print 'Nr of nodes ' + str(len(result.V))
    print 'Nr of edges ' + str(len(result.E))
    print 'Saved network'

    save_network(file_name, result, theme)


  nd = NodeData()
  nd.load(file_name)
  skel = GraphSkeleton()
  skel.load(file_name)
  skel.toporder()
  bn = DiscreteBayesianNetwork(skel, nd)

  return bn

def update_bayesian_network_structure(theme, nodes, edges, cpts):
  data = select_theme_data(theme)
  skel = GraphSkeleton()
  skel.load_skel(nodes, edges)
  skel.toporder()

  learner = PGMLearner()
  bn, passed = learner.discrete_mle_estimateparams(skel, data)

  return bn.Vdata, passed

def inference(bn, evidence):
  fn = TableCPDFactorization(bn)
  result = fn.gibbssample(evidence, GIBBS_ITERATIONS)
  agg = SampleAggregator()
  result = agg.aggregate(result)
  return json.dumps(result, indent=2)


if __name__ == "__main__":
  theme = raw_input("Enter theme.\n")
  bn = create_bayesian_network_structure(theme)
  # evidence = json.loads(raw_input("Evidence.\n"))
  # evidence = dict(InteractNet=0)
  # query = dict(HighValueCivilian=[1])
  # inference(bn, evidence)
