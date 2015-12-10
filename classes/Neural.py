from math import exp
from copy import deepcopy

class Node:
	def __init__(self):
		self.innovation = None
		self.value  = None
		self.inputs = []
		self.type   = None # 0 = input, 1 = output, 2 = hidden

	def evaluate(self):
		#print "Evaluating node #%s!" % self.innovation
		if len(self.inputs) == 0 and self.type != 0:
			self.value = 0
		if self.value is not None:
			#print "Already know the value of node #%s as %s" % (self.innovation, self.value)
			return self.value

		# Loop through connections and call evaluate on them
		self.value = 0
		for inp in self.inputs:
			node = inp['node']
			self.value += node.evaluate()*inp['weight']
		self.value = self.activationFunc(self.value)
		#print "Evaluated node #%s to %s" % (self.innovation, self.value)
		return self.value

	def activationFunc(self, x):
		return 2.0 / (1.0 + exp(-4.9 * x)) - 1 # sigmoid

class Network:
	def __init__(self, levels):
		self.nodes  = {}
		self.levels = levels

	def getInputNodes(self):
		inputs = []
		for node in self.nodes.iteritems():
			if node[1].type == 0: inputs.append(node[0])

		inputs.sort(key=lambda x: self.nodes[x].innovation)
		return inputs

	def getOutputNodes(self):
		outputs = []
		for node in self.nodes.iteritems():
			if node[1].type == 1: outputs.append(node[0])

		outputs.sort(key=lambda x: self.nodes[x].innovation)
		return outputs

	def evaluate(self, inputs):
		input_nodes = self.getInputNodes()
		output_nodes = self.getOutputNodes()

		network = deepcopy(self)

		if len(input_nodes) - 1 != len(inputs): raise Exception("Number of inputs must match number of input nodes!")

		for i, node_id in enumerate(input_nodes[:(len(input_nodes)-1)]):
			network.nodes[node_id].value = inputs[i]

		for node_id in output_nodes:
			network.nodes[node_id].evaluate()

		output = [network.nodes[node_id].value for node_id in output_nodes]
		return output