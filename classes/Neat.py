from copy import deepcopy
from random import randint, choice, getrandbits
from pprint import pprint
import pygame

from Genes import ConnectionGene
from Genes import NodeGene

class Organism:
	mutate_rate = 0.05
	def __init__(self):
		self.node_genes       = {}
		self.connection_genes = {}

		self.fitness = None

	def __str__(self):
		nodes = [str(node.innovation) for node in self.node_genes.values()]
		nodes = ", ".join(nodes)
		connections = ["%s: %s to %s, %s" % (connection.innovation, connection.into, connection.out, connection.disabled) for connection in self.connection_genes.values()]
		connections = "\n".join(connections)
		return "Nodes:\n" + nodes + "\nConnections:\n" + connections

	def addNode(self, node):
		self.node_genes[node.innovation] = deepcopy(node)

	def addConnection(self, connection):
		self.connection_genes[connection.innovation] = deepcopy(connection)

	def mateWith(self, mate):
		fit_parent   = None
		unfit_parent = None
		if self.fitness > mate.fitness:
			fit_parent   = self
			unfit_parent = mate
		else:
			fit_parent   = mate
			unfit_parent = self

		fit_nodes   = fit_parent.node_genes.keys()
		unfit_nodes = unfit_parent.node_genes.keys()
		fit_connections   = fit_parent.connection_genes.keys()
		unfit_connections = unfit_parent.connection_genes.keys()

		disjoint_nodes       = set(fit_nodes) - set(unfit_nodes)
		disjoint_connections = set(fit_connections) - set(unfit_connections)

		common_nodes       = set(fit_nodes).intersection(unfit_nodes)
		common_connections = set(fit_connections).intersection(unfit_connections)

		# Create child
		child = Organism()

		# Loop through disjoint_nodes and add to child from fit parent
		for node_id in disjoint_nodes:
			child.addNode(fit_parent.node_genes[node_id])
		# Loop through disjoint_connections and add to child from fit parent
		for connection_id in disjoint_connections:
			child.addConnection(fit_parent.connection_genes[connection_id])

		# Randomly inherit common nodes
		for node_id in common_nodes:
			if bool(getrandbits(1)):
				child.addNode(fit_parent.node_genes[node_id])
			else:
				child.addNode(unfit_parent.node_genes[node_id])

		# Randomly inherit common connections
		for connection_id in common_connections:
			connection = None
			if bool(getrandbits(1)):
				connection = fit_parent.connection_genes[connection_id]
			else:
				connection = unfit_parent.connection_genes[connection_id]

			if connection.disabled:
				if randint(0,3) == 3:
					connection.disabled = False

			child.addConnection(connection)

		return child

	def mutate(self):
		self.__mutateAddConnection()
		self.__mutateAddNode()
		# Add __mutateDisableConnection?

	def __mutateAddConnection(self):
		for i in self.node_genes:
			for j in self.node_genes:
				if i == j: continue

				# Find node in higher level
				into = None
				out = None
				if self.node_genes[i].level < self.node_genes[j].level:
					into = self.node_genes[i]
					out = self.node_genes[j]
				else:
					continue
				
				# Check if connection between into and out already exists
				flag = False
				for connection in self.connection_genes.values():
					if connection.into == into.innovation and connection.out == out.innovation:
						flag = True
						break

				if flag: continue
				if randint(0,100)+1 > Organism.mutate_rate*100.0: continue

				connection = ConnectionGene()
				connection.into   = into.innovation
				connection.out    = out.innovation
				connection.weight = float(randint(-10,10))/10.0
				print "New connection between %s and %s with weight %s" % (connection.into, connection.out, connection.weight) 
				self.addConnection(connection)

	def __mutateAddNode(self):
		# Loop through connections
		for connection_gene in self.connection_genes.values():
			into = self.node_genes[connection_gene.into]
			out  = self.node_genes[connection_gene.out]

			# If difference in node levels is greater than 1, add node with chance in random level between
			diff = out.level - into.level
			if diff > 1:
				if randint(0,100)+1 > Organism.mutate_rate*100.0: continue
				print "New node between %s and %s" % (connection_gene.into, connection_gene.out)
				# Disable connection gene
				connection_gene.disabled = True
				# Create new node in random level between
				new_node       = NodeGene()
				new_node.level = randint(into.level, out.level)
				self.node_genes[new_node.innovation] = new_node

				# Create two new connection genes connecting new node
				new_conn1        = ConnectionGene()
				new_conn1.into   = into.innovation
				new_conn1.out    = new_node.innovation
				new_conn1.weight = 1.0
				new_conn2        = ConnectionGene()
				new_conn2.into   = new_node.innovation
				new_conn2.out    = out.innovation
				new_conn2.weight = connection_gene.weight
				self.addConnection(new_conn1)
				self.addConnection(new_conn2)
	
	def __mutateDisableConnection(self):
		# If last connection into node is disabled, removed node.
		pass

class Neat:
	def __init__(self, n_inputs, n_outputs):
		self.generation = 0
		self.population = []
		self.history    = []

		self.options              = {}
		self.options['pop_size']  = 20
		self.options['n_survive'] = 5
		self.options['levels']    = 10
		Organism.mutate_rate      = 0.05

		# Create initial population
		base = Organism()

		# Create input NodeGenes
		for i in range(0, n_inputs):
			node = NodeGene()
			node.level = 1
			base.addNode(node)

		# Create output NodeGenes
		for i in range(0, n_outputs):
			node = NodeGene()
			node.level = self.options['levels']
			base.addNode(node)

		# Mutate base to create population
		for i in range(0, self.options['pop_size']):
			print i
			new_organism = deepcopy(base)
			new_organism.mutate()
			#base = new_organism
			self.population.append(new_organism)

	def epoch(self):
		sum_fitness = 0
		for organism in self.population:
			if organism.fitness is None: raise Exception("Cannot epoch when some fitness values have not been assigned!")
			sum_fitness += organism.fitness

		# Get top n organisms then recombine and mutate to create new population
		self.population.sort(key=lambda x: x.fitness*-1)
		top = self.population[:self.options['n_survive']]

		history = {}
		history['max_fitness'] = top[0].fitness
		history['avg_fitness'] = sum_fitness/len(self.population)
		self.history.append(history)

		# Reset population
		self.population = deepcopy(top)
		for organism in self.population:
			organism.fitness = None
		
		# For pop_size - n_survive times, mate two random top and then mutate child
		for i in range(0, self.options['pop_size']-self.options['n_survive']):
			# Select two random organisms from top
			temp_top = deepcopy(top)
			org1 = choice(temp_top)
			temp_top.remove(org1)
			org2 = choice(temp_top)
			if org1 == org2: raise Exception("Attempting to mate an organism with itself!")

			# Mate two organisms
			child = org1.mateWith(org2)
			child.mutate()
			self.population.append(child)