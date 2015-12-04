from copy import deepcopy
from random import randint
from pprint import pprint
import pygame

from Genes import ConnectionGene
from Genes import NodeGene

class Organism:
	mutate_rate = 0.05
	def __init__(self):
		self.node_genes       = {}
		self.connection_genes = []

		self.fitness = None

	def addNode(self, node):
		self.node_genes[node.innovation] = node

	def addConnection(self, connection):
		self.connection_genes.append(connection)

	def mutate(self):
		self.__mutateAddConnection()
		self.__mutateAddNode()
		# Add __mutateDisableConnection?

	def __mutateAddConnection(self):
		for i in range(1,len(self.node_genes)):
			for j in range(i+1,len(self.node_genes)):

				# Find node in higher level
				into = None
				out = None
				if self.node_genes[i].level < self.node_genes[j].level:
					into = self.node_genes[i]
					out = self.node_genes[j]
				elif self.node_genes[i].level == self.node_genes[j].level:
					continue
				else:
					into = self.node_genes[j]
					out = self.node_genes[i]
				
				# Check if connection between into and out already exists
				flag = False
				for connection in self.connection_genes:
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
				self.connection_genes.append(connection)

	def __mutateAddNode(self):
		# Loop through connections
		for connection_gene in self.connection_genes:
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
				self.connection_genes.append(new_conn1)
				self.connection_genes.append(new_conn2)
	
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
			sum_fitness += organism.fitness
			if organism.fitness is None: raise Exception("Cannot epoch when some fitness values have not been assigned!")

		# Get top n organisms then recombine and mutate to create new population
		self.population.sort(key=lambda x: x.fitness*-1)
		top = self.population[:self.options['n_survive']]

		history = {}
		history['max_fitness'] = top[0].fitness
		history['avg_fitness'] = sum_fitness/len(self.population)
		self.history.append(history)

		# Reset population
		self.population = top

		# For pop_size - n_survive times, mate two random top and then mutate child
		# TODO
		