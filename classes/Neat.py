from copy import deepcopy
from random import randint, choice, getrandbits, sample, random
from math import fabs, floor, ceil, log
from pprint import pprint
from time import sleep
from Genes import ConnectionGene, NodeGene
import pickle

from Neural import Network, Node
class Organism:

	organism_id = 0
	mutate_connection_rate = 0.3
	mutate_node_rate = 0.03
	def __init__(self, levels):
		self.node_genes       = {}
		self.connection_genes = {}
		self.levels  = levels
		self.fitness = None
		self.fitness_adj = None
		self.max_fitness = 0
		self.species = None

		self.id = Organism.organism_id
		Organism.organism_id += 1

	def __str__(self):
		nodes = [str(node.innovation) + ": %s" % node.level for node in self.node_genes.values()]
		nodes = "\n".join(nodes)
		connections = ["%s: %s to %s: %s, %s" % (connection.innovation, connection.out, connection.into, connection.weight, connection.disabled) for connection in self.connection_genes.values()]
		connections = "\n".join(connections)
		return "Nodes:\n" + nodes + "\nConnections:\n" + connections

	def save(self, file):
		with open(file,"a+") as f:
			pickle.dump(self, f)

	@classmethod
	def load(self, file):
		with open(file,"r") as f:
			return pickle.load(f)

	def setFitness(self, value):
		self.fitness = value
		if value > self.max_fitness: self.max_fitness = value

	def addNode(self, node):
		self.node_genes[node.innovation] = deepcopy(node)

	def addConnection(self, connection):
		self.connection_genes[connection.innovation] = deepcopy(connection)

	def totalGenes(self):
		return len(self.node_genes) + len(self.connection_genes)

	def distanceFrom(self, org2):
		c1 = 1.0
		c3 = 0.4
		N  = float(max([self.totalGenes(), org2.totalGenes()]))
		N = 1

		shared_connections = set(self.connection_genes.keys()).intersection(set(org2.connection_genes.keys()))
		W = 0.0
		for conn_id in shared_connections:
			diff = fabs(self.connection_genes[conn_id].weight - org2.connection_genes[conn_id].weight)
			W += diff
		W = W / len(shared_connections)


		self_keys = set(self.node_genes.keys() + self.connection_genes.keys())
		org2_keys = set(org2.node_genes.keys() + org2.connection_genes.keys())

		c = self_keys.union(org2_keys)
		d = self_keys.intersection(org2_keys)
		ED = len((c - d))

		return c1*ED/N + c3*W

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
		child = Organism(self.levels)

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
		return deepcopy(child)

	def createNetwork(self):
		network = Network(self.levels)
		# Loop through node genes and create nodes
		for node_gene in self.node_genes.values():
			node = Node()
			node.innovation = node_gene.innovation
			if node_gene.value is not None:
				node.value = node_gene.value
			# Assign node type
			if node_gene.level == 1:
				node.type = 0
			elif node_gene.level == self.levels:
				node.type = 1
			else:
				node.type = 2

			network.nodes[node_gene.innovation] = node

		# Loop through connection genes and populate node inputs
		for connection_gene in self.connection_genes.values():
			if connection_gene.disabled: continue
			network.nodes[connection_gene.into].inputs.append({
				'node': network.nodes[connection_gene.out],
				'weight': connection_gene.weight
			})

		return network

	def mutate(self):
		if randint(0,100)+1 < Organism.mutate_connection_rate*100.0:
			self.__mutateAddConnection()

		if randint(0,100)+1 < Organism.mutate_node_rate*100.0 and len(self.connection_genes) > 0:
			self.__mutateAddNode()

		#if randint(0,100)+1 < Organism.mutate_rate*100.0:
		#	self.__mutateDisableConnection()

		if randint(0,100)+1 < 0.8*100.0:
			self.__mutateConnectionWeight()

	def __mutateAddConnection(self):
		if len(self.node_genes) < 2: raise Exception("Cannot mutateAddConnection if there are less than 2 node_genes")

		out  = choice(self.node_genes.values())
		into = choice(self.node_genes.values())

		while out == into:
			out = choice(self.node_genes.values())
			into = choice(self.node_genes.values())

		# Find node in higher level
		if out.level < into.level:
			temp = out
			into = out
			out = into
		
		# Check if connection between into and out already exists
		flag = False
		for connection in self.connection_genes.values():
			if connection.into == into.innovation and connection.out == out.innovation:
				flag = True
				break

		if flag:
			self.__mutateAddConnection()
			return

		connection = ConnectionGene()
		connection.into   = into.innovation
		connection.out    = out.innovation
		connection.weight = float(randint(-40,40))/10.0
		#print "New connection between %s and %s with weight %s" % (connection.out, connection.into, connection.weight) 
		self.addConnection(connection)

	def __mutateAddNode(self):
		# Loop through connections
		connection_gene = choice(self.connection_genes.values())
		into = self.node_genes[connection_gene.into]
		out  = self.node_genes[connection_gene.out]

		# If difference in node levels is greater than 1, add node with chance in random level between
		diff = into.level - out.level
		if diff > 1:
			#print "New node between %s and %s" % (connection_gene.out, connection_gene.into)
			# Disable connection gene
			connection_gene.disabled = True
			# Create new node in random level between
			new_node       = NodeGene()
			new_node.level = randint(out.level+1, into.level-1)
			self.addNode(new_node)

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
		else:
			self.__mutateAddNode()
	
	def __mutateDisableConnection(self):
		# Loop through all connections and chance disable
		connection = choice(self.connection_genes.values())
		connection.disabled = True

	def __mutateConnectionWeight(self):
		# Loop through all connections and chance change weight
		for connection in self.connection_genes.values():
			if random() < 0.1:
				connection.weight = float(randint(-40,40))/10.0
			else:
				connection.weight += choice([-1,1]) * 0.1

class Species:

	species_id = 0
	crossover_chance = 0.75
	def __init__(self):
		self.members     = []
		self.max_fitness = 0
		self.avg_fitness = 0
		self.staleness  = 0

		self.id = Species.species_id
		Species.species_id += 1

	def getTopFitness(self):
		self.members.sort(key=lambda x: x.fitness*-1)
		return self.members[0].fitness

	def assignAdjFitness(self):
		for member in self.members:
			if member.fitness is None: raise Exception("Cannot assign adjusted fitness if fitness is not set!")
			member.fitness_adj = member.fitness/len(self.members)

	def addOrganism(self, organism):
		self.members.append(organism)
		organism.species = self.id

	def cull(self, n):
		self.rankMembers()
		top = self.members[:n]
		self.members = top

	def cullOne(self):
		top = self.members[0]
		top.fitness = None
		self.members = [top]

	def cullHalf(self):
		if len(self.members)/2 < 1: return
		self.members = self.members[:len(self.members)/2]

	def updateMaxFitness(self):
		self.staleness += 1
		for organism in self.members:
			if organism.max_fitness >= self.max_fitness:
				self.max_fitness = organism.max_fitness
				self.staleness = 0

	def rankMembers(self):
		self.members.sort(key=lambda x: x.max_fitness*-1)
		self.avg_fitness = reduce(lambda x, y: x + y.fitness, self.members, 0.0) / len(self.members)

	def organismBelongs(self, organism):
		if len(self.members) == 0: raise Exception("Cannot check if organism belongs to empty species!")
		rep = self.members[0]
		if rep.distanceFrom(organism) < 3.0:
			return True
		return False

	def breedChild(self):
		child = None
		if random() < Species.crossover_chance and len(self.members) > 1:
			if len(self.members) < 2: raise Exception("Not enough members of species to breed!")
			# Pick random parents from species
			temp_members = deepcopy(self.members)
			parent1 = choice(temp_members)
			#temp_members.remove(parent1)
			parent2 = choice(temp_members)
			child = parent1.mateWith(parent2)
			child.mutate()
		else:
			# Pick random parent from species
			parent = choice(self.members)
			child = Organism(parent.levels)
			child.node_genes = deepcopy(parent.node_genes)
			child.connection_genes = deepcopy(parent.connection_genes)
			child.mutate()

		return child

class Neat:
	def __init__(self):
		self.generation  = 0
		self.history     = []
		self.species     = []
		self.global_rank = []

		self.max_fitness = 0
		self.avg_fitness = 0

		self.options              = {}
		self.options['pop_size']  = 100
		self.options['levels']    = 10
		self.options['members_passed']    = 4
		self.options['species_passed']    = 7

		Species.crossover_chance  = 0.01

	def init(self, n_inputs, n_outputs):
		# Create initial population
		base = Organism(self.options['levels'])

		# Create input NodeGenes
		for i in range(0, n_inputs):
			node = NodeGene()
			node.level = 1
			base.addNode(node)

		# Create bias node
		node = NodeGene()
		node.level = 1
		node.value = 2.0
		base.addNode(node)

		# Create output NodeGenes
		for i in range(0, n_outputs):
			node = NodeGene()
			node.level = self.options['levels']
			base.addNode(node)

			
			# Create connections between all inputs and outputs
			for j in range(0, n_inputs):
				connection_gene = ConnectionGene()
				connection_gene.out  = j + 1
				connection_gene.into = node.innovation
				connection_gene.weight = 0.0
				base.addConnection(connection_gene)
			


		# Mutate base to create population
		for i in range(0, self.options['pop_size']):
			new_organism = deepcopy(base)
			new_organism.mutate()
			new_organism.id = Organism.organism_id
			Organism.organism_id += 1
			#base = new_organism
			self.addOrganism(new_organism)

	def addSpecies(self, species):
		self.species.append(species)

	def addOrganism(self, organism):
		# Loop through all species and compare
		for species in self.species:
			# Continue if organism already belongs to a species
			if organism.species is not None: continue
			if species.organismBelongs(organism):
				species.addOrganism(organism)

		# If organism does not belong to a species, create new
		if organism.species is None:
			species = Species()
			species.addOrganism(organism)
			self.addSpecies(species)

	def removeStaleSpecies(self):
		survived = []
		for species in self.species:
			# Rank members of species
			species.rankMembers()

			# Check if species fitness has improved, reset staleness if so, increment staleness if not
			if species.members[0].fitness > species.max_fitness:
				species.stalesness = 0
				species.max_fitness = species.members[0].fitness
			else:
				species.stalesness += 1

			# Check if species is stale or the best species
			if species.stalesness > 15 and species.max_fitness < self.max_fitness: continue

			# If species is ok, add to survived
			survived.append(species)

		# Replaced species with survived
		if len(survived) > 1: self.species = survived

	def removeWeakSpecies(self):
		survived = []
		for species in self.species:
			if species.avg_fitness < self.avg_fitness: continue
			survived.append(species)
		if len(survived) > 1: self.species = survived

	def getPopulation(self):
		population = []
		for species in self.species:
			for member in species.members:
				population.append(member)
		return population

	def epoch(self):
		
		# Make sure each organism has been assigned a fitness
		for species in self.species:
			for organism in species.members:
				if organism.fitness is None: raise Exception("Cannot epoch when some fitness values have not been assigned!")
				
				# Update max fitness for neat
				if organism.fitness > self.max_fitness: self.max_fitness = organism.fitness

			# Update each species fitness
			species.updateMaxFitness()

			#species.assignAdjFitness()



		# Sort species by their max_fitness **used to be by best current organism
		self.species.sort(key=lambda x: x.max_fitness*-1/log(x.staleness+15,15)) # used to be x.getBestOrganism()*-1

		# Created passed, a list of the top species
		passed = deepcopy(self.species[:self.options['species_passed']])
		for species in passed:
			species.cull(self.options['members_passed'])

		self.avg_fitness = reduce(lambda x, y: x + y.fitness, self.getPopulation(), 0.0)/len(self.getPopulation())
		self.max_fitness = passed[0].members[0].fitness

		print "\n===== Summary of Generation #%s =====" % self.generation
		print "Number of Species: %s" % len(self.species)	
		print "Overall Max Fitness: %s" % passed[0].max_fitness
		print "Max Fitness: %s" % self.max_fitness
		print "Avg Fitness: %s" % self.avg_fitness
		print "Passing Following Species:"
		for species in passed:
			print "Species %s: Fitness: %s, Staleness: %s" % (species.id, species.max_fitness, species.staleness)
			for member in species.members:
				print "     Member %s: Max Fitness: %s, Current Fitness: %s" % (member.id, member.max_fitness, member.fitness)
		print "======================================\n\n"

		'''
		# Clear fitness values of passed
		for species in passed:
			for organism in species.members:
				organism.fitness = None
		'''

		if self.generation % 25 == 0:
			passed[0].members[0].save("generation_%s.org" % self.generation)

		self.generation += 1


		children = []
		while len(passed)*self.options['members_passed'] + len(children) < self.options['pop_size']:
			# Same species breed or interspecies breed?
			child = None
			if len(passed) == 1:
				species = passed[0]
				child = species.breedChild()
			else:
				if random() < 0.1:
					temp_passed = deepcopy([species.members[0] for species in passed])
					parent1 = choice(temp_passed)
					temp_passed.remove(parent1)
					parent2 = choice(temp_passed)
					print "%s, %s" % (parent1.species, parent2.species)
					child = parent1.mateWith(parent2)
					child.mutate()
				else:
					species = choice(passed)
					child = species.breedChild()
			
			children.append(child)

		

		self.species = deepcopy(passed)
		for organism in children:
			self.addOrganism(organism)

		history = {}
		history['max_fitness'] = self.max_fitness
		history['avg_fitness'] = self.avg_fitness
		self.history.append(history)

	def epochOld(self):
		# Check that fitness values have been assigned
		for species in self.species:
			for organism in species.members:
				if organism.fitness is None: raise Exception("Cannot epoch when some fitness values have not been assigned!")
				# Update max fitness
				if organism.fitness > self.max_fitness: self.max_fitness = organism.fitness

		if len(self.species) > 1:
			self.removeStaleSpecies()
			self.rankGlobally()
			self.removeWeakSpecies()
		self.rankGlobally()

		children = []
		passed = filter(lambda x: x, [len(species.members) > 0 for species in self.species])
		if len(passed) > self.options['pop_size']*0.1:
			self.species.sort(key=lambda x: x.max_fitness*-1)
			self.species = self.species[:int(self.options['pop_size']*0.1)]
		print "passed: %s" % len(passed)
		# Generate children from species
		for species in self.species:
			#print "Species avg_fitness: %s, Population avg_fitness: %s" % (species.avg_fitness, self.avg_fitness)
			if self.avg_fitness == 0: self.avg_fitness = 0.0001
			breed = int(ceil(species.max_fitness / self.avg_fitness))**2
			for i in range(0, breed):
				children.append(species.breedChild())
				if len(children) + len(passed) >= self.options['pop_size']: break
			if len(children) + len(passed) >= self.options['pop_size']: break


		while len(children) + len(self.species) < self.options['pop_size']:
			if len(self.species) == 0: raise Exception("Species are empty!")
			species = choice(self.species)
			children.append(species.breedChild())

		

		for species in self.species:
			species.cullOne()

		for child in children:
			self.addOrganism(child)

		history = {}
		history['max_fitness'] = self.max_fitness
		history['avg_fitness'] = self.avg_fitness
		self.history.append(history)

		