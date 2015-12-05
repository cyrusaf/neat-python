from copy import deepcopy
from random import randint, choice, getrandbits, sample, random
from math import fabs
from pprint import pprint

from Genes import ConnectionGene, NodeGene

from Neural import Network, Node
class Organism:
	mutate_rate = 0.05
	def __init__(self, levels):
		self.node_genes       = {}
		self.connection_genes = {}
		self.levels  = levels
		self.fitness = None
		self.species = None

	def __str__(self):
		nodes = [str(node.innovation) + ": %s" % node.level for node in self.node_genes.values()]
		nodes = "\n".join(nodes)
		connections = ["%s: %s to %s: %s, %s" % (connection.innovation, connection.out, connection.into, connection.weight, connection.disabled) for connection in self.connection_genes.values()]
		connections = "\n".join(connections)
		return "Nodes:\n" + nodes + "\nConnections:\n" + connections

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
		if N < 20: N = 1.0

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

		return child

	def createNetwork(self):
		network = Network(self.levels)
		# Loop through node genes and create nodes
		for node_gene in self.node_genes.values():
			node = Node()
			node.innovation = node_gene.innovation
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
			network.nodes[connection_gene.into].inputs.append({
				'node': network.nodes[connection_gene.out],
				'weight': connection_gene.weight
			})

		return network

	def mutate(self):
		self.__mutateAddConnection()
		self.__mutateAddNode()
		self.__mutateDisableConnection()
		self.__mutateConnectionWeight()

	def __mutateAddConnection(self):
		for i in self.node_genes:
			for j in self.node_genes:
				if i == j: continue

				# Find node in higher level
				into = None
				out = None
				if self.node_genes[i].level < self.node_genes[j].level:
					out = self.node_genes[i]
					into = self.node_genes[j]
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
				connection.weight = float(randint(-100,100))/10.0
				#print "New connection between %s and %s with weight %s" % (connection.out, connection.into, connection.weight) 
				self.addConnection(connection)

	def __mutateAddNode(self):
		# Loop through connections
		for connection_gene in self.connection_genes.values():
			into = self.node_genes[connection_gene.into]
			out  = self.node_genes[connection_gene.out]

			# If difference in node levels is greater than 1, add node with chance in random level between
			diff = into.level - out.level
			if diff > 1:
				if randint(0,100)+1 > Organism.mutate_rate*100.0: continue
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
	
	def __mutateDisableConnection(self):
		# Loop through all connections and chance disable
		for connection in self.connection_genes.values():
			if randint(0,100)+1 > Organism.mutate_rate*100.0: continue
			connection.disabled = True

	def __mutateConnectionWeight(self):
		# Loop through all connections and chance change weight
		for connection in self.connection_genes.values():
			if randint(0,100)+1 > 0.8*100.0: continue
			connection.weight = float(randint(-10,10))/10.0

class Species:
	def __init__(self):
		self.members     = []
		self.max_fitness = 0
		self.avg_fitness = 0
		self.stalesness  = 0

	def addOrganism(self, organism):
		self.members.append(organism)

	def cullOne(self):
		pass

	def cullHalf(self):
		self.members = self.members[:len(self.members)/2]

	def rankMembers(self):
		self.members.sort(key=lambda x: x.fitness*-1)
		self.max_fitness = self.members[0].fitness
		self.avg_fitness = reduce(lambda x, y: x + y, self.members) / len(self.members)

	def organismBelongs(self, organism):
		if len(self.members) == 0: raise Exception("Cannot check if organism belongs to empty species!")
		rep = self.members[0]
		if rep.distanceFrom(organism) < 3.0:
			return True
		return False

	def breedChild(self):
		

class Neat:
	def __init__(self, n_inputs, n_outputs):
		self.generation  = 0
		self.history     = []
		self.species     = []
		self.global_rank = []

		self.max_fitness = 0

		self.options              = {}
		self.options['pop_size']  = 150
		self.options['n_survive'] = 10
		self.options['levels']    = 10
		Organism.mutate_rate      = 0.1

		# Create initial population
		base = Organism(self.options['levels'])

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

			# Create connections between all inputs and outputs
			for j in range(0, n_inputs):
				connection_gene = ConnectionGene()
				connection_gene.out  = j + 1
				connection_gene.into = node.innovation
				connection_gene.weight = 1.0
				base.addConnection(connection_gene)


		# Mutate base to create population
		for i in range(0, self.options['pop_size']):
			new_organism = deepcopy(base)
			new_organism.mutate()
			#base = new_organism
			self.addOrganism(new_organism)

	def addSpecies(self, species):
		self.species.append(species)

	def addOrganism(self, organism):
		# Loop through all species and compare
		for species in self.species:
			# Continue if organism already belongs to a species
			organism.species is not None: continue
			if species.organismBelongs(organism):
				species.addOrganism(organism)

		# If organism does not belong to a species, create new
		if organism.species is None:
			species = Species()
			species.addOrganism(organism)

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
		self.species = survived

	def removeWeakSpecies(self):
		survived = []
		for species in self.species:
			if species.avg_fitness < self.avg_fitness: continue
			survived.append(species)
		self.species = survived

	def rankGlobally(self):
		# Sum members of species
		population = reduce(lambda x, y: x.members + y.members, species)
		# Get avg_fitness
		self.avg_fitness = reduce(lambda x, y: x + y, population) / len(population)
		# Rank members
		population.sort(lambda x: -1*x.fitness)
		# Get max fitness
		self.max_fitness = population[0]

	def epoch(self):

		# Sum fitness for average and check that fitness values have been assigned
		sum_fitness = 0
		for species in self.species:
			for organism is species.members:
				if organism.fitness is None: raise Exception("Cannot epoch when some fitness values have not been assigned!")
				sum_fitness += organism.fitness
				# Update max fitness
				if organism.fitness > self.max_fitness: self.max_fitness = organism.fitness
			species.cullHalf()
			species.removeStaleSpecies()
			species.rankGlobally()
			species.removeWeakSpecies()
			species.rankGlobally()



		# Get top n organisms then recombine and mutate to create new population
		self.population.sort(key=lambda x: -1*x.fitness/self.numInSpecies(x)) #(x.fitness*-1, 1/(len(x.connection_genes) + len(x.node_genes)))
		print [organism.fitness/self.numInSpecies(organism) for organism in self.population]
		print "Total Species: %s" % len(self.species)
		top = self.population[:self.options['n_survive']]

		history = {}
		history['max_fitness'] = top[0].fitness
		history['avg_fitness'] = sum_fitness/len(self.population)
		history['best_performers'] = top
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

		self.determineSpecies()