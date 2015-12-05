from classes import Neat, Gene, Network
from math import fabs
from time import sleep

neat = Neat(2, 1)

for i in range(0, 300):
	print i
	for organism in neat.population:
		network = organism.createNetwork()

		'''
		result = network.evaluate([1])
		error = fabs(1.0 - result[0])

		if error < 0.00001:
			#raise Exception("GENERATION %s" % i)
			organism.fitness = 100000
			continue

		organism.fitness = 1.0 / error

		'''
		error = 0.0

		result = network.evaluate([0, 0])
		error += fabs(0.0 - result[0])

		result = network.evaluate([0, 1])
		error += fabs(1.0 - result[0])

		result = network.evaluate([1, 0])
		error += fabs(1.0 - result[0])

		result = network.evaluate([1, 1])
		error += fabs(0.0 - result[0])

		if error == 0:
			organism.fitness = 100000
			continue

		organism.fitness =  1.0 / error

		#print len(organism.node_genes) + len(organism.connection_genes)
	neat.epoch()
	print " "
	print " "
	print " "

i = 0
for history in neat.history:
	print i
	print history['max_fitness']
	print " "
	i += 1


print Gene.innovation

print neat.species