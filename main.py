from classes import Neat

neat = Neat(5, 5)

for i in range(0,5):
	for organism in neat.population:
		organism.fitness = len(organism.connection_genes) + len(organism.node_genes)
	neat.epoch()
	print len(neat.population)

print neat.history
