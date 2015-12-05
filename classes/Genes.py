class Gene(object):
	innovation = 1
	def __init__(self):
		self.innovation = Gene.innovation
		Gene.innovation += 1

class NodeGene(Gene):
	def __init__(self):
		Gene.__init__(self)
		self.level = None
		self.value = None

class ConnectionGene(Gene):
	def __init__(self):
		Gene.__init__(self)
		self.into     = None
		self.out      = None
		self.weight   = None
		self.disabled = False
