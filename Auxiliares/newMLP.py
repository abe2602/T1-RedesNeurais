import numpy as np 
import math

class MLP():
	"""docstring for MLP"""
	def __init__(self, n_nodeInput, n_nodeHidden, n_nodeOutput):
		self.n_nodeInput = n_nodeInput
		self.n_nodeHidden = n_nodeHidden
		self.n_nodeOutput = n_nodeOutput

		self.wh = np.random.rand(self.n_nodeInput, self.n_nodeHidden)
		#self.bh = 
		self.wout =  np.random.rand(self.n_nodeHidden, self.n_nodeOutput)
		#self.bout = 

	def sigmoide(self, x):
		return 1 / (1 + np.exp(-x))	

	def dsigmoide(self, f_net):
		return (f_net * (np.array(1)-f_net))

	def train(self, x, learnigRate = 0.5, threshold = 1e-3):
		gradError = 1

		while(gradError > threshold):
			for p in x:
				expected = np.matrix(p[1])
				inputs = np.matrix(p[0])

				result = np.dot(inputs, self.wh)
				
				hiddenlayer_activations = self.sigmoide(result)
				
				outputlayer_input = np.dot(hiddenlayer_activations, self.wout)
				output = self.sigmoide(outputlayer_input)

				gradError = 0.5*((expected - output)**2)
				error = (expected - output)

				slope_output_layer = self.dsigmoide(output)
				slope_hidden_layer = self.dsigmoide(np.asarray(hiddenlayer_activations))

				d_out = error*slope_output_layer
				error_hiddenLayer = np.dot(d_out, np.transpose(self.wout))

				d_hiddenLayer = np.asarray(error_hiddenLayer)*np.asarray(slope_hidden_layer)
				self.wout += np.dot(np.transpose(hiddenlayer_activations), d_out)*learnigRate
				self.wh += np.dot(np.transpose(inputs), d_hiddenLayer)*learnigRate
		#print(output)

	def backprop(self, dataset):
		self.train(dataset)

	def test(self):
		pat = [[1, 1], [0]]
		self.train(pat)

if __name__ == '__main__':

	x = [
		[[1, 0], [1]],
		[[0, 0], [0]],
		[[0, 1], [1]]
	]

	#print(np.asarray(x[0]).shape[1])

	nn = MLP(2, 3, 1)
	nn.backprop(x)
	nn.test()