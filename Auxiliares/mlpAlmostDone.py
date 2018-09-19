#Feito por: Bruno Abe
from numpy import exp, array, random, dot
import os
import glob
import math
import random

RATE_LEARNING = 0.5 
NUM_MAX_INTERACTIONS = 10
THETA = 0.1
random.seed(1)

#Sigmoide
def activationFunction(x):
	return math.tanh(x)
	#return 1 / (1 + exp(-x))

#Derivada da sigmoide 
def derivativeActivationFunction(x):
	return (1 - x)*x

def rand(x, y):
	return random.random()

def createMatrix(x, y):
	matrix = []

	for i in range(x):
		matrix.append([0]*y)

	return matrix

def startWeights(X, Y, weights, a, b):
	for i in range(X):
		for j in range(Y):
			weights[i][j] = rand(a, b)*0.1

#Valores na primeira camada
def firstLayerValues(X, inputValues, inputs):
	for i in range(X - 1):
		inputValues[i] = inputs[i]

#Valores intermediários
def hiddenLayerValues(X, Y, inputValues, inputWeights, hiddenValues):
	for j in range(X):
		sum = 0
		for i in range(Y):
			sum += inputValues[i] * inputWeights[i][j]
			
		hiddenValues[j] = activationFunction(sum)

#Valores na saída
def outputLayerValues(X, Y, hiddenValues, outputWeights, outputValues):
	for j in range(X):
		sum = 0
		for i in range(Y):
			sum += hiddenValues[i] * outputWeights[i][j]
			
		outputValues[j] = activationFunction(sum)

def momentum(X, Y, delta, activation, weights, N, M, momentumMatrix):
	for i in range(X):
		for j in range(Y):
			change = delta[j]*activation[i]
			weights[i][j] = weights[i][j] + N*change + M*momentumMatrix[i][j]
			momentumMatrix[i][j] = change

def findOutputDelta(outputDelta, X, targets, outputValues):
	for i in range(X):
		erro = (targets[i] - outputValues[i])
		outputDelta[i] = (-1)*derivativeActivationFunction(outputValues[i]) * erro 

def findHiddenDelta(X, Y, outputDelta, outputWeights, hiddenDelta, hiddenValues):
	for i in range(X):
		erro = 0
		for j in range(Y):
			erro = erro + outputDelta[j]*outputWeights[i][j]
		hiddenDelta[i] = (-1)*derivativeActivationFunction(hiddenValues[i])*erro

class MultiLayerPerceptron:
	#Construtor do MultiLayer
	def __init__(self, inputNodesNumber, hidenNodesNumber, outputNodesNumber):
		self.inputNodesNumber = inputNodesNumber + 1 #bias
		self.hidenNodesNumber = hidenNodesNumber
		self.outputNodesNumber = outputNodesNumber
		#Ativação dos nós
		self.inputValues = [1.0]*self.inputNodesNumber
		self.hiddenValues = [1.0]*self.hidenNodesNumber
		self.outputValues = [1.0]*self.outputNodesNumber
		#Matriz de pesos é criada
		self.inputWeights = createMatrix(self.inputNodesNumber, self.hidenNodesNumber)
		self.outputWeights = createMatrix(self.hidenNodesNumber, self.outputNodesNumber)
		#Coloca os pesos aleatórios na matrix
		startWeights(self.inputNodesNumber, self.hidenNodesNumber, self.inputWeights, -1, 1)
		startWeights(self.hidenNodesNumber, self.outputNodesNumber, self.outputWeights, -1, 1)
		#Matrizes para evitar o "zig-zag"
		self.momentumInput = createMatrix(self.inputNodesNumber, self.hidenNodesNumber)
		self.momentumOutput = createMatrix(self.hidenNodesNumber, self.outputNodesNumber)

	#Encontra os pesos para cada uma das camadas e atualiza os pesos(inicial, hidden e output)
	def findValues(self, inputs):
		#Encontra os valores
		firstLayerValues(self.inputNodesNumber, self.inputValues, inputs)
		hiddenLayerValues(self.hidenNodesNumber, self.inputNodesNumber, self.inputValues, self.inputWeights, self.hiddenValues)
		outputLayerValues(self.outputNodesNumber, self.hidenNodesNumber, self.hiddenValues, self.outputWeights, self.outputValues)

		#print("out:", self.outputWeights)
	#	input()
		return self.outputValues[:]

	#Propaga o erro do output para trás (output para hidden)
	def backPropagate(self, targets, N, M):
		outputDelta = [0]*self.outputNodesNumber
		findOutputDelta(outputDelta, self.outputNodesNumber, targets, self.outputValues)

		hiddenDelta = [0]*self.hidenNodesNumber
		findHiddenDelta(self.hidenNodesNumber, self.outputNodesNumber, outputDelta, self.outputWeights, hiddenDelta, self.hiddenValues)

		momentum(self.hidenNodesNumber, self.outputNodesNumber, outputDelta, self.hiddenValues, self.outputWeights, N, M, self.momentumOutput)
		momentum(self.inputNodesNumber, self.hidenNodesNumber, hiddenDelta, self.inputValues, self.inputWeights, N, M, self.momentumInput)

		#print("out2:", self.outputValues[:], outputDelta)

		#Encontra o erro
		erro = 0
		for i in range(len(targets)):
			erro = erro + 0.5*(targets[i] - self.outputValues[i])**2

		return erro

	def testMLP(self, patterns):
		print("Entrada: ", patterns[0])
		print("Resposta esperada: ", patterns[0][1])
		for p in patterns:
			x = self.findValues(p[0])
			y = x[0]
			print("Resposta rede: ", x, 2)

	def trainMLP(self, patterns):
		erro = 1
		while (erro > 0.0001):
			#erro = 0
			for j in patterns:
				inputs = j[0] #Carrega o input
				expectedResult = j[1] #Carrega a resposta esperada

				self.findValues(inputs) #Valores encontrados na rede
				erro = self.backPropagate(expectedResult, 0.5, 0.1) #Propaga o erro
				print(erro)


def andTrain():

	pat = [
	    [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
		[[0, 1, 0, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]],
	    [[0, 0, 1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]],
	    [[0, 0, 0, 1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]],
	    [[0, 0, 0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0, 0, 0, 0]],
	    [[0, 0, 0, 0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]],
	    [[0, 0, 0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 1, 0, 0, 0]],
	    [[0, 0, 0, 0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1, 0, 0]],
	    [[0, 0, 0, 0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 0, 0, 1, 0]],
	    [[0, 0, 0, 0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]]
	]

	pat2 = [
	     [[0, 0, 0, 0, 0, 0, 0, 0, 0, 1],[0, 0, 0, 0, 0, 0, 0, 0, 0, 1]]  
	]

	size = len(pat)
	print(math.ceil((math.log2(size))))
	n = MultiLayerPerceptron(size,math.ceil((math.log2(size))),size)
	n.trainMLP(pat)
	n.testMLP(pat2)

if __name__ == '__main__':
	andTrain()