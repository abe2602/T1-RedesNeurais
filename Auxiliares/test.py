# Criado por: Bruno Abe
# Baseado no algoritmo do livro: programming collective intelligence 
from numpy import exp, array, random, dot
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def derivativeSigmoid(x):
    return (1 - x)*x

#Recebe como parâmetros a quantidade de neuronios no input, meio e fim
class MPL:
    def __init__(self, numInput, numHidden, numOutput):
    	self.input = numInput
    	self.hidden = numHidden
    	self.output = numOutput
    	#Valores que ficarão nos neurônios
    	self.valuesInput = [1]*self.input
    	self.valuesHidden = [1]*self.hidden
    	self.valuesOutput = [1]*self.output
    	#Pesos
    	self.weightHidden = np.random.randn(self.input, self.hidden)
    	self.weightOutput = np.random.randn(self.hidden, self.output)
    	#Mudanças nas matrizes
    	self.changeInput = np.zeros((self.input, self.hidden))
    	self.changeOutput = np.zeros((self.hidden, self.output))

######################Funções para o Feedforward#############################
    #A alimentação da entrada é: copia-se todo o vetor
    def feedInput(self, inputs):
    	for i in range(self.input):
    		self.valuesInput[i] = inputs[i]
   
   #Aplica o somatório para criar os novos valores da rede (multiplica o valor pelo peso)
   #com o resultado do somatório, aplicar a sigmoide (função de ativação)
    def feedHidden(self):
    	for i in range(self.hidden):
    		sum = 0.0
    		for j in range(self.input):
    			sum += self.valuesInput[j] * self.weightHidden[j][i]

    		self.valuesHidden[i] = sigmoid(sum)

    #Aplica o somatório para criar os novos valores da rede (multiplica o valor pelo peso)
    def feedOutput(self):
    	for i in range(self.output):
    		sum = 0
    		for j in range(self.hidden):
    			sum+= self.valuesHidden[j] * self.weightOutput[j][i]
    		self.valuesOutput[i] = sigmoid(sum)
   	
   	#Realiza a alimentação da rede usando as funções auxiliares
    def feedforward(self, inputs):
    	self.feedInput(inputs)
    	self.feedHidden()
    	self.feedOutput()
    	return self.valuesOutput[:]
######################Fim do Feedforward#############################

######################Funções para o Backpropagate#############################
#As propagações de erro são diferentes nas diferentes camadas, então, separei-as
#em funções diferentes

#Propagação do erro para a primeira camada
    def backPropagateFirstLayer(self, output_deltas, expected):
    	for i in range(self.output):
    		error = -(expected[i] - self.valuesOutput[i])
    		output_deltas[i] = derivativeSigmoid(self.valuesOutput[i])*error

#Propagação do erro para a segunda camada
#Calculamos o delta (gradiente) de onde devemos ir, isso para a camada interna e de output
    def backPropagateHiddenLayer(self, hidden_deltas, output_deltas):
    	for j in range(self.hidden):
    		error = 0
    		for i in range(self.output):
    			error += output_deltas[i]*self.weightOutput[j][i]
    		hidden_deltas[j] = derivativeSigmoid(self.valuesHidden[j])*error

    def backPropagateChangeOutput(self, output_deltas, N):
    	for j in range(self.hidden):
    		for k in range(self.output):
    			change = output_deltas[k] * self.valuesHidden[j]
    			self.weightOutput[j][k] -= N*change + self.changeOutput[j][k]
    			self.changeOutput[j][k] = change

#Realiza as mudanças nos pesos e nos inputs
    def backPropagateChangeHidden(self, hidden_deltas, N):
    	for i in range(self.input):
    		for j in range(self.hidden):
    			change = hidden_deltas[j] * self.valuesInput[i]
    			self.weightHidden[i][j] -= N*change + self.changeInput[i][j]
    			self.changeInput[i][j] = change

    def findError(self, expected):
    	error = 0
    	for i in range(len(expected)):
    		error += 0.5 * (expected[i] - self.valuesOutput[i]) ** 2
    	return error	

    #Realiza o backpropagation
    def backPropagate(self, expected, N):
    	output_deltas = [0]*self.output
    	hidden_deltas = [0]*self.hidden

    	self.backPropagateFirstLayer(output_deltas, expected)
    	self.backPropagateHiddenLayer(hidden_deltas, output_deltas)
    	self.backPropagateChangeOutput(output_deltas, N)
    	self.backPropagateChangeHidden(hidden_deltas, N)
    	error = self.findError(expected)
    	
    	return error

    #Realiza os testes na rede
    def test(self, examples):
    	predictions = []

    	for p in examples:
    		x = self.feedforward(p[0])
    		y = x[0]
    		print("resp: ", round(y, 3))

    #Treina a rede
    def train(self, examples):
    	iterations=10000
    	ratio=0.5

    	for i in range(iterations):
    		for p in examples:
    			inputs = p[0]
    			expected = p[1]
    			self.feedforward(inputs)
    			self.backPropagate(expected, ratio)
    	return 0

    def readWine(self):
        return 0


def start():
	trainMatrix = [
        [[1, 1], [1]],
		[[0, 1], [0]],
        [[0, 0], [1]] 
	]

	testMatrix = [
        [[1, 0], [0]]
	]

	mpl = MPL(13, 8 , 1)
    mpl.readWine()

if __name__ == '__main__':
    start()

