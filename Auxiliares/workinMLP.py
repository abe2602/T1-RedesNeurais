# Back-Propagation Neural Networks
#
from numpy import exp, array, random, dot
import numpy as np
import math

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
        self.weightHidden = np.random.rand(self.input, self.hidden)
        self.weightOutput = np.random.rand(self.hidden, self.output)
        #Mudanças nas matrizes
        self.changeInput = np.zeros((self.input, self.hidden))
        self.changeOutput = np.zeros((self.hidden, self.output))

    #Funções auxiliares para realizar a alimentação da rede
    #A alimentação da entrada é: copia-se todo o vetor
    def feedInput(self, inputs):
        for i in range(self.input):
            self.valuesInput[i] = inputs[i]
   
   #Aplica o somatório (poderia ter utilizado a função "dot")
    def feedHidden(self):
        for i in range(self.hidden):
            sum = 0.0
            for j in range(self.input):
                sum += self.valuesInput[j] * self.weightHidden[j][i]

            self.valuesHidden[i] = sigmoid(sum)
   
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

    #Funções auxiliares do backPropagation

    def backPropagateFirstLayer(self, output_deltas, expected):
        for i in range(self.output):
            error = (expected[i] - self.valuesOutput[i])
            output_deltas[i] = -1*derivativeSigmoid(self.valuesOutput[i])*error

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
            #x = self.feedforward(p[0])
            print("result: ", self.feedforward(p[0]))

    #Treina a rede
    def train(self, examples):
        iterations=5000
        ratio=0.5
        errorQuad = 1

        #while error > 0.01:
        for i in range(iterations):
            for p in examples:
                inputs = p[0]
                expected = p[1]

                print(self.feedforward(inputs))
                self.feedforward(inputs)
                errorQuad = self.backPropagate(expected, ratio)

        return 0

#Temos o vetor com 70 de tamanho, onde os dois últimos representam a latitude. 
    def aprox(self, tracksString):
        #Treino
        file = np.loadtxt(tracksString, delimiter=", ", dtype = float)
        z = []
        z2 = []
        bigger = 0
        biggerLat = 0

                #Acha o maior da matriz para normalizar
        for i in range (len(file)):
            bigger2 = max(file[i][:-2])

            if(bigger2 > bigger):
                bigger = bigger2
        

        for i in range (len(file)):
            bigger2 = max(file[i][68:69])

            if(bigger2 > biggerLat):
                biggerLat = bigger2

        print(biggerLat)

        for i in range(len(file)):
            x = []

            x.append(file[i][68]/biggerLat)
            x.append(file[i][69]/biggerLat)

            #Normaliza os dados
            y = np.array((file[i][:-2])/bigger)
            l = y.tolist()
            w = [l, x]
            z.append(w)
        
        return z

def demo():
    trainFileMusic = "aprox/t1/tracks.txt"
    print("Starting training...")
    n = MPL(68, 30, 2)
    pat = n.aprox(trainFileMusic)
    n.train(pat)
   # n.test(pat2)

if __name__ == '__main__':
    demo()

