#Criado por: Bruno Bacelar Abe, número USP: 9292858
import numpy as np 
import math
import os

"""Recebe como parâmetros a quantidade de nós no input, camada do meio e output como parâmetros"""
class MLP():
	def __init__(self, n_nodeInput, n_nodeHidden, n_nodeOutput):
		#Definição do tamanho da rede
		self.n_nodeInput = n_nodeInput
		self.n_nodeHidden = n_nodeHidden
		self.n_nodeOutput = n_nodeOutput

		#Usados na ativação da rede
		self.hiddenlayer_activations = []
		self.outputlayer_input = []

		#Pesos e bias da camada hidden
		self.wh = np.random.rand(self.n_nodeInput, self.n_nodeHidden)
		self.bh = np.random.rand(1, self.n_nodeHidden) 

		#Pesos e bias da camada de output
		np.random.rand
		self.wout =  np.random.rand(self.n_nodeHidden, self.n_nodeOutput)
		self.bout = np.random.rand(1, self.n_nodeOutput) 

	#Função sigmoid
	def sigmoide(self, x):
		return 1 / (1 + np.exp(-x))	

	#Derivada da função 
	def dsigmoide(self, f_net):
		f_net = np.asarray(f_net)
		return (f_net * (1 -f_net))

	#Fast forward 
	def ff(self, inputs):
		#Multiplicação entre os inputs e os pesos
		result = np.dot(inputs, self.wh)
		result = result + self.bh

		#Função de ativação
		self.hiddenlayer_activations = self.sigmoide(result)

		#Multiplicação entre a saída da layer do meio e os pesos da saída
		self.outputlayer_input = np.dot(self.hiddenlayer_activations, self.wout)
		self.outputlayer_input = self.outputlayer_input + self.bout

		#Aplicação da sigmoid novamente
		output = self.sigmoide(self.outputlayer_input)	

		return output

	#Função de treino, executa o backpropagation de fato
	def train(self, x, learnigRate, momentum, itDefined, threshold = 1e-3):
		gradError = []
		gradErrorWhile = 1
		it = 0

		while(it < itDefined):
		#while(gradErrorWhile > threshold):
			for p in x:
				expected = np.matrix(p[1])
				inputs = np.matrix(p[0])

				#fast foward
				output = self.ff(inputs)
				#Erro quadrado
				gradError = (np.power((expected - output), 2))
				gradErrorWhile = np.max(np.asarray(gradError[:]))
				error = (expected - output)

				#Deltas
				delta_output_layer = np.asmatrix(self.dsigmoide(output))
				delta_hidden_layer = self.dsigmoide(np.asarray(self.hiddenlayer_activations))
				d_out = (-1)*np.asarray(error)*np.asarray(delta_output_layer)
				error_hiddenLayer = np.dot(d_out, np.transpose(self.wout))
				d_hiddenLayer = np.asarray(error_hiddenLayer)*np.asarray(delta_hidden_layer)

				#Atualização dos pesos
				self.wout = self.wout - momentum *np.dot(np.transpose(self.hiddenlayer_activations), d_out)*learnigRate
				self.wh = self.wh - momentum*np.dot(np.transpose(inputs), d_hiddenLayer)*learnigRate

			it += 1
			if(it % 500 == 0):
				print("erro quad: ", gradError)

		print("erro quad: ", gradError)
		print("total it: ", it)
		return gradError

	#Função que prevê
	def testApx(self, examples, auxInput):
		#file = open("aprox/t3/resultado_10000_pto8_pto6.txt", "w")
		file.write(str("Erro quadrático: ").rstrip('\n'))
		file.write(str(auxInput))
		file.write("\n\n")

		for p in examples:
			expected = p[1]
			data = p[0]
			out = self.ff(p[0])

			print("Esperado: ", p[1])
			print("Obtidao: ", np.asarray(out))
			print("Acur: ", np.asarray(out)/p[1], "\n")
			
			"""
			file.write(str("Esperado: ").rstrip('\n'))
			file.write(str(np.asarray(p[1])))
			file.write(str(" Obtido: ").rstrip('\n'))
			file.write(str(np.asarray(out)))
			file.write("\n\n")"""

	#Função que prevê
	def testWine(self, examples, normaOut):
		#file = open("testes/t3/resultado_5000_pto8_pto6.txt", "w")
		accur = 0

		for p in examples:
			expected = p[1]
			data = p[0]
			out = self.ff(p[0])

			print("Esperado: ", np.asarray(p[1]).max())
			print("Obtidao: ", out.max())

			if( out.max() > np.asarray(p[1]).max()):
				print("Acur: ", np.asarray(p[1]).max()/out.max(), "\n")
				accur = np.asarray(p[1]).max()/out.max()
			else:
				print("Acur: ", out.max()/np.asarray(p[1]).max(), "\n")
				accur = out.max()/np.asarray(p[1]).max()
			
			"""
			file.write(str("Esperado: ").rstrip('\n'))
			file.write(str(np.asarray(p[1]).max()))
			file.write(str(" Obtido: ").rstrip('\n'))
			file.write(str(out.max()))
			file.write(str(" Accuracia: ").rstrip('\n'))
			file.write(str(accur))
			file.write("\n\n")"""


	def readWine_testAll(self):
		file = np.loadtxt("testes/t3/wine.txt", delimiter=", ")
		z = []
		z2 = []
		bigger = 0

		#Acha o maior da matriz para normalizar
		for i in range (len(file)):
			bigger2 = max(file[i][1:])

			if(bigger2 > bigger):
				bigger = bigger2

		size = np.round(len(file) - len(file)*0.25)
		print(size)
				
		for i in range(int(size)):
			x = []

			x.append(file[i][0]/3)

			#Normaliza os dados
			y = np.array((file[i][1:])/bigger)
			l = y.tolist()
			w = [l, x]

			z.append(w)

		w = []
		l = []
		y = []

		for i in range (int(size), len(file)):
			x = []

			x.append(file[i][0]/3)

			#Normaliza os dados
			y = np.array((file[i][1:])/bigger)
			l = y.tolist()
			w = [l, x]

			z2.append(w)			

		return z, z2


	def readWine_test(self, trainFile, testFile):
		#Treino
		file = np.loadtxt(trainFile, delimiter=", ")
		z = []
		z2 = []
		bigger = 0

		#Acha o maior da matriz para normalizar
		for i in range (len(file)):
			bigger2 = max(file[i][1:])

			if(bigger2 > bigger):
				bigger = bigger2

		size = np.round(len(file))			
		for i in range(int(size)):
			x = []

			x.append(file[i][0]/3)

			#Normaliza os dados
			y = np.array((file[i][1:])/bigger)
			l = y.tolist()
			w = [l, x]

			z.append(w)

		#Testes
		file = np.loadtxt(testFile, delimiter=", ")
		w = []
		l = []
		y = []

		for i in range (len(file)):
			x = []

			x.append(file[i][0]/3)

			#Normaliza os dados
			y = np.array((file[i][1:])/bigger)
			l = y.tolist()
			w = [l, x]

			z2.append(w)			

		return z, z2

	#Temos o vetor com 70 de tamanho, onde os dois últimos representam a latitude. 
	def aprox(self, tracksString):
		#Treino
		file = np.loadtxt(tracksString, delimiter=", ", dtype = float)
		z = []
		z2 = []
		bigger = 0
		biggerLat = 0
		biggerLong = 0

				#Acha o maior da matriz para normalizar
		for i in range (len(file)):
			bigger2 = max(file[i][:-2])

			if(bigger2 > bigger):
				bigger = bigger2
		
		for i in range (len(file)):
			bigger3 = file[i][68]
			bigger2 = file[i][69]

			if(bigger2 > biggerLat):
				biggerLat = bigger2

			if(bigger3 > biggerLong):
				biggerLong = bigger3

		for i in range(len(file)):
			x = []

			x.append(file[i][68]/biggerLong)
			x.append(file[i][69]/biggerLat)

			#Normaliza os dados
			y = np.array((file[i][:-2])/bigger)
			l = y.tolist()
			w = [l, x]
			z.append(w)
		
		return z, biggerLat, biggerLong

	def trainXOR(self, x, learnigRate, momentum, itDefined, threshold = 1e-3):
		gradError = []
		gradErrorWhile = 1
		it = 0

		#while(it < itDefined):
		while(gradErrorWhile > threshold):
			for p in x:
				expected = np.matrix(p[1])
				inputs = np.matrix(p[0])

				#fast foward
				output = self.ff(inputs)
				#Erro quadrado
				gradError = (np.power((expected - output), 2))
				gradErrorWhile = np.max(np.asarray(gradError[:]))
				error = (expected - output)

				#Deltas
				delta_output_layer = np.asmatrix(self.dsigmoide(output))
				delta_hidden_layer = self.dsigmoide(np.asarray(self.hiddenlayer_activations))
				d_out = (-1)*np.asarray(error)*np.asarray(delta_output_layer)
				error_hiddenLayer = np.dot(d_out, np.transpose(self.wout))
				d_hiddenLayer = np.asarray(error_hiddenLayer)*np.asarray(delta_hidden_layer)

				#Atualização dos pesos
				self.wout = self.wout - momentum *np.dot(np.transpose(self.hiddenlayer_activations), d_out)*learnigRate
				self.wh = self.wh - momentum*np.dot(np.transpose(inputs), d_hiddenLayer)*learnigRate

			it += 1
			if(it % 500 == 0):
				print("erro quad: ", gradError)

		print("erro quad: ", gradError)
		print("total it: ", it)
		return gradError


if __name__ == '__main__':
	trainFile2 = "testes/t2/wine2.txt"
	testFile2 = "testes/t2/teste2.txt"
	
	trainFile1 = "testes/t1/wine1.txt"
	testFile1 = "testes/t1/teste1.txt"

	trainFileMusic = "aprox/t1/tracks.txt"
	testFileMusic = "aprox/t1/test.txt"

	trainFileMusic2 = "aprox/t2/tracks.txt"
	testFileMusic2= "aprox/t2/test.txt"

	trainFileMusic3 = "aprox/t3/tracks.txt"
	testFileMusic3 = "aprox/t3/test.txt"

	while (1):
		print("Digite a sua opção: \n")
		print("1. Base de dados 1 - Wines\n")
		print("2. Base de dados 2 - Música\n")
		print("3. Teste padrão - XOR")
		opt = input()

		print("Escolha a quantidade de ciclos: ")
		epo = int(input())
		
		print("Escolha o momentum: (0 < momentum <= 1)")
		mm = float(input())

		print("Escolha o learnig rate: (0 < learnig rate <= 1)")
		lr = float(input())

		if (opt == "1"):
			nn = MLP(13, 10, 1)
			print("Escolha o caso de teste: \n")
			print("1. Base dividida em 70/30%\n")
			print("2. Base dividida em 50%\n")
			print("3. Base inteira\n")
			opt2 = int(input())
			
			if (opt2 == 1):
				print("Treinando com 2/3 dos dados")
				z, z2 = nn.readWine_test(trainFile1, testFile1)

				print("Treinando a rede...\n")
				nn.train(z, lr, mm, epo)
				print("Classificando o exemplo...\n")
				nn.testWine(z2, 1)

			elif (opt2 == 2):
				print("Treinando com metade dos dados")
				z, z2 = nn.readWine_test(trainFile2, testFile2)

				print("Treinando a rede...\n")
				nn.train(z, lr, mm, epo)
				print("Classificando o exemplo...\n")
				nn.testWine(z2, 1)

			elif (opt2 == 3):
				print("Treinando com todos os dados...")
				z, z2= nn.readWine_testAll()
				print("Treinando a rede...\n")
				nn.train(z, lr, mm, epo)
				print("Classificando o exemplo...\n")
				nn.testWine(z2, 1)
			else:
				print("Opção inválida")

		elif (opt == "2"):
			nn = MLP(68, 45, 2)
			print("Escolha o caso de teste: \n")
			print("1. Base dividida em 70/30%\n")
			print("2. Base dividida em 50%\n")
			print("3. Base inteira\n")
			opt2 = int(input())
			
			if (opt2 == 1):
				print("Treinando com 2/3 dos dados")
				z, normaOutLat, normaOutLong = nn.aprox(trainFileMusic)
				z2, normaOutLat2, normaOutLong2 = nn.aprox(testFileMusic)

				print("Treinando a rede...\n")
				gradError = nn.train(z, lr, mm, epo)
				print("Classificando o exemplo...\n")
				nn.testApx(z2, gradError)

			elif (opt2 == 2):
				print("Treinando com metade dos dados")
				z, normaOutLat, normaOutLong = nn.aprox(trainFileMusic2)
				z2, normaOutLat2, normaOutLong2 = nn.aprox(testFileMusic2)

				print("Treinando a rede...\n")
				gradError = nn.train(z, lr, mm, epo)
				print("Classificando o exemplo...\n")
				nn.testApx(z2, gradError)

			elif (opt2 == 3):
				print("Treinando com todos os dados...")
				z, normaOutLat, normaOutLong = nn.aprox(trainFileMusic3)
				z2, normaOutLat2, normaOutLong2 = nn.aprox(trainFileMusic3)

				print("Treinando a rede...\n")
				gradError = nn.train(z, lr, mm, epo)
				print("Classificando o exemplo...\n")
				nn.testApx(z2, gradError)
			else:
				print("Opção inválida")
		else:
			x = [
				[[1, 0], [1]],
				[[0, 0], [0]],
				[[0, 1], [1]],
				[[1, 1], [0]]
			]
			print("Iniciando a rede... \n")
			nn = MLP(2, 2, 1)
			print("Treinando a rede...\n")
			nn.trainXOR(x, lr, mm, epo)
			print("Classificando o exemplo...\n")
			nn.testWine(x, 1)