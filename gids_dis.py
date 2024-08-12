import torch
from torch import nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from io import StringIO

def set_device():
	global DEVICE
	if torch.cuda.is_available(): DEVICE = torch.device('cuda')
	else: DEVICE = torch.device('cpu')
	return DEVICE

DEVICE = set_device()

def input_to_pandas(ler, escrever, save = True):
	# Precisa ter a pasta original_files

	with open(ler, 'r') as file:
		# Grab everything and then change to csv
		everything = file.read().strip()
		everything = everything.replace('(', '').replace(')', '').replace(' can0','').replace('#', ' ')
		everything = everything.replace(' ', ',')
		# read in csv
		frame = pd.read_csv(StringIO(everything), header=None)
		frame = frame.rename(columns = {0:'TIME', 1: 'ID', 2: 'DATA', 3: 'NORMAL'})
		# Changing R and T's
		mask_t = frame['NORMAL'] == 'T'; mask_r = frame['NORMAL'] == 'R'
		frame.loc[mask_r, 'NORMAL'] = True; frame.loc[mask_t, 'NORMAL'] = False;
		if save: frame.to_csv(escrever, index=False, header = True)
		frame = None # Fechando o frame para salvar memória

U_MISS = ['000', '007', '008', '00D', '00E', '014', '015', '016', '017', '041', '055', '056', '05B', '05C', '05D']
# UNIQUE = ['000', '007', '008', '00D', '00E', '014', '015', '016', '017', '041', '055', '056', '05B', '05C', '05D', 'NO']
SET = {'000', '007', '008', '00D', '00E', '014', '015', '016', 
			'017', '041', '055', '056', '05B', '05C', '05D'}
VALID_IDS = ['007', '008', '00D', '00E', '014', '015', '016', '017', '041', '055', '056', '05B', '05C', '05D']

def one_hot_encoded(values):
	# Todos os outros vão em NO
	dummies = pd.get_dummies([x if x in SET else '000' for x in values], dtype=np.float32)
	return dummies.reindex(columns=U_MISS, fill_value=0.0)

def one_hot_encoded_to_tensor(valores):
	entrada = one_hot_encoded(valores).values
	entrada = torch.tensor(entrada, dtype=torch.float32).transpose(0,1)
	return torch.stack([entrada])

def get_information(frame):
	hot = one_hot_encoded(frame['ID'].values).values
	tipos = all(frame['NORMAL'].values) * 1.0 # Vê se tem um ataque
	return torch.tensor(hot, dtype=torch.float32).transpose(0,1), torch.tensor([tipos], dtype=torch.float32)

class Dataset():
	def __init__(self, window, datas : list, division = 0.8):
		self.window_size = window
		self.datasets = []
		for i in datas:
			self.datasets.append(pd.read_csv(i))
		self.middle = []
		for i in self.datasets:
			self.middle.append(int(division * (len(i) // window)) - self.window_size)
		self.end = []
		for i in self.datasets:
			self.end.append((len(i) // window) - self.window_size)

	def __getitem__(self, place):
		table, index = place
		index *= self.window_size # offset
		frame = self.datasets[table][index : index + self.window_size]

		# Pegar informações
		return get_information(frame)
	
class Dataloader():
	def __init__(self, dataset):
		self.dataset = dataset
		self.iter_t = [0,0,0,0,0] #normal, dos, false, fuzz, impersonate

	def reset_others(self, training = False):
		if training: self.iter_t[0] = 0
		self.iter_t[1] = 0
		self.iter_t[2] = 0
		self.iter_t[3] = 0
		self.iter_t[4] = 0
		
	def get_training(self, normal = 16, ataque = 4):
		# Pegar em valor 3 de cada		
		while True:
			values = []
			# Pegar os valores
			values += [self.dataset[0, x] for x in range(self.iter_t[0], self.iter_t[0] + normal)]
			values += [self.dataset[y, x] for y in [1,2,3,4] for x in range(self.iter_t[y], self.iter_t[y] + ataque) ]
			# Offset de iteradores
			self.iter_t[0] = self.iter_t[0] + normal if self.iter_t[0] + normal < self.dataset.middle[0] else 0
			
			sair = False
			for i in [1,2,3,4]:
				if self.iter_t[i] + ataque < self.dataset.middle[i]:
					self.iter_t[i] = self.iter_t[i] + ataque
				else: sair = True
			if sair: break
			
			# separar data e label
			data, label = zip(*values)
			yield torch.stack(data), torch.stack(label)

	def get_validation(self):
		# Pegar em valor 3 de cada
		for i in range(5):
			values = [self.dataset[i, x] for x in 
				range(self.dataset.middle[i], self.dataset.end[i], self.dataset.window_size)]
			data, label = zip(*values)
			yield torch.stack(data), torch.stack(label)


def init_weigh_normal(model):
	for layer in model:
		if isinstance(layer, nn.Conv1d):
			nn.init.kaiming_normal_(layer.weight, nonlinearity="leaky_relu")
		if isinstance(layer, nn.BatchNorm1d):
			nn.init.ones_(layer.weight)
			nn.init.zeros_(layer.bias)

class Discriminator(nn.Module):
	def __init__(self, window):
		super(Discriminator, self).__init__()

		self.convolutions = nn.Sequential(
			# 11,1,5 ou 13,1,6 ou 15,1,7 ou 17,1,8
			nn.Conv1d(15,9, 17, 1, 8, bias=False),
			nn.BatchNorm1d(9),
			nn.LeakyReLU(0.2, inplace=True),
			# Proxima camada
			nn.Conv1d(9,5, 17, 1, 8, bias=False),
			nn.BatchNorm1d(5),
			nn.LeakyReLU(0.2, inplace=True),
			# Proxima camada
			nn.Conv1d(5,3, 17, 1, 8, bias=False),
			nn.BatchNorm1d(3),
			nn.LeakyReLU(0.2, inplace=True),
		)
		
		init_weigh_normal(self.convolutions) # Para melhorar Talvez

		self.fullyConnected = nn.Sequential(
			nn.Linear(window * 3, window * 2),
			nn.BatchNorm1d(window * 2),
			nn.LeakyReLU(0.2, inplace=True),
			# Proxima camada
			nn.Linear(window * 2, 1),
			nn.Sigmoid()
		)

		init_weigh_normal(self.fullyConnected)

	def forward(self, x):
		y = self.convolutions(x)
		y = y.reshape(y.shape[0], -1)
		return self.fullyConnected(y)
		
def graph_validation(model, dataloader : Dataloader, DEVICE, graph_range = (0,1)):
	model.eval()
	attack = []
	normal = []
	total = 0
	# Pegar os pontos
	for data, labels  in dataloader.get_validation():
		predictions = model(data.to(DEVICE))
		for label, prediction in zip(labels, predictions):
			if label.item() == 1.0: normal.append(prediction.item())
			else: attack.append(prediction.item())
			total += 1
	
	# Fazer os graficos
	plt.hist(normal, bins = 100, color='b', alpha = 0.4, range = graph_range)
	plt.hist(attack, bins = 100, color='r', alpha = 0.7, range = graph_range)	
	
def save_model(model, nome):
	torch.save(model.cpu(), f"{nome}")

def load_model(arquivo : Discriminator, window, device='cpu'):
	# Precisa ser da mesma janela
	model = Discriminator(window)
	model = torch.load(arquivo).to(device)
	criterion = nn.BCELoss() # Usado para regressão logística
	optmizer = torch.optim.SGD(model.parameters(), lr=2e-4, momentum=0.9, weight_decay=1e-3) # Precisa ter um grande número de épocas para servir
	return model, criterion, optmizer

LISTA_TREINAMENTO = [r"training\normal.csv", r"training\dos.csv", r"training\false.csv", r"training\fuzz.csv", r"training\impersonate.csv"]
def create_everything(window, device = 'cpu', Adam = False, files = LISTA_TREINAMENTO):
	model = Discriminator(window).to(device)
	criterion = nn.BCELoss()
	# OTIMO 2e-4 para lr
	# Momentum de 0.9 normal
	# Weight_decay com 1e-5
	if Adam: optimizer = torch.optim.Adam(model.parameters(), lr=3e-4) # Valores pegos de exemplo
	else:  optimizer = torch.optim.SGD(model.parameters(), lr = 2e-4, momentum=0.9, weight_decay=1e-3) # Só testar
	dataset = Dataset(window, LISTA_TREINAMENTO)
	dataloader = Dataloader(dataset)

	return model, criterion, optimizer, dataset, dataloader

def training(model, criterion, optmizer, dataloader : Dataloader, device, LIMIT = 0.05):
	training_losses = []
	validation_losses = []
	loss_training = 100
	loss_validation = 100
	sair = False
	
	for epoch in range(3):
		dataloader.reset_others()
		for iteration, (data,labels) in enumerate(dataloader.get_training()):
			model.train()
			optmizer.zero_grad() # Zeroing gradients
			prediction = model(data.to(device))
			
			#Compute loss
			loss = criterion(prediction, labels.to(device))
			loss.backward()
			optmizer.step()
			
			training_losses.append(loss.detach().cpu().numpy())
			if (iteration + 1) % 16 == 0:
				model.eval()
				for data, labels  in dataloader.get_validation():
					prediction = model(data.to(device))
					loss = criterion(prediction, labels.to(device))
					validation_losses.append(loss.detach().cpu().numpy())
				
				loss_training = np.mean(training_losses) 
				loss_validation = np.mean(validation_losses)
				print(f"[{epoch + 1}° Epoch] | {iteration + 1}° batch: Training loss = {loss_training : .6f} | Validation loss = {loss_validation : .5f}")
				
				training_losses.clear(); validation_losses.clear()
			
				if loss_validation < LIMIT or loss_training < LIMIT: 
					sair = True
					print("Saiu porque atingiu o limite")
					break
		
		if sair: break # Acabar
		
	print("Terminou o treinamento")
	
def testing(model, path_to_files, window, limit = 0.5, device = 'cpu'):
	positivo, negativo, falso_positivo, falso_negativo = 0,0,0,0
	hits, total = 0,0
	model.eval()
	normal = []
	attack = [[],[],[],[]]

	# Começar a checar
	for dirpath, dirname, filenames in os.walk(path_to_files):
		print("Entrou em ", dirpath)
		for i, filename in enumerate(filenames):
			cur_hit, cur_total = 0,0
			if i != 3: 	print(f"{filename} \t\t ", end = "")
			else: 		print(f"{filename} \t ", end = "") # Impersonate
			# Lendo arquivo
			frame = pd.read_csv(f"{os.path.join(dirpath, filename)}")
			separar = [frame[i : i + window] for i in range(0, len(frame) - (len(frame) % window), window)]
			data, label = zip(*[get_information(x) for x in separar])
			# Predict
			result = model(torch.stack(data).to(device))
			# label = torch.stack(label)
			for x, y in zip(result, label):
				r, l = x.item(), y.item()
				# Checar valores
				if 	 (r >= limit and l == 1.0): positivo += 1; hits += 1; cur_hit += 1
				elif (r < limit and l == 0.0): negativo += 1; hits += 1; cur_hit += 1
				elif (r >= limit and l == 0.0): falso_positivo += 1
				elif (r < limit and l == 1.0): falso_negativo += 1
				total += 1; cur_total += 1
				# Colocar nos valores
				if l == 1.0: normal.append(r)
				else: attack[i].append(r)
			print(f"Acurácia {cur_hit/cur_total * 100}% | Total: {hits/total * 100}%")
	# Retornando
	if total: return hits/total, hits, total, [normal] + attack, [positivo, falso_positivo, falso_negativo, negativo] # Porcentagem de certo

def make_graph(datas, confusion_matrix, graph_range = (0,1)):
	total = sum(confusion_matrix)
	print(f"Positivo: {confusion_matrix[0]/total :.2f} \t\t| Falso Positivo: {confusion_matrix[1]/total :.2f}")
	print(f"Falso Negativo: {confusion_matrix[2]/total :.2f} \t| Negativo: {confusion_matrix[3]/total :.2f}")
	
	# Fazendo gráfico
	plt.hist(datas[0], bins = 100, color='b',  range = graph_range)
	plt.hist(datas[1], bins = 100, color='r',  range = graph_range, label='dos')	
	plt.hist(datas[2], bins = 100, color='g',  range = graph_range, label='false')	
	plt.hist(datas[3], bins = 100, color='y',  range = graph_range, label='fuzz')	
	plt.hist(datas[4], bins = 100, color='m',  range = graph_range, label='impersonate')	
	plt.legend(); plt.yscale('log')

import seaborn as sns