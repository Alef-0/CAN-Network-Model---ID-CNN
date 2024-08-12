# CAN-Network-Model---ID-CNN
###  Grupo: Alef Gabryel e Samuel Simões

Repositório para o projeto da cadeira de redes intraveiculares.
O objetivo foi implementar um sistema de detecção de intrusão para a rede CAN. 
Foi criado um modelo de machine learning utilizando pytorch, que retorna um valor de 0 a 1, com 1 sendo normal se não houve nenhum ataque em uma série de mensagens na rede CAN.

O código principal está no arquivo **gids_dis.py**, enquanto os outros servem como um passo a passo. [A base de dados foi disponibilizada pelo professor]
- **main_data**: Uso de funções na ordem para filtrar os dados e treinar.
- **main_linear**: Chamar o modelo já treinado, o arquivo .pth foi o melhor modelo treinado.
- **visalizacao_XXX**: Arquivos de python notebook para analisar os dados.

# Inputs de dados
O modelo através da função ```input_to_pandas``` pega arquivos de texto e os transforma em arquivos .csv para melhor leitura. Ele espera que todos os IDs estejam incluidos entre esses
> 007 | 008 | 00D | 00E | 014 | 015 | 016 | 017 | 041 | 055 | 056 | 05B | 05C | 05D

Caso contrário será considerado um valor falso, a variável pode ser editada no arquivo para considerar novos tipos de entrada.

# Propriedades do modelo
Utiliza-se na entrada uma rede convolucional, e depois uma parte totalmente conectada. Utiliza-se *batch normalization* entre as camadas, e a função de ativação é a *LeakyReLU*. 
```
Discriminator(
  (convolutions): Sequential(
    (0): Conv1d(15, 9, kernel_size=(17,), stride=(1,), padding=(8,), bias=False)
    (1): BatchNorm1d(9, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): LeakyReLU(negative_slope=0.2, inplace=True)
    (3): Conv1d(9, 5, kernel_size=(17,), stride=(1,), padding=(8,), bias=False)
    (4): BatchNorm1d(5, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (5): LeakyReLU(negative_slope=0.2, inplace=True)
    (6): Conv1d(5, 3, kernel_size=(17,), stride=(1,), padding=(8,), bias=False)
    (7): BatchNorm1d(3, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (8): LeakyReLU(negative_slope=0.2, inplace=True)
  )
  (fullyConnected): Sequential(
    (0): Linear(in_features=192, out_features=128, bias=True)
    (1): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): LeakyReLU(negative_slope=0.2, inplace=True)
    (3): Linear(in_features=128, out_features=1, bias=True)
    (4): Sigmoid()
  )
)
```

# Utilizando em uma placa
Colocar o arquivo ```read-messages.py``` junto com o ```gids_dis.py``` e rodar em uma placa raspberry pi.
Precisa estar conectada em uma can transmitter e estar configurada para enviar mensagens can.