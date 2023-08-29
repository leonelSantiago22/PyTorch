import torch as th
import numpy as np
'''
En PyTorch, los tensores son estructuras fundamentales que se utilizan para representar datos multidimensionales, como 
matrices o vectores, de manera eficiente en términos de cálculos y operaciones matemáticas. Los tensores son similares a 
las matrices numéricas, pero están diseñados para funcionar de manera eficiente en arquitecturas de GPU, lo que los 
hace especialmente útiles para el procesamiento paralelo y el entrenamiento de modelos de aprendizaje profundo.
'''
'''
escalar = th.tensor(4.0)
print(f'Tensor de orden 0 (escalar): {escalar}.\nTipo: {type(escalar)}')
vector = th.tensor([2.0, 3.0, 4.0])
print(f'Tensor de orden 1 (vector): {vector}')
matriz = th.tensor([[1, 2, 3], [4, 5, 6]])
print(f'Tensor de orden 2 (matriz): {matriz}')
tensor3 = th.tensor([[[1,2,3], [4,5,6]], [[7,8,9], [10,11,12]]])
print(f'Tensor de orden 3: {tensor3}')
'''
#unif_ten = th.rand((4, 3))
#print(f'Matriz con elementos muestreados uniformemente de [0,1): {unif_ten}')

#Los elementos de los tensores pueden ser de distintos tipos básicos y precisiones. Por ej. flotantes de 16 bits (float16)
# o enteros sin signo de 8 bits (uint8).
'''
ten_f64 = th.arange(start=-3, end=3, step=0.5, dtype=th.float64)
ten_i32 = th.arange(start=-3, end=3, step=1, dtype=th.int32)
print(f'Tipo flotante de 64: {ten_f64.dtype}')
print(f'Tipo entero de 32: {ten_i32.dtype}')


x_orig = th.arange(30)
print(f'Forma de tensor original: {x_orig.shape}')
'''

xcpu = th.rand((100, 100))
print(xcpu.device)
xgpu = th.rand((100, 100), device='cuda:0')
xgpu.device
ygpu = th.rand((100, 100), device='cuda:0')
(xgpu + ygpu).device
print(th.cuda.is_available())
