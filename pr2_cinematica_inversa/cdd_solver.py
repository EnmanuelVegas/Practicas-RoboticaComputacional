#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Enmanuel Vegas Acosta (alu0101281698)
# Robótica Computacional
# Grado en Ingeniería Informática
# Práctica: Resolución de la cinemática inversa mediante CCD
#           (Cyclic Coordinate Descent).

import sys
from math import *
import numpy as np
import matplotlib.pyplot as plt
import colorsys as cs

# ******************************************************************************
# Declaración de funciones

def muestra_origenes(O,final=0):
  # Muestra los orígenes de coordenadas para cada articulación
  print('Origenes de coordenadas:')
  for i in range(len(O)):
    print('(O'+str(i)+')0\t= '+str([round(j,3) for j in O[i]]))
  if final:
    print('E.Final = '+str([round(j,3) for j in final]))

def muestra_robot(O, obj):
  # Usa una figura persistente y modo interactivo
  plt.ion()
  fig = plt.gcf()
  fig.clf()
  size = 12
  plt.xlim(-size, size)
  plt.ylim(-size, size)
  T = [np.array(o).T.tolist() for o in O]
  for i, t in enumerate(T):
    plt.plot(t[0], t[1], '-o', color=cs.hsv_to_rgb(i / float(len(T)), 1, 1))
  plt.plot(obj[0], obj[1], '*')
  plt.waitforbuttonpress()

def matriz_T(d,th,a,al):
  # Calcula la matriz T. Los ángulos son en radianes directamente.
  return [[cos(th), -sin(th)*cos(al),  sin(th)*sin(al), a*cos(th)]
         ,[sin(th),  cos(th)*cos(al), -sin(al)*cos(th), a*sin(th)]
         ,[      0,          sin(al),          cos(al),         d]
         ,[      0,                0,                0,         1]
         ]

def cin_dir(th,a):
  # Sea 'th' el vector de thetas
  # Sea 'a'  el vector de longitudes
  # Devuelve lista de coordenadas X e Y de cada uno de los puntos O respecto al punto inicial.
  # 0 = [[x00, y00], [x10, y10], [x20, y20], ...]

  # T = np.identity(4)
  # o = [[0,0]]
  # for i in range(len(th)):
  #   T = np.dot(T,matriz_T(0,th[i],a[i],0))
  #   tmp=np.dot(T,[0,0,0,1])
  #   o.append([tmp[0],tmp[1]])
  # return o

  T0i_1 = np.identity(4) # T0i - 1
  origins = [[0,0]]
  for i in range(len(th)):
    Ti_1i = matriz_T(0, th[i], a[i], 0)
    Oii = [0,0,0,1]
    T0i = np.dot(T0i_1, Ti_1i)
    Oi0 = np.dot(T0i, Oii)
    xi0, yi0 = Oi0[0], Oi0[1]
    origins.append([xi0,yi0])
    T0i_1 = T0i
  return origins

# ******************************************************************************
# Cálculo de la cinemática inversa de forma iterativa por el método CCD

# valores articulares arbitrarios para la cinemática directa inicial
### EJERCICIO 1
# isPrismatic = [False, False, True, False]
# a =  [1., 4., 1., 1.]
# th = [10*pi/180, 25*pi/180, 0., 0.]
# qMin = np.array([0*pi/180,   -45*pi/180, 1, -90*pi/180]) 
# qMax = np.array([180*pi/180,  45*pi/180, 4,  90*pi/180])

### EJERCICIO 2
isPrismatic = [False, True, False, False]
a =  [2., 1., 3., 1.]
th = [15*pi/180, 0., 25*pi/180, 0.]
qMin = np.array([-90*pi/180, 0, -25*pi/180, -90*pi/180]) 
qMax = np.array([ 90*pi/180, 3,  65*pi/180,  90*pi/180])

### EJERCICIO 3
# isPrismatic = [False, True, False, False]
# a =  [2., 1., 3., 1.]
# th = [15*pi/180, 0., 25*pi/180, 0.]
# qMin = np.array([-90*pi/180, 0, -25*pi/180, -90*pi/180]) 
# qMax = np.array([ 90*pi/180, 3,  65*pi/180,  90*pi/180])

L = sum(a) # variable para representación gráfica
EPSILON = .01

#plt.ion() # modo interactivo

# introducción del punto para la cinemática inversa
if len(sys.argv) != 3:
  sys.exit("python " + sys.argv[0] + " x y")
objetivo=[float(i) for i in sys.argv[1:]]
O=cin_dir(th,a)
#O=zeros(len(th)+1) # Reservamos estructura en memoria
# Calculamos la posicion inicial
print ("- Posicion inicial:")
muestra_origenes(O)

numArtic = len(th) # número de articulaciones del manipulador
dist = float("inf")
prev = 0.
iteracion = 1
while (dist > EPSILON and abs(prev-dist) > EPSILON/100.):
  prev = dist
  posiciones = [cin_dir(th,a)]

  # Para cada combinación de articulaciones:
  for i in range(numArtic):
    artActual = numArtic - i - 1
    O_act = posiciones[-1]
    if not isPrismatic[artActual]:
      pos_art = np.array(O_act[artActual])
      pos_ext = np.array(O_act[-1])
      v1 = pos_ext - pos_art
      v2 = np.array(objetivo) - pos_art

      if np.linalg.norm(v1) < 1e-12 or np.linalg.norm(v2) < 1e-12:
        posiciones.append(cin_dir(th, a))
        continue

      ang1 = atan2(v1[1], v1[0])
      ang2 = atan2(v2[1], v2[0])
      delta = ang2 - ang1
      
      # Calcular el nuevo ángulo
      thActual = th[artActual] + delta

      # Normalizar para el rango [-pi, pi]
      while thActual > pi:
        thActual -= 2*pi
      while thActual < -pi:
        thActual += 2*pi

      # Limitar ángulos
      if thActual < qMin[artActual]:
        thActual = qMin[artActual]
      if thActual > qMax[artActual]:
        thActual = qMax[artActual]

      th[artActual] = thActual

    else:
      pos_art = np.array(O_act[artActual])
      pos_ext = np.array(O_act[-1])
      
      omega = sum(th[j] for j in range(artActual) if not isPrismatic[j])
      
      unit_vector = np.array([cos(omega), sin(omega)])
      v_objetivo = np.array(objetivo) - pos_ext
      distance = np.dot(v_objetivo, unit_vector)      
      distanciaNueva = a[artActual] + distance

      # Limitar distance según qmin y qmax
      if distanciaNueva < qMin[artActual]:
        distanciaNueva = qMin[artActual]
      elif distanciaNueva > qMax[artActual]:
        distanciaNueva = qMax[artActual]

      a[artActual] = distanciaNueva

    # Guarda la nueva posición tras el ajuste
    posiciones.append(cin_dir(th, a))

  dist = np.linalg.norm(np.subtract(objetivo,posiciones[-1][-1]))
  print ("\n- Iteracion " + str(iteracion) + ':')
  muestra_origenes(posiciones[-1])
  for i in range(len(th)):
    print ("  theta" + str(i+1) + " = " + str(round(th[i],3)) + " -> " + str(round(((th[i] * 180) / pi), 3)) + "°")
  muestra_robot(posiciones,objetivo)
  print ("Distancia al objetivo = " + str(round(dist,5)))
  iteracion+=1
  posiciones[0]=posiciones[-1]

if dist <= EPSILON:
  print ("\n" + str(iteracion) + " iteraciones para converger.")
else:
  print ("\nNo hay convergencia tras " + str(iteracion) + " iteraciones.")
print ("- Umbral de convergencia epsilon: " + str(EPSILON))
print ("- Distancia al objetivo:          " + str(round(dist,5)))
print ("- Valores finales de las articulaciones:")
for i in range(len(th)):
  print ("  theta" + str(i+1) + " = " + str(round(th[i],3)) + " -> " + str(round(((th[i] * 180) / pi), 3)) + "°")
for i in range(len(th)):
  print ("  L" + str(i+1) + "     = " + str(round(a[i],3)))
muestra_robot(posiciones,objetivo)
