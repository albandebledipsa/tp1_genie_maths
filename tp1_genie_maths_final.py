# -*- coding: utf-8 -*-
"""
Created on Sat Mar 13 14:07:06 2021

@author: alban Debled, Lara Chouraqui
"""

import numpy as np
from numpy import linalg as LA
import time
import matplotlib.pyplot as plt

"""Partie 1"""

def ReductionGauss(Aaug):
    A = np.array(Aaug)
    for j in range(len(A[0])-1):
        for i in range(j+1, len(A)):
            pivot = A[j, j]
            g = A[i, j] / pivot
            A[i] = A[i] - g * A[j]
    return A

def ResolutionSystTriSup(Taug):
        
    n,m = np.shape(Taug)
    X = np.zeros(n)
    for i in range(n-1, -1, -1):
        somme = 0
        for j in range(i, n):
            somme += Taug[i][j] * X[j]
        X[i] = (Taug[i,n] - somme) / Taug[i][i]
    return X

def Gauss(matrice_augmente, A, B):
    startt_CPU = time.process_time()
    startt_eff = time.time()
    resultat = ReductionGauss(matrice_augmente)
    solution = ResolutionSystTriSup(resultat)
    stopt_CPU = time.process_time()
    stopt_eff = time.time()
    erreur = calcul_erreur(A, solution, B)
    temps = stopt_eff - startt_eff
    return temps, erreur

"""partie 2"""


def DecompositionLU(A):
    U = np.array(A)
    L = np.zeros((np.size(A[0]),np.size(A[0])))
    np.fill_diagonal(L, 1)    
    for j in range(len(U[0])):
        for i in range(j+1, len(U)):
            g = U[i, j] / U[j, j]
            U[i] = U[i] - g * U[j]
            L[i, j] = g
    return U,L


def ResolutionLU(L,U,B):

    n = len(B)
    X = np.zeros(n)
    Y = np.zeros(n)
    
    for i in range(0, n, 1):
        somme = 0
        for j in range(0 , i+1, 1):
            somme += L[i, j] * Y[j]
        Y[i] = (B[i] - somme) / L[i, i]

    for i in range(n-1, -1, -1):
        somme = 0
        for j in range(i , n):
            somme += U[i, j] * X[j]
        X[i] = (Y[i] - somme) / U[i, i]
    return X


def LU(A, B):
    startt_CPU = time.process_time()
    startt_eff = time.time()
    [U, L] = DecompositionLU(A)
    solution = ResolutionLU(L,U,B)
    stopt_CPU = time.process_time()
    stopt_eff = time.time()
    erreur = calcul_erreur(A, solution, B)
    temps = stopt_eff - startt_eff 
    return temps, erreur
    
""" Partie 3"""

def ReductionGaussPivotPartiel(Aaug):
    A = np.array(Aaug)
    m = 0
    for j in range(0, len(A)-1):
        L=[]
        for i in range(m, len(A)):
            L.append(A[i][j])
        pivot_max = L.index(max(L))
        if pivot_max != j:
            memoire = A[pivot_max + m, :].copy()
            A[pivot_max + m, :] = A[j, :]
            A[j, :] = memoire
        for i in range(j + 1, len(A)):
            pivot_max = A[j, j]
            g = A[i, j] / pivot_max
            A[i] = A[i] - g * A[j]
        m += 1
    return A

def ResolutionSystTriSupPivotPartiel(Taug):
    n,m = np.shape(Taug)
    X = np.zeros(n)
    for i in range(n-1, -1, -1):
        somme = 0
        for j in range(i, n):
            somme += Taug[i][j] * X[j]
        X[i] = (Taug[i,n] - somme) / Taug[i][i]
    return X

def GaussChoixPivotPartiel(matrice_augmente, A, B):
    startt_CPU = time.process_time()
    startt_eff = time.time()
    resultat = ReductionGaussPivotPartiel(matrice_augmente)
    solution = ResolutionSystTriSupPivotPartiel(resultat)
    stopt_CPU = time.process_time()
    stopt_eff = time.time()
    erreur = calcul_erreur(A, solution, B)
    temps = stopt_eff - startt_eff 
    return temps, erreur

    
""" Partie 4"""

def ReductionGaussPivotTotal(Aaug):
    A = np.array(Aaug)
    m = 0 
    for j in range(0, len(A)-1):
        L = []
        C = []
        for i in range(m, len(A)):
            L.append(A[i][j])
        for k in range(m, len(A)):
            C.append(A[m][k])
        pivot_max_ligne = L.index(max(L))
        pivot_max_colonne = C.index(max(C))
        if L[pivot_max_ligne] > C[pivot_max_colonne] :
            memoire = A[pivot_max_ligne + m, :].copy()
            A[pivot_max_ligne + m, :] = A[j, :]
            A[j, :] = memoire
        if L[pivot_max_ligne] < C[pivot_max_colonne] :
            memoire = A[:, pivot_max_colonne + m].copy()
            A[:, pivot_max_colonne + m] = A[: ,j]
            A[:, j] = memoire
        if L[pivot_max_ligne] == C[pivot_max_colonne] :
            memoire = A[pivot_max_ligne + m, :].copy()
            A[pivot_max_ligne + m, :] = A[j, :]
            A[j, :] = memoire
        
        for i in range(j + 1, len(A)):
            pivot_max = A[j, j]
            g = A[i, j] / pivot_max
            A[i] = A[i] - g * A[j]
        m += 1
    return A

def ResolutionSystTriSupPivotTotal(Taug):
    
    n,m = np.shape(Taug)
    X = np.zeros(n)
    for i in range(n-1, -1, -1):
        somme = 0
        for j in range(i, n):
            somme += Taug[i][j] * X[j]
        X[i] = (Taug[i,n] - somme) / Taug[i][i]
    return X

def reorganisation_solution(A, resultat, solution_desordre):
    for k in range(len(A)):
        for l in range(len(A)):
            if resultat[0,0] == A[k, l]:
                indice = k
                break
    for i in range(len(A[0])):
        if A[indice, i] != resultat[0, i]:
            for j in range(i, len(A)):
                if A[indice, i] == resultat[0, j]:
                    resultat[0, i], resultat[0, j] = resultat[0, j], resultat[0, i]
                    solution_desordre[i], solution_desordre[j] = solution_desordre[j], solution_desordre[i]
                    break
    return solution_desordre
        
        
    

def GaussChoixPivotTotal(matrice_augmente, A, B):
        startt_CPU = time.process_time()
        startt_eff = time.time()
        resultat = ReductionGaussPivotTotal(matrice_augmente)
        solution_desordre = ResolutionSystTriSupPivotTotal(resultat)
        solution = reorganisation_solution(A, resultat, solution_desordre)
        stopt_CPU = time.process_time()
        stopt_eff = time.time()
        erreur = calcul_erreur(A, solution, B)
        temps = stopt_eff - startt_eff
        return temps, erreur


"""Partie 5 : linalg_solve"""

def linalg_solve(A, B):
        startt_CPU = time.process_time()
        startt_eff = time.time()
        X = LA.solve(A, B)
        stopt_CPU = time.process_time()
        stopt_eff = time.time()
        erreur = calcul_erreur(A, X, B)
        temps = stopt_eff - startt_eff
        return temps, erreur

""" Programme principal"""

def main():
    temps_Gauss = []
    temps_LU = []
    temps_pivot_partiel = []
    temps_pivot_total = []
    temps_linalg_solve = []
    erreur_Gauss = []
    erreur_LU = []
    erreur_pivot_partiel = []
    erreur_pivot_total = []
    erreur_linalg_solve = []
    taille_de_la_matrice = []
    
    
    for i in range(200, 1000, 50):
        
        print(i)
        A = []
        B = []
        A = np.random.rand(i,i)
        B = np.random.rand(i,1)
        matrice_augmente = np.append(A, B, axis = 1)
        
        taille_de_la_matrice.append(i)
        temps_Gauss.append(Gauss(matrice_augmente, A, B)[0])
        erreur_Gauss.append(Gauss(matrice_augmente, A, B)[1])
        temps_LU.append(LU(A, B)[0])
        erreur_LU.append(LU(A, B)[1])
        temps_pivot_partiel.append(GaussChoixPivotPartiel(matrice_augmente, A, B)[0])
        erreur_pivot_partiel.append(GaussChoixPivotPartiel(matrice_augmente, A, B)[1])
        temps_linalg_solve.append(linalg_solve(A, B)[0])
        erreur_linalg_solve.append(linalg_solve(A, B)[1])
        temps_pivot_total.append(GaussChoixPivotTotal(matrice_augmente, A, B)[0])
        erreur_pivot_total.append(GaussChoixPivotTotal(matrice_augmente, A, B)[1])
      
    print("t1 :", temps_Gauss)
    print("t2 :", temps_LU)
    print("t3 :", temps_pivot_partiel)
    print("t4 :", temps_pivot_total)
    print("t5 :", temps_linalg_solve)
    
    fig, ax = plt.subplots()
    ax.plot(taille_de_la_matrice, temps_Gauss, label = "Gauss")
    ax.plot(taille_de_la_matrice, temps_LU, label = "LU")
    ax.plot(taille_de_la_matrice, temps_pivot_partiel, label = "Gauss pivot partiel")
    ax.plot(taille_de_la_matrice, temps_pivot_total, label = "Gauss pivot total")    
    ax.plot(taille_de_la_matrice, temps_linalg_solve, label = "linalg_solve")
        
    plt.legend()
    plt.title("Temps en fonction de la taille de la matrice pour toutes les méthodes")
    plt.xlabel("taille des matrices")
    plt.ylabel("temps (secondes)")
    plt.savefig("graph temps toutes méthodes")
    
    fig, ax2 = plt.subplots()
    ax2.plot(taille_de_la_matrice, erreur_Gauss, label = "Gauss")
    ax2.plot(taille_de_la_matrice, erreur_LU, label = "LU")
    ax2.plot(taille_de_la_matrice, erreur_pivot_partiel, label = "Gauss pivot partiel")
    ax2.plot(taille_de_la_matrice, erreur_pivot_total, label = "Gauss pivot total")    
    ax2.plot(taille_de_la_matrice, erreur_linalg_solve, label = "linalg_solve")

    plt.legend()
    plt.title("Erreur en fonction de la taille de la matrice pour toutes les méthodes")
    plt.xlabel("taille des matrices")
    plt.ylabel("erreur")
    plt.savefig("graph erreur toutes méthodes")

    plt.show()
        

def calcul_erreur(A, X, B):
    produit = np.dot(A, X)
    for i in range(len(produit)):
        produit[i] = produit[i] - B[i]
    erreur = LA.norm(produit)
    return erreur

main()