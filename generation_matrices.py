import numpy as np
from scipy import sparse




def generate_D_matrices(X,P,exactly=False):
    """
    Function: generate_D_matrices
    Generate exactly P matrices D_i of size n*n, where n is the number of samples in X.
    
    parameters:
        param X: np.array of size n*d
        param P: int, number of matrices to generate
    return: list of np.array of size n*n
    """
    n, d = X.shape
    D_matrices = []
    
    # On vérifie quand même pour éviter une boucle infinie
    assert P <= 2**d or not exactly, "P must be less than 2^d"

    # Génération des matrices
    while len(D_matrices) < P:
        
        # Génération d'un vecteur aléatoire, suivant une loi normale centrée réduite
        # Pourra être modifié par la suite
        u = np.random.randn(d)
        
        Xu = X.dot(u) #simple produit scalaire
        
        activation = (Xu >= 0).astype(int)
        
        # ligne à optimiser ! Renvoie une matrice de taille n*n, ce qui fait beaucoup, alors que la majorité des éléments sont nuls.
        # Voir avec le prof comment faire ! 
        # Stocker les vecteurs directement 
        # Dans numpy ustiliser les matrices sparse 
        
        
        #D_i = np.diag(activation) 
        
        #pour éviter les doublons
        if len([1  for f in D_matrices if (activation==f).all()]) == 0 :
            D_matrices.append(activation)
        else:
            if not exactly:
                P -= 1 # on retire un P, comme si on avait bien trouvé une matrice
            print("Doublon trouvé")
        
    return D_matrices