import cvxpy as cp
import numpy as np


def solve(X,y,D, beta):
    P_tilde = len(D)
    n, d = X.shape
    
    # Variables d'optimisation
    v = [cp.Variable(d) for _ in range(P_tilde)]
    w = [cp.Variable(d) for _ in range(P_tilde)]
    
    
    constraints = []


    # Expression pour la somme
    expr = 0
    for i in range(P_tilde):
        activation = D[i]
        
        D_i_X = activation[:, np.newaxis] * X

        #D_i_X = activation @ X  # Ici est ce qu'il faut utiliser np.dot ou @ ? 
        expr += D_i_X @ (v[i] - w[i])  # (n,)
        
        diag_2D_minus_I = (2 * activation - 1)  
        # Contraintes : (2D_i - I_n) X v_i >= 0
        constraints.append(cp.multiply(diag_2D_minus_I, X @ v[i]) >= 0)
        constraints.append(cp.multiply(diag_2D_minus_I, X @ w[i]) >= 0)

    # Fonction de loss, on prend la norme 2 
    loss = cp.sum_squares(expr - y)

    # On y ajoute la régularisation, on reprend la norme 2 -> comme on a vu elle conduit aussi à la sparsité
    # On fera le test aussi avec la norme 1 pour voir la différence
    reg = beta * cp.sum([cp.sum_squares(v[i]) + cp.sum_squares(w[i]) for i in range(P_tilde)])

    # On féfinit le problème
    objective = cp.Minimize(loss + reg)
    problem = cp.Problem(objective, constraints)

    # Résolution du problème, on a trouvé SCS mais on doit voir si c'est bien le bon ...
    problem.solve(solver=cp.ECOS)  
    if problem.status in ["infeasible", "unbounded"]:
        raise Exception("Problem is infeasible or unbounded")
    

    # Solution finale ! 
    v_opti = [vi.value for vi in v]
    w_opti = [wi.value for wi in w]
    
    return v_opti, w_opti

def construction_reseau(v_opti,w_opti,P_tilde):
    # Puis on revient à notre réseau ! On applique juste la formule, il fuadrait le mettre dans un autre fichier à terme 
    u_list = []
    alpha_list = []

    for i in range(P_tilde):
        vi_norm = np.linalg.norm(v_opti[i])
        if vi_norm >  1e-6: # toujours vérifier que le vecteur n'est pas 0 quand on divise 
            u = v_opti[i] / np.sqrt(vi_norm)
            alpha = np.sqrt(vi_norm)
            u_list.append(u)
            alpha_list.append(alpha)
        
        wi_norm = np.linalg.norm(w_opti[i])
        if wi_norm >  1e-6:
            u = w_opti[i] / np.sqrt(wi_norm)
            alpha = -np.sqrt(wi_norm) # seul changement avec les vi, on prend l'opposé
            u_list.append(u)
            alpha_list.append(alpha)
    return u_list, alpha_list
    