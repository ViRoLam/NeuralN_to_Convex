import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


np.random.seed(1)

# On génère les données, en 2D
n_samples = 500
mean = [0, 0]

# La matrice de covariance, elle est cruciale, car c'est elle qui va déterminer si la PCA est pertinente ou non
# On peut imaginer que plus la matrice de covariance sera asymétrique (plus il y a une variance forte dans une direction), plus la PCA sera pertinente
cov = [[3, 2], 
       [2, 1]]

X = np.random.multivariate_normal(mean, cov, n_samples)

# On fait la PCA
pca = PCA(n_components=2)
pca.fit(X)
X_pca = pca.transform(X)

# On extrait les composantes principales
pc1 = pca.components_[0]  
pc2 = pca.components_[1]


# On va maintenant générer des hyperplans, pour voir comment ils coupent les données
# On va comparer deux méthodes:
# 1. Gaussian: u ~ N(0, I)
# 2. PCA 

def sample_isotropic(d=2):
    u = np.random.randn(d)
    u /= np.linalg.norm(u)
    return u

def sample_pca(d=2, lambda1_ratio=0.8, lambda2_ratio=0.5):
    if np.random.rand() < lambda1_ratio:
        # Slight random noise around PC1
        direction = pc1 + 0.1*np.random.randn(d)
        direction /= np.linalg.norm(direction)
        return direction
    else:
        if np.random.rand() < lambda2_ratio:
            # Slight random noise around PC1
            direction = pc2 + 0.1*np.random.randn(d)
            direction /= np.linalg.norm(direction)
            return direction
        else:
            u = np.random.randn(d)
            u /= np.linalg.norm(u)
            return u

# We want to visualize how each sampling method picks hyperplanes that partition the data.
# We'll draw a few hyperplanes from each distribution and see how they intersect with the data.

num_lines = 10

fig, ax = plt.subplots(1, 2, figsize=(12, 5))
ax[0].set_xlim(X[:,0].min(), X[:,0].max())
ax[0].set_ylim(X[:,1].min(), X[:,1].max())
ax[0].set_aspect('equal')

ax[0].scatter(X[:,0], X[:,1], alpha=0.5)
ax[0].set_title("Hyperplanes from Gaussian Sampling")

for _ in range(num_lines):
    u_iso = sample_isotropic(d=2)
    # Hyperplane: u_iso^T x = 0
    # Let's plot the line passing through origin perpendicular to u_iso.
    # Equation of line: u_iso[0]*x + u_iso[1]*y = 0
    # Solve for y: y = (-u_iso[0]/u_iso[1])*x if u_iso[1]!=0
    xs = np.linspace(X[:,0].min(), X[:,0].max(), 100)
    if np.abs(u_iso[1]) > 1e-8:
        ys = -(u_iso[0]/u_iso[1])*xs
    else:
        # vertical line if u_iso[1] ~ 0
        xs = np.ones(100)*(-0/u_iso[0]) 
        ys = np.linspace(X[:,1].min(), X[:,1].max(), 100)
    ax[0].plot(xs, ys, 'r-', alpha=0.5)

ax[1].scatter(X[:,0], X[:,1], alpha=0.5)
ax[1].set_title("Hyperplanes from PCA-based Sampling")
ax[1].set_xlim(X[:,0].min(), X[:,0].max())
ax[1].set_ylim(X[:,1].min(), X[:,1].max())
ax[1].set_aspect('equal')

for _ in range(num_lines):
    u_pca_samp = sample_pca(d=2)
    xs = np.linspace(X[:,0].min(), X[:,0].max(), 100)
    if np.abs(u_pca_samp[1]) > 1e-8:
        ys = -(u_pca_samp[0]/u_pca_samp[1])*xs
    else:
        xs = np.ones(100)*0
        ys = np.linspace(X[:,1].min(), X[:,1].max(), 100)
    ax[1].plot(xs, ys, 'g-', alpha=0.5)

plt.tight_layout()
plt.show()

# Show how PCA "converges" faster in terms of explained variance:
# We simulate a scenario where we pick directions incrementally from each method 
# and compute how much variance they "explain" of the data in that direction.

def explained_variance_in_direction(X, u):
    # Project data on u and compute variance
    proj = X.dot(u)
    return np.var(proj)

num_samples_for_convergence = 100
explained_var_iso = []
explained_var_pca_samp = []
for k in range(1, num_samples_for_convergence+1):
    # Collect k directions and measure total explained variance
    # For isotropic:
    directions_iso = [sample_isotropic(d=2) for _ in range(k)]
    directions_pca_ = [sample_pca(d=2) for _ in range(k)]
    
    # Compute sum of variances along these directions
    # (This is a simplistic measure; real "explained variance" would be orthogonal directions)
    var_iso = sum(explained_variance_in_direction(X, u_dir) for u_dir in directions_iso)
    var_pca_ = sum(explained_variance_in_direction(X, u_dir) for u_dir in directions_pca_)
    explained_var_iso.append(var_iso)
    explained_var_pca_samp.append(var_pca_)

# Plot the convergence (in terms of explained variance)
plt.figure(figsize=(6,4))
plt.plot(range(1, num_samples_for_convergence+1), explained_var_iso, label='Gaussian')
plt.plot(range(1, num_samples_for_convergence+1), explained_var_pca_samp, label='PCA-based')
plt.xlabel("Number of sampled directions")
plt.ylabel("Sum of Variance Explained")
plt.title("Variance Explained vs. Number of Directions")
plt.legend()
plt.show()


# On va maintenant comparer l'asymétrie des données, et l'efficacité de la PCA
from sklearn.linear_model import LinearRegression


asymetries = np.linspace(1,12,200)
coeff_efficacy = [0 for _ in range(len(asymetries))]
number_of_attempts = 5 # On va faire plusieurs essais pour chaque asymétrie, afin de lisser les résultats

for idn,i in enumerate(asymetries):
    cov = [[i, 0], 
           [0, 1]]
    for _ in range(number_of_attempts):
        X = np.random.multivariate_normal(mean, cov, n_samples)
        pca = PCA(n_components=2)
        pca.fit(X)
        X_pca = pca.transform(X)
        
        # On extrait les composantes principales
        pc1 = pca.components_[0]  
        pc2 = pca.components_[1]
        
        
        # On va calculer le coefficient d'efficacité de la PCA
        
        num_samples_for_convergence = 50
        
        explained_var_iso = []
        explained_var_pca_samp = []
        for k in range(1, num_samples_for_convergence+1):
            # Collect k directions and measure total explained variance
            # For isotropic:
            directions_iso = [sample_isotropic(d=2) for _ in range(k)]
            directions_pca_ = [sample_pca(d=2) for _ in range(k)]
            
            # Compute sum of variances along these directions
            # (This is a simplistic measure; real "explained variance" would be orthogonal directions)
            var_iso = sum(explained_variance_in_direction(X, u_dir) for u_dir in directions_iso)
            var_pca_ = sum(explained_variance_in_direction(X, u_dir) for u_dir in directions_pca_)
            explained_var_iso.append(var_iso)
            explained_var_pca_samp.append(var_pca_)
            
        # On va maintenant faire une régression linéaire pour la loi gaussienne
        # Create a linear regression model
        model = LinearRegression()

        # Fit the model to the data
        model.fit(np.array(range(num_samples_for_convergence)).reshape(-1, 1), np.array(explained_var_iso))

        # Get the slope (coefficient)
        slope1 = model.coef_[0]
        
        # On va maintenant faire une régression linéaire pour la loi PCA
        model = LinearRegression()
        model.fit(np.array(range(num_samples_for_convergence)).reshape(-1, 1), explained_var_pca_samp)
        slope2 = model.coef_[0]
        if coeff_efficacy[idn] == 0:
            coeff_efficacy[idn] = slope2/slope1
        else:
            coeff_efficacy[idn] += slope2/slope1
        #coeff_efficacy.append(slope2/slope1)
        print(f"Asymétrie: {i}, Coefficient d'efficacité: {slope2/slope1}")
    
coeff_efficacy = [x/number_of_attempts for x in coeff_efficacy] # On re-normalise

# On va maintenant afficher le coefficient d'efficacité en fonction de l'asymétrie
plt.figure(figsize=(6,4))
plt.plot(asymetries, coeff_efficacy)
plt.xlim(asymetries[0], asymetries[-1])
plt.xlabel("Asymétrie")
plt.ylabel("Coefficient d'efficacité")
plt.title("Coefficient d'efficacité de la PCA en fonction de l'asymétrie")
plt.show()
    
    
    
  