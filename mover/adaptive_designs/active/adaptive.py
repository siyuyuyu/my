import numpy as np
import matplotlib.pyplot as plt

# Binh-Korn test function
def binh_korn(x, y):
    f1 = 4 * x**2 + 4 * y**2
    f2 = (x - 5)**2 + (y - 5)**2
    return np.array([f1, f2])

# Constraint functions
def g1(x, y):
    return (x - 5)**2 + y**2 - 25

def g2(x, y):
    return (x - 8)**2 + (y + 3)**2 - 7.7

# Multi-objective design algorithm
def DESIGN(Xtrain, Ytrain, Xsearch):
    # Build surrogate model
    f = surrogate_model(Xtrain, Ytrain)
    
    # Initialize variables
    n_search = len(Xsearch)
    E_I = np.zeros(n_search)
    I_max = np.zeros(n_search)
    I_cent = np.zeros(n_search)
    P_I = np.zeros(n_search)
    
    # Iterate over unmeasured materials
    for i in range(n_search):
        # Bootstrap the predictions
        g = bootstrap(f, Xsearch[i])
        
        # Calculate mean and uncertainty
        mu = np.mean(g)
        sigma = np.sqrt(np.mean((g - mu) ** 2))
        
        # Calculate probability of improvement
        z = (np.min(Ytrain, axis=0) - mu) / sigma
        P_I[i] = norm.cdf(z)
        
        # Calculate improvement
        I_max[i] = np.max(np.abs(mu - Ytrain), axis=0)
        I_cent[i] = np.sum(np.abs(mu - Ytrain), axis=0) / len(Ytrain)
        
        # Calculate expected improvement
        E_I[i] = P_I[i] * I_max[i] # or I_cent[i]
    
    # Select material with highest expected improvement
    x_selected = Xsearch[np.argmax(E_I)]
    
    return x_selected

# Surrogate model
def surrogate_model(Xtrain, Ytrain):
    # Build surrogate model
    # Here, we assume a simple linear regression model
    return lambda X: np.dot(X, np.linalg.lstsq(Xtrain, Ytrain, rcond=None)[0])

# Bootstrap function
def bootstrap(f, X):
    # Bootstrap the predictions
    # Here, we assume a simple Gaussian noise model
    n_samples = 100
    noise = np.random.normal(loc=0.0, scale=0.01, size=(n_samples, 2))
    return f(X) + noise

# Test the algorithm on the Binh-Korn test function
Xtrain = np.random.uniform(low=[0, 0], high=[5, 3], size=(100, 2))
Ytrain = np.array([binh_korn(x, y) for x, y in Xtrain])

# Validation dataset
Xvalid = np.random.uniform(low=[0, 0], high=[5, 3], size=(200, 2))

# Apply constraint functions
Xvalid = Xvalid[(g1(Xvalid[:, 0], Xvalid[:, 1]) <= 0) & (g2(Xvalid[:, 0], Xvalid[:, 1]) >= 0), :]

# Find Pareto front using brute force search
Fvalid = np.array([binh_korn(x, y) for x, y in Xvalid])
PF = []
for i in range(len(Fvalid)):
    is_pareto = True
    for j in range(len(Fvalid)):
        if i != j and (Fvalid[j, 0] <= Fvalid[i, 0] or Fvalid[j, 1] <= Fvalid[i, 1]):
            is_pareto = False
            break
    if is_pareto:
        PF.append(Fvalid[i])

PF = np.array(PF)


plt.scatter(PF[:, 0], PF[:, 1])
plt.xlabel('f1(x, y)')
plt.ylabel('f2(x, y)')
plt.title('Pareto Front')
plt.show()
