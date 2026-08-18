import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

v1 = np.random.rand(4)
v2 = np.random.rand(4)

A1 = v1.reshape(2,2)
A2 = v2.reshape(2,2)

# Vector distances
dist_L1 = np.sum(np.abs(v1 - v2))
dist_L2 = np.sqrt(np.sum((v1 - v2)**2))

# Matrix distances
dist_mat1 = np.max(np.sum(np.abs(A1 - A2), axis=0))   # 1-norm
dist_fro = np.sqrt(np.sum((A1 - A2)**2))              # Frobenius


# Step 4: Simple 2D visualization (fix first two components)
xr = v1
x = np.linspace(-2,2,200)
y = np.linspace(-2,2,200)
X, Y = np.meshgrid(x,y)

# L1 ball
L1_mask = np.abs(X) + np.abs(Y) <= 1
# L2 ball
L2_mask = np.sqrt((X)**2 + (Y)**2) <= 1

plt.figure(figsize=(8,4))
plt.subplot(1,2,1)
plt.contourf(X, Y, L1_mask, cmap='pink')
plt.title("L1 unit ball section")
plt.subplot(1,2,2)
plt.contourf(X, Y, L2_mask, cmap='Blues')
plt.title("L2 unit ball section")
plt.show()
