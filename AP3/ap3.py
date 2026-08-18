

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os



# 1D function
def f(x):
    return np.sin(x)

def f_prime_exact(x):
    return np.cos(x)

# 2D function
def F(x, y):
    return np.exp(-(x - 1.0)**2 - (y + 0.5)**2)

def Fx_exact(x, y):
    return -2 * (x - 1.0) * F(x, y)

def Fy_exact(x, y):
    return -2 * (y + 0.5) * F(x, y)


#Difference Formulas


def forward_diff_1d(fun, x, h):
    return (fun(x + h) - fun(x)) / h

def central_diff_1d(fun, x, h):
    return (fun(x + h) - fun(x - h)) / (2 * h)

def five_point_diff_1d(fun, x, h):
    return (-fun(x + 2*h) + 8*fun(x + h) - 8*fun(x - h) + fun(x - 2*h)) / (12*h)

#derivatives
def central_partial(fun, x, y, h, var='x'):
    if var == 'x':
        return (fun(x + h, y) - fun(x - h, y)) / (2 * h)
    else:
        return (fun(x, y + h) - fun(x, y - h)) / (2 * h)

def five_point_partial(fun, x, y, h, var='x'):
    if var == 'x':
        return (-fun(x + 2*h, y) + 8*fun(x + h, y) - 8*fun(x - h, y) + fun(x - 2*h, y)) / (12 * h)
    else:
        return (-fun(x, y + 2*h) + 8*fun(x, y + h) - 8*fun(x, y - h) + fun(x, y - 2*h)) / (12 * h)



x0 = 1.0
y0 = -0.5
z0 = F(x0, y0)
hs = np.logspace(-1, -8, 50)

output_dir = "normals_tangents_results"
os.makedirs(output_dir, exist_ok=True)



rows_1d = []
for h in hs:
    fd = forward_diff_1d(f, x0, h)
    cd = central_diff_1d(f, x0, h)
    fp = five_point_diff_1d(f, x0, h)
    exact = f_prime_exact(x0)
    rows_1d.append((h, fd, cd, fp, exact, abs(fd-exact), abs(cd-exact), abs(fp-exact)))

df1 = pd.DataFrame(rows_1d, columns=[
    "h", "forward", "central", "five_point", "exact",
    "err_forward", "err_central", "err_5point"
])
df1.to_csv(os.path.join(output_dir, "errors_1d.csv"), index=False)

# Error vs h


plt.figure(figsize=(6,4))
plt.loglog(df1["h"], df1["err_forward"], "o-", label="Forward Difference")
plt.loglog(df1["h"], df1["err_central"], "o-", label="Central Difference")
plt.loglog(df1["h"], df1["err_5point"], "o-", label="5-Point Difference")
plt.gca().invert_xaxis()
plt.xlabel("Step size h")
plt.ylabel("Absolute Error")
plt.title("1D Derivative Approximation Error")
plt.legend()
plt.grid(True, which="both", ls="--", lw=0.4)
plt.savefig(os.path.join(output_dir, "error_vs_h_1d.png"), dpi=200)
plt.close()

#  Tangent Line


x_vals = np.linspace(x0 - 1, x0 + 1, 400)
y_vals = f(x_vals)
slope_exact = f_prime_exact(x0)
tangent_exact = f(x0) + slope_exact * (x_vals - x0)

h_used = 1e-4
slope_fd = central_diff_1d(f, x0, h_used)
tangent_fd = f(x0) + slope_fd * (x_vals - x0)

plt.figure(figsize=(6,4))
plt.plot(x_vals, y_vals, label="y = sin(x)")
plt.plot(x_vals, tangent_exact, "--", label=f"Exact Tangent (slope={slope_exact:.4f})")
plt.plot(x_vals, tangent_fd, ":", label=f"FD Tangent (h={h_used}, slope={slope_fd:.4f})")
plt.scatter([x0], [f(x0)], color="red", zorder=5, label="Point")
plt.title("Tangent Line at x₀ = 1.0")
plt.xlabel("x"); plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(output_dir, "tangent_line.png"), dpi=200)
plt.close()

# 2D Derivative


rows_2d = []
for h in hs:
    cx = central_partial(F, x0, y0, h, "x")
    cy = central_partial(F, x0, y0, h, "y")
    fx5 = five_point_partial(F, x0, y0, h, "x")
    fy5 = five_point_partial(F, x0, y0, h, "y")
    exact_x = Fx_exact(x0, y0)
    exact_y = Fy_exact(x0, y0)
    rows_2d.append((h, cx, cy, fx5, fy5, exact_x, exact_y,
                    abs(cx-exact_x), abs(cy-exact_y),
                    abs(fx5-exact_x), abs(fy5-exact_y)))

df2 = pd.DataFrame(rows_2d, columns=[
    "h","central_x","central_y","five_x","five_y",
    "exact_x","exact_y","err_cx","err_cy","err_5x","err_5y"
])
df2.to_csv(os.path.join(output_dir, "errors_2d.csv"), index=False)


# Error vs h


plt.figure(figsize=(6,4))
plt.loglog(df2["h"], df2["err_cx"], "o-", label="Central Fx")
plt.loglog(df2["h"], df2["err_cy"], "o-", label="Central Fy")
plt.loglog(df2["h"], df2["err_5x"], "o-", label="5-Point Fx")
plt.loglog(df2["h"], df2["err_5y"], "o-", label="5-Point Fy")
plt.gca().invert_xaxis()
plt.xlabel("Step size h")
plt.ylabel("Absolute Error")
plt.title("2D Partial Derivative Approximation Error")
plt.legend()
plt.grid(True, which="both", ls="--", lw=0.4)
plt.savefig(os.path.join(output_dir, "error_vs_h_2d.png"), dpi=200)
plt.close()

#Tangent Plane


fig = plt.figure(figsize=(7,5))
ax = fig.add_subplot(111, projection="3d")

xs = np.linspace(x0 - 0.8, x0 + 0.8, 40)
ys = np.linspace(y0 - 0.8, y0 + 0.8, 40)
XS, YS = np.meshgrid(xs, ys)
ZS = F(XS, YS)

Fx = Fx_exact(x0, y0)
Fy = Fy_exact(x0, y0)
plane = z0 + Fx * (XS - x0) + Fy * (YS - y0)

ax.plot_surface(XS, YS, ZS, alpha=0.8, cmap="viridis")
ax.plot_surface(XS, YS, plane, alpha=0.5, color="orange")
ax.scatter(x0, y0, z0, color="red", s=50, label="Point (x₀,y₀,z₀)")
ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
ax.set_title("Surface and Tangent Plane")
plt.legend()
plt.savefig(os.path.join(output_dir, "tangent_plane.png"), dpi=200)
plt.close()


print(" All computations complete.")
print(f"Results saved in folder: {output_dir}")
print("Generated files:")
for file in os.listdir(output_dir):
    print(" -", file)
