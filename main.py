# ==== main.py ====
import numpy as np
import matplotlib.pyplot as plt

def analytical_solution(N):
    """Return analytical solution on an N x N grid (including boundaries)."""
    x = np.linspace(0, 1, N)
    y = np.linspace(0, 1, N)
    X, Y = np.meshgrid(x, y, indexing='ij')
    return np.sin(np.pi * X) * np.sinh(np.pi * Y) / np.sinh(np.pi)

def solve_laplace(N, max_iter=10000, tol=1e-8, omega=1.5):
    """Solve Laplace equation on unit square with Dirichlet BCs using SOR.
    Returns solution array of shape (N, N) including boundaries.
    """
    h = 1.0 / (N - 1)
    u = np.zeros((N, N), dtype=np.float64)
    # Apply Dirichlet BCs
    x = np.linspace(0, 1, N)
    u[:, 0] = np.sin(np.pi * x)  # y=0 bottom boundary
    # other boundaries are already zero
    for it in range(max_iter):
        max_diff = 0.0
        for i in range(1, N - 1):
            for j in range(1, N - 1):
                old = u[i, j]
                new = 0.25 * (u[i+1, j] + u[i-1, j] + u[i, j+1] + u[i, j-1])
                u[i, j] = old + omega * (new - old)
                diff = abs(u[i, j] - old)
                if diff > max_diff:
                    max_diff = diff
        if max_diff < tol:
            break
    return u

def plot_solution_comparison(u_num, u_exact, filename):
    plt.figure(figsize=(6,5))
    X = np.linspace(0,1,u_num.shape[0])
    Y = np.linspace(0,1,u_num.shape[1])
    Xg, Yg = np.meshgrid(X, Y, indexing='ij')
    cp = plt.contourf(Xg, Yg, u_num, levels=50, cmap='viridis')
    plt.colorbar(cp, label='Numerical solution')
    # overlay analytical contours in black
    plt.contour(Xg, Yg, u_exact, colors='k', linewidths=0.5)
    plt.title('Laplace solution: numerical (color) vs analytical (black contours)')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

def plot_convergence(h_vals, err_vals, filename):
    plt.figure(figsize=(6,5))
    plt.loglog(h_vals, err_vals, 'o-', label='L2 error')
    # reference slope 2 line
    h_ref = np.array([h_vals[0], h_vals[-1]])
    err_ref = err_vals[0] * (h_ref / h_vals[0])**2
    plt.loglog(h_ref, err_ref, '--', label='O(h^2)')
    plt.xlabel('Grid spacing h')
    plt.ylabel('L2 error')
    plt.title('Grid convergence study')
    plt.legend()
    plt.grid(True, which='both', ls='--')
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

def compute_l2_error(u_num, u_exact):
    diff = u_num - u_exact
    return np.sqrt(np.mean(diff**2))

def main():
    # Experiment 1: analytical benchmark
    N_exp1 = 81  # a reasonably fine grid for visualization
    u_num = solve_laplace(N_exp1)
    u_exact = analytical_solution(N_exp1)
    plot_solution_comparison(u_num, u_exact, 'laplace_solution_comparison.png')

    # Experiment 2: grid convergence study
    Ns = [20, 40, 80, 160]
    h_vals = []
    err_vals = []
    for N in Ns:
        u_num = solve_laplace(N)
        u_exact = analytical_solution(N)
        err = compute_l2_error(u_num, u_exact)
        h = 1.0 / (N - 1)
        h_vals.append(h)
        err_vals.append(err)
    plot_convergence(h_vals, err_vals, 'error_convergence_loglog.png')

    # Primary numeric answer: L2 error for the finest grid (N=160)
    answer = err_vals[-1]
    print('Answer:', answer)

if __name__ == '__main__':
    main()

