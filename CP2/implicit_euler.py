
import numpy as np
import time


class ImplicitEuler:

    def __init__(self, model, solver_type='fixed_point'):

        self.model = model
        self.solver_type = solver_type
        self.stats = {
            'total_iterations': 0,
            'total_steps': 0,
            'solve_time': 0.0
        }

    def fixed_point_iteration(self, y_n, t_next, h, max_iter=100, tol=1e-6):

        y_old = y_n.copy()

        for iteration in range(max_iter):
            # Fixed point iteration: y^(k+1) = y_n + h * f(t_{n+1}, y^(k))
            y_new = y_n + h * self.model.ode_system(t_next, y_old)

            # Check convergence
            error = np.linalg.norm(y_new - y_old) / (np.linalg.norm(y_new) + 1e-10)

            if error < tol:
                self.stats['total_iterations'] += iteration + 1
                return y_new, True, iteration + 1

            y_old = y_new.copy()

        # Did not converge
        self.stats['total_iterations'] += max_iter
        return y_new, False, max_iter

    def newton_method(self, y_n, t_next, h, max_iter=20, tol=1e-6):
   
        y_old = y_n.copy()
        epsilon = 1e-7  # For numerical Jacobian

        for iteration in range(max_iter):
            # Compute G(y)
            G = y_old - y_n - h * self.model.ode_system(t_next, y_old)

            # Check convergence
            if np.linalg.norm(G) < tol:
                self.stats['total_iterations'] += iteration + 1
                return y_old, True, iteration + 1

            # Compute Jacobian numerically: J = I - h * df/dy
            n = len(y_old)
            J = np.eye(n)
            f0 = self.model.ode_system(t_next, y_old)

            for i in range(n):
                y_perturb = y_old.copy()
                y_perturb[i] += epsilon
                f_perturb = self.model.ode_system(t_next, y_perturb)
                J[:, i] -= h * (f_perturb - f0) / epsilon

            # Newton step: y_new = y_old - J^{-1} * G
            try:
                delta = np.linalg.solve(J, G)
                y_old = y_old - delta
            except np.linalg.LinAlgError:
                # Singular matrix, fall back to fixed point
                return self.fixed_point_iteration(y_n, t_next, h, max_iter, tol)

        # Did not converge
        self.stats['total_iterations'] += max_iter
        return y_old, False, max_iter

    def solve(self, t_span, y0, h):

        t0, tf = t_span
        t = t0
        y = y0.copy()

        # Storage
        t_values = [t]
        y_values = [y.copy()]

        # Statistics
        convergence_failures = 0
        iterations_per_step = []

        start_time = time.time()

        while t < tf:
            # Adjust last step
            if t + h > tf:
                h = tf - t

            t_next = t + h

            # Solve implicit equation
            if self.solver_type == 'fixed_point':
                y_next, converged, iters = self.fixed_point_iteration(y, t_next, h)
            else:  # newton
                y_next, converged, iters = self.newton_method(y, t_next, h)

            if not converged:
                convergence_failures += 1

            iterations_per_step.append(iters)

            # Update
            t = t_next
            y = y_next

            t_values.append(t)
            y_values.append(y.copy())

            self.stats['total_steps'] += 1

        self.stats['solve_time'] = time.time() - start_time
        self.stats['avg_iterations'] = np.mean(iterations_per_step)
        self.stats['convergence_failures'] = convergence_failures

        return np.array(t_values), np.array(y_values), self.stats