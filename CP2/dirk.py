

import numpy as np
import time


class DIRK2:

    def __init__(self, model, solver_type='fixed_point'):
        """
        model: EvaporationPondModel instance
        solver_type: 'fixed_point' or 'newton'
        """
        self.model = model
        self.solver_type = solver_type

        # SDIRK2 coefficients
        self.gamma = 1.0 - 1.0 / np.sqrt(2.0)

        self.stats = {
            'total_iterations': 0,
            'total_steps': 0,
            'solve_time': 0.0
        }

    def solve_stage_fixed_point(self, y_n, t, h, a, b, max_iter=100, tol=1e-6):
        """
        Solve implicit stage equation using fixed-point iteration
        k = f(t + c*h, y_n + h*(a*k + b))
        where b is known from previous stages
        """
        k_old = self.model.ode_system(t, y_n)

        for iteration in range(max_iter):
            y_stage = y_n + h * (a * k_old + b)
            k_new = self.model.ode_system(t, y_stage)

            error = np.linalg.norm(k_new - k_old) / (np.linalg.norm(k_new) + 1e-10)

            if error < tol:
                self.stats['total_iterations'] += iteration + 1
                return k_new, True, iteration + 1

            k_old = k_new.copy()

        self.stats['total_iterations'] += max_iter
        return k_new, False, max_iter

    def solve_stage_newton(self, y_n, t, h, a, b, max_iter=20, tol=1e-6):
        """
        Solve implicit stage equation using Newton's method
        """
        k_old = self.model.ode_system(t, y_n)
        epsilon = 1e-7

        for iteration in range(max_iter):
            y_stage = y_n + h * (a * k_old + b)
            G = k_old - self.model.ode_system(t, y_stage)

            if np.linalg.norm(G) < tol:
                self.stats['total_iterations'] += iteration + 1
                return k_old, True, iteration + 1

            # Numerical Jacobian
            n = len(k_old)
            J = np.eye(n)
            f0 = self.model.ode_system(t, y_stage)

            for i in range(n):
                y_perturb = y_stage.copy()
                y_perturb[i] += epsilon
                f_perturb = self.model.ode_system(t, y_perturb)
                J[:, i] += h * a * (f_perturb - f0) / epsilon

            try:
                delta = np.linalg.solve(J, G)
                k_old = k_old - delta
            except np.linalg.LinAlgError:
                # Fall back to fixed point
                return self.solve_stage_fixed_point(y_n, t, h, a, b, max_iter, tol)

        self.stats['total_iterations'] += max_iter
        return k_old, False, max_iter

    def solve(self, t_span, y0, h):
        """
        Solve ODE using SDIRK2
        """
        t0, tf = t_span
        t = t0
        y = y0.copy()

        t_values = [t]
        y_values = [y.copy()]

        convergence_failures = 0
        iterations_per_step = []

        start_time = time.time()

        while t < tf:
            if t + h > tf:
                h = tf - t

            # Stage 1: k1 with c1 = γ
            t1 = t + self.gamma * h
            if self.solver_type == 'fixed_point':
                k1, conv1, iters1 = self.solve_stage_fixed_point(
                    y, t1, h, self.gamma, 0.0
                )
            else:
                k1, conv1, iters1 = self.solve_stage_newton(
                    y, t1, h, self.gamma, 0.0
                )

            if not conv1:
                convergence_failures += 1

            # Stage 2: k2 with c2 = 1
            t2 = t + h
            b2 = (1.0 - self.gamma) * k1
            if self.solver_type == 'fixed_point':
                k2, conv2, iters2 = self.solve_stage_fixed_point(
                    y, t2, h, self.gamma, b2
                )
            else:
                k2, conv2, iters2 = self.solve_stage_newton(
                    y, t2, h, self.gamma, b2
                )

            if not conv2:
                convergence_failures += 1

            # Update: y_{n+1} = y_n + h * ((1-γ)*k1 + γ*k2)
            y_next = y + h * ((1.0 - self.gamma) * k1 + self.gamma * k2)

            total_iters = iters1 + iters2
            iterations_per_step.append(total_iters)

            t = t + h
            y = y_next

            t_values.append(t)
            y_values.append(y.copy())

            self.stats['total_steps'] += 1

        self.stats['solve_time'] = time.time() - start_time
        self.stats['avg_iterations'] = np.mean(iterations_per_step)
        self.stats['convergence_failures'] = convergence_failures

        return np.array(t_values), np.array(y_values), self.stats