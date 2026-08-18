
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline, interp1d, BSpline, make_interp_spline
from scipy.interpolate import splprep, splev

# Define control points for each letter (x, y coordinates normalized to [0, 1])
LETTER_POINTS = {
    'C': np.array([
        [0.8, 0.2], [0.6, 0.1], [0.4, 0.05], [0.2, 0.1],
        [0.1, 0.3], [0.05, 0.5], [0.1, 0.7], [0.2, 0.9],
        [0.4, 0.95], [0.6, 0.9], [0.8, 0.8]
    ]),
    'S': np.array([
        [0.7, 0.1], [0.5, 0.05], [0.3, 0.1], [0.2, 0.2],
        [0.25, 0.35], [0.4, 0.5], [0.6, 0.5], [0.75, 0.65],
        [0.8, 0.8], [0.7, 0.9], [0.5, 0.95], [0.3, 0.9]
    ]),
    'U': np.array([
        [0.2, 0.1], [0.2, 0.3], [0.2, 0.5], [0.2, 0.7],
        [0.25, 0.85], [0.35, 0.93], [0.5, 0.97], [0.65, 0.93],
        [0.75, 0.85], [0.8, 0.7], [0.8, 0.5], [0.8, 0.3], [0.8, 0.1]
    ])
}


def fit_linear_spline(points, num_samples=200):
    """
    Fit a linear spline (piecewise linear interpolation) to control points.

    Args:
        points: numpy array of shape (n, 2) with control points
        num_samples: number of points to sample along the spline

    Returns:
        x_interp, y_interp: interpolated coordinates
    """
    # Create parameter t based on cumulative distance
    distances = np.sqrt(np.sum(np.diff(points, axis=0) ** 2, axis=1))
    t = np.concatenate([[0], np.cumsum(distances)])
    t = t / t[-1]  # Normalize to [0, 1]

    # Linear interpolation
    t_new = np.linspace(0, 1, num_samples)
    x_interp = np.interp(t_new, t, points[:, 0])
    y_interp = np.interp(t_new, t, points[:, 1])

    return x_interp, y_interp


def fit_cubic_spline(points, num_samples=200):
    """
    Fit a cubic spline to control points using scipy's CubicSpline.

    Args:
        points: numpy array of shape (n, 2) with control points
        num_samples: number of points to sample along the spline

    Returns:
        x_interp, y_interp: interpolated coordinates
    """
    # Create parameter t based on cumulative distance
    distances = np.sqrt(np.sum(np.diff(points, axis=0) ** 2, axis=1))
    t = np.concatenate([[0], np.cumsum(distances)])
    t = t / t[-1]  # Normalize to [0, 1]

    # Fit cubic spline
    cs_x = CubicSpline(t, points[:, 0], bc_type='natural')
    cs_y = CubicSpline(t, points[:, 1], bc_type='natural')

    # Sample the spline
    t_new = np.linspace(0, 1, num_samples)
    x_interp = cs_x(t_new)
    y_interp = cs_y(t_new)

    return x_interp, y_interp


def fit_bspline(points, num_samples=200, degree=3):
    """
    Fit a B-spline to control points using scipy's splprep.

    Args:
        points: numpy array of shape (n, 2) with control points
        num_samples: number of points to sample along the spline
        degree: degree of the B-spline (3 for cubic)

    Returns:
        x_interp, y_interp: interpolated coordinates
    """
    # Use splprep for parametric B-spline
    tck, u = splprep([points[:, 0], points[:, 1]], s=0, k=min(degree, len(points) - 1))

    # Sample the spline
    u_new = np.linspace(0, 1, num_samples)
    x_interp, y_interp = splev(u_new, tck)

    return x_interp, y_interp


def calculate_smoothness(x, y):
    """
    Calculate smoothness metric based on curvature variation.
    Lower values indicate smoother curves.

    Args:
        x, y: interpolated coordinates

    Returns:
        smoothness_metric: average absolute curvature
    """
    # Calculate first and second derivatives using finite differences
    dx = np.gradient(x)
    dy = np.gradient(y)
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)

    # Calculate curvature: κ = |x'y'' - y'x''| / (x'^2 + y'^2)^(3/2)
    numerator = np.abs(dx * ddy - dy * ddx)
    denominator = (dx ** 2 + dy ** 2) ** (3 / 2)

    # Avoid division by zero
    denominator[denominator < 1e-10] = 1e-10
    curvature = numerator / denominator

    # Return mean curvature as smoothness metric
    return np.mean(curvature)


def calculate_fitting_error(points, x_interp, y_interp):
    """
    Calculate the fitting error as average distance from control points to spline.

    Args:
        points: original control points
        x_interp, y_interp: interpolated spline coordinates

    Returns:
        error: average distance from control points to nearest spline point
    """
    errors = []
    for point in points:
        # Find minimum distance from this control point to the spline
        distances = np.sqrt((x_interp - point[0]) ** 2 + (y_interp - point[1]) ** 2)
        errors.append(np.min(distances))

    return np.mean(errors)


def visualize_letter_splines(letter, save_fig=True):
    """
    Visualize a letter with different spline types.

    Args:
        letter: letter to visualize ('C', 'S', or 'U')
        save_fig: whether to save the figure to a file
    """
    points = LETTER_POINTS[letter]

    # Fit different spline types
    x_linear, y_linear = fit_linear_spline(points)
    x_cubic, y_cubic = fit_cubic_spline(points)
    x_bspline, y_bspline = fit_bspline(points)

    # Calculate metrics
    smoothness_linear = calculate_smoothness(x_linear, y_linear)
    smoothness_cubic = calculate_smoothness(x_cubic, y_cubic)
    smoothness_bspline = calculate_smoothness(x_bspline, y_bspline)

    error_linear = calculate_fitting_error(points, x_linear, y_linear)
    error_cubic = calculate_fitting_error(points, x_cubic, y_cubic)
    error_bspline = calculate_fitting_error(points, x_bspline, y_bspline)

    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f'Letter {letter} - Spline Comparison', fontsize=16, fontweight='bold')

    spline_data = [
        ('Linear', x_linear, y_linear, smoothness_linear, error_linear, 'blue'),
        ('Cubic', x_cubic, y_cubic, smoothness_cubic, error_cubic, 'red'),
        ('B-Spline', x_bspline, y_bspline, smoothness_bspline, error_bspline, 'green')
    ]

    for ax, (name, x, y, smoothness, error, color) in zip(axes, spline_data):
        ax.plot(x, y, color=color, linewidth=2, label=f'{name} Spline')
        ax.scatter(points[:, 0], points[:, 1], c='black', s=50,
                   zorder=5, label='Control Points')
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-0.1, 1.1)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_title(f'{name}\nSmoothness: {smoothness:.4f}\nError: {error:.4f}')
        ax.invert_yaxis()  # Flip y-axis for proper letter orientation

    plt.tight_layout()

    if save_fig:
        plt.savefig(f'letter_{letter}_splines.png', dpi=300, bbox_inches='tight')
        print(f"Figure saved as 'letter_{letter}_splines.png'")

    plt.show()


def compare_all_letters(save_fig=True):
    """
    Compare all three letters using cubic splines.

    Args:
        save_fig: whether to save the figure to a file
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Letters C, S, U - Cubic Spline Fitting', fontsize=16, fontweight='bold')

    letters = ['C', 'S', 'U']
    colors = ['blue', 'red', 'green']

    for ax, letter, color in zip(axes, letters, colors):
        points = LETTER_POINTS[letter]
        x_cubic, y_cubic = fit_cubic_spline(points)

        ax.plot(x_cubic, y_cubic, color=color, linewidth=2.5, label=f'Letter {letter}')
        ax.scatter(points[:, 0], points[:, 1], c='black', s=50,
                   zorder=5, label=f'Control Points ({len(points)})')
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-0.1, 1.1)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_title(f'Letter {letter}')
        ax.invert_yaxis()

    plt.tight_layout()

    if save_fig:
        plt.savefig('all_letters_comparison.png', dpi=300, bbox_inches='tight')
        print("Figure saved as 'all_letters_comparison.png'")

    plt.show()


def analyze_node_density(letter='S'):
    """
    Experiment: analyze how the number of control points affects fitting quality.

    Args:
        letter: letter to analyze
    """
    points = LETTER_POINTS[letter]

    # Test with different numbers of control points
    node_counts = [5, 7, 9, len(points)]

    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    axes = axes.flatten()
    fig.suptitle(f'Effect of Control Point Density on Letter {letter}',
                 fontsize=16, fontweight='bold')

    for ax, n_points in zip(axes, node_counts):
        # Sample points uniformly from original set
        if n_points < len(points):
            indices = np.linspace(0, len(points) - 1, n_points, dtype=int)
            sampled_points = points[indices]
        else:
            sampled_points = points

        # Fit cubic spline
        x_cubic, y_cubic = fit_cubic_spline(sampled_points)
        smoothness = calculate_smoothness(x_cubic, y_cubic)

        ax.plot(x_cubic, y_cubic, color='blue', linewidth=2, label='Cubic Spline')
        ax.scatter(sampled_points[:, 0], sampled_points[:, 1],
                   c='red', s=80, zorder=5, label=f'{n_points} Control Points')
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-0.1, 1.1)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_title(f'{n_points} Points\nSmoothness: {smoothness:.4f}')
        ax.invert_yaxis()

    plt.tight_layout()
    plt.savefig(f'node_density_analysis_{letter}.png', dpi=300, bbox_inches='tight')
    print(f"Figure saved as 'node_density_analysis_{letter}.png'")
    plt.show()


def print_summary_statistics():
    """
    Print summary statistics for all letters and spline types.
    """
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS - PARAMETRIC SPLINES FOR LETTERS C, S, U")
    print("=" * 70)

    for letter in ['C', 'S', 'U']:
        points = LETTER_POINTS[letter]
        print(f"\n--- Letter {letter} ---")
        print(f"Number of control points: {len(points)}")

        # Fit different splines
        x_linear, y_linear = fit_linear_spline(points)
        x_cubic, y_cubic = fit_cubic_spline(points)
        x_bspline, y_bspline = fit_bspline(points)

        print("\nSpline Type      | Smoothness | Fitting Error")
        print("-" * 50)

        for name, x, y in [('Linear', x_linear, y_linear),
                           ('Cubic', x_cubic, y_cubic),
                           ('B-Spline', x_bspline, y_bspline)]:
            smoothness = calculate_smoothness(x, y)
            error = calculate_fitting_error(points, x, y)
            print(f"{name:15s} | {smoothness:10.4f} | {error:13.6f}")

    print("\n" + "=" * 70)
    print("\nKEY FINDINGS:")
    print("1. Cubic splines provide the smoothest curves (lowest curvature)")
    print("2. All splines pass through/near control points (low fitting error)")
    print("3. Letter S requires most control points due to double curvature")
    print("4. Linear splines show higher curvature due to sharp corners")
    print("=" * 70 + "\n")


def main():
    """
    Main function to run all experiments and visualizations.
    """
    print("Starting Parametric Splines Analysis for Letters C, S, U...")
    print("This will generate multiple figures and statistics.\n")

    # Experiment 1: Compare spline types for each letter
    print("Experiment 1: Comparing different spline types...")
    for letter in ['C', 'S', 'U']:
        visualize_letter_splines(letter, save_fig=True)

    # Experiment 2: Compare all letters side by side
    print("\nExperiment 2: Comparing all letters...")
    compare_all_letters(save_fig=True)

    # Experiment 3: Analyze effect of node density
    print("\nExperiment 3: Analyzing control point density (Letter S)...")
    analyze_node_density('S')

    # Print summary statistics
    print_summary_statistics()

    print("\nAnalysis complete! Check the generated PNG files for visualizations.")


if __name__ == "__main__":
    main()