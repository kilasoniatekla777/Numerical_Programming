import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import UnivariateSpline, splrep, splev
from scipy.integrate import simpson
import cv2
from PIL import Image
import os
import warnings

warnings.filterwarnings('ignore')


class CandleVolumeReconstructor:
    """
    Class for reconstructing 3D candle volume from 2D image.
    """

    def __init__(self, image_path=None, image_array=None):
        """
        Initialize with either image path or numpy array.

        Args:
            image_path: path to image file
            image_array: numpy array of image (H, W, 3) or (H, W)
        """
        if image_path and os.path.exists(image_path):
            self.image = cv2.imread(image_path)
            if self.image is None:
                raise ValueError(f"Could not load image from {image_path}")
            self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
            self.image_source = f"File: {image_path}"
        elif image_array is not None:
            self.image = image_array
            self.image_source = "Provided array"
        else:
            # Create synthetic candle image for demonstration
            print("No image provided. Generating synthetic candle...")
            self.image = self.create_synthetic_candle()
            self.image_source = "Synthetic candle"

        self.edges = None
        self.contour_points = None
        self.spline_y = None
        self.spline_r = None
        self.volume = None

    def create_synthetic_candle(self, height=400, width=300):
        """
        Create a synthetic pillar candle image for demonstration.

        Args:
            height: image height in pixels
            width: image width in pixels

        Returns:
            RGB image array
        """
        image = np.ones((height, width, 3), dtype=np.uint8) * 255

        # Define candle profile (slightly tapered cylinder)
        center_x = width // 2
        top_radius = 60
        bottom_radius = 65
        candle_top = 50
        candle_bottom = 350

        # Draw candle body with gradient
        for y in range(candle_top, candle_bottom):
            # Linear taper from top to bottom
            progress = (y - candle_top) / (candle_bottom - candle_top)
            radius = int(top_radius + (bottom_radius - top_radius) * progress)

            # Create gradient color (cream to light brown)
            color_val = int(230 - progress * 30)
            color = (color_val, color_val - 20, color_val - 40)

            cv2.circle(image, (center_x, y), radius, color, -1)

        # Add wick at top
        wick_height = 20
        cv2.rectangle(image,
                      (center_x - 2, candle_top - wick_height),
                      (center_x + 2, candle_top),
                      (50, 50, 50), -1)

        # Add flame
        flame_pts = np.array([
            [center_x, candle_top - wick_height - 15],
            [center_x - 8, candle_top - wick_height - 5],
            [center_x + 8, candle_top - wick_height - 5]
        ], np.int32)
        cv2.fillPoly(image, [flame_pts], (255, 200, 50))

        return image

    def detect_edges(self, low_threshold=50, high_threshold=150, blur_kernel=5):
        """
        Detect edges in the image using Canny edge detection.

        Args:
            low_threshold: lower threshold for Canny
            high_threshold: upper threshold for Canny
            blur_kernel: Gaussian blur kernel size (must be odd)

        Returns:
            edges: binary edge image
        """
        # Convert to grayscale
        if len(self.image.shape) == 3:
            gray = cv2.cvtColor(self.image, cv2.COLOR_RGB2GRAY)
        else:
            gray = self.image

        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (blur_kernel, blur_kernel), 0)

        # Canny edge detection
        self.edges = cv2.Canny(blurred, low_threshold, high_threshold)

        return self.edges

    def extract_contour(self, axis_x=None):
        """
        Extract the right-side contour of the candle (one half due to symmetry).

        Args:
            axis_x: x-coordinate of symmetry axis (default: image center)

        Returns:
            contour_points: array of (y, r) coordinates
        """
        if self.edges is None:
            self.detect_edges()

        if axis_x is None:
            axis_x = self.image.shape[1] // 2

        height, width = self.edges.shape
        contour_points = []

        # For each row, find the rightmost edge point (right side of candle)
        for y in range(height):
            row = self.edges[y, :]
            edge_indices = np.where(row > 0)[0]

            if len(edge_indices) > 0:
                # Find edges on the right side of the axis
                right_edges = edge_indices[edge_indices > axis_x]
                if len(right_edges) > 0:
                    x = right_edges[0]  # Take the first edge on right side
                    r = abs(x - axis_x)  # Radius from axis
                    contour_points.append([y, r])

        self.contour_points = np.array(contour_points)

        # Filter out outliers and sort by y
        if len(self.contour_points) > 0:
            self.contour_points = self.contour_points[self.contour_points[:, 0].argsort()]

            # Remove duplicate y values, keep median r
            unique_y = np.unique(self.contour_points[:, 0])
            filtered_points = []
            for y in unique_y:
                r_values = self.contour_points[self.contour_points[:, 0] == y, 1]
                filtered_points.append([y, np.median(r_values)])
            self.contour_points = np.array(filtered_points)

        return self.contour_points

    def fit_spline(self, smoothing=300, degree=3):
        """
        Fit a smooth spline r(y) for the candle profile.
        This version gives MUCH smoother results.
        """

        if self.contour_points is None or len(self.contour_points) == 0:
            raise ValueError("No contour points found. Run extract_contour first.")

        y = self.contour_points[:, 0]  # height
        r = self.contour_points[:, 1]  # radius


        # Sort by y

        order = np.argsort(y)
        y = y[order]
        r = r[order]


        #  Remove duplicate y-values

        unique_y, idx = np.unique(y, return_index=True)
        y = unique_y
        r = r[idx]


        # Remove outliers (optional but helps a LOT)

        q1, q3 = np.percentile(r, [25, 75])
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        mask = (r >= lower) & (r <= upper)
        y = y[mask]
        r = r[mask]


        #  Fit a smoothing spline

        self.spline_tck = splrep(y, r, k=min(degree, len(y) - 1), s=smoothing)

        # store spline
        self.spline_y = y
        self.spline_r = lambda y_val: splev(y_val, self.spline_tck)

        return self.spline_r

    def compute_volume(self, num_samples=1000):
        """
        Compute volume of revolution using the disk method.

        Volume = π ∫ r(y)² dy

        Args:
            num_samples: number of sample points for integration

        Returns:
            volume: computed volume in cubic pixels
        """
        if self.spline_r is None:
            raise ValueError("Spline not fitted. Run fit_spline first.")

        # Sample y values
        y_min = self.spline_y.min()
        y_max = self.spline_y.max()
        y_samples = np.linspace(y_min, y_max, num_samples)

        # Evaluate radius at each y
        r_samples = self.spline_r(y_samples)

        # Ensure non-negative radii
        r_samples = np.maximum(r_samples, 0)

        # Disk method: V = π ∫ r² dy
        # Using Simpson's rule for integration
        integrand = np.pi * r_samples ** 2
        self.volume = simpson(integrand, x=y_samples)

        return self.volume

    def generate_3d_mesh(self, num_theta=80, num_z=200):
        """
        Generate a 3D mesh by rotating the candle's profile around
        the vertical Z-axis (correct physical orientation).
        """

        if self.spline_r is None:
            raise ValueError("Spline not fitted. Run fit_spline first.")

        #  Sample vertical positions (Z-axis, height)
        z_min = self.spline_y.min()
        z_max = self.spline_y.max()
        z = np.linspace(z_min, z_max, num_z)  # shape (num_z,)

        #  Sample rotation angles
        theta = np.linspace(0, 2 * np.pi, num_theta)  # shape (num_theta,)

        #  Create meshgrid
        Z, TH = np.meshgrid(z, theta)  # shapes: (num_theta, num_z)

        #  Evaluate radius at each Z
        R = self.spline_r(z)  # shape (num_z,)
        R = np.maximum(R, 0)
        R = np.tile(R, (num_theta, 1))  # shape (num_theta, num_z)

        #  Convert to Cartesian coordinates
        X = R * np.cos(TH)
        Y = R * np.sin(TH)
        # Z is already correct shape

        return X, Y, Z

    def visualize_processing_steps(self, save_fig=True):
        """
        Visualize the image processing steps.

        Args:
            save_fig: whether to save the figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f'Image Processing Pipeline\nSource: {self.image_source}',
                     fontsize=16, fontweight='bold')

        # Original image
        axes[0, 0].imshow(self.image)
        axes[0, 0].set_title('1. Original Image')
        axes[0, 0].axis('off')

        # Edge detection
        if self.edges is not None:
            axes[0, 1].imshow(self.edges, cmap='gray')
            axes[0, 1].set_title('2. Edge Detection (Canny)')
            axes[0, 1].axis('off')

        # Extracted contour
        if self.contour_points is not None:
            axes[1, 0].imshow(self.image, alpha=0.5)
            axes[1, 0].plot(self.image.shape[1] // 2 + self.contour_points[:, 1],
                            self.contour_points[:, 0],
                            'r.', markersize=2, label='Extracted Points')
            axes[1, 0].axvline(x=self.image.shape[1] // 2, color='blue',
                               linestyle='--', label='Symmetry Axis')
            axes[1, 0].set_title('3. Contour Extraction')
            axes[1, 0].legend()
            axes[1, 0].axis('off')

        # Spline fit
        if self.spline_r is not None:
            y_plot = np.linspace(self.spline_y.min(), self.spline_y.max(), 500)
            r_plot = self.spline_r(y_plot)

            axes[1, 1].plot(self.contour_points[:, 1], self.contour_points[:, 0],
                            'b.', markersize=4, label='Original Points', alpha=0.5)
            axes[1, 1].plot(r_plot, y_plot, 'r-', linewidth=2, label='Fitted Spline')
            axes[1, 1].set_xlabel('Radius (pixels)')
            axes[1, 1].set_ylabel('Height (pixels)')
            axes[1, 1].set_title('4. Spline Fitting')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
            axes[1, 1].invert_yaxis()

        plt.tight_layout()

        if save_fig:
            plt.savefig('candle_processing_steps.png', dpi=300, bbox_inches='tight')
            print("Saved: candle_processing_steps.png")

        plt.show()

    def visualize_3d_reconstruction(self, save_fig=True, elevation=20, azimuth=45):
        """
        Visualize the 3D reconstructed candle.

        Args:
            save_fig: whether to save the figure
            elevation: viewing elevation angle
            azimuth: viewing azimuth angle
        """
        X, Y, Z = self.generate_3d_mesh()

        fig = plt.figure(figsize=(14, 6))

        # 3D surface plot
        ax1 = fig.add_subplot(121, projection='3d')
        surf = ax1.plot_surface(X, Y, Z, cmap=cm.YlOrBr,
                                linewidth=0, antialiased=True, alpha=0.9)
        ax1.set_xlabel('X (pixels)')
        ax1.set_ylabel('Y (pixels)')
        ax1.set_zlabel('Z (pixels)')
        ax1.set_title('3D Reconstruction - Surface')
        ax1.view_init(elev=elevation, azim=azimuth)
        fig.colorbar(surf, ax=ax1, shrink=0.5, aspect=5)

        # Wireframe plot
        ax2 = fig.add_subplot(122, projection='3d')
        ax2.plot_wireframe(X, Y, Z, color='brown', linewidth=0.5, alpha=0.7)
        ax2.set_xlabel('X (pixels)')
        ax2.set_ylabel('Y (pixels)')
        ax2.set_zlabel('Z (pixels)')
        ax2.set_title('3D Reconstruction - Wireframe')
        ax2.view_init(elev=elevation, azim=azimuth)

        plt.suptitle(f'3D Pillar Candle Reconstruction\nVolume: {self.volume:.2f} cubic pixels',
                     fontsize=14, fontweight='bold')
        plt.tight_layout()

        if save_fig:
            plt.savefig('candle_3d_reconstruction.png', dpi=300, bbox_inches='tight')
            print("Saved: candle_3d_reconstruction.png")

        plt.show()

    def visualize_volume_integration(self, save_fig=True):
        """
        Visualize the volume computation using disk method.

        Args:
            save_fig: whether to save the figure
        """
        y_samples = np.linspace(self.spline_y.min(), self.spline_y.max(), 100)
        r_samples = self.spline_r(y_samples)
        r_samples = np.maximum(r_samples, 0)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Profile with disks
        ax1 = axes[0]
        ax1.plot(r_samples, y_samples, 'b-', linewidth=2, label='Candle Profile')
        ax1.fill_betweenx(y_samples, 0, r_samples, alpha=0.3, color='orange')

        # Draw sample disks
        disk_indices = np.linspace(0, len(y_samples) - 1, 10, dtype=int)
        for idx in disk_indices:
            y_disk = y_samples[idx]
            r_disk = r_samples[idx]
            ax1.plot([0, r_disk], [y_disk, y_disk], 'r-', linewidth=1.5, alpha=0.7)

        ax1.set_xlabel('Radius (pixels)')
        ax1.set_ylabel('Height (pixels)')
        ax1.set_title('Disk Method Visualization')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.invert_yaxis()

        # Cross-sectional area vs height
        ax2 = axes[1]
        areas = np.pi * r_samples ** 2
        ax2.fill_between(y_samples, 0, areas, alpha=0.5, color='blue')
        ax2.plot(y_samples, areas, 'b-', linewidth=2)
        ax2.set_xlabel('Height (pixels)')
        ax2.set_ylabel('Cross-sectional Area (sq pixels)')
        ax2.set_title('Cross-sectional Area Distribution')
        ax2.grid(True, alpha=0.3)

        plt.suptitle(f'Volume Integration: V = π ∫ r²(y) dy = {self.volume:.2f} cubic pixels',
                     fontsize=12, fontweight='bold')
        plt.tight_layout()

        if save_fig:
            plt.savefig('candle_volume_integration.png', dpi=300, bbox_inches='tight')
            print("Saved: candle_volume_integration.png")

        plt.show()


def get_user_image():
    """
    Prompt user for image path and validate it.

    Returns:
        image_path: valid path to image file, or None for synthetic
    """
    print("\n" + "=" * 70)
    print("CANDLE IMAGE INPUT")
    print("=" * 70)
    print("\nOptions:")
    print("  1. Enter path to your candle image file")
    print("  2. Press Enter to use synthetic candle (default)")
    print("-" * 70)

    user_input = input("Enter image path (or press Enter for synthetic): ").strip()

    if not user_input:
        print("\nUsing synthetic candle for demonstration.")
        return None

    # Remove quotes if user copied path with them
    user_input = user_input.strip('"').strip("'")

    if os.path.exists(user_input):
        print(f"\n✓ Image found: {user_input}")
        return user_input
    else:
        print(f"\n✗ Error: File not found at '{user_input}'")
        print("Using synthetic candle instead.")
        return None


def analyze_candle_image(image_path=None, low_threshold=30, high_threshold=100):
    """
    Complete analysis pipeline for a candle image.

    Args:
        image_path: path to candle image (None for synthetic)
        low_threshold: Canny lower threshold
        high_threshold: Canny upper threshold

    Returns:
        reconstructor: CandleVolumeReconstructor object with results
    """
    print(f"\n{'=' * 70}")
    print("STARTING CANDLE ANALYSIS")
    print(f"{'=' * 70}\n")

    # Initialize reconstructor
    reconstructor = CandleVolumeReconstructor(image_path=image_path)

    print("Step 1: Detecting edges...")
    reconstructor.detect_edges(low_threshold=low_threshold, high_threshold=high_threshold)
    print(f"  ✓ Edge detection complete")

    print("\nStep 2: Extracting contour...")
    reconstructor.extract_contour()
    print(f"  ✓ Found {len(reconstructor.contour_points)} contour points")

    if len(reconstructor.contour_points) < 5:
        print("  ✗ Error: Insufficient contour points detected")
        print("  Try adjusting edge detection thresholds")
        return None

    print("\nStep 3: Fitting cubic spline...")
    reconstructor.fit_spline()
    print(f"  ✓ Spline fitted successfully")

    print("\nStep 4: Computing volume...")
    volume = reconstructor.compute_volume()
    print(f"  ✓ Volume: {volume:.2f} cubic pixels")

    # If synthetic, compare with theoretical
    if image_path is None:
        h = 300  # height
        r1 = 60  # top radius
        r2 = 65  # bottom radius
        theoretical_volume = (np.pi * h / 3) * (r1 ** 2 + r1 * r2 + r2 ** 2)
        error = abs(volume - theoretical_volume) / theoretical_volume * 100

        print(f"\n  Comparison with theoretical volume:")
        print(f"    Computed:    {volume:.2f} cubic pixels")
        print(f"    Theoretical: {theoretical_volume:.2f} cubic pixels")
        print(f"    Error:       {error:.2f}%")

    print("\n" + "=" * 70)
    print("GENERATING VISUALIZATIONS")
    print("=" * 70 + "\n")

    print("Generating processing steps visualization...")
    reconstructor.visualize_processing_steps()

    print("\nGenerating 3D reconstruction...")
    reconstructor.visualize_3d_reconstruction()

    print("\nGenerating volume integration visualization...")
    reconstructor.visualize_volume_integration()

    return reconstructor


def main():
    """
    Main function with interactive user input.
    """
    print("\n" + "=" * 70)
    print("3D PILLAR CANDLE VOLUME RECONSTRUCTION")
    print("AP#6 - Numerical Programming")
    print("=" * 70)

    # Get image from user
    image_path = get_user_image()

    # Run analysis
    reconstructor = analyze_candle_image(image_path)

    if reconstructor:
        print("\n" + "=" * 70)
        print("ANALYSIS COMPLETE ✓")
        print("=" * 70)
        print("\nGenerated files:")
        print("  1. candle_processing_steps.png")
        print("  2. candle_3d_reconstruction.png")
        print("  3. candle_volume_integration.png")
        print(f"\nFinal Volume: {reconstructor.volume:.2f} cubic pixels")
        print("=" * 70 + "\n")
    else:
        print("\n" + "=" * 70)
        print("ANALYSIS FAILED")
        print("=" * 70 + "\n")


if __name__ == "__main__":
    main()