import numpy as np
from scipy.spatial import ConvexHull, convex_hull_plot_2d
import matplotlib.pyplot as plt


def spherical_flip(p_i: np.ndarray, radius: float, center: np.ndarray) -> np.ndarray:
    """"""
    p_i_hat: np.ndarray = np.empty_like(p_i)
    # Store each p_i vector as a row in the matrix, iterate through the rows
    for i in range(p_i_hat.shape[0]):
        p_i_hat[i] = p_i[i] + 2 * (radius - np.linalg.norm(p_i[i] - center)) * (
            (p_i[i] - center) / np.linalg.norm(p_i[i] - center))

    return p_i_hat


def visible_points(p_i: np.ndarray, center: np.ndarray, const_alpha: bool = False, alpha: float = 1.05) -> np.ndarray:
    """"""
    radius = calculate_radius(p_i, center, const_alpha, alpha)
    p_i_hat = spherical_flip(p_i, radius, center)
    hull = ConvexHull(p_i_hat)
    return hull


def calculate_radius(p_i: np.ndarray, center: np.ndarray, const_alpha: bool = False, alpha: float = 1.05) -> np.ndarray:
    """Calculates a radius for the circle that """
    center_of_mass = np.empty((p_i.shape[1],))
    for i in range(center_of_mass.shape[0]):
        center_of_mass[i] = p_i.transpose()[i].sum() / \
            p_i.transpose()[i].size()
    # radius = norm(- center of mass + maximum p_i point) + norm(max norm point)
    distance = np.empty((p_i.shape[0]))
    for i in range(p_i.shape[0]):
        distance[i] = np.linalg.norm(p_i[i] - center)
    max_p_i_distance = distance.max()
    if not const_alpha:
        """Calculate automatic alpha/radius using center of mass."""
        distance_mass_circle = np.empty((p_i.shape[0]))
        for i in range(p_i.shape[0]):
            distance_mass_circle[i] = np.linalg.norm(p_i[i] - center_of_mass)
        return max(distance_mass_circle) + max_p_i_distance
    else:
        return alpha * max_p_i_distance

def visualize_2d(p_i: np.ndarray, p_i_hat: np.ndarray, hull: ConvexHull):
    """"""
    


def main():
    TEST_SHAPE = (20, 2)
    TEST_SHAPE_CENTER = [4, 0]
    RADIUS_P_I = 1
    CENTER = [-3, 0]
    p_i = np.empty(TEST_SHAPE)
    for i in range(p_i.shape[0]):
        p_i[i] = [TEST_SHAPE_CENTER[0] + (RADIUS_P_I * np.cos((2 * i / (TEST_SHAPE[0] - 1)) * np.pi)),
                  TEST_SHAPE_CENTER[1] + (RADIUS_P_I * np.sin((2 * i / (TEST_SHAPE[0] - 1)) * np.pi))]

    p_i_hat = visible_points(p_i, CENTER)

    x0 = p_i.transpose()[0]
    y0 = p_i.transpose()[1]

    x1 = p_i_hat.transpose()[0]
    y1 = p_i_hat.transpose()[1]

    plt.scatter(x0, y0)
    plt.scatter(x1, y1)
    plt.scatter(CENTER[0], CENTER[1])
    plt.yticks(np.arange(-5, 5, 1))
    plt.xticks(np.arange(-10, 10, 1))
    plt.show()


if __name__ == "__main__":
    main()
