import simple_3dviz
import simple_3dviz.behaviours.io
import simple_3dviz.behaviours.trajectory
import simple_3dviz.behaviours.movements
import numpy as np
import numba
from simple_3dviz import Spherecloud
from simple_3dviz.behaviours import Behaviour


scene_for_module = None
DEFAULT_SIZE = 1000


_CITATION = """
@article{https://doi.org/10.7916/v9ym-tq84,
  doi = {10.7916/V9YM-TQ84},
  url = {https://academiccommons.columbia.edu/doi/10.7916/v9ym-tq84},
  author = {{Watkins, David Joseph}},
  title = {Learning Mobile Manipulation},
  publisher = {Columbia University},
  year = {2022}
}
"""


@numba.njit(nopython=True)
def minmax(array):
    # Ravel the array and return early if it's empty
    array = array.ravel()
    length = array.size
    if not length:
        return
    # We want to process two elements at once so we need
    # an even sized array, but we preprocess the first and
    # start with the second element, so we want it "odd"
    odd = length % 2
    if not odd:
        length -= 1
    # Initialize min and max with the first item
    minimum = maximum = array[0]
    i = 1
    while i < length:
        # Get the next two items and swap them if necessary
        x = array[i]
        y = array[i + 1]
        if x > y:
            x, y = y, x
        # Compare the min with the smaller one and the max
        # with the bigger one
        minimum = min(x, minimum)
        maximum = max(y, maximum)
        i += 2
    # If we had an even sized array we need to compare the
    # one remaining item too.
    if not odd:
        x = array[length]
        minimum = min(x, minimum)
        maximum = max(x, maximum)

        return minimum, maximum


def initial_scene_for_module(size=(DEFAULT_SIZE, DEFAULT_SIZE)):
    global scene_for_module
    if scene_for_module is None:
        scene_for_module = simple_3dviz.scenes.Scene(size)


class FrameBuffer(Behaviour):
    """Save the rendered frames to a numpy array.

    Arguments
    ---------
        number_of_frames: Number of frames to save
        every_n: int, Save every n frames instead of all frames.
    """

    def __init__(self, number_of_frames, image_width, image_height, every_n=1):
        self.frame_buffer = np.zeros((number_of_frames, image_width, image_height, 3))
        self._every_n = every_n
        self._i = 0

    def behave(self, params):
        if (self._i % self._every_n) != 0:
            return

        self.frame_buffer[self._i // self._every_n] = params.frame()[:, :, 0:3]

        self._i += 1


def render_pointcloud_still_np(
    points,
    save_file_path=None,
    color=np.array([1, 0, 0]),
    size=(DEFAULT_SIZE, DEFAULT_SIZE),
    camera_position=(1.1, 1.1, 1.1),
    sphere_size=0.02,
    scale_points=True,
):
    global scene_for_module
    if scene_for_module is None:
        scene_for_module = simple_3dviz.scenes.Scene(size)

    # Fit points inside unit cube TODO: make this work with a fixed scale across views
    if scale_points:
        x_min, x_max = minmax(points[:, 0])
        y_min, y_max = minmax(points[:, 1])
        z_min, z_max = minmax(points[:, 2])

        scale = 0.6 / (max([x_max, y_max, z_max]) - min([x_min, y_min, z_min]))
        points = points * scale

    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    sizes = np.ones(points.shape[0]) * sphere_size
    centers = np.stack([x, y, z]).reshape(3, -1).T
    colors = np.tile(color, (points.shape[0], 1))

    point_cloud = Spherecloud(centers=centers, sizes=sizes, colors=colors)

    behaviours = []
    frame_buffer = FrameBuffer(1, size[0], size[1])
    behaviours.append(frame_buffer)
    if save_file_path is not None:
        behaviours.append(simple_3dviz.behaviours.io.SaveFrames(save_file_path))

    camera_target = np.average(points, axis=0)

    simple_3dviz.render(
        point_cloud,
        behaviours=behaviours,
        camera_position=camera_position,
        camera_target=camera_target,
        light=(-2, -2, 2),
        n_frames=1,
        size=size,
        scene=scene_for_module,
    )

    return frame_buffer.frame_buffer[0]


def render_mesh_still(
    mesh,
    save_file_path=None,
    color=np.array([1, 0, 0]),
    size=(DEFAULT_SIZE, DEFAULT_SIZE),
    camera_position=(0.35, 0.35, 0.35),
    camera_target=None,
):
    global scene_for_module
    if scene_for_module is None:
        scene_for_module = simple_3dviz.scenes.Scene(size)

    behaviours = []
    frame_buffer = FrameBuffer(1, size[0], size[1])
    behaviours.append(frame_buffer)
    if save_file_path is not None:
        behaviours.append(simple_3dviz.behaviours.io.SaveFrames(save_file_path))

    if camera_target is None:
        camera_target = np.average(
            [np.min(mesh.vertices, axis=0), np.max(mesh.vertices, axis=0)], axis=0
        )
    mesh = simple_3dviz.Mesh.from_faces(mesh.vertices, mesh.faces, color)
    simple_3dviz.render(
        mesh,
        behaviours=behaviours,
        camera_position=camera_position,
        camera_target=camera_target,
        light=(-2, -2, 2),
        n_frames=1,
        size=size,
        scene=scene_for_module,
    )

    return frame_buffer.frame_buffer[0]


def render_mesh_movie(
    mesh,
    save_file_path,
    color=np.array([1, 0, 0]),
    size=(DEFAULT_SIZE, DEFAULT_SIZE),
    camera_position=(0.35, 0.35, 0.35),
    camera_target=(0, 0, 0),
    number_of_images=64,
    rotation_axis="z",
):
    global scene_for_module
    if scene_for_module is None:
        scene_for_module = simple_3dviz.scenes.Scene(size)

    mesh = simple_3dviz.Mesh.from_faces(mesh.vertices, mesh.faces, color)

    simple_3dviz.render(
        mesh,
        behaviours=[
            simple_3dviz.behaviours.movements.RotateModel(
                axis=rotation_axis, speed=2 * np.pi / number_of_images
            ),
            simple_3dviz.behaviours.io.SaveGif(save_file_path, duration=100),
        ],
        camera_position=camera_position,
        camera_target=camera_target,
        n_frames=number_of_images,
        light=(-2, -2, 2),
        size=size,
        scene=scene_for_module,
    )
