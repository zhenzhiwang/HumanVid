import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import torch

def visualize_camera_pose(visualizer, cam_params, dir_path, index):
    transform_matrix = np.asarray([[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]]).reshape(4, 4)
    c2ws = [transform_matrix @ cam_param for cam_param in cam_params]
    length = len(c2ws)
    for frame_idx, c2w in enumerate(c2ws):
        visualizer.extrinsic2pyramid(c2w, frame_idx / length, hw_ratio=9/16, base_xval=0.05,
                                     zval=0.1)
    visualizer.colorbar(length)
    visualizer.savefig(f"{dir_path}/img_camera_{index}.png")
    
class CameraPoseVisualizer:
    def __init__(self, xlim, ylim, zlim):
        self.fig = plt.figure(figsize=(10, 7))
        self.ax = self.fig.add_subplot(projection='3d')
        self.plotly_data = None  # plotly data traces
        self.ax.set_aspect("auto")
        self.ax.set_xlim(xlim)
        self.ax.set_ylim(ylim)
        self.ax.set_zlim(zlim)
        self.ax.set_xlabel('x')
        self.ax.set_ylabel('y')
        self.ax.set_zlabel('z')
        print('initialize camera pose visualizer')

    def extrinsic2pyramid(self, extrinsic, color_map='red', hw_ratio=9/16, base_xval=1, zval=3):
        vertex_std = np.array([[0, 0, 0, 1],
                               [base_xval, -base_xval * hw_ratio, zval, 1],
                               [base_xval, base_xval * hw_ratio, zval, 1],
                               [-base_xval, base_xval * hw_ratio, zval, 1],
                               [-base_xval, -base_xval * hw_ratio, zval, 1]])
        vertex_transformed = vertex_std @ extrinsic.T
        meshes = [[vertex_transformed[0, :-1], vertex_transformed[1][:-1], vertex_transformed[2, :-1]],
                            [vertex_transformed[0, :-1], vertex_transformed[2, :-1], vertex_transformed[3, :-1]],
                            [vertex_transformed[0, :-1], vertex_transformed[3, :-1], vertex_transformed[4, :-1]],
                            [vertex_transformed[0, :-1], vertex_transformed[4, :-1], vertex_transformed[1, :-1]],
                            [vertex_transformed[1, :-1], vertex_transformed[2, :-1], vertex_transformed[3, :-1], vertex_transformed[4, :-1]]]

        color = color_map if isinstance(color_map, str) else plt.cm.rainbow(color_map)

        self.ax.add_collection3d(
            Poly3DCollection(meshes, facecolors=color, linewidths=0.8, edgecolors=color, alpha=0.35))

    def customize_legend(self, list_label):
        list_handle = []
        for idx, label in enumerate(list_label):
            color = plt.cm.rainbow(idx / len(list_label))
            patch = Patch(color=color, label=label)
            list_handle.append(patch)
        plt.legend(loc='right', bbox_to_anchor=(1.8, 0.5), handles=list_handle)

    def colorbar(self, max_frame_length):
        cmap = mpl.cm.rainbow
        norm = mpl.colors.Normalize(vmin=0, vmax=max_frame_length)
        self.fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=self.ax, orientation='vertical', label='Frame Number')

    def savefig(self, path):
        plt.title('Extrinsic Parameters')
        plt.savefig(path)

def pca_visualize(img, n=3):
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt
    img = img.cpu().numpy()
    # Reshape features from (6, width, height) to (6, width*height)
    features_reshaped = img.reshape(6, -1).T

    # Apply PCA
    pca = PCA(n_components=3)
    principal_components = pca.fit_transform(features_reshaped)
    principal_images = principal_components.T.reshape(3,img.shape[1], img.shape[2])

    # Visualize PCA components as RGB image
    pca_image = np.stack(principal_images, axis=-1)
    pca_image = (pca_image - pca_image.min()) / (pca_image.max() - pca_image.min()) * 255

    return pca_image.astype(np.uint8)

def to_image(tensor, is_normed):
    if is_normed:
        tensor = tensor * 0.5 + 0.5  # Un-normalize if normalized
    tensor = tensor.clamp(0, 1)  # Clamp values to ensure they are between 0 and 1
    tensor = tensor.permute(1, 2, 0)  # Change from (C, H, W) to (H, W, C)
    image = (tensor.numpy() * 255).astype(np.uint8)  # Scale to [0, 255] and convert to uint8
    return image