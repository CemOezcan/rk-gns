
import matplotlib.colors
import matplotlib.pyplot as plt
import torch

from matplotlib.collections import PatchCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon


from src.util.types import NodeType

def visualize_graph(data):
    mask = torch.where(data.node_type == NodeType.MESH)[0]
    collider_mask = torch.where(data.node_type == NodeType.COLLIDER)[0]
    point_index = data.point_index# torch.where(data.node_type == NodeType.POINT)[0]
    cell_mask = torch.where(data.cell_type == NodeType.MESH)[0]
    obst_cell_mask = torch.where(data.cell_type == NodeType.COLLIDER)[0]

    faces = data.cells[cell_mask]
    obst_faces = data.cells[obst_cell_mask]

    r, g, b = matplotlib.colors.to_rgb('dimgrey')

    fig, ax = plt.subplots()

    gt_pos = data.pos
    # TODO: data.y insert points?

    #obst_collection = [Polygon(face, closed=True) for face in gt_pos[obst_faces]]
    #gt_collection = [Polygon(face, closed=True) for face in gt_pos[faces]]

    #obst_collection = PatchCollection(obst_collection, ec=(r, g, b, 0.1), fc=(0, 0, 0, 0.5))
    #gt_collection = PatchCollection(gt_collection, alpha=0.5, ec='dimgrey', fc='yellow')

    #ax.add_collection(obst_collection)
    #ax.add_collection(gt_collection)

    ax.scatter(gt_pos[mask][:, 0], gt_pos[mask][:, 1], label='Ground Truth Position', c='blue', alpha=0.5, s=5)
    ax.scatter(gt_pos[collider_mask][:, 0], gt_pos[collider_mask][:, 1], label='Ground Truth Position', c='dimgrey', alpha=0.5, s=5)
    ax.scatter(gt_pos[point_index:][:, 0], gt_pos[point_index:][:, 1], label='Ground Truth Position', c='green', s=5)

    colors = ['dimgrey', 'red', 'blue', 'green', 'yellow', 'red', 'brown', 'blue']
    for i, (x, y) in enumerate(zip(data.edge_index[0], data.edge_index[1])):
        fst = gt_pos[x]
        snd = gt_pos[y]
        color = colors[int(data.edge_type[i])]
        ax.add_line(Line2D([fst[0], snd[0]], [fst[1], snd[1]], alpha=0.3, color=color))


    ax.legend()
    plt.show()
