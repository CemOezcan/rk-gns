import torch
import matplotlib.colors

import matplotlib.pyplot as plt
import torch_geometric

from matplotlib.lines import Line2D

from src.util.types import NodeType

def visualize_graph(data):
    """mask = torch.where(data.node_type == NodeType.MESH)[0]
    collider_mask = torch.where(data.node_type == NodeType.COLLIDER)[0]
    point_index = data.point_index# torch.where(data.node_type == NodeType.POINT)[0]"""
    #cell_mask = torch.where(data.cell_type == NodeType.MESH)[0]
    #obst_cell_mask = torch.where(data.cell_type == NodeType.COLLIDER)[0]
    #data = data.get_example(0)
    points = torch.where(data['mesh'].node_type == NodeType.POINT)[0]
    obst_mask = torch.where(data['mesh'].node_type == NodeType.COLLIDER)[0]
    #points = torch.cat([points, obst_mask], dim=0)
    #subgraph = data.subgraph({'mesh': points})
    subgraph = data

    #x, y = torch_geometric.utils.subgraph(torch.where(data.node_type == NodeType.POINT)[0], edge_index=data.edge_index, return_edge_mask=True)


    #faces = data.cells[cell_mask]
    #obst_faces = data.cells[obst_cell_mask]

    r, g, b = matplotlib.colors.to_rgb('dimgrey')

    fig, ax = plt.subplots()
    plt.xlim([-1.5, 1.5])
    plt.ylim([-1.5, 1.5])
    gt_pos = subgraph['mesh'].pos
    # TODO: data.y insert points?

    #obst_collection = [Polygon(face, closed=True) for face in gt_pos[obst_faces]]
    #gt_collection = [Polygon(face, closed=True) for face in gt_pos[faces]]

    #obst_collection = PatchCollection(obst_collection, ec=(r, g, b, 0.1), fc=(0, 0, 0, 0.5))
    #gt_collection = PatchCollection(gt_collection, alpha=0.5, ec='dimgrey', fc='yellow')

    #ax.add_collection(obst_collection)
    #ax.add_collection(gt_collection)

    #ax.scatter(gt_pos[mask][:, 0], gt_pos[mask][:, 1], label='Ground Truth Position', c='blue', alpha=0.5, s=5)
    ax.scatter(gt_pos[:, 0], gt_pos[:, 1], label='Ground Truth Position', c='dimgray', alpha=0.3, s=5)

    #ax.scatter(data.this[:, 0], data.this[:, 1], label='Ground Truth Position', c='red', alpha=0.5, s=5)
    #ax.scatter(gt_pos[collider_mask][:, 0], gt_pos[collider_mask][:, 1], label='Ground Truth Position', c='dimgrey', alpha=0.5, s=5)
    #ax.scatter(gt_pos[point_index:][:, 0], gt_pos[point_index:][:, 1], label='Ground Truth Position', c='green', s=5)

    colors = ['dimgrey', 'yellow', 'blue', 'blue', 'green', 'green', 'purple', 'purple', 'orange', 'red', 'cyan', 'cyan', 'lime', 'brown']
    for i, (x, y) in enumerate(zip(subgraph[('mesh', '0', 'mesh')].edge_index[0], subgraph[('mesh', '0', 'mesh')].edge_index[1])):
        if int(subgraph[('mesh', '0', 'mesh')].edge_type[i]) <= 100:
            fst = gt_pos[x]
            snd = gt_pos[y]
            color = colors[int(subgraph[('mesh', '0', 'mesh')].edge_type[i])]
            ax.add_line(Line2D([fst[0], snd[0]], [fst[1], snd[1]], alpha=0.1, color=color))


    ax.legend()
    plt.show()
