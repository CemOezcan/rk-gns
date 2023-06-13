
import math
import os
import pickle
import re
from typing import Tuple

import matplotlib.colors
from matplotlib.animation import PillowWriter, FuncAnimation
import matplotlib.pyplot as plt
import torch
import wandb
from matplotlib import tri
from matplotlib.collections import PolyCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon

from src.algorithms.abstract_simulator import AbstractSimulator
from src.algorithms.mesh_simulator import MeshSimulator
from src.algorithms.get_simulator import get_simulator
from src.data.data_utils import transform_position_to_edges, convert_to_hetero_data
from src.data.get_data import get_directories, get_data
from src.algorithms.abstract_task import AbstractTask
from tqdm import trange

from src.util.types import ConfigDict, ScalarDict
from src.util.util import get_from_nested_dict, device, triangles_to_edges


class MeshTask(AbstractTask):
    """
    Training and evaluation loops for mesh simulators.
    """

    def __init__(self, config: ConfigDict):
        """
        Initializes all necessary data for a mesh simulation task.

        Parameters
        ----------
            config : ConfigDict
                A (potentially nested) dictionary containing the "params" section of the section in the .yaml file
                used by cw2 for the current run.

        """
        super().__init__(config=config)
        self._config = config
        self._epochs = config.get('task').get('epochs')
        self._trajectories = config.get('task').get('trajectories')
        self._dataset_name = config.get('task').get('dataset')
        _, self._out_dir = get_directories(self._dataset_name)

        self._num_val_trajectories = config.get('task').get('validation').get('trajectories')
        self._num_val_rollouts = self._config.get('task').get('validation').get('rollouts')
        self._num_val_n_step_rollouts = self._config.get('task').get('validation').get('n_step_rollouts')
        self._val_n_steps = self._config.get('task').get('validation').get('n_steps')

        self._num_test_trajectories = config.get('task').get('test').get('trajectories')
        self._num_test_rollouts = config.get('task').get('test').get('rollouts')
        self._num_n_step_rollouts = config.get('task').get('test').get('n_step_rollouts')
        self._n_steps = config.get('task').get('test').get('n_steps')
        self.n_viz = self._config.get('task').get('validation').get('n_viz')

        self.train_loader = get_data(config=config)
        self._test_loader = get_data(config=config, split='test', raw=True)
        self._valid_loader = get_data(config=config, split='test')

        self._mp = get_from_nested_dict(config, ['model', 'message_passing_steps'])
        aggr = get_from_nested_dict(config, ['model', 'aggregation'])
        lr = get_from_nested_dict(config, ['model', 'learning_rate'])
        use_global = get_from_nested_dict(config, ['model', 'use_global'])
        heterogeneous = get_from_nested_dict(config, ['model', 'heterogeneous'])
        self._task_name = f'{self._dataset_name}_aggr:{aggr}_lr:{lr}_global:{use_global}_hetero:{heterogeneous}_mp:{self._mp}_epoch:'

        retrain = config.get('retrain')
        epochs = list() if retrain else [
            int(file.split('_epoch:')[1][:-4])
            for file in os.listdir(self._out_dir)
            if re.match(rf'model_{self._task_name}[0-9]+\.pkl', file) and 'final' not in self._task_name
        ]

        if epochs:
            self._current_epoch = max(epochs)
            model_path = os.path.join(self._out_dir, f'model_{self._task_name}{self._current_epoch}.pkl')
            with open(model_path, 'rb') as file:
                self._algorithm = pickle.load(file)
        else:
            self._algorithm = get_simulator(config)
            self._current_epoch = 0

        self._algorithm.initialize(task_information=config)
        wandb.init(reinit=False)

    def run_iterations(self) -> None:
        """
        Run all training epochs of the mesh simulator.
        Continues the training after the given epoch, if necessary.
        """
        assert isinstance(self._algorithm, MeshSimulator), 'Need a classifier to train on a classification task'
        start_epoch = self._current_epoch
        for e in trange(start_epoch, self._epochs, desc='Epochs', leave=True):
            task_name = f'{self._task_name}{e + 1}'

            self._algorithm.fit_iteration(train_dataloader=self.train_loader)
            one_step = self._algorithm.one_step_evaluator(self._valid_loader, self._num_val_trajectories, task_name)
            rollout = self._algorithm.rollout_evaluator(self._test_loader, self._num_val_rollouts, task_name)
            #n_step = self._algorithm.n_step_evaluator(self._test_loader, task_name, n_steps=self._val_n_steps, n_traj=self._num_val_n_step_rollouts)

            dir_dict = self.select_plotting(task_name)

            animation = {f"video_{key}": wandb.Video(value, fps=5, format="gif") for key, value in dir_dict.items()}
            data = {k: v for dictionary in [one_step, rollout, animation] for k, v in dictionary.items()}
            data['epoch'] = e + 1
            self._algorithm.save(task_name)
            self._algorithm.log_epoch(data)
            self._current_epoch = e + 1

    def get_scalars(self) -> None:
        """
        Estimate and document the one-step, rollout and n-step losses of the mesh simulator.

        Returns
        -------

        """
        assert isinstance(self._algorithm, MeshSimulator)
        task_name = f'{self._task_name}final'

        self._algorithm.one_step_evaluator(self._valid_loader, self._num_test_trajectories, task_name, logging=False)
        self._algorithm.rollout_evaluator(self._test_loader, self._num_test_rollouts, task_name, logging=False)
        self._algorithm.n_step_evaluator(self._test_loader, task_name, n_steps=self._n_steps, n_traj=self._num_n_step_rollouts, logging=False)

        self.select_plotting(task_name)

    def select_plotting(self, task_name):
        a, w = self.plot(task_name)
        dir_1 = self._save_plot(a, w, task_name)

        return {'': dir_1}

    def plot(self, task_name: str) -> Tuple[FuncAnimation, PillowWriter]:
        """
        Simulates and visualizes predicted trajectories as well as their respective ground truth trajectories.
        The predicted trajectories are produced by the current state of the mesh simulator.

        Parameters
        ----------
            task_name : str
                The name of the task

        Returns
        -------
            Tuple[FuncAnimation, PillowWriter]
                The simulations

        """
        rollouts = os.path.join(self._out_dir, f'{task_name}_rollouts.pkl')

        with open(rollouts, 'rb') as fp:
            rollout_data = pickle.load(fp)

        rollout_data = rollout_data[0]
        mask = torch.where(rollout_data['node_type'] == 0)[0]

        cell_mask = torch.where(rollout_data['cell_type'] == 0)[0]
        obst_cell_mask = torch.where(rollout_data['cell_type'] == 1)[0]
        faces = rollout_data['cells'][cell_mask]
        obst_faces = rollout_data['cells'][obst_cell_mask]

        fig, ax = plt.subplots()
        scatter = ax.scatter([], [])
        r, g, b = matplotlib.colors.to_rgb('dimgrey')
        x_min, y_min = rollout_data['gt_pos'].numpy().min(axis=(0, 1))
        x_max, y_max = rollout_data['gt_pos'].numpy().max(axis=(0, 1))

        def update(frame):
            ax.cla()
            ax.set_xlim(x_min * 1.1, x_max * 1.1)
            ax.set_ylim(y_min * 1.1, y_max * 1.1)

            pred_pos = rollout_data['pred_pos'][frame]
            gt_pos = rollout_data['gt_pos'][frame]

            for face in gt_pos[obst_faces]:
                triangle = Polygon(face, closed=True, edgecolor=(r, g, b, 0.1), facecolor=(0, 0, 0, 0.5))
                ax.add_patch(triangle)

            for face in gt_pos[faces]:
                triangle = Polygon(face, closed=True, alpha=0.5, edgecolor='dimgrey', facecolor='yellow')
                ax.add_patch(triangle)

            for face in pred_pos[faces]:
                triangle = Polygon(face, closed=True, alpha=0.3, edgecolor='dimgrey', facecolor='red')
                ax.add_patch(triangle)

            ax.scatter(pred_pos[mask][:, 0], pred_pos[mask][:, 1], label='Predicted Position', color='red', alpha=0.3, s=5)
            ax.scatter(gt_pos[mask][:, 0], gt_pos[mask][:, 1], label='Ground Truth Position', color='blue', alpha=0.5, s=5)

            return scatter,

        anima = FuncAnimation(fig, update, frames=len(rollout_data['pred_pos']), blit=True)
        writergif = PillowWriter(fps=10)

        return anima, writergif

    def _save_plot(self, animation: FuncAnimation, writervideo: PillowWriter, task_name: str) -> str:
        """
        Saves a simulation as a .gif file.

        Parameters
        ----------
            animation : FuncAnimation
                The animation
            writervideo : PillowWriter
                The writer
            task_name : str
                The task name

        Returns
        -------
            str
                The path to the .gif file

        """
        dir = os.path.join(self._out_dir, f'{task_name}_animation.gif')
        animation.save(dir, writer=writervideo)
        plt.show(block=True)
        return dir

