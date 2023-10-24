import copy
import os
import pickle
import re
import torch
import wandb
import matplotlib.colors

import matplotlib.pyplot as plt

from typing import Tuple
from matplotlib.animation import PillowWriter, FuncAnimation
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
from tqdm import trange

from src.algorithms.abstract_simulator import AbstractSimulator
from src.algorithms.mesh_simulator import MeshSimulator
from src.algorithms.get_simulator import get_simulator
from src.data.get_data import get_directories, get_data
from src.algorithms.abstract_task import AbstractTask
from src.util.types import ConfigDict, NodeType
from src.util.util import get_from_nested_dict


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
        self._dataset_name = config.get('directory')
        _, self._out_dir = get_directories(self._dataset_name)

        self._validation_interval = get_from_nested_dict(config, ['task', 'validation', 'interval'])
        self._num_val_trajectories = config.get('task').get('validation').get('trajectories')
        self._num_val_rollouts = self._config.get('task').get('validation').get('rollouts')
        self._num_val_n_step_rollouts = self._config.get('task').get('validation').get('n_step_rollouts')
        self._val_n_steps = self._config.get('task').get('validation').get('n_steps')

        self._num_test_trajectories = config.get('task').get('test').get('trajectories')
        self._num_test_rollouts = config.get('task').get('test').get('rollouts')
        self._num_n_step_rollouts = config.get('task').get('test').get('n_step_rollouts')
        self._n_steps = config.get('task').get('test').get('n_steps')
        self.n_viz = self._config.get('task').get('validation').get('n_viz')
        self.test_viz = self._config.get('task').get('test').get('n_viz')
        self.viz_interval = self._config.get('task').get('validation').get('viz_interval')

        self.train_loader = get_data(config=config)

        self._rollout_loader = get_data(config=config, split='eval', raw=True)
        self._valid_loader = get_data(config=config, split='eval')

        self._mp = get_from_nested_dict(config, ['model', 'message_passing_steps'])
        aggr = get_from_nested_dict(config, ['model', 'aggregation'])
        self.task_type = get_from_nested_dict(config, ['task', 'task'])
        lr = get_from_nested_dict(config, ['task', 'learning_rate'])
        wd = get_from_nested_dict(config, ['task', 'weight_decay'])
        feature_norm = get_from_nested_dict(config, ['model', 'feature_norm'])
        layer_norm = get_from_nested_dict(config, ['model', 'layer_norm'])
        layers = get_from_nested_dict(config, ['model', 'layers'])
        poisson = get_from_nested_dict(config, ['task', 'poisson_ratio'])
        ggns = get_from_nested_dict(config, ['task', 'ggns'])
        self.model_type = get_from_nested_dict(config, ['task', 'model'])
        seq = get_from_nested_dict(config, ['task', 'sequence'])
        batch_size = config.get('task').get('batch_size')
        rnn_type = config.get('task').get('recurrence')
        reduced = config.get('task').get('reduced')
        #self._task_name = f'm:{self.task_type}_l:{layers}_fn:{feature_norm}_ln:{layer_norm}_b:{batch_size}_t:{self.model_type}_a:{aggr}_lr:{lr}_seq:{seq}_ggns:{ggns}_red:{reduced}_poisson:{poisson}_mp:{self._mp}_epoch:'
        self._task_name = f'm:{self.task_type}_b:{batch_size}_t:{self.model_type}_lr:{lr}_wd:{wd}_seq:{seq}_ggns:{ggns}_red:{reduced}_p:{poisson}_r:{rnn_type}_mp:{self._mp}_epoch:'

        self.frequency_list = [0] if self.task_type != 'poisson' and not ggns and self.model_type == 'mgn' else get_from_nested_dict(config, ['task', 'imputation'])

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
        assert isinstance(self._algorithm, AbstractSimulator), 'Need a classifier to train on a classification task'
        start_epoch = self._current_epoch

        #if start_epoch == 0:
        #    self._algorithm.pretraining(train_dataloader=self.train_loader)
        for e in trange(start_epoch, self._epochs, desc='Epochs', leave=True):
            task_name = f'{self._task_name}{e + 1}'
            epoch_loss = self._algorithm.fit_iteration(train_dataloader=self.train_loader)
            evaluation_data = [{'training/epoch_loss': epoch_loss}]

            if (e + 1) % self._validation_interval == 0:
                one_step = self._algorithm.one_step_evaluator(self._valid_loader, self._num_val_trajectories, task_name)
                n_steps = [self._algorithm.n_step_evaluator(self._rollout_loader, task_name, n_steps=self._val_n_steps, n_traj=self._num_val_n_step_rollouts, freq=freq)
                            for freq in self.frequency_list]
                rollouts = [self._algorithm.rollout_evaluator(self._rollout_loader, self._num_val_rollouts, task_name, freq=freq)
                            for freq in self.frequency_list]

                dir_dict = self.select_plotting(task_name, self.frequency_list, self.n_viz) if (e + 1) % self.viz_interval == 0 else {}
                animation = {f"video_{key}": wandb.Video(value, fps=10, format="gif") for key, value in dir_dict.items()}

                evaluation_data = evaluation_data + [one_step] + rollouts + n_steps + [animation]
                self._algorithm.save(task_name)

            data = {k: v for dictionary in evaluation_data for k, v in dictionary.items()}
            data['epoch'] = e + 1
            self._algorithm.log_epoch(data)
            self._current_epoch = e + 1

    def get_model(self):
        if self._algorithm.best_models[5][1] is None:
            return self._algorithm._network

        return self._algorithm.best_models[5][1]

    def set_model(self, model):
        if model is not None:
            self._algorithm.global_model = model

    def finish(self):
        wandb.finish()

    def get_scalars(self) -> None:
        """
        Estimate and document the one-step, rollout and n-step losses of the mesh simulator.

        Returns
        -------

        """
        assert isinstance(self._algorithm, AbstractSimulator)
        task_name = f'{self._task_name}final'

        self._test_rollout_loader = get_data(config=self._config, split='test', raw=True)
        self._test_loader = get_data(config=self._config, split='test')

        old_model = copy.deepcopy(self._algorithm._network)
        self._algorithm._network = self._algorithm.best_models[0][1] if 0 in self.frequency_list else self._algorithm.best_models[2][1]
        one_step = self._algorithm.one_step_evaluator(self._test_loader, self._num_test_trajectories, task_name)

        n_step = list()
        rollout = list()
        for freq in self.frequency_list:
            self._algorithm._network = self._algorithm.best_models[freq][1]
            n_step.append(self._algorithm.n_step_evaluator(self._test_rollout_loader, task_name, n_steps=self._n_steps, n_traj=self._num_n_step_rollouts, freq=freq))
            rollout.append(self._algorithm.rollout_evaluator(self._test_rollout_loader, self._num_test_rollouts, task_name, freq=freq))

        self._algorithm._network = old_model
        dir_dict = self.select_plotting(task_name, self.frequency_list, self.test_viz)
        animation = {f"video_{key}": wandb.Video(value, fps=10, format="gif") for key, value in dir_dict.items()}

        evaluation_data = [one_step] + rollout + n_step + [animation]
        data = {'test/' + k: v for dictionary in evaluation_data for k, v in dictionary.items()}

        self._algorithm.log_epoch(data)

    def select_plotting(self, task_name: str, freq_list: list, n_viz):
        if self.task_type == 'poisson' and self.model_type != 'supervised':
            return {}
        if n_viz == 0:
            return {}

        out = dict.fromkeys(freq_list)
        for freq in freq_list:
            a, w = self.plot(task_name, freq, n_viz)
            out[freq] = self._save_plot(a, w, task_name, freq)

        return out

    def plot(self, task_name: str, freq: int, n_viz: int) -> Tuple[FuncAnimation, PillowWriter]:
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
        rollouts = os.path.join(self._out_dir, f'{task_name}_rollouts_k={freq}.pkl')
        with open(rollouts, 'rb') as fp:
            rollout_data = pickle.load(fp)[:n_viz]

        mask = torch.where(rollout_data[0]['node_type'] == NodeType.MESH)[0]
        cell_mask = torch.where(rollout_data[0]['cell_type'] == NodeType.MESH)[0]
        obst_cell_mask = torch.where(rollout_data[0]['cell_type'] == NodeType.COLLIDER)[0]

        faces = rollout_data[0]['cells'][cell_mask]
        obst_faces = rollout_data[0]['cells'][obst_cell_mask]

        r, g, b = matplotlib.colors.to_rgb('dimgrey')
        num_steps = rollout_data[0]['pred_pos'].shape[0]

        bounds = list()
        for trajectory in rollout_data:
            x_min, y_min = trajectory['gt_pos'].numpy().min(axis=(0, 1))
            x_max, y_max = trajectory['gt_pos'].numpy().max(axis=(0, 1))
            bounds.append((x_min, y_min, x_max, y_max))

        fig, ax = plt.subplots()
        scatter = ax.scatter([], [])

        def update(frame):
            step = frame % num_steps
            traj = frame // num_steps
            x_min, y_min, x_max, y_max = bounds[traj]

            ax.cla()
            ax.set_xlim(x_min * 1.1, x_max * 1.1)
            ax.set_ylim(y_min * 1.1, y_max * 1.1)

            pred_pos = rollout_data[traj]['pred_pos'][step]
            gt_pos = rollout_data[traj]['gt_pos'][step]

            obst_collection = [Polygon(face, closed=True) for face in gt_pos[obst_faces]]
            gt_collection = [Polygon(face, closed=True) for face in gt_pos[faces]]
            pred_collection = [Polygon(face, closed=True) for face in pred_pos[faces]]

            obst_collection = PatchCollection(obst_collection, ec=(r, g, b, 0.1), fc=(0, 0, 0, 0.5))
            gt_collection = PatchCollection(gt_collection, alpha=0.5, ec='dimgrey', fc='yellow')
            pred_collection = PatchCollection(pred_collection, alpha=0.3, ec='dimgrey', fc='red')

            ax.add_collection(obst_collection)
            ax.add_collection(gt_collection)
            ax.add_collection(pred_collection)

            ax.scatter(pred_pos[mask][:, 0], pred_pos[mask][:, 1], label='Predicted Position', c='red', alpha=0.3, s=5)
            ax.scatter(gt_pos[mask][:, 0], gt_pos[mask][:, 1], label='Ground Truth Position', c='blue', alpha=0.5, s=5)

            ax.legend()

            return scatter,

        anima = FuncAnimation(fig, update, frames=num_steps * len(rollout_data), blit=True)
        writergif = PillowWriter(fps=10)

        return anima, writergif

    def _save_plot(self, animation: FuncAnimation, writer_video: PillowWriter, task_name: str, freq: int) -> str:
        """
        Saves a simulation as a .gif file.

        Parameters
        ----------
            animation : FuncAnimation
                The animation
            writer_video : PillowWriter
                The writer
            task_name : str
                The task name

        Returns
        -------
            str
                The path to the .gif file

        """
        dir = os.path.join(self._out_dir, f'{task_name}_animation_k={freq}.gif')
        animation.save(dir, writer=writer_video)
        plt.show(block=True)
        return dir

