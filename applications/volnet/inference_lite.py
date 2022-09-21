"""
Inference: loads models from hdf5-files and renders them
"""

import sys
import os
sys.path.append(f'{os.getcwd()}/..') # Hotfix for pymodule issues

import numpy as np
import torch
import torch.nn.functional as F
from typing import Union, List, Any, Optional
import enum
import h5py
import io
import collections
from functools import lru_cache
import logging
import subprocess

from volnet.network import SceneRepresentationNetwork, InputParametrization
from volnet.input_data import TrainingInputData
from volnet.raytracing import Raytracing


class LoadedModel:
    """
    Class to load trained models from hdf5-checkpoints,
    evaluate them in world and screen space and
    convert them to compiled tensorcore implementations.

    Note: for time-dependent volumes,
    the time-indices are the actual timestep from the underlying dataset.
    That is, the integer values represent actual ground truth data.
    As the latent space variables are usually only defined sparsely,
    the
    """

    class EvaluationMode(enum.Enum):
        TENSORCORES_SHARED = enum.auto()
        TENSORCORES_MIXED = enum.auto()
        PYTORCH32 = enum.auto()
        PYTORCH16 = enum.auto()

    @staticmethod
    def _get_input_data(opt, force_config_file:str, _CACHE=dict()):
        # setup config file mapper
        def mapper(name:str, force_config_file=force_config_file):
            if force_config_file is not None:
                return force_config_file
            #else use from checkpoint
            if os.path.exists(name): return name
            # replace "server" config files with normal config files
            return name.replace('-server.json', '.json')
        TrainingInputData.set_config_file_mapper(mapper)
        # translate volume filenames if trained on the server, evaluated locally
        volume_filenames = opt['volume_filenames']
        if volume_filenames is not None and os.name=='nt':
            base_data_folder = os.path.abspath(os.path.join(os.path.split(__file__)[0], '../../..'))
            volume_filenames = volume_filenames.replace("/home/weiss", base_data_folder)
        # filter out options only for TrainingInputData for better caching
        opt2 = {
            'settings': opt['settings'],
            'tf_directory': opt['tf_directory'],
            'volume_filenames': volume_filenames,
            'ensembles': opt['ensembles'],
            'time_keyframes': opt['time_keyframes'],
            'time_train': opt['time_train'],
            'time_val': opt['time_val']
        }

        opt_string = str(opt2)
        d = _CACHE.get(opt_string, None)
        if d is None:
            d = TrainingInputData(opt2, check_volumes_exist=False)
            _CACHE[opt_string] = d
        return d

    @staticmethod
    def setup_config_file_mapper():
        if LoadedModel._config_file_mapper_initialized: return
        def mapper(name:str):
            if os.path.exists(name): return name
            # replace "server" config files with normal config files
            return name.replace('-server.json', '.json')
        TrainingInputData.set_config_file_mapper(mapper)
        LoadedModel._config_file_mapper_initialized = True

    def __init__(self, filename_or_hdf5:Union[str, h5py.File],
                 epoch:int=-1, grid_encoding=None,
                 force_config_file:str=None):
        """
        Loads the network from the filename or directly h5py file.
        :param filename_or_hdf5: the filename
        :param epoch: the epoch to read the weights from
        :param grid_encoding: the grid encoding for TensorCores
        :param force_config_file: if not None, the path to the .json config file
          that is enforced. This overwrites the TF and camera,
          filenames of the volumes are retained.
        """
        if isinstance(filename_or_hdf5, str):
            assert filename_or_hdf5.endswith(".hdf5")
            self._filename = os.path.splitext(os.path.split(filename_or_hdf5)[1])[0]
            print("Load network from", filename_or_hdf5)
            with h5py.File(filename_or_hdf5, 'r') as f:
                self._init_from_hdf5(f, epoch, grid_encoding, force_config_file)
        elif isinstance(filename_or_hdf5, h5py.File):
            self._filename = None
            self._init_from_hdf5(filename_or_hdf5, epoch, grid_encoding, force_config_file)
        else:
            raise ValueError("Unknown argument", filename_or_hdf5)

    def _init_from_hdf5(self, f:h5py.File, epoch:int, grid_encoding, force_config_file:str):
        self._dtype = torch.float32
        self._device = torch.device("cuda")
        self._opt = collections.defaultdict(lambda: None)
        self._opt.update(f.attrs)
        self._input_data = LoadedModel._get_input_data(self._opt, force_config_file)
        self._image_evaluator = self._input_data.default_image_evaluator()
        # self._image_evaluator.selected_channel = pyrenderer.IImageEvaluator.ChannelMode.Color

        total_losses = f['total']
        if total_losses[-1] == 0:
            print("WARNING: Last loss is zero, training most likely didn't finish. Filename: "+f.filename)
        self._training_time = float(f['times'][-1])

        # hack, fix for old networks
        is_new_network = True
        git_hash = self._opt['git'] or ""
        if len(git_hash)>0:
            try:
                exit_code = subprocess.run(["git", "merge-base", "--is-ancestor", "59fc3010267a00d111a16bce591fd6a0e7cd6c8b", git_hash]).returncode
                is_new_network = True if exit_code==0 else False
                print("Based on the git-commit of the checkpoint, it is a %s network"%("new" if is_new_network else "old"))
            except:
                print("unable to check git commit, assume new network architecture")
        InputParametrization.PREMULTIPLY_2_PI = is_new_network


        self._network = SceneRepresentationNetwork(self._opt, self._input_data, self._dtype, self._device)
        self._has_direction = self._network.use_direction()
        weights_np = f['weights'][epoch, :]
        weights_bytes = io.BytesIO(weights_np.tobytes())
        self._network.load_state_dict(
            torch.load(weights_bytes, map_location=self._device), strict=True)
        self._network.to(device=self._device)

        weights_bytes = io.BytesIO(weights_np.tobytes())
        self._network16 = SceneRepresentationNetwork(self._opt, self._input_data, self._dtype, self._device)
        self._network16.load_state_dict(
            torch.load(weights_bytes, map_location=self._device), strict=True)
        self._network16.to(dtype=torch.float16, device=self._device)

        self._volume_grid = self._image_evaluator.volume

        # lite inference does not support tensorcores network
        self._tensorcores_available = False
        
        print("Loaded, output mode:", self._network.output_mode())

        self._network_output_mode = self._network.output_mode().split(':')[0]  # trim options
        self._raytracing = Raytracing(self._input_data.default_image_evaluator(),
                                      self._network_output_mode, 0.01, 128, 128,
                                      self._dtype, self._device)

        def get_attr_or_None(a):
            return f.attrs[a] if a in f.attrs else None
        self.time_keyframes = get_attr_or_None('time_keyframes')
        self.time_train = get_attr_or_None('time_train')

    def filename(self):
        return self._filename

    def training_time_seconds(self):
        return self._training_time

    def fill_weights(self, weights, epoch:int):
        weights_np = weights[epoch, :]

        weights_bytes = io.BytesIO(weights_np.tobytes())
        self._network.load_state_dict(
            torch.load(weights_bytes, map_location=self._device), strict=True)

        weights_bytes = io.BytesIO(weights_np.tobytes())
        self._network16.load_state_dict(
            torch.load(weights_bytes, map_location=self._device), strict=True)
        self._network16.to(dtype=torch.float16, device=self._device)

    def is_time_dependent(self):
        """
        Returns true iff the network/data is time- or ensemble-dependent.
        :return:
        """
        return self._input_data.volume_filenames() is not None

    def min_timestep(self):
        """
        If time-dependent, returns the minimal timestep index (inclusive)
        """
        assert self.is_time_dependent()
        return self._input_data.time_keyframe_indices()[0]

    def max_timestep(self):
        """
        If time-dependent, returns the maximal timestep index (inclusive)
        """
        assert self.is_time_dependent()
        return self._input_data.time_keyframe_indices()[-1]

    def min_ensemble(self):
        """
        If time-dependent, returns the minimal timestep index (inclusive)
        """
        assert self.is_time_dependent()
        return self._input_data.ensemble_indices()[0]

    def max_ensemble(self):
        """
        If time-dependent, returns the maximal timestep index (inclusive)
        """
        assert self.is_time_dependent()
        return self._input_data.ensemble_indices()[-1]

    def timestep_interpolation_index(self, timestep: Union[float, int]):
        """
        Given the current timestep (self.min_timestep() <= timestep <= self.max_timestep()),
        returns the interpolation index into the latent space vector or grid
        (in [0, self.get_input_data().num_timekeyframes]).
        :param timestep: the timestep of the data
        :return: the index into the latent space grid
        """
        assert self.is_time_dependent()
        return self._input_data.timestep_to_index(timestep)

    def ensemble_interpolation_index(self, ensemble: Union[float, int]):
        """
        Given the current ensemble (self.min_ensemble() <= ensemble <= self.max_ensemble()),
        returns the interpolation index into the latent space vector or grid
        (in [0, self.get_input_data().num_ensembles()-1]).
        :param ensemble: the ensemble of the data
        :return: the index into the latent space grid
        """
        assert self.is_time_dependent()
        return self._input_data.ensemble_to_index(ensemble)

    def timestep_training_type(self, timestep: int):
        """
        Evaluates how that timestep was used during training.
        Returns a tuple of two booleans
            is_keyframe, is_trained = self.timestep_training_type(timestep)
        Where 'is_keyframe' is true iff there was a keyframe / latent vector at that timestep;
        and 'is_trained' is true iff that timestep was used in the training data
         (either directly via a keyframe or interpolated).
        :param timestep: the timestep to check
        :return: is_keyframe, is_trained
        """
        assert self.is_time_dependent()
        is_keyframe = timestep in self._input_data.time_keyframe_indices()
        is_trained = timestep in self._input_data.time_train_indices()
        return is_keyframe, is_trained

    def warps_mixed(self):
        return self._warps_mixed

    def warps_shared(self):
        return self._warps_shared

    def num_parameters(self):
        return self._num_parameters

    def is_tensorcores_available(self):
        return self._tensorcores_available

    def get_image_evaluator(self):
        return self._input_data.default_image_evaluator()

    def get_input_data(self):
        return self._input_data

    def get_raytracing(self) -> Raytracing:
        return self._raytracing

    def get_network_pytorch(self):
        return self._network, self._network16

    def set_network_pytorch(self, network32, network16):
        self._network = network32
        self._network16 = network16
        self._network_output_mode = self._network.output_mode().split(':')[0]  # trim options
        self._raytracing = Raytracing(self._input_data.default_image_evaluator(),
                                      self._network_output_mode, 0.01, 128, 128,
                                      self._dtype, self._device)

    def get_grid_encoding_error(self):
        return self._grid_encoding_error



    def evaluate(self, positions: torch.Tensor, mode:EvaluationMode, tf=0, timestep=0, ensemble=0):
        """
        Evaluates the network in world-space at the given positions
        :param positions: the positions of shape (N,3)
        :param mode: the evaluation mode (TensorCore<->PyTorch)
        :param tf: the TF index (currently unused)
        :param timestep: the timestep index, self.min_timestep()<=timestep<=self.max_timestep()
        :param ensemble: the ensemble index, self.min_ensemble()<=ensemble<=self.max_ensemble()
        :return: the values at the given position of shape (N,C), C=1 for densities, C=4 for color
        """
        assert len(positions.shape)==2
        assert positions.shape[1] == 3

        # convert from actual index to interpolation index
        if self.is_time_dependent():
            timestep = self.timestep_interpolation_index(timestep)
            ensemble = self.ensemble_interpolation_index(ensemble)
        else:
            if timestep!=0 or ensemble!=0:
                logging.warning(f"The current network is not time- or ensemble-dependent, but specified a timestep index (value {timestep}) != 0 or ensemble index (value {ensemble}) != 0.")

        # indices for torch
        dtype_time = torch.float16 if mode == LoadedModel.EvaluationMode.PYTORCH16 else torch.float32
        tf_index = torch.full((positions.shape[0],), tf, dtype=torch.int32, device=self._device)
        time_index = torch.full((positions.shape[0],), timestep, dtype=dtype_time, device=self._device)
        ensemble_index = torch.full((positions.shape[0],), ensemble, dtype=dtype_time, device=self._device)
        network_args = [tf_index, time_index, ensemble_index, 'screen']

        if mode == LoadedModel.EvaluationMode.PYTORCH16:
            pos2 = positions.to(dtype=torch.float16)
            if self._has_direction:
                pos2 = torch.cat((pos2, torch.zeros_like(pos2)), dim=1)
            with torch.no_grad():
                return self._network16(pos2, *network_args).to(dtype=self._dtype)
        elif mode == LoadedModel.EvaluationMode.PYTORCH32:
            pos2 = positions
            if self._has_direction:
                pos2 = torch.cat((pos2, torch.zeros_like(pos2)), dim=1)
            with torch.no_grad():
                return self._network(pos2, *network_args)
        else:
            return None


    def get_max_steps(self, camera:Optional[torch.Tensor], width:int, height:int, stepsize:float):
        """
        Returns the maximal number of steps through the volume
        :param camera:
        :param width:
        :param height:
        :param stepsize:
        :return:
        """
        self._raytracing.set_stepsize(stepsize)
        self._raytracing.set_resolution(width, height)
        return self._raytracing.get_max_steps(camera)


if __name__ == '__main__':

    # app dir
    appdir = '/home/qadwu/Work/fV-SRN/applications/'

    # file name
    fn = os.path.join(appdir,'volnet/results/eval_CompressionTeaser/hdf5/rm60-Hybrid.hdf5')
    cfn = os.path.join(appdir, 'config-files/RichtmyerMeshkov-t60-v1-dvr.json')

    # out dir
    outdir = os.path.join(appdir, 'volnet/results/eval_CompressionTeaser/reconstruction/')

    # volume dimensions
    X,Y,Z = 256, 256, 256
    
    #test, example network trained in eval_VolumetricFeatures.py
    ln = LoadedModel(fn, force_config_file=cfn)
    num_refine = 0

    tf = 0
    ensemble = ln.min_ensemble() if ln.is_time_dependent() else 0
    time = ln.min_timestep() if ln.is_time_dependent() else 0

    # points
    N = X*Y*Z
    linx = torch.linspace(0,1,X, dtype=ln._dtype, device=ln._device)
    liny = torch.linspace(0,1,Y, dtype=ln._dtype, device=ln._device)
    linz = torch.linspace(0,1,Z, dtype=ln._dtype, device=ln._device)
    positions = torch.stack(torch.meshgrid(linx,liny,linz), dim=-1).reshape(N, 3)
    points_torch32 = ln.evaluate(positions, LoadedModel.EvaluationMode.PYTORCH32, tf=tf, timestep=time, ensemble=ensemble)
    points_torch16 = ln.evaluate(positions, LoadedModel.EvaluationMode.PYTORCH16, tf=tf, timestep=time, ensemble=ensemble)

    print("Shape:", points_torch32.shape)
    print("Difference torch32-torch16:", F.mse_loss(points_torch32, points_torch16).item())

    print("Saving outputs")
    points_torch32.cpu().numpy().astype(np.float32).tofile(os.path.join(outdir, f'points_{X}x{Y}x{Z}_fp32.bin'))
    points_torch16.cpu().numpy().astype(np.float16).tofile(os.path.join(outdir, f'points_{X}x{Y}x{Z}_fp16.bin'))


    print("Done")
