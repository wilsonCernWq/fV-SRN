"""
Inference: loads models from hdf5-files and renders them
"""

import sys
import os
sys.path.append(f'{os.getcwd()}') # Hotfix for pymodule issues

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
import pdb
import copy

from volnet.network import SceneRepresentationNetwork, InputParametrization
from volnet.input_data import TrainingInputData
from volnet.raytracing import Raytracing

from inference import LoadedModel
import pyrenderer

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class MyLoadedModel(LoadedModel):
    def __init__(self, pth, fn, **kwargs):
        # The hdf5 file reader or writer is incorrect. Read the binary using pytorch directly
        state = torch.load(pth) # {'epoch': epoch + 1, 'model': network, 'parameters': opt}
        opt = state['parameters']

        self.from_hdf5(opt, **kwargs)

        self._network16 = copy.deepcopy(state['model']).to(torch.float16)
        self._network   = copy.deepcopy(state['model']).to(torch.float32)

        # create tensorcores network
        grid_encoding = pyrenderer.SceneNetwork.LatentGrid.Float
        try:
            self._scene_network, self._grid_encoding_error = self._network.export_to_pyrenderer(self._opt, grid_encoding, return_grid_encoding_error = True)
            self._num_parameters = self._scene_network.num_parameters()

            def to_float3(v): return pyrenderer.float3(v.x, v.y, v.z)
            self._scene_network.box_min  = to_float3(self._image_evaluator.volume.box_min())
            self._scene_network.box_size = to_float3(self._image_evaluator.volume.box_size())

            self._warps_shared = self._scene_network.compute_max_warps(False)
            self._warps_mixed  = self._scene_network.compute_max_warps(True)
            print("Warps shared:", self._warps_shared, ", warps mixed:", self._warps_mixed)

            self._volume_network = pyrenderer.VolumeInterpolationNetwork()
            self._volume_network.set_network(self._scene_network)

        except Exception as ex:
            print("Unable to load tensor core implementation:", ex)

        print("Loaded, output mode:", self._network.output_mode())

        self.save_compiled_network(fn.replace('.hdf5', '.volnet'))

    def from_hdf5(self, opt, epoch=-1, force_config_file=None, grid_encoding=None):
        self._dtype  = torch.float32
        self._device = torch.device("cuda")

        self._opt = collections.defaultdict(lambda: None)
        self._opt.update(opt)

        self._input_data = LoadedModel._get_input_data(self._opt, force_config_file)
        self._image_evaluator = self._input_data.default_image_evaluator()
        # self._image_evaluator.selected_channel = pyrenderer.IImageEvaluator.ChannelMode.Color

        self._training_time = 0.0

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

        # weights_np = f['weights'][epoch, :]

        self._network = SceneRepresentationNetwork(self._opt, self._input_data, self._dtype, self._device)
        # weights_bytes = io.BytesIO(weights_np.tobytes())
        # self._network.load_state_dict(torch.load(weights_bytes, map_location=self._device), strict=True)
        # self._network.to(device=self._device)

        self._network16 = SceneRepresentationNetwork(self._opt, self._input_data, self._dtype, self._device)
        # weights_bytes = io.BytesIO(weights_np.tobytes())
        # self._network16.load_state_dict(torch.load(weights_bytes, map_location=self._device), strict=True)
        # self._network16.to(dtype=torch.float16, device=self._device)

        self._has_direction = self._network.use_direction()

        self._volume_grid = self._image_evaluator.volume

        # create tensorcores network
        self._tensorcores_available = False
        if grid_encoding is None: grid_encoding = pyrenderer.SceneNetwork.LatentGrid.Float

        try:
            self._scene_network, self._grid_encoding_error = self._network.export_to_pyrenderer(self._opt, grid_encoding, return_grid_encoding_error = True)
            self._num_parameters = self._scene_network.num_parameters()

            def to_float3(v): return pyrenderer.float3(v.x, v.y, v.z)
            self._scene_network.box_min  = to_float3(self._image_evaluator.volume.box_min())
            self._scene_network.box_size = to_float3(self._image_evaluator.volume.box_size())

            self._warps_shared = self._scene_network.compute_max_warps(False)
            self._warps_mixed  = self._scene_network.compute_max_warps(True)
            print("Warps shared:", self._warps_shared, ", warps mixed:", self._warps_mixed)

            self._volume_network = pyrenderer.VolumeInterpolationNetwork()
            self._volume_network.set_network(self._scene_network)
            self._tensorcores_available = True
        except Exception as ex:
            print("Unable to load tensor core implementation:", ex)

        print("Loaded, output mode:", self._network.output_mode())

        self._network_output_mode = self._network.output_mode().split(':')[0]  # trim options
        self._raytracing = Raytracing(
            self._input_data.default_image_evaluator(),
            self._network_output_mode, 0.01, 128, 128,
            self._dtype, 
            self._device
        )

        self.time_keyframes = None
        self.time_train = None


if __name__ == '__main__':

    # args
    if len(sys.argv) < 5:
        print('Provide a result name and x,y,z dimensions')
        sys.exit(-1)

    # result
    result = sys.argv[1]

    # volume dimensions
    X = int(sys.argv[2])
    Y = int(sys.argv[3])
    Z = int(sys.argv[4])

    # app dir
    appdir = '/home/qadwu/Work/fV-SRN/applications/'

    # file name
    fn = os.path.join(appdir,f'volnet/results/{result}_hybrid/hdf5/run00003.hdf5')
    # fn = '/home/qadwu/Work/fV-SRN/applications/volnet/results/eval_CompressionTeaser/hdf5/rm60-Hybrid.hdf5'

    pth = os.path.join(appdir,f'volnet/results/{result}_hybrid/model/run00003/model_epoch_200.pth')
    # pth = '/home/qadwu/Work/fV-SRN/applications/volnet/results/eval_CompressionTeaser/model/rm60-Hybrid/model_epoch_200.pth'

    cfn = os.path.join(appdir, f'config-files/instant-vnr/{result}.json')
    # cfn = '/home/qadwu/Work/fV-SRN/applications/config-files/RichtmyerMeshkov-t60-v1-dvr.json'

    outdir = os.path.join(appdir, f'volnet/results/{result}_hybrid/reconstruction/')
    # outdir = '/home/qadwu/Work/fV-SRN/applications/volnet/results/eval_CompressionTeaser/reconstruction'
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    # test, example network trained in eval_VolumetricFeatures.py
    ln = MyLoadedModel(pth, fn, force_config_file=cfn)

    # exit() # stop here to avoid decoding the volume

    num_refine = 0
    tf = 0
    ensemble = ln.min_ensemble() if ln.is_time_dependent() else 0
    time = ln.min_timestep() if ln.is_time_dependent() else 0

    # points
    N = X*Y*Z
    linx = torch.linspace(0,1,X, dtype=ln._dtype)
    liny = torch.linspace(0,1,Y, dtype=ln._dtype)
    linz = torch.linspace(0,1,Z, dtype=ln._dtype)
    # positions = torch.stack(torch.meshgrid(linz,liny,linx), dim=-1).reshape(N, 3)
    positions = torch.stack(torch.meshgrid(linx,liny,linz), dim=-1).permute(2, 1, 0, 3).reshape(N, 3).to(ln._device)
    # pdb.set_trace()
    points_torch32 = torch.zeros((N,1))
    points_torch16 = torch.zeros((N,1))
    points_mixed   = torch.zeros((N,1))
    stride = 256*256
    for offset in range(0, N, stride):
        end = min(offset+stride, N)
        points_torch32[offset:end] = ln.evaluate(positions[offset:end], LoadedModel.EvaluationMode.PYTORCH32, tf=tf, timestep=time, ensemble=ensemble)
        points_torch16[offset:end] = ln.evaluate(positions[offset:end], LoadedModel.EvaluationMode.PYTORCH16, tf=tf, timestep=time, ensemble=ensemble)
        points_mixed  [offset:end] = ln.evaluate(positions[offset:end], LoadedModel.EvaluationMode.TENSORCORES_MIXED, tf=tf, timestep=time, ensemble=ensemble).float()

    print("Shape:", points_torch32.shape)
    print("Difference torch32-torch16:", F.mse_loss(points_torch32, points_torch16).item())
    print("Difference torch32-mixed:  ", F.mse_loss(points_torch32, points_mixed  ).item())

    print("Saving outputs")
    points_torch32.cpu().numpy().astype(np.float32).tofile(os.path.join(outdir, f'points_{X}x{Y}x{Z}_fp32.raw'))
    points_torch16.cpu().numpy().astype(np.float32).tofile(os.path.join(outdir, f'points_{X}x{Y}x{Z}_fp16.raw'))
    points_mixed  .cpu().numpy().astype(np.float32).tofile(os.path.join(outdir, f'points_{X}x{Y}x{Z}_mixed.raw'))

    print("Done")
