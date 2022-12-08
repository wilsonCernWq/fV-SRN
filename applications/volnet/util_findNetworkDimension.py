import numpy as np
import sys, os
sys.path.insert(0, os.getcwd())

def findNetworkDimension(target_num_parameters, channels_last):
    """
    Finds the number of channels+layers for the ResidualSine-architecture
    by Lu et al 2021 "Compressive Neural Representations of Volumetric Scalar Fields"
    """
    # Based on the correspondence with Matthew Berger, always 8 residual blocks were used
    NUM_RESIDUAL_BLOCKS = 8
    def getLayerStr(num_channels:int):
        """Returns the layer specification string for the InnerNetwork parametrization"""
        #+1 because the first layer is a Sine from the input dimension
        return ":".join(["%d"%num_channels]*(NUM_RESIDUAL_BLOCKS+1))

    # Lu et al don't use fourier features or other parametrization
    # Hence, all parameters are in the InnerNetwork
    # For simplicity, binary search until the channel count is matched
    from volnet.network import InnerNetwork

    def getNumParameters(num_channels:int):
        layers = getLayerStr(num_channels)
        net = InnerNetwork(input_channels=3, output_channels=channels_last,
                           layers=layers, activation="ResidualSine",
                           latent_size=0) #, split_density_and_auxiliary=False)
        params = 0
        for p in net.parameters(recurse=True):
            params += p.numel()
        return params

    # Phase one: double until we exceed the target
    high_channels = 8
    high_params = getNumParameters(high_channels)
    assert high_params<target_num_parameters, "Already a tiny network is too big"
    low_params = None
    low_channels = None
    while high_params<target_num_parameters:
        low_channels = high_channels
        low_params = high_params
        high_channels = low_channels*2
        high_params = getNumParameters(high_channels)

    # Phase two: binary search
    while high_channels-low_channels>1:
        mid_channels = np.clip((low_channels+high_channels)//2, low_channels+1, high_channels-1)
        mid_params = getNumParameters(mid_channels)
        if mid_params<target_num_parameters:
            low_channels = mid_channels
            low_params = mid_params
        else:
            high_channels = mid_channels
            high_params = mid_params

    # now pick the closest match
    if (target_num_parameters-low_params) < (high_params-target_num_parameters):
        best_channels = low_channels
        best_params = low_params
    else:
        best_channels = high_channels
        best_params = high_params

    return getLayerStr(best_channels), best_params


# # Based on the correspondence with Matthew Berger, always 8 residual blocks were used
# NUM_RESIDUAL_BLOCKS = 8
# def getLayerStr(num_channels:int):
#     """Returns the layer specification string for the InnerNetwork parametrization"""
#     #+1 because the first layer is a Sine from the input dimension
#     return ":".join(["%d"%num_channels]*(NUM_RESIDUAL_BLOCKS+1))
# # Lu et al don't use fourier features or other parametrization
# # Hence, all parameters are in the InnerNetwork
# # For simplicity, binary search until the channel count is matched
# from volnet.network import InnerNetwork
# def getNumParameters(num_channels:int, channels_last=1):
#     layers = getLayerStr(num_channels)
#     net = InnerNetwork(input_channels=3, output_channels=channels_last,
#                        layers=layers, activation="ResidualSine",
#                        latent_size=0) #, split_density_and_auxiliary=False)
#     params = 0
#     for p in net.parameters(recurse=True):
#         params += p.numel()
#     return params
# 
# print(getNumParameters(128)) # 264833
