import numpy as np

import os
import os.path as osp
from .globals import dir_path
import ctypes
ctypes.CDLL(osp.join(dir_path, "libamirstan_plugin.so"))

import tensorrt as trt
import torchvision.ops


def create_roipool_plugin(layer_name,
                            out_size,
                            featmap_strides,
                            roi_scale_factor,
                            finest_scale):

    creator = trt.get_plugin_registry().get_plugin_creator(
        'RoiPoolPluginDynamic', '1', '')

    pfc = trt.PluginFieldCollection()

    pf_out_size = trt.PluginField("out_size", np.array(
        [out_size], dtype=np.int32), trt.PluginFieldType.INT32)
    pfc.append(pf_out_size)

    pf_featmap_strides = trt.PluginField("featmap_strides", np.array(
        featmap_strides).astype(np.float32), trt.PluginFieldType.FLOAT32)
    pfc.append(pf_featmap_strides)

    pf_roi_scale_factor = trt.PluginField("roi_scale_factor", np.array(
        [roi_scale_factor], dtype=np.float32), trt.PluginFieldType.FLOAT32)
    pfc.append(pf_roi_scale_factor)

    pf_finest_scale = trt.PluginField("finest_scale", np.array(
        [finest_scale], dtype=np.int32), trt.PluginFieldType.INT32)
    pfc.append(pf_finest_scale)

    return creator.create_plugin(layer_name, pfc)