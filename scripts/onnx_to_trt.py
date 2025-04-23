#
# SPDX-FileCopyrightText: Copyright (c) 1993-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import os
import sys
import logging
import argparse
import platform
import ctypes

import tensorrt as trt
import numpy as np

# Set up Python logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("EngineBuilder")


def load_plugins(logger: trt.Logger):
    """
    Load custom TensorRT plugins from shared libraries.
    """
    if platform.system().lower() == 'linux':
        ctypes.CDLL(
            "/home/user/grid-sample3d-trt-plugin/build/libgrid_sample_3d_plugin.so",
            mode=ctypes.RTLD_GLOBAL
        )
    else:
        ctypes.CDLL(
            "./checkpoints/liveportrait_onnx/grid_sample_3d_plugin.dll",
            mode=ctypes.RTLD_GLOBAL,
            winmode=0
        )
    trt.init_libnvinfer_plugins(logger, "")


class EngineBuilder:
    """
    Parses an ONNX graph and builds a TensorRT engine from it.
    """

    def __init__(self, verbose: bool = False):
        """
        :param verbose: If enabled, set TensorRT logger to VERBOSE.
        """
        # Initialize the TensorRT logger
        self.trt_logger = trt.Logger(trt.Logger.INFO)
        if verbose:
            self.trt_logger.min_severity = trt.Logger.Severity.VERBOSE

        # Load plugins and initialize plugin registry
        trt.init_libnvinfer_plugins(self.trt_logger, namespace="")
        load_plugins(self.trt_logger)

        # Create builder and config
        self.builder = trt.Builder(self.trt_logger)
        self.config = self.builder.create_builder_config()

        # Set maximum workspace size via memory pool (TRT ≥ 8.4)
        self.config.set_memory_pool_limit(
            trt.MemoryPoolType.WORKSPACE,
            12 * (2**30)  # 12 GB
        )

        # Create an optimization profile (for dynamic shapes)
        profile = self.builder.create_optimization_profile()
        # Uncomment and set shapes as needed:
        # profile.set_shape("data", (1,3,192,192), (1,3,192,192), (1,3,192,192))
        # profile.set_shape("input.1", (1,3,512,512), (1,3,512,512), (1,3,512,512))
        self.config.add_optimization_profile(profile)

        self.batch_size = None
        self.network = None
        self.parser = None

    def create_network(self, onnx_path: str):
        """
        Parse the ONNX graph and create the TensorRT network definition.
        :param onnx_path: Path to the ONNX model file.
        """
        network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        self.network = self.builder.create_network(network_flags)
        self.parser = trt.OnnxParser(self.network, self.trt_logger)

        onnx_path = os.path.realpath(onnx_path)
        with open(onnx_path, "rb") as f:
            if not self.parser.parse(f.read()):
                log.error(f"Failed to load ONNX file: {onnx_path}")
                for i in range(self.parser.num_errors):
                    log.error(self.parser.get_error(i))
                sys.exit(1)

        # Log network I/O
        inputs = [self.network.get_input(i) for i in range(self.network.num_inputs)]
        outputs = [self.network.get_output(i) for i in range(self.network.num_outputs)]

        log.info("Network description:")
        for inp in inputs:
            self.batch_size = inp.shape[0]
            log.info(f"  Input '{inp.name}': shape={inp.shape}, dtype={inp.dtype}")
        for out in outputs:
            log.info(f"  Output '{out.name}': shape={out.shape}, dtype={out.dtype}")

    def create_engine(self, engine_path: str, precision: str):
        """
        Build and serialize a TensorRT engine.
        :param engine_path: Output path for the serialized engine.
        :param precision: 'fp32', 'fp16', or 'int8'.
        """
        engine_path = os.path.realpath(engine_path)
        os.makedirs(os.path.dirname(engine_path), exist_ok=True)
        log.info(f"Building {precision.upper()} engine to: {engine_path}")

        if precision == 'fp16':
            if not self.builder.platform_has_fast_fp16:
                log.warning("FP16 not supported on this platform/device")
            else:
                self.config.set_flag(trt.BuilderFlag.FP16)

        serialized_engine = self.builder.build_serialized_network(
            self.network, self.config
        )
        with open(engine_path, 'wb') as f:
            log.info(f"Serializing engine to file: {engine_path}")
            f.write(serialized_engine)


def main():
    parser = argparse.ArgumentParser(description="Convert ONNX to TensorRT engine.")
    parser.add_argument(
        '-o', '--onnx', required=True,
        help="Input ONNX model file"
    )
    parser.add_argument(
        '-e', '--engine',
        help="Output path for TensorRT engine"
    )
    parser.add_argument(
        '-p', '--precision', default='fp16', choices=['fp32','fp16','int8'],
        help="Precision mode: fp32, fp16, or int8 (default: fp16)"
    )
    parser.add_argument(
        '-v', '--verbose', action='store_true',
        help="Enable verbose TensorRT logs"
    )
    args = parser.parse_args()

    if args.engine is None:
        args.engine = args.onnx.rsplit('.', 1)[0] + '.trt'

    builder = EngineBuilder(verbose=args.verbose)
    builder.create_network(args.onnx)
    builder.create_engine(args.engine, args.precision)


if __name__ == '__main__':
    main()
