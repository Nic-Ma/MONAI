# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pprint
from typing import Dict, Optional, Sequence, Tuple, Union

import torch
from monai.apps.utils import get_logger
from monai.bundle.config_parser import ConfigParser
from monai.utils.type_conversion import get_equivalent_dtype

logger = get_logger(module_name=__name__)


def _update_args(args: Optional[Union[str, Dict]] = None, ignore_none: bool = True, **kwargs) -> Dict:
    """
    Update the `args` with the input `kwargs`.
    For dict data, recursively update the content based on the keys.

    Args:
        args: source args to update.
        ignore_none: whether to ignore input args with None value, default to `True`.
        kwargs: destination args to update.

    """
    args_: Dict = args if isinstance(args, dict) else {}  # type: ignore
    if isinstance(args, str):
        # args are defined in a structured file
        args_ = ConfigParser.load_config_file(args)

    # recursively update the default args with new args
    for k, v in kwargs.items():
        if ignore_none and v is None:
            continue
        if isinstance(v, dict) and isinstance(args_.get(k), dict):
            args_[k] = _update_args(args_[k], ignore_none, **v)
        else:
            args_[k] = v
    return args_


def _log_input_summary(tag: str, args: Dict):
    logger.info(f"\n--- input summary of monai.bundle.scripts.{tag} ---")
    for name, val in args.items():
        logger.info(f"> {name}: {pprint.pformat(val)}")
    logger.info("---\n\n")


def _get_fake_spatial_shape(shape: Sequence[Union[str, int]], p: int = 1, n: int = 1, any: int = 1):
    ret = []
    for i in shape:
        if isinstance(i, int):
            ret.append(i)
        elif isinstance(i, str):
            ret.append(any if i == "*" else eval(i, {"p": p, "n": n}))
        else:
            raise ValueError(f"spatial shape items must be int or string, but got: {type(i)} {i}.")
    return tuple(ret)


def run(
    runner_id: Optional[str] = None,
    meta_file: Optional[Union[str, Sequence[str]]] = None,
    config_file: Optional[Union[str, Sequence[str]]] = None,
    args_file: Optional[str] = None,
    **override,
):
    """
    Specify `meta_file` and `config_file` to run monai bundle components and workflows.

    Typical usage examples:

    .. code-block:: bash

        # Execute this module as a CLI entry:
        python -m monai.bundle run trainer --meta_file <meta path> --config_file <config path>

        # Override config values at runtime by specifying the component id and its new value:
        python -m monai.bundle run trainer --net#input_chns 1 ...

        # Override config values with another config file `/path/to/another.json`:
        python -m monai.bundle run evaluator --net %/path/to/another.json ...

        # Override config values with part content of another config file:
        python -m monai.bundle run trainer --net %/data/other.json#net_arg ...

        # Set default args of `run` in a JSON / YAML file, help to record and simplify the command line.
        # Other args still can override the default args at runtime:
        python -m monai.bundle run --args_file "/workspace/data/args.json" --config_file <config path>

    Args:
        runner_id: ID name of the runner component or workflow, it must have a `run` method.
        meta_file: filepath of the metadata file, if `None`, must be provided in `args_file`.
            if it is a list of file paths, the content of them will be merged.
        config_file: filepath of the config file, if `None`, must be provided in `args_file`.
            if it is a list of file paths, the content of them will be merged.
        args_file: a JSON or YAML file to provide default values for `runner_id`, `meta_file`,
            `config_file`, and override pairs. so that the command line inputs can be simplified.
        override: id-value pairs to override or add the corresponding config content.
            e.g. ``--net#input_chns 42``.

    """

    _args = _update_args(args=args_file, runner_id=runner_id, meta_file=meta_file, config_file=config_file, **override)
    for k in ("meta_file", "config_file"):
        if k not in _args:
            raise ValueError(f"{k} is required for 'monai.bundle run'.\n{run.__doc__}")
    _log_input_summary(tag="run", args=_args)

    parser = ConfigParser()
    parser.read_config(f=_args.pop("config_file"))
    parser.read_meta(f=_args.pop("meta_file"))
    id = _args.pop("runner_id", "")

    # the rest key-values in the _args are to override config content
    for k, v in _args.items():
        parser[k] = v

    workflow = parser.get_parsed_content(id=id)
    if not hasattr(workflow, "run"):
        raise ValueError(f"The parsed workflow {type(workflow)} does not have a `run` method.\n{run.__doc__}")
    workflow.run()


def verify_net_in_out(
    net_id: Optional[str] = None,
    meta_file: Optional[Union[str, Sequence[str]]] = None,
    config_file: Optional[Union[str, Sequence[str]]] = None,
    device: Optional[str] = None,
    p: Optional[int] = None,
    n: Optional[int] = None,
    any: Optional[int] = None,
    args_file: Optional[str] = None,
    **override,
):
    """
    Verify the input and output shape of network defined in the metadata.
    Will test with fake Tensor data according to the network args with id:
    input data:
    "_meta_#network_data_format#inputs#image#num_channels"
    "_meta_#network_data_format#inputs#image#spatial_shape"
    "_meta_#network_data_format#inputs#image#dtype"
    "_meta_#network_data_format#inputs#image#value_range"
    output data:
    "_meta_#network_data_format#outputs#pred#num_channels"
    "_meta_#network_data_format#outputs#pred#dtype"
    "_meta_#network_data_format#outputs#pred#value_range"

    Args:
        net_id: ID name of the network component to verify, it must be `torch.nn.Module`.
        meta_file: filepath of the metadata file to get network args, if `None`, must be provided in `args_file`.
            if it is a list of file paths, the content of them will be merged.
        config_file: filepath of the config file to get network definition, if `None`, must be provided in `args_file`.
            if it is a list of file paths, the content of them will be merged.
        device: target device to run the network forward computation, if None, prefer to "cuda" if existing.
        p: power factor to generate fake data shape if dim of expected shape is "x**p", default to 1.
        p: multiply factor to generate fake data shape if dim of expected shape is "x*n", default to 1.
        any: specified size to generate fake data shape if dim of expected shape is "*", default to 1.
        args_file: a JSON or YAML file to provide default values for `meta_file`, `config_file`,
            `net_id` and override pairs. so that the command line inputs can be simplified.
        override: id-value pairs to override or add the corresponding config content.
            e.g. ``--_meta#network_data_format#inputs#image#num_channels 3``.

    """

    _args = _update_args(
        args=args_file,
        net_id=net_id,
        meta_file=meta_file,
        config_file=config_file,
        device=device,
        p=p,
        n=n,
        any=any,
        **override,
    )
    _log_input_summary(tag="verify_net_in_out", args=_args)

    parser = ConfigParser()
    parser.read_config(f=_args.pop("config_file"))
    parser.read_meta(f=_args.pop("meta_file"))
    id = _args.pop("net_id", "")
    device = torch.device(_args.pop("device", "cuda" if torch.cuda.is_available() else "cpu"))
    p = _args.pop("p", 1)
    n = _args.pop("n", 1)
    any = _args.pop("any", 1)

    # the rest key-values in the _args are to override config content
    for k, v in _args.items():
        parser[k] = v

    net = parser[id].to(device)
    input_channels = parser["_meta_#network_data_format#inputs#image#num_channels"]
    input_spatial_shape = tuple(parser["_meta_#network_data_format#inputs#image#spatial_shape"])
    input_dtype = get_equivalent_dtype(parser["_meta_#network_data_format#inputs#image#dtype"], data_type=torch.Tensor)
    input_value_range = tuple(parser["_meta_#network_data_format#inputs#image#value_range"])

    output_channels = parser["_meta_#network_data_format#outputs#pred#num_channels"]
    output_dtype = get_equivalent_dtype(parser["_meta_#network_data_format#output#pred#dtype"], data_type=torch.Tensor)
    output_value_range = tuple(parser["_meta_#network_data_format#output#pred#value_range"])

    net.eval()
    with torch.no_grad():
        spatial_shape = _get_fake_spatial_shape(input_spatial_shape, p=p, n=n, any=any)
        test_data = torch.rand(*(input_channels, *spatial_shape), dtype=input_dtype, device=device)
        output = net(test_data)
        if output.shape[0] != output_channels:
            raise ValueError(f"output channel number `{output.shape[0]}` doesn't match: `{output_channels}`.")
        if output.dtype != output_dtype:
            raise ValueError(f"dtype of output data `{output.dtype}` doesn't match: {output_dtype}.")
