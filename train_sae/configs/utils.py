import argparse
import json
from typing import Literal, get_args, get_origin

from pydantic import BaseModel

from train_sae.configs.base import RunConfig


def config_to_argparser(config: BaseModel) -> argparse.Namespace:
    """Convert a Pydantic config to an argparse namespace."""
    parser = argparse.ArgumentParser()
    for name, field in config.model_fields.items():
        arg_name = f"--{name}"
        arg_params = {
            "default": field.default,
            "help": field.description,
        }

        # Handle Literal types
        if get_origin(field.annotation) is Literal:
            choices = get_args(field.annotation)
            arg_params.update({"type": type(choices[0]), "choices": choices})
        else:
            arg_params["type"] = field.annotation

        # Add argument to parser
        parser.add_argument(arg_name, **arg_params)
    return parser


def parse_config(config: RunConfig) -> RunConfig:
    """Load the configuration from the command line. Optionally load from a file and
    overridewith command line arguments."""
    parser = config_to_argparser(config)
    parser.add_argument(
        "--config_file", type=str, default=None, help="Path to a configuration file."
    )
    args = parser.parse_args()

    if args.config_file is not None:
        with open(args.config_file) as f:
            config_dict = json.load(f)
        config = config.model_validate(config_dict)

        for name, field in config.model_fields.items():
            if getattr(args, name) != field.default:
                setattr(config, name, getattr(args, name))
    else:
        config = config.model_validate(args.__dict__)

    return config
