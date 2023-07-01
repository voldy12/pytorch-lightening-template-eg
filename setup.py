#!/usr/bin/env python

from setuptools import find_packages, setup

setup(
    name="radium",
    version="0.0.1",
    description="PyTorch Lightning With Cifar10",
    author="",
    author_email="",
    url="https://github.com/user/project",
    install_requires=["lightning", "hydra-core"],
    packages=find_packages(),
    # use this to customize global commands available in the terminal after installing the package
    entry_points={
        "console_scripts": [
            "radium_train = radium.train:main",
            "radium_eval = radium.eval:main",
            "radium_predict = radium.predict:main"
        ]
    },
)
