#!/usr/bin/env python

from setuptools import find_packages, setup

setup(
    name="misalign",
    version="0.0.1",
    description="Misalignment robust training",
    author="Kanghyun Ryu",
    author_email="khryu@kist.re.kr",
    url="https://github.com/KHRyu8985/misalign-benchmark",
    install_requires=["lightning", "hydra-core"],
    packages=find_packages(),
    # use this to customize global commands available in the terminal after installing the package
    entry_points={
        "console_scripts": [
            "train_command = misalign.train:main",
            "eval_command = misalign.eval:main",
        ]
    },
)
