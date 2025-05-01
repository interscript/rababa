import re
from os import environ

import setuptools
from setuptools import find_packages

with open("README.adoc", "r", encoding="utf-8") as fh:
    LONG_DESCRIPTION = fh.read()

PKG_VERSION = "0.1.0"

GIT_TAG = environ.get("GITHUB_REF", "")
TAG_VERSION = re.match(r'^refs/tags/v([0-9]+\.[0-9a-z]+\.[0-9a-z]+)$', GIT_TAG)

if TAG_VERSION:
    PKG_VERSION = TAG_VERSION.group(1)

setuptools.setup(
    name='rababa-arabic',
    version=PKG_VERSION,
    author="Ribose",
    author_email="open.source@ribose.com",
    license='MIT',
    description='Rababa for Arabic diacriticization',
    packages=find_packages(include=[
        "*",
        "models.*",
        "modules.*",
        "util.*",
    ]),
    url='https://www.interscript.org',
    python_requires='>=3.8, <4',
    project_urls={
        'Documentation': 'https://github.com/interscript/rababa',
        'Source': 'https://github.com/interscript/rababa',
        'Tracker': 'https://github.com/interscript/rababa/issues',
    },
    install_requires=[
      'torch>=1.9.0,<3.0.0',
      'numpy>=1.20.0,<2.0.0',
      'matplotlib>=3.3.3',
      'pandas>=1.3.0',
      'ruamel.yaml>=0.16.12',
      'tensorboard>=2.4.0',
      'diacritization-evaluation>=0.5',
      'tqdm>=4.56.0',
      'onnx>=1.9.0',
      'onnxruntime>=1.8.1',
      'pyyaml>=5.4.1',
    ],
    setup_requires=['pytest-runner'],
)
