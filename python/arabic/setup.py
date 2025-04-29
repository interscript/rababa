import re
from os import environ

import setuptools

with open("README.adoc", "r", encoding="utf-8") as fh:
    LONG_DESCRIPTION = fh.read()

PKG_VERSION = "0.1.0"

GIT_TAG = environ.get("GITHUB_REF", "")
TAG_VERSION = re.match(r'^refs/tags/v([0-9]+\.[0-9a-z]+\.[0-9a-z]+)$', GIT_TAG)

if TAG_VERSION:
    PKG_VERSION = TAG_VERSION.group(1)

setuptools.setup(
    name='rababa',
    version=PKG_VERSION,
    author="Ribose",
    author_email="open.source@ribose.com",
    license='MIT',
    description='Rababa for Arabic diacriticization',
    # packages=['rababa'],
    url='https://www.interscript.org',
    python_requires='>=3.8, <4',
    project_urls={
        'Documentation': 'https://github.com/interscript/rababa',
        'Source': 'https://github.com/interscript/rababa',
        'Tracker': 'https://github.com/interscript/rababa/issues',
    },
    install_requires=[
      'torch>=1.9.0',
      'numpy',
      'matplotlib',
      'pandas',
      'ruamel.yaml',
      'tensorboard',
      'diacritization-evaluation',
      'tqdm',
      'onnx',
      'onnxruntime',
      'pyyaml',
    ],
    # extras_require={'plotting': ['matplotlib>=2.2.0', 'jupyter']},
    setup_requires=['pytest-runner'],
    tests_require=['pytest'],
    # entry_points={
    #     'console_scripts': ['my-command=exampleproject.example:main']
    # },
    # package_data={'exampleproject': ['data/schema.json']}
)
