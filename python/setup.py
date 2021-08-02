from setuptools import setup, find_packages

setup(
    name='rababa',
    version='0.1.0',
    description='Rababa for Arabic diacriticization',
    author='Ribose',
    author_email='open.source@ribose.com',
    url='https://www.interscript.org',
    # packages=find_packages(include=['exampleproject', 'exampleproject.*']),
    python_requires='>=3.6, <4',
    install_requires=[
      'torch==1.9.0',
      'numpy==1.19.5',
      'matplotlib==3.3.3',
      'pandas==1.1.5',
      'ruamel.yaml==0.16.12',
      'tensorboard==2.4.0',
      'diacritization-evaluation==0.5',
      'tqdm==4.56.0',
      'onnx==1.9.0',
      'onnxruntime==1.8.1',
      'pyyaml==5.4.1',
    ],
    # extras_require={'plotting': ['matplotlib>=2.2.0', 'jupyter']},
    setup_requires=['pytest-runner'],
    tests_require=['pytest'],
    # entry_points={
    #     'console_scripts': ['my-command=exampleproject.example:main']
    # },
    # package_data={'exampleproject': ['data/schema.json']}
)
