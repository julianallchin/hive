from setuptools import setup, find_packages

setup(
    name="clean_rl_train",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'torch',
        'numpy',
        'gymnasium',
        'tyro',
        'tensorboard',
        'matplotlib',
        'tqdm',
        'pandas',
    ],
    python_requires='>=3.8',
)
