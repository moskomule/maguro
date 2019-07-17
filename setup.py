from setuptools import setup, find_packages

with open("README.md") as f:
    readme = f.read()

setup(
    name="maguro",
    version="0.3.1",
    author="moskomule",
    author_email="hataya@nlab.jp",
    packages=find_packages(),
    url="https://github.com/moskomule/maguro",
    description="A minimal job scheduler with GPUs",
    long_description=readme,
    license="BSD",
    entry_points={
        'console_scripts': ['maguro = maguro.maguro:main']
    }
)
