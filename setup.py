from setuptools import setup, find_packages

setup(
    name='toolbox_ml',
    version='1.0.0',
    description='Toolbox de EDA y selección de features para ML',
    author='Equipo TC5 - The Bridge',
    license='MIT',
    python_requires='>=3.10',
    packages=find_packages(),
    install_requires=open('requirements.txt').read().splitlines()
)