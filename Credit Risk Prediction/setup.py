from setuptools import setup, find_packages
from typing import List


HYPEN_E_DOT = '-e .'

def get_requirements(file_path:str)->list:
    '''This function will return the list of requirements'''
    with open(file_path) as file:
        requirements = file.readlines()
        requirements = [req.replace('\n','') for req in requirements]
        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)
    return requirements


setup(    name='Student Performance Prediction',
    version='1.0',
    author='Ani',
    package=find_packages(),
    install_requires=get_requirements('requirements.txt')
)
