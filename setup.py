'''The setup.py is a Python script typically included with Python-written libraries or apps. Its objective is to
 ensure that the program is installed correctly. With the aid of pip , we can use the setup.py to install
 any module without having to call setup.py directly. The setup.py is a standard Python file.'''

from setuptools import find_packages,setup
from typing import List


HYPEN_E_DOT = '-e .'

def get_requirements(file_path:str)->List[str]:
    '''
    This function return the required packages
    '''
    requirements =[]
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("\n","") for req in requirements] 

        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)

    return requirements

setup(
name='MLProject',
version='0.0.1',
author='Avinash Vyas',
author_email='avinash1999.vyas@gmail.com',
packages=find_packages(),
install_requires=get_requirements('requirements.txt')
)