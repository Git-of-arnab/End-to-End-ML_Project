from setuptools import find_packages,setup
from typing import List

HYPHEN_E_DOT = '-e .'

def get_requirements(file_path:str)->List[str]:
    '''
    This function takes the file path of requirement.txt as string and 
    returns the list of required packages for the project as string

    i/p -> str
    o/p -> list of str
    '''
    
    requirements = []
    with open(file_path) as file_obj:
        requirements=file_obj.readlines()
        #readlines will read including \n for new line, we need to remove the same
        requirements=[req.replace("\n","") for req in requirements]

        if HYPHEN_E_DOT in requirements: #This need not be read, its just for triggering setup.py
            requirements.remove(HYPHEN_E_DOT)
    
    return requirements

setup(        
    name='e2eMLproject',
    version='0.0.1',
    author='Arnab Chandra',
    author_email='hello.arnab.chandra@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)