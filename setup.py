from setuptools import find_packages,setup
from typing import List

HYPHEN_E_DOT = '-e .'

def get_requirements(file_path:str) -> List[str]:
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        [req.replace("\n", "")for req in requirements]

        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)

    return requirements


setup(
    name = 'proj-1',
    version = '0.0.1',
    author = 'Sebi Samuel',
    author_email = 'sebisamuel99@gmail.com',
    packages = find_packages(),
    install_requires = get_requirements('requirements.txt')
)