from setuptools import find_packages, setup

setup(
    name='null_space_projection_operators',
    packages=find_packages(include=['null_space_projection_operators']),
    version='0.0.1',
    description='',
    author='',
    license='GPLv3.0',
    install_requires=['numpy', 'scipy', 'tensorflow'],
    setup_requires=['pytest-runner'],
    tests_require=['pytest'],
    test_suite='tests',
    )
