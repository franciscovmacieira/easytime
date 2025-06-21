from setuptools import setup, find_packages

setup(
    name='ete_ts',
    version='0.2.0',
    packages=find_packages(),
    install_requires=[
        'numpy~=1.26.4',
        'antropy~=0.1.9',
        'pycatch22~=0.4.5',
        'tsfel~=0.1.9',
        'tsfeatures~=0.4.5',
        'statsmodels~=0.14.4', 
        'ruptures~=1.1.9',
        'scikit-learn~=1.6.1',

    ],
    author='Francisco Macieira',
    description='A package for analysing different caractheristics of time series data.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/franciscovmacieira/easytime.git',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.9',
)


