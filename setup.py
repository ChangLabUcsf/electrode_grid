from setuptools import setup, find_packages

setup(
    name='electrodeGrid',
    version='0.0.1',
    description='Analyses for neural control of larynx',
    author='Ben Dichter',
    author_email='ben.dichter@gmail.com',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.6',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.2',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
    ],
    packages=find_packages(),
    install_requires=['matplotlib', 'numpy', 'tqdm', 'brewer2mpl', 'seaborn'],
)
