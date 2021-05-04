from setuptools import setup, find_packages


with open('README.md') as f:
    readme = f.read()

with open('License.md') as f:
    license = f.read()

setup(
    name='EIT',
    version='0.1.0',
    description='Solving the Calderon problem via PDE-constrained optimization',
    long_description=readme,
    author='Noah Cohen Kalafut',  # add your name here!
    author_email='nkalafut@wisc.edu', # and your email too
    url='https://github.com/Oafish1/CSC-715',
    license=license,
    install_requires=['numpy', 'scipy'], # add extra requirements here
    packages=find_packages(),
    classifiers=["Programming Language :: Python :: 3",
                "License :: MIT License",
                "Operating System :: OS Independent",],
)