from setuptools import setup, find_packages
setup(name='tight_binding',
      version='1.0',
      packages=find_packages(),
      install_requires=['numpy','scipy','matplotlib'],
      description='Useful functions for coding tight binding models',
      author='Laurens Smulders',
      author_email='laurenssmulders21@gmail.com')