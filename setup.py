"""
Install command: pip --user install -e .

"""

from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = ['matplotlib', 'numpy', 'bokeh', 'tensorboardX', 'sklearn', 'torchvision',
                     'torch', 'wfdb', 'google-api-python-client', 'opencv-python', 'pandas', 'scipy']

setup(name='simg_gan',
      version='0.1',
      install_requires=REQUIRED_PACKAGES,
      description='SimGAN',
      url='http://github.com/tomergolany/sim_gan',
      author='Tomer Golany',
      author_email='tomer.golany@gmail.com',
      license='Technion',
      packages=find_packages(),
      include_package_data=True,
      zip_safe=False)
