from setuptools import setup
from catkin_pkg.python_setup import generate_distutils_setup


setup_args = generate_distutils_setup(
    packages=[
        'kl_planning',
        'kl_planning.environments',
        'kl_planning.planning',
        'kl_planning.util',
        'kl_planning.visualization',
        'kl_planning.models'
    ],
    package_dir={'': 'src'}
)

setup(**setup_args)
