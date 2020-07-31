from setuptools import setup


setup(
    name='crowdnav',
    version='0.0.1',
    packages=[
        'envs',
        'envs.crowd_nav',
        'envs.crowd_nav.configs',
        'envs.crowd_nav.policy',
        'envs.crowd_nav.utils',
        'envs.crowd_nav.empowerment',
        'envs.crowd_sim',
        'envs.crowd_sim.configs',
        'envs.crowd_sim.policy',
        'envs.crowd_sim.utils',
        'envs.crowd_sim.visualization',
    ],
    install_requires=[
        'gitpython',
        'gym',
        'matplotlib',
        'numpy',
        'scipy',
        'torch',
        'torchvision',
    ],
    extras_require={
        'test': [
            'pylint',
            'pytest',
        ],
    },
)
