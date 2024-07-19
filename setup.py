from setuptools import setup


setup(
    # 应用名
    name='crowdnav',
    # 版本号
    version='0.0.1',
    # 包括在安装包内的python包
    packages=[
        'crowd_nav',
        'crowd_nav.configs',
        'crowd_nav.policy',
        'crowd_nav.utils',
        'crowd_sim',
        'crowd_sim.envs',
        'crowd_sim.envs.policy',
        'crowd_sim.envs.utils',
    ],

    install_requires=[
        'gitpython',
        'gym==0.18.3',
        'matplotlib',
        'numpy==1.21.0',
        'scipy',
      #  'torch==1.9.0',
       # 'torchvision==0.10.0',
        'seaborn',
        'tqdm',
        'tensorboardX==2.4'
        ],
    # 而 extras_require 不会，这里仅表示该模块会依赖这些包
    # 但是这些包通常不会使用到，只有当你深度使用模块时，才会用到，这里需要你手动安装
    extras_require={
        'test': [
            'pylint',
            'pytest',
            ],
        },
    )
