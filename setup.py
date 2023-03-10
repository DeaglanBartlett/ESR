from setuptools import setup

setup(
    name='esr',
    version='0.1.0',    
    description='A Python package to perform Exhaustive Symbolic Regression',
    url='https://github.com/DeaglanBartlett/ESR',
    author='Deaglan Bartlett and Harry Desmond',
    author_email='deaglan.bartlett@physics.ox.ac.uk',
    license='MIT licence',
    packages=['esr'],
    install_requires=['sympy',
		'myst-parser',
		'numpy',
		'scipy',
		'matplotlib',
		'pandas',
		'sphinx>=5.0',
		'pympler',
		'psutil',
		'prettytable',
		'numdifftools',
		'mpi4py',
		'astropy',
                'treelib',
                'networkx'
		],

    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'License :: MIT License',  
        'Operating System :: POSIX :: Linux',        
        'Programming Language :: Python :: 3',
    ],
)

