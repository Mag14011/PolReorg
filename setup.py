from setuptools import setup, find_packages

setup(
    name="PolReorgEngAnalysis",
    version="1.0.0",
    description="Compute the influence of heme polarizability on electron transfer reorganization energy",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "plotly",
        "matplotlib",
    ],
    entry_points={
        'console_scripts': [
            'polreorg=pol_reorg_eng.ComputePolarizedReorgEng:main',
        ],
    },
    author="Matthew Guberman-Pfeffer",
    python_requires=">=3.7",
)

