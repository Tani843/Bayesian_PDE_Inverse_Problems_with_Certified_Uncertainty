from setuptools import setup, find_packages

setup(
    name="bayesian-pde-solver",
    version="1.0.0",
    author="Tanisha Gupta",
    description="Bayesian methods for solving inverse problems in PDEs with certified uncertainty quantification",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "scipy>=1.7.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "plotly>=5.0.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
        "emcee>=3.1.0",
        "arviz>=0.11.0",
        "fenics>=2019.1.0",
        "pymc>=4.0.0",
        "jax>=0.3.0",
        "jaxlib>=0.3.0",
        "tensorflow-probability>=0.14.0",
        "dolfin-adjoint>=2019.1.0",
        "pytest>=6.0.0",
        "jupyter>=1.0.0",
        "tqdm>=4.60.0",
        "h5py>=3.0.0",
        "pyyaml>=5.4.0"
    ],
    extras_require={
        "dev": [
            "pytest-cov",
            "black",
            "flake8",
            "sphinx",
            "sphinx-rtd-theme"
        ]
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Physics"
    ]
)