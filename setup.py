from setuptools import setup, find_packages

setup(
    name="astropack2",
    version="1.0.0",
    packages=find_packages(),
    author="icaromeidem",
    description="Versão otimizada do Astropack (cordeirossauro/astropack) para varios modelos e surveys",
    install_requires=[
        "numpy",
        "pandas",
        "scikit-learn",
        "joblib",
        "tqdm",
        "seaborn",
        "matplotlib"
    ],
    python_requires=">=3.7",
)