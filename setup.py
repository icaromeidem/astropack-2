from setuptools import setup, find_packages

setup(
    name="minas",
    version="1.0.0",
    packages=find_packages(),
    author="icaromeidem",
    description="MINAS - Machine learning for INference with Astronomical Surveys. Optimized version of Astropack (cordeirosauro/astropack) for astronomical surveys and new machine learning techniques",
    install_requires=[
        "numpy",
        "pandas",
        "scikit-learn",
        "joblib",
        "tqdm",
        "seaborn",
        "matplotlib",
        "xgboost"
        # adicione outros pacotes necessários aqui
    ],
    python_requires=">=3.7",
)