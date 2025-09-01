setup(
    name="astropack2",
    version="1.0.0",
    packages=find_packages(),
    author="icaromeidem",
    description="Versão otimizada do Astropack (cordeirossauro/astropack) para filtros do J-PAS",
    install_requires=[
        "numpy",
        "pandas",
        "scikit-learn",
        "joblib",
        "tqdm",
        "seaborn",
        "matplotlib"
        # adicione outros pacotes necessários aqui
    ],
    python_requires=">=3.7",
)