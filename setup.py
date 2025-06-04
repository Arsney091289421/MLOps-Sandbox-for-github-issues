from setuptools import setup, find_packages

setup(
    name="mlops_github_issues",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "PyGithub",
        "python-dotenv",
        "pandas",
        "pyarrow",
        "boto3",
        "pqdm",
        "optuna",
        "xgboost",
        "scikit-learn",
        "moto[s3]==4.2.13",
        "pytest",
        "prefect"
    ],
    python_requires=">=3.7",  
)
