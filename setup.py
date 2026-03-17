from setuptools import setup, find_packages

setup(
    name="customer-churn-ml",
    version="1.0.0",
    author="ML Engineering Team",
    description="Production-ready Customer Churn Prediction ML System",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.10",
    install_requires=[
        "numpy",
        "pandas",
        "scikit-learn",
        "xgboost",
        "mlflow",
        "fastapi",
        "uvicorn",
        "pydantic",
        "pyyaml",
        "loguru",
        "joblib",
    ],
)
