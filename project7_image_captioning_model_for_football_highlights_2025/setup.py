from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="football-captioning-2025",
    version="1.0.0",
    author="AI Developer",
    author_email="your.email@example.com",
    description="Advanced Image Captioning Model for Football Highlights using CNN+LSTM",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/football-captioning-2025",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Multimedia :: Video",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "black>=22.0.0",
            "flake8>=4.0.0",
            "pytest>=7.1.0",
            "pytest-cov>=4.0.0",
        ],
        "gpu": [
            "tensorflow-gpu>=2.10.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "football-caption-demo=src.demo:main",
            "football-caption-train=src.train:main",
            "football-caption-eval=src.evaluate:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.json", "*.txt", "*.md"],
    },
    keywords="computer-vision nlp deep-learning image-captioning football sports-ai",
    project_urls={
        "Bug Reports": "https://github.com/your-username/football-captioning-2025/issues",
        "Source": "https://github.com/your-username/football-captioning-2025",
        "Documentation": "https://github.com/your-username/football-captioning-2025#readme",
    },
)