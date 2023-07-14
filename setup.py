from setuptools import setup, find_packages

VERSION = '0.0.2' 
DESCRIPTION = 'Quantum Image Classifier'

with open("README.md", "r") as fh:
    LONG_DESCRIPTION = fh.read()

# Setting up
setup(
       # the name must match the folder name 'verysimplemodule'
        name="qimgclassifier", 
        version=VERSION,
        author="Sarthak Prasad Malla",
        author_email="spm9513@nyu.edu",
        url="https://github.com/Sarthak-Malla/quantum-image-classification",
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        packages=["qimgclassifier"],
        install_requires=[
            "numpy",
            "torch",
            "torchvision",
            "qiskit",
            "matplotlib",
            "qiskit_machine_learning",
        ],
        
        keywords= ['python', 
            'qiskit', 
            'quantum', 
            'machine learning', 
            'image classification', 
            'quantum image classification', 
            'quantum machine learning', 
            'quantum computing', 
            'quantum image classifier', 
            'quantum image classifier',
        ],
        classifiers= [
            "Development Status :: 3 - Alpha",
            "Intended Audience :: Education",
            "Programming Language :: Python :: 2",
            "Programming Language :: Python :: 3",
            "Operating System :: MacOS :: MacOS X",
            "Operating System :: Microsoft :: Windows",
        ]
)