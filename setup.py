from setuptools import find_packages, setup

setup(
    name='Bead Spring Analytics',  # Replace with your package's name
    version='0.1.0',  # Package version
    author='Utku Gürel',  # Your name or your organization's name
    author_email='utkugurel@gmail.com',  # Your email or your organization's contact email
    description='Analysis of beadspring polymers',  # A short description of the package
    long_description=open('README.md').read(),  # A long description from README.md
    long_description_content_type='text/markdown',  # Content type for the long description, e.g., markdown or reStructuredText
    url='https://github.com/utkugurel/bead-spring',  # URL to the repository or package homepage
    packages=find_packages(),  # Automatically find and include all packages

    classifiers=[
        'Development Status :: 3 - Alpha',  # Choose the appropriate status
        'Intended Audience :: Developers',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.11.6',
    ],
    python_requires='>=3.11.6',  # Minimum version requirement of the package
)
