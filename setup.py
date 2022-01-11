import os
import io
from setuptools import setup
from pkg_resources import parse_requirements

source_dir = os.path.abspath(os.path.dirname(__file__))

# read the version and other strings from _version.py
version_info = {}
with open(os.path.join(source_dir, "signalanalysis/_version.py")) as o:
    exec(o.read(), version_info)

# read install requirements from requirements.txt
with open(os.path.join(source_dir, "requirements.txt")) as o:
    requirements = [str(r) for r in parse_requirements(o.read())]

def read(*filenames, **kwargs):
    encoding = kwargs.get('encoding', 'utf-8')
    sep = kwargs.get('sep', '\n')
    buf = []
    for filename in filenames:
        with io.open(filename, encoding=encoding) as f:
            buf.append(f.read())
    return sep.join(buf)

long_description = read('README.rst')

setup(
    name='signalanalysis',
    version=version_info['__version__'],
    url='https://github.com/philip-gemmell/signalanalysis',
    license='Apache Software License',
    author='Philip Gemmell',
    install_requires=requirements,
    author_email='philip.gemmell@kcl.ac.uk',
    description='A Python package for the reading, analysis and plotting of ECG and VCG data',
    long_description=long_description,
    packages=['signalanalysis'],
    package_dir={'signalanalysis': 'signalanalysis'},
    package_data={'signalanalysis': ['data/**/*']},
    include_package_data=True,
    platforms='any',
    classifiers=[
        'Programming Language :: Python',
        'Development Status :: 4 - Beta',
        'Natural Language :: English',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
        ]
)
