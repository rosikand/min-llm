from setuptools import setup
import os 


def read_requirements_file(filename):
    req_file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                 filename)
    with open(req_file_path) as f:
        return [line.strip() for line in f]



setup(name='min_llm',
      version='0.0.13',
      description='Minimal implementations of large language models',
      url='https://github.com/rosikand/min-llm',
      author='Rohan Sikand',
      author_email='rsikand29@gmail.com',
      license='MIT',
      packages=['min_llm'],
    #   install_requires=read_requirements_file('./min_llm/requirements.txt'),
      zip_safe=False)