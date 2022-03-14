from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='chamfer_3D',
    ext_modules=[
        CUDAExtension(
            name= 'chamfer_3D', 
            sources=["/".join(__file__.split('/')[:-1] + ['chamfer_cuda.cpp']),
                     "/".join(__file__.split('/')[:-1] + ['chamfer3D.cu']),],
            extra_compile_args={'cxx': ['-g'], 'nvcc': ['-O2']}
            ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })