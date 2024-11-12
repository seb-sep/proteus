from setuptools import setup, find_packages
from mlx import extension

if __name__ == "__main__":
    setup(
        name="c_extensions",
        version="0.0.0",
        description="Sample C++ and Metal extensions for MLX primitives.",
        ext_modules=[
            extension.CMakeExtension("c_extensions._ext", sourcedir="./c_extensions")
        ],
        cmdclass={"build_ext": extension.CMakeBuild},
        package_dir={
            "c_extensions": "c_extensions/c_extensions",
            "proteus": "proteus",
        },
        packages=find_packages(where="c_extensions") + find_packages(where="proteus"),
        package_data={"c_extensions": ["*.so", "*.dylib", "*.metallib"]},
        zip_safe=False,
        python_requires=">=3.8",
    )
