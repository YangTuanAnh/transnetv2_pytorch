from setuptools import setup, find_packages

setup(
    name="transnetv2_pytorch",
    version="0.1.0",
    packages=find_packages(),
    include_package_data=True,
    # install_requires=[
    #     "torch",
    #     "ffmpeg-python",
    #     "Pillow"
    # ],
    entry_points={
        "console_scripts": [
            "transnetv2_pytorch=transnetv2_pytorch.transnetv2_infer:main"
        ]
    },
    author="",
    description="PyTorch inference script for TransNetV2 (scene detection)",
)
