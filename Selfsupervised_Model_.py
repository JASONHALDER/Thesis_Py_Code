{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "gpuType": "T4"
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    },
    "accelerator": "GPU"
  },
  "cells": [
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "vb9l-nlsXcvA",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "c7788af4-ee01-4ada-8591-4fcdbdd13c39"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Mounted at /content/drive\n"
          ]
        }
      ],
      "source": [
        "from google.colab import drive\n",
        "drive.mount('/content/drive')"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "!pip install -q vit-pytorch torchmetrics matplotlib seaborn --no-cache-dir\n",
        "!pip install -U torch torchvision --no-cache-dir"
      ],
      "metadata": {
        "id": "H9CfFU3tXuQ-",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "78057c06-a9bd-4be2-ed69-ca8a57b03199"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\u001b[2K     \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m69.7/69.7 kB\u001b[0m \u001b[31m79.8 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m140.8/140.8 kB\u001b[0m \u001b[31m17.3 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m962.6/962.6 kB\u001b[0m \u001b[31m79.0 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m363.4/363.4 MB\u001b[0m \u001b[31m266.0 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m13.8/13.8 MB\u001b[0m \u001b[31m276.7 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m24.6/24.6 MB\u001b[0m \u001b[31m237.0 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m883.7/883.7 kB\u001b[0m \u001b[31m225.7 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m664.8/664.8 MB\u001b[0m \u001b[31m271.1 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m211.5/211.5 MB\u001b[0m \u001b[31m293.7 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m56.3/56.3 MB\u001b[0m \u001b[31m279.0 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m127.9/127.9 MB\u001b[0m \u001b[31m299.5 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m207.5/207.5 MB\u001b[0m \u001b[31m259.3 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m21.1/21.1 MB\u001b[0m \u001b[31m264.3 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[?25hRequirement already satisfied: torch in /usr/local/lib/python3.11/dist-packages (2.6.0+cu124)\n",
            "Collecting torch\n",
            "  Downloading torch-2.7.1-cp311-cp311-manylinux_2_28_x86_64.whl.metadata (29 kB)\n",
            "Requirement already satisfied: torchvision in /usr/local/lib/python3.11/dist-packages (0.21.0+cu124)\n",
            "Collecting torchvision\n",
            "  Downloading torchvision-0.22.1-cp311-cp311-manylinux_2_28_x86_64.whl.metadata (6.1 kB)\n",
            "Requirement already satisfied: filelock in /usr/local/lib/python3.11/dist-packages (from torch) (3.18.0)\n",
            "Requirement already satisfied: typing-extensions>=4.10.0 in /usr/local/lib/python3.11/dist-packages (from torch) (4.14.0)\n",
            "Collecting sympy>=1.13.3 (from torch)\n",
            "  Downloading sympy-1.14.0-py3-none-any.whl.metadata (12 kB)\n",
            "Requirement already satisfied: networkx in /usr/local/lib/python3.11/dist-packages (from torch) (3.5)\n",
            "Requirement already satisfied: jinja2 in /usr/local/lib/python3.11/dist-packages (from torch) (3.1.6)\n",
            "Requirement already satisfied: fsspec in /usr/local/lib/python3.11/dist-packages (from torch) (2025.3.2)\n",
            "Collecting nvidia-cuda-nvrtc-cu12==12.6.77 (from torch)\n",
            "  Downloading nvidia_cuda_nvrtc_cu12-12.6.77-py3-none-manylinux2014_x86_64.whl.metadata (1.5 kB)\n",
            "Collecting nvidia-cuda-runtime-cu12==12.6.77 (from torch)\n",
            "  Downloading nvidia_cuda_runtime_cu12-12.6.77-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl.metadata (1.5 kB)\n",
            "Collecting nvidia-cuda-cupti-cu12==12.6.80 (from torch)\n",
            "  Downloading nvidia_cuda_cupti_cu12-12.6.80-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl.metadata (1.6 kB)\n",
            "Collecting nvidia-cudnn-cu12==9.5.1.17 (from torch)\n",
            "  Downloading nvidia_cudnn_cu12-9.5.1.17-py3-none-manylinux_2_28_x86_64.whl.metadata (1.6 kB)\n",
            "Collecting nvidia-cublas-cu12==12.6.4.1 (from torch)\n",
            "  Downloading nvidia_cublas_cu12-12.6.4.1-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl.metadata (1.5 kB)\n",
            "Collecting nvidia-cufft-cu12==11.3.0.4 (from torch)\n",
            "  Downloading nvidia_cufft_cu12-11.3.0.4-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl.metadata (1.5 kB)\n",
            "Collecting nvidia-curand-cu12==10.3.7.77 (from torch)\n",
            "  Downloading nvidia_curand_cu12-10.3.7.77-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl.metadata (1.5 kB)\n",
            "Collecting nvidia-cusolver-cu12==11.7.1.2 (from torch)\n",
            "  Downloading nvidia_cusolver_cu12-11.7.1.2-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl.metadata (1.6 kB)\n",
            "Collecting nvidia-cusparse-cu12==12.5.4.2 (from torch)\n",
            "  Downloading nvidia_cusparse_cu12-12.5.4.2-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl.metadata (1.6 kB)\n",
            "Collecting nvidia-cusparselt-cu12==0.6.3 (from torch)\n",
            "  Downloading nvidia_cusparselt_cu12-0.6.3-py3-none-manylinux2014_x86_64.whl.metadata (6.8 kB)\n",
            "Collecting nvidia-nccl-cu12==2.26.2 (from torch)\n",
            "  Downloading nvidia_nccl_cu12-2.26.2-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl.metadata (2.0 kB)\n",
            "Collecting nvidia-nvtx-cu12==12.6.77 (from torch)\n",
            "  Downloading nvidia_nvtx_cu12-12.6.77-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl.metadata (1.6 kB)\n",
            "Collecting nvidia-nvjitlink-cu12==12.6.85 (from torch)\n",
            "  Downloading nvidia_nvjitlink_cu12-12.6.85-py3-none-manylinux2010_x86_64.manylinux_2_12_x86_64.whl.metadata (1.5 kB)\n",
            "Collecting nvidia-cufile-cu12==1.11.1.6 (from torch)\n",
            "  Downloading nvidia_cufile_cu12-1.11.1.6-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl.metadata (1.5 kB)\n",
            "Collecting triton==3.3.1 (from torch)\n",
            "  Downloading triton-3.3.1-cp311-cp311-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl.metadata (1.5 kB)\n",
            "Requirement already satisfied: setuptools>=40.8.0 in /usr/local/lib/python3.11/dist-packages (from triton==3.3.1->torch) (75.2.0)\n",
            "Requirement already satisfied: numpy in /usr/local/lib/python3.11/dist-packages (from torchvision) (2.0.2)\n",
            "Requirement already satisfied: pillow!=8.3.*,>=5.3.0 in /usr/local/lib/python3.11/dist-packages (from torchvision) (11.2.1)\n",
            "Requirement already satisfied: mpmath<1.4,>=1.1.0 in /usr/local/lib/python3.11/dist-packages (from sympy>=1.13.3->torch) (1.3.0)\n",
            "Requirement already satisfied: MarkupSafe>=2.0 in /usr/local/lib/python3.11/dist-packages (from jinja2->torch) (3.0.2)\n",
            "Downloading torch-2.7.1-cp311-cp311-manylinux_2_28_x86_64.whl (821.2 MB)\n",
            "\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m821.2/821.2 MB\u001b[0m \u001b[31m297.1 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[?25hDownloading nvidia_cublas_cu12-12.6.4.1-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl (393.1 MB)\n",
            "\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m393.1/393.1 MB\u001b[0m \u001b[31m269.7 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[?25hDownloading nvidia_cuda_cupti_cu12-12.6.80-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl (8.9 MB)\n",
            "\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m8.9/8.9 MB\u001b[0m \u001b[31m313.9 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[?25hDownloading nvidia_cuda_nvrtc_cu12-12.6.77-py3-none-manylinux2014_x86_64.whl (23.7 MB)\n",
            "\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m23.7/23.7 MB\u001b[0m \u001b[31m291.4 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[?25hDownloading nvidia_cuda_runtime_cu12-12.6.77-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl (897 kB)\n",
            "\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m897.7/897.7 kB\u001b[0m \u001b[31m371.1 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[?25hDownloading nvidia_cudnn_cu12-9.5.1.17-py3-none-manylinux_2_28_x86_64.whl (571.0 MB)\n",
            "\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m571.0/571.0 MB\u001b[0m \u001b[31m218.0 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[?25hDownloading nvidia_cufft_cu12-11.3.0.4-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl (200.2 MB)\n",
            "\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m200.2/200.2 MB\u001b[0m \u001b[31m214.0 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[?25hDownloading nvidia_cufile_cu12-1.11.1.6-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl (1.1 MB)\n",
            "\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m1.1/1.1 MB\u001b[0m \u001b[31m236.4 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[?25hDownloading nvidia_curand_cu12-10.3.7.77-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl (56.3 MB)\n",
            "\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m56.3/56.3 MB\u001b[0m \u001b[31m261.7 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[?25hDownloading nvidia_cusolver_cu12-11.7.1.2-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl (158.2 MB)\n",
            "\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m158.2/158.2 MB\u001b[0m \u001b[31m161.1 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[?25hDownloading nvidia_cusparse_cu12-12.5.4.2-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl (216.6 MB)\n",
            "\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m216.6/216.6 MB\u001b[0m \u001b[31m238.3 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[?25hDownloading nvidia_cusparselt_cu12-0.6.3-py3-none-manylinux2014_x86_64.whl (156.8 MB)\n",
            "\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m156.8/156.8 MB\u001b[0m \u001b[31m113.6 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[?25hDownloading nvidia_nccl_cu12-2.26.2-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl (201.3 MB)\n",
            "\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m201.3/201.3 MB\u001b[0m \u001b[31m284.2 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[?25hDownloading nvidia_nvjitlink_cu12-12.6.85-py3-none-manylinux2010_x86_64.manylinux_2_12_x86_64.whl (19.7 MB)\n",
            "\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m19.7/19.7 MB\u001b[0m \u001b[31m238.3 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[?25hDownloading nvidia_nvtx_cu12-12.6.77-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl (89 kB)\n",
            "\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m89.3/89.3 kB\u001b[0m \u001b[31m321.7 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[?25hDownloading triton-3.3.1-cp311-cp311-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl (155.7 MB)\n",
            "\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m155.7/155.7 MB\u001b[0m \u001b[31m199.7 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[?25hDownloading torchvision-0.22.1-cp311-cp311-manylinux_2_28_x86_64.whl (7.5 MB)\n",
            "\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m7.5/7.5 MB\u001b[0m \u001b[31m225.9 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[?25hDownloading sympy-1.14.0-py3-none-any.whl (6.3 MB)\n",
            "\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m6.3/6.3 MB\u001b[0m \u001b[31m231.6 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[?25hInstalling collected packages: nvidia-cusparselt-cu12, triton, sympy, nvidia-nvtx-cu12, nvidia-nvjitlink-cu12, nvidia-nccl-cu12, nvidia-curand-cu12, nvidia-cufile-cu12, nvidia-cuda-runtime-cu12, nvidia-cuda-nvrtc-cu12, nvidia-cuda-cupti-cu12, nvidia-cublas-cu12, nvidia-cusparse-cu12, nvidia-cufft-cu12, nvidia-cudnn-cu12, nvidia-cusolver-cu12, torch, torchvision\n",
            "  Attempting uninstall: nvidia-cusparselt-cu12\n",
            "    Found existing installation: nvidia-cusparselt-cu12 0.6.2\n",
            "    Uninstalling nvidia-cusparselt-cu12-0.6.2:\n",
            "      Successfully uninstalled nvidia-cusparselt-cu12-0.6.2\n",
            "  Attempting uninstall: triton\n",
            "    Found existing installation: triton 3.2.0\n",
            "    Uninstalling triton-3.2.0:\n",
            "      Successfully uninstalled triton-3.2.0\n",
            "  Attempting uninstall: sympy\n",
            "    Found existing installation: sympy 1.13.1\n",
            "    Uninstalling sympy-1.13.1:\n",
            "      Successfully uninstalled sympy-1.13.1\n",
            "  Attempting uninstall: nvidia-nvtx-cu12\n",
            "    Found existing installation: nvidia-nvtx-cu12 12.4.127\n",
            "    Uninstalling nvidia-nvtx-cu12-12.4.127:\n",
            "      Successfully uninstalled nvidia-nvtx-cu12-12.4.127\n",
            "  Attempting uninstall: nvidia-nvjitlink-cu12\n",
            "    Found existing installation: nvidia-nvjitlink-cu12 12.4.127\n",
            "    Uninstalling nvidia-nvjitlink-cu12-12.4.127:\n",
            "      Successfully uninstalled nvidia-nvjitlink-cu12-12.4.127\n",
            "  Attempting uninstall: nvidia-nccl-cu12\n",
            "    Found existing installation: nvidia-nccl-cu12 2.21.5\n",
            "    Uninstalling nvidia-nccl-cu12-2.21.5:\n",
            "      Successfully uninstalled nvidia-nccl-cu12-2.21.5\n",
            "  Attempting uninstall: nvidia-curand-cu12\n",
            "    Found existing installation: nvidia-curand-cu12 10.3.5.147\n",
            "    Uninstalling nvidia-curand-cu12-10.3.5.147:\n",
            "      Successfully uninstalled nvidia-curand-cu12-10.3.5.147\n",
            "  Attempting uninstall: nvidia-cuda-runtime-cu12\n",
            "    Found existing installation: nvidia-cuda-runtime-cu12 12.4.127\n",
            "    Uninstalling nvidia-cuda-runtime-cu12-12.4.127:\n",
            "      Successfully uninstalled nvidia-cuda-runtime-cu12-12.4.127\n",
            "  Attempting uninstall: nvidia-cuda-nvrtc-cu12\n",
            "    Found existing installation: nvidia-cuda-nvrtc-cu12 12.4.127\n",
            "    Uninstalling nvidia-cuda-nvrtc-cu12-12.4.127:\n",
            "      Successfully uninstalled nvidia-cuda-nvrtc-cu12-12.4.127\n",
            "  Attempting uninstall: nvidia-cuda-cupti-cu12\n",
            "    Found existing installation: nvidia-cuda-cupti-cu12 12.4.127\n",
            "    Uninstalling nvidia-cuda-cupti-cu12-12.4.127:\n",
            "      Successfully uninstalled nvidia-cuda-cupti-cu12-12.4.127\n",
            "  Attempting uninstall: nvidia-cublas-cu12\n",
            "    Found existing installation: nvidia-cublas-cu12 12.4.5.8\n",
            "    Uninstalling nvidia-cublas-cu12-12.4.5.8:\n",
            "      Successfully uninstalled nvidia-cublas-cu12-12.4.5.8\n",
            "  Attempting uninstall: nvidia-cusparse-cu12\n",
            "    Found existing installation: nvidia-cusparse-cu12 12.3.1.170\n",
            "    Uninstalling nvidia-cusparse-cu12-12.3.1.170:\n",
            "      Successfully uninstalled nvidia-cusparse-cu12-12.3.1.170\n",
            "  Attempting uninstall: nvidia-cufft-cu12\n",
            "    Found existing installation: nvidia-cufft-cu12 11.2.1.3\n",
            "    Uninstalling nvidia-cufft-cu12-11.2.1.3:\n",
            "      Successfully uninstalled nvidia-cufft-cu12-11.2.1.3\n",
            "  Attempting uninstall: nvidia-cudnn-cu12\n",
            "    Found existing installation: nvidia-cudnn-cu12 9.1.0.70\n",
            "    Uninstalling nvidia-cudnn-cu12-9.1.0.70:\n",
            "      Successfully uninstalled nvidia-cudnn-cu12-9.1.0.70\n",
            "  Attempting uninstall: nvidia-cusolver-cu12\n",
            "    Found existing installation: nvidia-cusolver-cu12 11.6.1.9\n",
            "    Uninstalling nvidia-cusolver-cu12-11.6.1.9:\n",
            "      Successfully uninstalled nvidia-cusolver-cu12-11.6.1.9\n",
            "  Attempting uninstall: torch\n",
            "    Found existing installation: torch 2.6.0+cu124\n",
            "    Uninstalling torch-2.6.0+cu124:\n",
            "      Successfully uninstalled torch-2.6.0+cu124\n",
            "  Attempting uninstall: torchvision\n",
            "    Found existing installation: torchvision 0.21.0+cu124\n",
            "    Uninstalling torchvision-0.21.0+cu124:\n",
            "      Successfully uninstalled torchvision-0.21.0+cu124\n",
            "\u001b[31mERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.\n",
            "fastai 2.7.19 requires torch<2.7,>=1.10, but you have torch 2.7.1 which is incompatible.\n",
            "torchaudio 2.6.0+cu124 requires torch==2.6.0, but you have torch 2.7.1 which is incompatible.\u001b[0m\u001b[31m\n",
            "\u001b[0mSuccessfully installed nvidia-cublas-cu12-12.6.4.1 nvidia-cuda-cupti-cu12-12.6.80 nvidia-cuda-nvrtc-cu12-12.6.77 nvidia-cuda-runtime-cu12-12.6.77 nvidia-cudnn-cu12-9.5.1.17 nvidia-cufft-cu12-11.3.0.4 nvidia-cufile-cu12-1.11.1.6 nvidia-curand-cu12-10.3.7.77 nvidia-cusolver-cu12-11.7.1.2 nvidia-cusparse-cu12-12.5.4.2 nvidia-cusparselt-cu12-0.6.3 nvidia-nccl-cu12-2.26.2 nvidia-nvjitlink-cu12-12.6.85 nvidia-nvtx-cu12-12.6.77 sympy-1.14.0 torch-2.7.1 torchvision-0.22.1 triton-3.3.1\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "# Import libraries"
      ],
      "metadata": {
        "id": "k-uJ8dALX5Rp"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "import os\n",
        "import torch\n",
        "import numpy as np\n",
        "from PIL import Image\n",
        "from tqdm import tqdm\n",
        "import matplotlib.pyplot as plt\n",
        "import seaborn as sns\n",
        "from sklearn.metrics import classification_report, confusion_matrix\n",
        "from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, random_split\n",
        "import torch.nn as nn\n",
        "import torchvision.transforms as transforms\n",
        "from torchvision import models\n",
        "from vit_pytorch import ViT"
      ],
      "metadata": {
        "id": "-2APvgIAXuO0"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "# Enable Tensor Cores and mixed precision for T4 GPU"
      ],
      "metadata": {
        "id": "tx0ocPW4X-yD"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "torch.backends.cudnn.benchmark = True\n",
        "torch.set_float32_matmul_precision('medium')"
      ],
      "metadata": {
        "id": "vsSSQ3vuXuMg"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "# Set random seeds for reproducibility"
      ],
      "metadata": {
        "id": "1NCI_MxvYC6U"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "torch.manual_seed(42)\n",
        "np.random.seed(42)"
      ],
      "metadata": {
        "id": "ktXwGeRxXuKJ"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "# Device configuration (will use T4 GPU if available)"
      ],
      "metadata": {
        "id": "vZUJljqHYGjD"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n",
        "print(f\"Using device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})\")"
      ],
      "metadata": {
        "id": "QfLpQJ1kXuHl",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "aeacff19-06d4-4ac8-fcb4-24973a204131"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Using device: cuda (Tesla T4)\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "# Data path in Google Drive"
      ],
      "metadata": {
        "id": "51Qn3mFvYKO8"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "data_dir = \"/content/drive/MyDrive/Augmented\"  # Update with your path\n",
        "classes = ['glioma', 'meningioma', 'pituitary', 'no_tumor']"
      ],
      "metadata": {
        "id": "TsKoA2DnXuFO"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "# Verify dataset structure"
      ],
      "metadata": {
        "id": "ZCGUAQQmYOOM"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "for cls in classes:\n",
        "    cls_path = os.path.join(data_dir, cls)\n",
        "    if not os.path.exists(cls_path):\n",
        "        raise FileNotFoundError(f\"Directory {cls_path} not found\")\n",
        "    print(f\"{cls}: {len(os.listdir(cls_path))} images\")"
      ],
      "metadata": {
        "id": "Wp2UqYLHXuCz",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "a18e978d-8562-4f7f-fa74-5921f95d5c71"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "glioma: 2437 images\n",
            "meningioma: 2438 images\n",
            "pituitary: 2402 images\n",
            "no_tumor: 1897 images\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "# WHO Grading System Mapping"
      ],
      "metadata": {
        "id": "BT3KyhQrYRbO"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "grade_mapping = {\n",
        "    'glioma': 3,      # Grade IV (Glioblastoma)\n",
        "    'meningioma': 0,  # Grade I\n",
        "    'pituitary': 0,   # Grade I\n",
        "    'no_tumor': -1    # Special case (no tumor)\n",
        "}\n",
        "\n",
        "grade_descriptions = [\n",
        "    \"Grade I (Least aggressive, benign)\",\n",
        "    \"Grade II (Moderately aggressive)\",\n",
        "    \"Grade III (Aggressive)\",\n",
        "    \"Grade IV (Most aggressive, malignant)\"\n",
        "]"
      ],
      "metadata": {
        "id": "EnrQXkrsXuAW"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "# Optimized Dataset Class with caching for T4 GPU"
      ],
      "metadata": {
        "id": "5a7wutQdYYYy"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "class BrainTumorDataset(Dataset):\n",
        "    def __init__(self, root_dir, transform=None, is_pretrain=False):\n",
        "        self.root_dir = root_dir\n",
        "        self.transform = transform\n",
        "        self.is_pretrain = is_pretrain\n",
        "        self.samples = []\n",
        "\n",
        "        for class_idx, class_name in enumerate(classes):\n",
        "            class_dir = os.path.join(root_dir, class_name)\n",
        "            for img_name in os.listdir(class_dir):\n",
        "                img_path = os.path.join(class_dir, img_name)\n",
        "                self.samples.append({\n",
        "                    'path': img_path,\n",
        "                    'type': class_idx,\n",
        "                    'grade': grade_mapping[class_name]\n",
        "                })\n",
        "\n",
        "    def __len__(self):\n",
        "        return len(self.samples)\n",
        "\n",
        "    def __getitem__(self, idx):\n",
        "        sample = self.samples[idx]\n",
        "        try:\n",
        "            image = Image.open(sample['path']).convert('RGB')\n",
        "\n",
        "            if self.is_pretrain:\n",
        "                # For pretraining, we return two augmented views\n",
        "                if self.transform:\n",
        "                    view1 = self.transform(image)\n",
        "                    view2 = self.transform(image)\n",
        "                return view1, view2\n",
        "            else:\n",
        "                # For supervised training\n",
        "                if self.transform:\n",
        "                    image = self.transform(image)\n",
        "                return {\n",
        "                    'image': image,\n",
        "                    'type': sample['type'],\n",
        "                    'grade': sample['grade']\n",
        "                }\n",
        "        except Exception as e:\n",
        "            print(f\"Error loading {sample['path']}: {e}\")\n",
        "            return self[np.random.randint(0, len(self))]\n",
        "\n",
        "# Data augmentations for self-supervised learning\n",
        "pretrain_transform = transforms.Compose([\n",
        "    transforms.Resize((224, 224)),\n",
        "    transforms.RandomHorizontalFlip(),\n",
        "    transforms.RandomVerticalFlip(),\n",
        "    transforms.RandomRotation(15),\n",
        "    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),\n",
        "    transforms.GaussianBlur(kernel_size=9),\n",
        "    transforms.ToTensor(),\n",
        "    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])\n",
        "])\n",
        "\n",
        "# Transformations for supervised training\n",
        "finetune_transform = transforms.Compose([\n",
        "    transforms.Resize((224, 224)),\n",
        "    transforms.RandomHorizontalFlip(),\n",
        "    transforms.RandomVerticalFlip(),\n",
        "    transforms.RandomRotation(15),\n",
        "    transforms.ColorJitter(brightness=0.1, contrast=0.1),\n",
        "    transforms.ToTensor(),\n",
        "    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])\n",
        "])\n",
        "\n",
        "# Load full dataset for pretraining\n",
        "pretrain_dataset = BrainTumorDataset(data_dir, transform=pretrain_transform, is_pretrain=True)\n",
        "\n",
        "# Split dataset (80% train, 20% test)\n",
        "train_size = int(0.8 * len(pretrain_dataset))\n",
        "test_size = len(pretrain_dataset) - train_size\n",
        "pretrain_train_dataset, pretrain_val_dataset = random_split(pretrain_dataset, [train_size, test_size])\n",
        "\n",
        "# Optimized DataLoader settings for T4 GPU\n",
        "num_workers = 2 if torch.cuda.is_available() else 0\n",
        "pin_memory = torch.cuda.is_available()\n",
        "\n",
        "pretrain_train_loader = DataLoader(\n",
        "    pretrain_train_dataset,\n",
        "    batch_size=32,\n",
        "    shuffle=True,\n",
        "    num_workers=num_workers,\n",
        "    pin_memory=pin_memory,\n",
        "    persistent_workers=True if num_workers > 0 else False\n",
        ")\n",
        "\n",
        "pretrain_val_loader = DataLoader(\n",
        "    pretrain_val_dataset,\n",
        "    batch_size=32,\n",
        "    shuffle=False,\n",
        "    num_workers=num_workers,\n",
        "    pin_memory=pin_memory,\n",
        "    persistent_workers=True if num_workers > 0 else False\n",
        ")\n",
        "\n",
        "print(f\"\\nPretraining dataset split:\")\n",
        "print(f\"Training samples: {len(pretrain_train_dataset)}\")\n",
        "print(f\"Testing samples: {len(pretrain_val_dataset)}\")"
      ],
      "metadata": {
        "id": "QL1RAuySXt95",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "4277ed25-4c01-4a88-8e65-b24cd2c17744"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "Pretraining dataset split:\n",
            "Training samples: 7339\n",
            "Validation samples: 1835\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "# Data augmentations for self-supervised learning"
      ],
      "metadata": {
        "id": "r7Fkm5y6Ylia"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "pretrain_transform = transforms.Compose([\n",
        "    transforms.Resize((224, 224)),\n",
        "    transforms.RandomHorizontalFlip(),\n",
        "    transforms.RandomVerticalFlip(),\n",
        "    transforms.RandomRotation(15),\n",
        "    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),\n",
        "    transforms.GaussianBlur(kernel_size=9),\n",
        "    transforms.ToTensor(),\n",
        "    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])\n",
        "])\n",
        "\n",
        "# Transformations for supervised training\n",
        "finetune_transform = transforms.Compose([\n",
        "    transforms.Resize((224, 224)),\n",
        "    transforms.RandomHorizontalFlip(),\n",
        "    transforms.RandomVerticalFlip(),\n",
        "    transforms.RandomRotation(15),\n",
        "    transforms.ColorJitter(brightness=0.1, contrast=0.1),\n",
        "    transforms.ToTensor(),\n",
        "    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])\n",
        "])\n",
        "\n",
        "# Load full dataset for pretraining\n",
        "pretrain_dataset = BrainTumorDataset(data_dir, transform=pretrain_transform, is_pretrain=True)\n",
        "\n",
        "# Split dataset (80% train, 20% test)\n",
        "train_size = int(0.8 * len(pretrain_dataset))\n",
        "test_size = len(pretrain_dataset) - train_size\n",
        "pretrain_train_dataset, pretrain_val_dataset = random_split(pretrain_dataset, [train_size, test_size])\n",
        "\n",
        "# Optimized DataLoader settings for T4 GPU\n",
        "num_workers = 2 if torch.cuda.is_available() else 0\n",
        "pin_memory = torch.cuda.is_available()\n",
        "\n",
        "pretrain_train_loader = DataLoader(\n",
        "    pretrain_train_dataset,\n",
        "    batch_size=32,\n",
        "    shuffle=True,\n",
        "    num_workers=num_workers,\n",
        "    pin_memory=pin_memory,\n",
        "    persistent_workers=True if num_workers > 0 else False\n",
        ")\n",
        "\n",
        "pretrain_val_loader = DataLoader(\n",
        "    pretrain_val_dataset,\n",
        "    batch_size=32,\n",
        "    shuffle=False,\n",
        "    num_workers=num_workers,\n",
        "    pin_memory=pin_memory,\n",
        "    persistent_workers=True if num_workers > 0 else False\n",
        ")\n",
        "\n",
        "print(f\"\\nPretraining dataset split:\")\n",
        "print(f\"Training samples: {len(pretrain_train_dataset)}\")\n",
        "print(f\"Validation samples: {len(pretrain_val_dataset)}\")"
      ],
      "metadata": {
        "id": "iPHK7YEhYX5Y",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "c2e8aacf-8dcc-475a-cbb1-3a4e42fb9386"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "Pretraining dataset split:\n",
            "Training samples: 7339\n",
            "Validation samples: 1835\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "# Self-Supervised Model (SimMIM approach)"
      ],
      "metadata": {
        "id": "tqwSr6-fYtjS"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "class SelfSupervisedViT(nn.Module):\n",
        "    def __init__(self):\n",
        "        super().__init__()\n",
        "\n",
        "        # Create the encoder (ViT)\n",
        "        self.encoder = ViT(\n",
        "            image_size=224,\n",
        "            patch_size=32,\n",
        "            num_classes=768,\n",
        "            dim=768,\n",
        "            depth=4,\n",
        "            heads=8,\n",
        "            mlp_dim=1536,\n",
        "            dropout=0.1,\n",
        "            emb_dropout=0.1\n",
        "        )\n",
        "\n",
        "        # Projection head for contrastive learning\n",
        "        self.projection_head = nn.Sequential(\n",
        "            nn.Linear(768, 768),\n",
        "            nn.GELU(),\n",
        "            nn.Linear(768, 256)\n",
        "        )\n",
        "\n",
        "    def forward(self, x):\n",
        "        features = self.encoder(x)\n",
        "        projections = self.projection_head(features)\n",
        "        return projections\n",
        "\n",
        "# Initialize self-supervised model\n",
        "ss_model = SelfSupervisedViT().to(device)\n",
        "\n",
        "# Contrastive loss (NT-Xent loss)\n",
        "class ContrastiveLoss(nn.Module):\n",
        "    def __init__(self, temperature=0.1):\n",
        "        super().__init__()\n",
        "        self.temperature = temperature\n",
        "        self.criterion = nn.CrossEntropyLoss()\n",
        "\n",
        "    def forward(self, z_i, z_j):\n",
        "        batch_size = z_i.shape[0]\n",
        "\n",
        "        # Concatenate projections\n",
        "        projections = torch.cat([z_i, z_j], dim=0)\n",
        "\n",
        "        # Compute similarity matrix\n",
        "        sim_matrix = torch.matmul(projections, projections.T) / self.temperature\n",
        "\n",
        "        # Create labels for the contrastive loss\n",
        "        labels = torch.cat([torch.arange(batch_size) for _ in range(2)], dim=0)\n",
        "        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float().to(device)\n",
        "\n",
        "        # Remove diagonal (self-similarity)\n",
        "        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(device)\n",
        "        labels = labels[~mask].view(labels.shape[0], -1)\n",
        "        sim_matrix = sim_matrix[~mask].view(sim_matrix.shape[0], -1)\n",
        "\n",
        "        # Select positives and negatives\n",
        "        positives = sim_matrix[labels.bool()].view(labels.shape[0], -1)\n",
        "        negatives = sim_matrix[~labels.bool()].view(sim_matrix.shape[0], -1)\n",
        "\n",
        "        logits = torch.cat([positives, negatives], dim=1)\n",
        "        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(device)\n",
        "\n",
        "        loss = self.criterion(logits, labels)\n",
        "        return loss\n",
        "\n",
        "# Pretraining setup\n",
        "ss_optimizer = torch.optim.AdamW(ss_model.parameters(), lr=1e-4, weight_decay=1e-5)\n",
        "ss_criterion = ContrastiveLoss(temperature=0.1)\n",
        "scaler = torch.cuda.amp.GradScaler()\n",
        "\n",
        "# Pretraining function\n",
        "def pretrain_epoch(epoch):\n",
        "    ss_model.train()\n",
        "    running_loss = 0.0\n",
        "    total = 0\n",
        "\n",
        "    pbar = tqdm(pretrain_train_loader, desc=f\"Pretrain Epoch {epoch+1}\")\n",
        "    for view1, view2 in pbar:\n",
        "        view1 = view1.to(device, non_blocking=True)\n",
        "        view2 = view2.to(device, non_blocking=True)\n",
        "\n",
        "        ss_optimizer.zero_grad(set_to_none=True)\n",
        "\n",
        "        # Mixed precision training\n",
        "        with torch.amp.autocast(device_type='cuda', dtype=torch.float16):\n",
        "            z1 = ss_model(view1)\n",
        "            z2 = ss_model(view2)\n",
        "            loss = ss_criterion(z1, z2)\n",
        "\n",
        "        scaler.scale(loss).backward()\n",
        "        scaler.step(ss_optimizer)\n",
        "        scaler.update()\n",
        "\n",
        "        running_loss += loss.item() * view1.size(0)\n",
        "        total += view1.size(0)\n",
        "        pbar.set_postfix({'Loss': running_loss/total, 'LR': ss_optimizer.param_groups[0]['lr']})\n",
        "\n",
        "    return running_loss / total"
      ],
      "metadata": {
        "id": "9EDpxwMMXt7g",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "e46c6849-4bb2-465e-91c8-1034413b822c"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "<ipython-input-12-65187176>:71: FutureWarning: `torch.cuda.amp.GradScaler(args...)` is deprecated. Please use `torch.amp.GradScaler('cuda', args...)` instead.\n",
            "  scaler = torch.cuda.amp.GradScaler()\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "# Pretraining loop"
      ],
      "metadata": {
        "id": "hgUEOfdgY4Ch"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "num_pretrain_epochs = 10  # Can be increased for better performance\n",
        "print(\"\\nStarting self-supervised pretraining...\")\n",
        "for epoch in range(num_pretrain_epochs):\n",
        "    train_loss = pretrain_epoch(epoch)\n",
        "    print(f\"Pretrain Epoch {epoch+1}/{num_pretrain_epochs} Loss: {train_loss:.4f}\")\n",
        "\n",
        "print(\"\\nSelf-supervised pretraining completed!\")\n",
        "\n",
        "# Save pretrained weights\n",
        "save_dir = \"/content/drive/MyDrive/BrainTumorModels\"\n",
        "os.makedirs(save_dir, exist_ok=True)\n",
        "pretrain_path = os.path.join(save_dir, \"pretrained_vit.pth\")\n",
        "torch.save(ss_model.encoder.state_dict(), pretrain_path)\n",
        "print(f\"Saved pretrained encoder to {pretrain_path}\")\n",
        "\n",
        "# Now load the supervised dataset for fine-tuning\n",
        "finetune_dataset = BrainTumorDataset(data_dir, transform=finetune_transform, is_pretrain=False)\n",
        "\n",
        "# Split dataset (80% train, 20% test)\n",
        "train_size = int(0.8 * len(finetune_dataset))\n",
        "test_size = len(finetune_dataset) - train_size\n",
        "train_dataset, test_dataset = random_split(finetune_dataset, [train_size, test_size])\n",
        "\n",
        "# Calculate class weights for imbalanced dataset\n",
        "# Get the targets from your dataset\n",
        "targets = [sample['type'] for sample in finetune_dataset.samples]  # Get all targets\n",
        "train_targets = [finetune_dataset.samples[i]['type'] for i in train_dataset.indices]  # Get train targets only\n",
        "\n",
        "# Calculate class weights\n",
        "class_counts = torch.bincount(torch.tensor(train_targets))\n",
        "num_samples = sum(class_counts)\n",
        "class_weights = num_samples / (len(class_counts) * (class_counts + 1e-6))  # Add small epsilon to avoid division by zero\n",
        "weights = class_weights[torch.tensor(train_targets)]\n",
        "\n",
        "# Create sampler\n",
        "sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)\n",
        "\n",
        "# DataLoaders for fine-tuning\n",
        "train_loader = DataLoader(\n",
        "    train_dataset,\n",
        "    batch_size=32,\n",
        "    sampler=sampler,  # Using weighted sampler instead of shuffle\n",
        "    num_workers=num_workers,\n",
        "    pin_memory=pin_memory,\n",
        "    persistent_workers=True if num_workers > 0 else False\n",
        ")\n",
        "\n",
        "test_loader = DataLoader(\n",
        "    test_dataset,\n",
        "    batch_size=32,\n",
        "    shuffle=False,\n",
        "    num_workers=num_workers,\n",
        "    pin_memory=pin_memory,\n",
        "    persistent_workers=True if num_workers > 0 else False\n",
        ")\n",
        "\n",
        "print(f\"\\nFine-tuning dataset split:\")\n",
        "print(f\"Training samples: {len(train_dataset)}\")\n",
        "print(f\"Testing samples: {len(test_dataset)}\")\n",
        "print(f\"Class distribution in training set: {dict(zip(classes, class_counts.tolist()))}\")"
      ],
      "metadata": {
        "id": "7cwitS-EXt48",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "30fd305a-5073-49b7-ecad-c2e730adcbd3"
      },
      "execution_count": null,
      "outputs": [
        {
          "metadata": {
            "tags": null
          },
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "\n",
            "Starting self-supervised pretraining...\n"
          ]
        },
        {
          "metadata": {
            "tags": null
          },
          "name": "stderr",
          "output_type": "stream",
          "text": [
            "Pretrain Epoch 1: 100%|██████████| 230/230 [18:42<00:00,  4.88s/it, Loss=2.55, LR=0.0001]\n"
          ]
        },
        {
          "metadata": {
            "tags": null
          },
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "Pretrain Epoch 1/10 Loss: 2.5483\n"
          ]
        },
        {
          "metadata": {
            "tags": null
          },
          "name": "stderr",
          "output_type": "stream",
          "text": [
            "Pretrain Epoch 2: 100%|██████████| 230/230 [03:22<00:00,  1.14it/s, Loss=1.26, LR=0.0001]\n"
          ]
        },
        {
          "metadata": {
            "tags": null
          },
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "Pretrain Epoch 2/10 Loss: 1.2648\n"
          ]
        },
        {
          "metadata": {
            "tags": null
          },
          "name": "stderr",
          "output_type": "stream",
          "text": [
            "Pretrain Epoch 3: 100%|██████████| 230/230 [03:19<00:00,  1.15it/s, Loss=0.778, LR=0.0001]\n"
          ]
        },
        {
          "metadata": {
            "tags": null
          },
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "Pretrain Epoch 3/10 Loss: 0.7782\n"
          ]
        },
        {
          "metadata": {
            "tags": null
          },
          "name": "stderr",
          "output_type": "stream",
          "text": [
            "Pretrain Epoch 4: 100%|██████████| 230/230 [03:14<00:00,  1.19it/s, Loss=0.572, LR=0.0001]\n"
          ]
        },
        {
          "metadata": {
            "tags": null
          },
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "Pretrain Epoch 4/10 Loss: 0.5723\n"
          ]
        },
        {
          "metadata": {
            "tags": null
          },
          "name": "stderr",
          "output_type": "stream",
          "text": [
            "Pretrain Epoch 5: 100%|██████████| 230/230 [03:13<00:00,  1.19it/s, Loss=0.492, LR=0.0001]\n"
          ]
        },
        {
          "metadata": {
            "tags": null
          },
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "Pretrain Epoch 5/10 Loss: 0.4923\n"
          ]
        },
        {
          "metadata": {
            "tags": null
          },
          "name": "stderr",
          "output_type": "stream",
          "text": [
            "Pretrain Epoch 6: 100%|██████████| 230/230 [03:12<00:00,  1.20it/s, Loss=0.416, LR=0.0001]\n"
          ]
        },
        {
          "metadata": {
            "tags": null
          },
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "Pretrain Epoch 6/10 Loss: 0.4156\n"
          ]
        },
        {
          "metadata": {
            "tags": null
          },
          "name": "stderr",
          "output_type": "stream",
          "text": [
            "Pretrain Epoch 7: 100%|██████████| 230/230 [03:17<00:00,  1.17it/s, Loss=0.35, LR=0.0001]\n"
          ]
        },
        {
          "metadata": {
            "tags": null
          },
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "Pretrain Epoch 7/10 Loss: 0.3502\n"
          ]
        },
        {
          "metadata": {
            "tags": null
          },
          "name": "stderr",
          "output_type": "stream",
          "text": [
            "Pretrain Epoch 8: 100%|██████████| 230/230 [03:18<00:00,  1.16it/s, Loss=0.321, LR=0.0001]\n"
          ]
        },
        {
          "metadata": {
            "tags": null
          },
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "Pretrain Epoch 8/10 Loss: 0.3214\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "Pretrain Epoch 9: 100%|██████████| 230/230 [03:17<00:00,  1.16it/s, Loss=0.289, LR=0.0001]\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Pretrain Epoch 9/10 Loss: 0.2887\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "Pretrain Epoch 10: 100%|██████████| 230/230 [03:18<00:00,  1.16it/s, Loss=0.28, LR=0.0001]\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Pretrain Epoch 10/10 Loss: 0.2798\n",
            "\n",
            "Self-supervised pretraining completed!\n",
            "Saved pretrained encoder to /content/drive/MyDrive/BrainTumorModels/pretrained_vit.pth\n",
            "\n",
            "Fine-tuning dataset split:\n",
            "Training samples: 7339\n",
            "Testing samples: 1835\n",
            "Class distribution in training set: {'glioma': 1934, 'meningioma': 1965, 'pituitary': 1924, 'no_tumor': 1516}\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "# Hybrid ViT-CNN Model with pretrained ViT"
      ],
      "metadata": {
        "id": "rp5zok3oZDty"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "class HybridTumorModel(nn.Module):\n",
        "    def __init__(self, num_types=4, num_grades=4):\n",
        "        super().__init__()\n",
        "\n",
        "        # CNN Backbone (EfficientNet-B0)\n",
        "        self.cnn = models.efficientnet_b0(pretrained=True)\n",
        "        self.cnn_features = nn.Sequential(*list(self.cnn.children())[:-2])\n",
        "\n",
        "        # Vision Transformer (Initialized with pretrained weights)\n",
        "        self.vit = ViT(\n",
        "            image_size=224,\n",
        "            patch_size=32,\n",
        "            num_classes=768,\n",
        "            dim=768,\n",
        "            depth=4,\n",
        "            heads=8,\n",
        "            mlp_dim=1536,\n",
        "            dropout=0.1,\n",
        "            emb_dropout=0.1\n",
        "        )\n",
        "\n",
        "        # Load pretrained weights\n",
        "        vit_state_dict = torch.load(pretrain_path)\n",
        "        self.vit.load_state_dict(vit_state_dict)\n",
        "\n",
        "        # Feature fusion with proper dimensions\n",
        "        self.cnn_proj = nn.Linear(1280, 384)\n",
        "        self.vit_proj = nn.Linear(768, 384)\n",
        "\n",
        "        # Classifiers\n",
        "        self.type_classifier = nn.Sequential(\n",
        "            nn.Linear(384, 256),\n",
        "            nn.GELU(),\n",
        "            nn.Dropout(0.3),\n",
        "            nn.Linear(256, num_types)\n",
        "        )\n",
        "\n",
        "        self.grade_classifier = nn.Sequential(\n",
        "            nn.Linear(384, 256),\n",
        "            nn.GELU(),\n",
        "            nn.Dropout(0.3),\n",
        "            nn.Linear(256, num_grades)\n",
        "        )\n",
        "\n",
        "    def forward(self, x):\n",
        "        # CNN features\n",
        "        cnn_feats = self.cnn_features(x)\n",
        "        cnn_feats = cnn_feats.mean(dim=[2, 3])\n",
        "        cnn_feats = self.cnn_proj(cnn_feats)\n",
        "\n",
        "        # ViT features\n",
        "        vit_feats = self.vit(x)\n",
        "        vit_feats = self.vit_proj(vit_feats)\n",
        "\n",
        "        # Feature fusion (element-wise addition)\n",
        "        fused = cnn_feats + vit_feats\n",
        "\n",
        "        # Multi-task output\n",
        "        type_logits = self.type_classifier(fused)\n",
        "        grade_logits = self.grade_classifier(fused)\n",
        "\n",
        "        return type_logits, grade_logits"
      ],
      "metadata": {
        "id": "ge_MrVxvXt2Y"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "# Initialize model with mixed precision"
      ],
      "metadata": {
        "id": "gTWzk13XZLXw"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "model = HybridTumorModel().to(device)\n",
        "scaler = torch.cuda.amp.GradScaler()"
      ],
      "metadata": {
        "id": "bVgacMDTXtzn",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "b23ecb62-d722-4016-cae6-50ff8408d00a"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "/usr/local/lib/python3.11/dist-packages/torchvision/models/_utils.py:208: UserWarning: The parameter 'pretrained' is deprecated since 0.13 and may be removed in the future, please use 'weights' instead.\n",
            "  warnings.warn(\n",
            "/usr/local/lib/python3.11/dist-packages/torchvision/models/_utils.py:223: UserWarning: Arguments other than a weight enum or `None` for 'weights' are deprecated since 0.13 and may be removed in the future. The current behavior is equivalent to passing `weights=EfficientNet_B0_Weights.IMAGENET1K_V1`. You can also use `weights=EfficientNet_B0_Weights.DEFAULT` to get the most up-to-date weights.\n",
            "  warnings.warn(msg)\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Downloading: \"https://download.pytorch.org/models/efficientnet_b0_rwightman-7f5810bc.pth\" to /root/.cache/torch/hub/checkpoints/efficientnet_b0_rwightman-7f5810bc.pth\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "100%|██████████| 20.5M/20.5M [00:00<00:00, 100MB/s]\n",
            "<ipython-input-15-1021322196>:2: FutureWarning: `torch.cuda.amp.GradScaler(args...)` is deprecated. Please use `torch.amp.GradScaler('cuda', args...)` instead.\n",
            "  scaler = torch.cuda.amp.GradScaler()\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "# Print model summary"
      ],
      "metadata": {
        "id": "N2G7Ex02ZPQG"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "total_params = sum(p.numel() for p in model.parameters())\n",
        "trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)\n",
        "print(f\"\\nModel architecture:\")\n",
        "print(model)\n",
        "print(f\"\\nTotal parameters: {total_params:,}\")\n",
        "print(f\"Trainable parameters: {trainable_params:,}\")\n",
        "\n",
        "# Loss functions with class weighting\n",
        "type_weights = torch.tensor([1.0, 1.28, 1.0, 1.01]).to(device)  # Rough inverse frequency weights\n",
        "type_criterion = nn.CrossEntropyLoss(weight=type_weights)\n",
        "grade_criterion = nn.CrossEntropyLoss(ignore_index=-1)\n",
        "\n",
        "# Optimizer (AdamW with weight decay)\n",
        "optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)\n",
        "\n",
        "# Learning rate scheduler\n",
        "scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(\n",
        "    optimizer,\n",
        "    mode='min',\n",
        "    factor=0.5,\n",
        "    patience=3\n",
        ")\n",
        "\n",
        "# Training loop with mixed precision for T4 GPU\n",
        "def train_epoch(epoch):\n",
        "    model.train()\n",
        "    running_loss = 0.0\n",
        "    type_correct = 0\n",
        "    grade_correct = 0\n",
        "    total = 0\n",
        "\n",
        "    pbar = tqdm(train_loader, desc=f\"Fine-tuning Epoch {epoch+1}\")\n",
        "    for batch in pbar:\n",
        "        images = batch['image'].to(device, non_blocking=True)\n",
        "        types = batch['type'].to(device, non_blocking=True)\n",
        "        grades = batch['grade'].to(device, non_blocking=True)\n",
        "\n",
        "        optimizer.zero_grad(set_to_none=True)\n",
        "\n",
        "        # Mixed precision training\n",
        "        with torch.amp.autocast(device_type='cuda', dtype=torch.float16):\n",
        "            type_pred, grade_pred = model(images)\n",
        "            loss_type = type_criterion(type_pred, types)\n",
        "            loss_grade = grade_criterion(grade_pred, grades)\n",
        "            loss = 0.6 * loss_type + 0.4 * loss_grade\n",
        "\n",
        "        scaler.scale(loss).backward()\n",
        "        scaler.step(optimizer)\n",
        "        scaler.update()\n",
        "\n",
        "        # Metrics\n",
        "        running_loss += loss.item() * images.size(0)\n",
        "        _, type_predicted = torch.max(type_pred, 1)\n",
        "        type_correct += (type_predicted == types).sum().item()\n",
        "\n",
        "        grade_mask = grades != -1\n",
        "        if grade_mask.any():\n",
        "            _, grade_predicted = torch.max(grade_pred[grade_mask], 1)\n",
        "            grade_correct += (grade_predicted == grades[grade_mask]).sum().item()\n",
        "\n",
        "        total += images.size(0)\n",
        "        pbar.set_postfix({\n",
        "            'Loss': running_loss/total,\n",
        "            'Type Acc': type_correct/total,\n",
        "            'LR': optimizer.param_groups[0]['lr']\n",
        "        })\n",
        "\n",
        "    epoch_loss = running_loss / total\n",
        "    type_acc = type_correct / total\n",
        "    grade_acc = grade_correct / (grades != -1).sum().item() if (grades != -1).any() else 0\n",
        "\n",
        "    return epoch_loss, type_acc, grade_acc\n",
        "\n",
        "def evaluate():\n",
        "    model.eval()\n",
        "    running_loss = 0.0\n",
        "    type_correct = 0\n",
        "    grade_correct = 0\n",
        "    total = 0\n",
        "\n",
        "    all_type_preds = []\n",
        "    all_type_labels = []\n",
        "    all_grade_preds = []\n",
        "    all_grade_labels = []\n",
        "\n",
        "    with torch.no_grad():\n",
        "        for batch in tqdm(test_loader, desc=\"Evaluating\"):\n",
        "            images = batch['image'].to(device, non_blocking=True)\n",
        "            types = batch['type'].to(device, non_blocking=True)\n",
        "            grades = batch['grade'].to(device, non_blocking=True)\n",
        "\n",
        "            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):\n",
        "                type_pred, grade_pred = model(images)\n",
        "                loss_type = type_criterion(type_pred, types)\n",
        "                loss_grade = grade_criterion(grade_pred, grades)\n",
        "                loss = 0.6 * loss_type + 0.4 * loss_grade\n",
        "\n",
        "            running_loss += loss.item() * images.size(0)\n",
        "            _, type_predicted = torch.max(type_pred, 1)\n",
        "            type_correct += (type_predicted == types).sum().item()\n",
        "\n",
        "            grade_mask = grades != -1\n",
        "            if grade_mask.any():\n",
        "                _, grade_predicted = torch.max(grade_pred[grade_mask], 1)\n",
        "                grade_correct += (grade_predicted == grades[grade_mask]).sum().item()\n",
        "\n",
        "            total += images.size(0)\n",
        "\n",
        "            all_type_preds.extend(type_predicted.cpu().numpy())\n",
        "            all_type_labels.extend(types.cpu().numpy())\n",
        "            if grade_mask.any():\n",
        "                all_grade_preds.extend(grade_predicted.cpu().numpy())\n",
        "                all_grade_labels.extend(grades[grade_mask].cpu().numpy())\n",
        "\n",
        "    avg_loss = running_loss / total\n",
        "    type_acc = type_correct / total\n",
        "    grade_acc = grade_correct / (grades != -1).sum().item() if (grades != -1).any() else 0\n",
        "\n",
        "    return avg_loss, type_acc, grade_acc, all_type_preds, all_type_labels, all_grade_preds, all_grade_labels\n",
        "\n",
        "# Training parameters\n",
        "num_epochs = 50  # Reduced epochs for Colab demo\n",
        "best_acc = 0.0\n",
        "\n",
        "train_losses = []\n",
        "val_losses = []\n",
        "train_type_accs = []\n",
        "val_type_accs = []\n",
        "train_grade_accs = []\n",
        "val_grade_accs = []\n",
        "\n",
        "# Fine-tuning loop\n",
        "print(\"\\nStarting fine-tuning...\")\n",
        "for epoch in range(num_epochs):\n",
        "    train_loss, train_type_acc, train_grade_acc = train_epoch(epoch)\n",
        "    val_loss, val_type_acc, val_grade_acc, _, _, _, _ = evaluate()\n",
        "    scheduler.step(val_loss)\n",
        "\n",
        "    # Store metrics\n",
        "    train_losses.append(train_loss)\n",
        "    val_losses.append(val_loss)\n",
        "    train_type_accs.append(train_type_acc)\n",
        "    val_type_accs.append(val_type_acc)\n",
        "    train_grade_accs.append(train_grade_acc)\n",
        "    val_grade_accs.append(val_grade_acc)\n",
        "\n",
        "    print(f\"\\nEpoch {epoch+1}/{num_epochs} Summary:\")\n",
        "    print(f\"Train Loss: {train_loss:.4f} | Type Acc: {train_type_acc:.2%} | Grade Acc: {train_grade_acc:.2%}\")\n",
        "    print(f\"Val Loss: {val_loss:.4f} | Type Acc: {val_type_acc:.2%} | Grade Acc: {val_grade_acc:.2%}\")\n",
        "\n",
        "    # Save best model\n",
        "    if val_type_acc > best_acc:\n",
        "        best_acc = val_type_acc\n",
        "        model_path = os.path.join(save_dir, \"best_self_supervised_model_t4.pth\")\n",
        "        torch.save(model.state_dict(), model_path)\n",
        "        print(f\"Saved new best model to {model_path}\")\n",
        "\n",
        "print(\"\\nFine-tuning completed!\")"
      ],
      "metadata": {
        "id": "3zHH8qqRXtw_",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "5cee203d-7e82-4be4-c839-a3601e2993e9"
      },
      "execution_count": null,
      "outputs": [
        {
          "metadata": {
            "tags": null
          },
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "\n",
            "Model architecture:\n",
            "HybridTumorModel(\n",
            "  (cnn): EfficientNet(\n",
            "    (features): Sequential(\n",
            "      (0): Conv2dNormActivation(\n",
            "        (0): Conv2d(3, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)\n",
            "        (1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n",
            "        (2): SiLU(inplace=True)\n",
            "      )\n",
            "      (1): Sequential(\n",
            "        (0): MBConv(\n",
            "          (block): Sequential(\n",
            "            (0): Conv2dNormActivation(\n",
            "              (0): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=32, bias=False)\n",
            "              (1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n",
            "              (2): SiLU(inplace=True)\n",
            "            )\n",
            "            (1): SqueezeExcitation(\n",
            "              (avgpool): AdaptiveAvgPool2d(output_size=1)\n",
            "              (fc1): Conv2d(32, 8, kernel_size=(1, 1), stride=(1, 1))\n",
            "              (fc2): Conv2d(8, 32, kernel_size=(1, 1), stride=(1, 1))\n",
            "              (activation): SiLU(inplace=True)\n",
            "              (scale_activation): Sigmoid()\n",
            "            )\n",
            "            (2): Conv2dNormActivation(\n",
            "              (0): Conv2d(32, 16, kernel_size=(1, 1), stride=(1, 1), bias=False)\n",
            "              (1): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n",
            "            )\n",
            "          )\n",
            "          (stochastic_depth): StochasticDepth(p=0.0, mode=row)\n",
            "        )\n",
            "      )\n",
            "      (2): Sequential(\n",
            "        (0): MBConv(\n",
            "          (block): Sequential(\n",
            "            (0): Conv2dNormActivation(\n",
            "              (0): Conv2d(16, 96, kernel_size=(1, 1), stride=(1, 1), bias=False)\n",
            "              (1): BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n",
            "              (2): SiLU(inplace=True)\n",
            "            )\n",
            "            (1): Conv2dNormActivation(\n",
            "              (0): Conv2d(96, 96, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=96, bias=False)\n",
            "              (1): BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n",
            "              (2): SiLU(inplace=True)\n",
            "            )\n",
            "            (2): SqueezeExcitation(\n",
            "              (avgpool): AdaptiveAvgPool2d(output_size=1)\n",
            "              (fc1): Conv2d(96, 4, kernel_size=(1, 1), stride=(1, 1))\n",
            "              (fc2): Conv2d(4, 96, kernel_size=(1, 1), stride=(1, 1))\n",
            "              (activation): SiLU(inplace=True)\n",
            "              (scale_activation): Sigmoid()\n",
            "            )\n",
            "            (3): Conv2dNormActivation(\n",
            "              (0): Conv2d(96, 24, kernel_size=(1, 1), stride=(1, 1), bias=False)\n",
            "              (1): BatchNorm2d(24, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n",
            "            )\n",
            "          )\n",
            "          (stochastic_depth): StochasticDepth(p=0.0125, mode=row)\n",
            "        )\n",
            "        (1): MBConv(\n",
            "          (block): Sequential(\n",
            "            (0): Conv2dNormActivation(\n",
            "              (0): Conv2d(24, 144, kernel_size=(1, 1), stride=(1, 1), bias=False)\n",
            "              (1): BatchNorm2d(144, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n",
            "              (2): SiLU(inplace=True)\n",
            "            )\n",
            "            (1): Conv2dNormActivation(\n",
            "              (0): Conv2d(144, 144, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=144, bias=False)\n",
            "              (1): BatchNorm2d(144, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n",
            "              (2): SiLU(inplace=True)\n",
            "            )\n",
            "            (2): SqueezeExcitation(\n",
            "              (avgpool): AdaptiveAvgPool2d(output_size=1)\n",
            "              (fc1): Conv2d(144, 6, kernel_size=(1, 1), stride=(1, 1))\n",
            "              (fc2): Conv2d(6, 144, kernel_size=(1, 1), stride=(1, 1))\n",
            "              (activation): SiLU(inplace=True)\n",
            "              (scale_activation): Sigmoid()\n",
            "            )\n",
            "            (3): Conv2dNormActivation(\n",
            "              (0): Conv2d(144, 24, kernel_size=(1, 1), stride=(1, 1), bias=False)\n",
            "              (1): BatchNorm2d(24, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n",
            "            )\n",
            "          )\n",
            "          (stochastic_depth): StochasticDepth(p=0.025, mode=row)\n",
            "        )\n",
            "      )\n",
            "      (3): Sequential(\n",
            "        (0): MBConv(\n",
            "          (block): Sequential(\n",
            "            (0): Conv2dNormActivation(\n",
            "              (0): Conv2d(24, 144, kernel_size=(1, 1), stride=(1, 1), bias=False)\n",
            "              (1): BatchNorm2d(144, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n",
            "              (2): SiLU(inplace=True)\n",
            "            )\n",
            "            (1): Conv2dNormActivation(\n",
            "              (0): Conv2d(144, 144, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), groups=144, bias=False)\n",
            "              (1): BatchNorm2d(144, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n",
            "              (2): SiLU(inplace=True)\n",
            "            )\n",
            "            (2): SqueezeExcitation(\n",
            "              (avgpool): AdaptiveAvgPool2d(output_size=1)\n",
            "              (fc1): Conv2d(144, 6, kernel_size=(1, 1), stride=(1, 1))\n",
            "              (fc2): Conv2d(6, 144, kernel_size=(1, 1), stride=(1, 1))\n",
            "              (activation): SiLU(inplace=True)\n",
            "              (scale_activation): Sigmoid()\n",
            "            )\n",
            "            (3): Conv2dNormActivation(\n",
            "              (0): Conv2d(144, 40, kernel_size=(1, 1), stride=(1, 1), bias=False)\n",
            "              (1): BatchNorm2d(40, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n",
            "            )\n",
            "          )\n",
            "          (stochastic_depth): StochasticDepth(p=0.037500000000000006, mode=row)\n",
            "        )\n",
            "        (1): MBConv(\n",
            "          (block): Sequential(\n",
            "            (0): Conv2dNormActivation(\n",
            "              (0): Conv2d(40, 240, kernel_size=(1, 1), stride=(1, 1), bias=False)\n",
            "              (1): BatchNorm2d(240, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n",
            "              (2): SiLU(inplace=True)\n",
            "            )\n",
            "            (1): Conv2dNormActivation(\n",
            "              (0): Conv2d(240, 240, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=240, bias=False)\n",
            "              (1): BatchNorm2d(240, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n",
            "              (2): SiLU(inplace=True)\n",
            "            )\n",
            "            (2): SqueezeExcitation(\n",
            "              (avgpool): AdaptiveAvgPool2d(output_size=1)\n",
            "              (fc1): Conv2d(240, 10, kernel_size=(1, 1), stride=(1, 1))\n",
            "              (fc2): Conv2d(10, 240, kernel_size=(1, 1), stride=(1, 1))\n",
            "              (activation): SiLU(inplace=True)\n",
            "              (scale_activation): Sigmoid()\n",
            "            )\n",
            "            (3): Conv2dNormActivation(\n",
            "              (0): Conv2d(240, 40, kernel_size=(1, 1), stride=(1, 1), bias=False)\n",
            "              (1): BatchNorm2d(40, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n",
            "            )\n",
            "          )\n",
            "          (stochastic_depth): StochasticDepth(p=0.05, mode=row)\n",
            "        )\n",
            "      )\n",
            "      (4): Sequential(\n",
            "        (0): MBConv(\n",
            "          (block): Sequential(\n",
            "            (0): Conv2dNormActivation(\n",
            "              (0): Conv2d(40, 240, kernel_size=(1, 1), stride=(1, 1), bias=False)\n",
            "              (1): BatchNorm2d(240, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n",
            "              (2): SiLU(inplace=True)\n",
            "            )\n",
            "            (1): Conv2dNormActivation(\n",
            "              (0): Conv2d(240, 240, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=240, bias=False)\n",
            "              (1): BatchNorm2d(240, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n",
            "              (2): SiLU(inplace=True)\n",
            "            )\n",
            "            (2): SqueezeExcitation(\n",
            "              (avgpool): AdaptiveAvgPool2d(output_size=1)\n",
            "              (fc1): Conv2d(240, 10, kernel_size=(1, 1), stride=(1, 1))\n",
            "              (fc2): Conv2d(10, 240, kernel_size=(1, 1), stride=(1, 1))\n",
            "              (activation): SiLU(inplace=True)\n",
            "              (scale_activation): Sigmoid()\n",
            "            )\n",
            "            (3): Conv2dNormActivation(\n",
            "              (0): Conv2d(240, 80, kernel_size=(1, 1), stride=(1, 1), bias=False)\n",
            "              (1): BatchNorm2d(80, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n",
            "            )\n",
            "          )\n",
            "          (stochastic_depth): StochasticDepth(p=0.0625, mode=row)\n",
            "        )\n",
            "        (1): MBConv(\n",
            "          (block): Sequential(\n",
            "            (0): Conv2dNormActivation(\n",
            "              (0): Conv2d(80, 480, kernel_size=(1, 1), stride=(1, 1), bias=False)\n",
            "              (1): BatchNorm2d(480, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n",
            "              (2): SiLU(inplace=True)\n",
            "            )\n",
            "            (1): Conv2dNormActivation(\n",
            "              (0): Conv2d(480, 480, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=480, bias=False)\n",
            "              (1): BatchNorm2d(480, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n",
            "              (2): SiLU(inplace=True)\n",
            "            )\n",
            "            (2): SqueezeExcitation(\n",
            "              (avgpool): AdaptiveAvgPool2d(output_size=1)\n",
            "              (fc1): Conv2d(480, 20, kernel_size=(1, 1), stride=(1, 1))\n",
            "              (fc2): Conv2d(20, 480, kernel_size=(1, 1), stride=(1, 1))\n",
            "              (activation): SiLU(inplace=True)\n",
            "              (scale_activation): Sigmoid()\n",
            "            )\n",
            "            (3): Conv2dNormActivation(\n",
            "              (0): Conv2d(480, 80, kernel_size=(1, 1), stride=(1, 1), bias=False)\n",
            "              (1): BatchNorm2d(80, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n",
            "            )\n",
            "          )\n",
            "          (stochastic_depth): StochasticDepth(p=0.07500000000000001, mode=row)\n",
            "        )\n",
            "        (2): MBConv(\n",
            "          (block): Sequential(\n",
            "            (0): Conv2dNormActivation(\n",
            "              (0): Conv2d(80, 480, kernel_size=(1, 1), stride=(1, 1), bias=False)\n",
            "              (1): BatchNorm2d(480, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n",
            "              (2): SiLU(inplace=True)\n",
            "            )\n",
            "            (1): Conv2dNormActivation(\n",
            "              (0): Conv2d(480, 480, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=480, bias=False)\n",
            "              (1): BatchNorm2d(480, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n",
            "              (2): SiLU(inplace=True)\n",
            "            )\n",
            "            (2): SqueezeExcitation(\n",
            "              (avgpool): AdaptiveAvgPool2d(output_size=1)\n",
            "              (fc1): Conv2d(480, 20, kernel_size=(1, 1), stride=(1, 1))\n",
            "              (fc2): Conv2d(20, 480, kernel_size=(1, 1), stride=(1, 1))\n",
            "              (activation): SiLU(inplace=True)\n",
            "              (scale_activation): Sigmoid()\n",
            "            )\n",
            "            (3): Conv2dNormActivation(\n",
            "              (0): Conv2d(480, 80, kernel_size=(1, 1), stride=(1, 1), bias=False)\n",
            "              (1): BatchNorm2d(80, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n",
            "            )\n",
            "          )\n",
            "          (stochastic_depth): StochasticDepth(p=0.08750000000000001, mode=row)\n",
            "        )\n",
            "      )\n",
            "      (5): Sequential(\n",
            "        (0): MBConv(\n",
            "          (block): Sequential(\n",
            "            (0): Conv2dNormActivation(\n",
            "              (0): Conv2d(80, 480, kernel_size=(1, 1), stride=(1, 1), bias=False)\n",
            "              (1): BatchNorm2d(480, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n",
            "              (2): SiLU(inplace=True)\n",
            "            )\n",
            "            (1): Conv2dNormActivation(\n",
            "              (0): Conv2d(480, 480, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=480, bias=False)\n",
            "              (1): BatchNorm2d(480, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n",
            "              (2): SiLU(inplace=True)\n",
            "            )\n",
            "            (2): SqueezeExcitation(\n",
            "              (avgpool): AdaptiveAvgPool2d(output_size=1)\n",
            "              (fc1): Conv2d(480, 20, kernel_size=(1, 1), stride=(1, 1))\n",
            "              (fc2): Conv2d(20, 480, kernel_size=(1, 1), stride=(1, 1))\n",
            "              (activation): SiLU(inplace=True)\n",
            "              (scale_activation): Sigmoid()\n",
            "            )\n",
            "            (3): Conv2dNormActivation(\n",
            "              (0): Conv2d(480, 112, kernel_size=(1, 1), stride=(1, 1), bias=False)\n",
            "              (1): BatchNorm2d(112, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n",
            "            )\n",
            "          )\n",
            "          (stochastic_depth): StochasticDepth(p=0.1, mode=row)\n",
            "        )\n",
            "        (1): MBConv(\n",
            "          (block): Sequential(\n",
            "            (0): Conv2dNormActivation(\n",
            "              (0): Conv2d(112, 672, kernel_size=(1, 1), stride=(1, 1), bias=False)\n",
            "              (1): BatchNorm2d(672, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n",
            "              (2): SiLU(inplace=True)\n",
            "            )\n",
            "            (1): Conv2dNormActivation(\n",
            "              (0): Conv2d(672, 672, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=672, bias=False)\n",
            "              (1): BatchNorm2d(672, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n",
            "              (2): SiLU(inplace=True)\n",
            "            )\n",
            "            (2): SqueezeExcitation(\n",
            "              (avgpool): AdaptiveAvgPool2d(output_size=1)\n",
            "              (fc1): Conv2d(672, 28, kernel_size=(1, 1), stride=(1, 1))\n",
            "              (fc2): Conv2d(28, 672, kernel_size=(1, 1), stride=(1, 1))\n",
            "              (activation): SiLU(inplace=True)\n",
            "              (scale_activation): Sigmoid()\n",
            "            )\n",
            "            (3): Conv2dNormActivation(\n",
            "              (0): Conv2d(672, 112, kernel_size=(1, 1), stride=(1, 1), bias=False)\n",
            "              (1): BatchNorm2d(112, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n",
            "            )\n",
            "          )\n",
            "          (stochastic_depth): StochasticDepth(p=0.1125, mode=row)\n",
            "        )\n",
            "        (2): MBConv(\n",
            "          (block): Sequential(\n",
            "            (0): Conv2dNormActivation(\n",
            "              (0): Conv2d(112, 672, kernel_size=(1, 1), stride=(1, 1), bias=False)\n",
            "              (1): BatchNorm2d(672, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n",
            "              (2): SiLU(inplace=True)\n",
            "            )\n",
            "            (1): Conv2dNormActivation(\n",
            "              (0): Conv2d(672, 672, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=672, bias=False)\n",
            "              (1): BatchNorm2d(672, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n",
            "              (2): SiLU(inplace=True)\n",
            "            )\n",
            "            (2): SqueezeExcitation(\n",
            "              (avgpool): AdaptiveAvgPool2d(output_size=1)\n",
            "              (fc1): Conv2d(672, 28, kernel_size=(1, 1), stride=(1, 1))\n",
            "              (fc2): Conv2d(28, 672, kernel_size=(1, 1), stride=(1, 1))\n",
            "              (activation): SiLU(inplace=True)\n",
            "              (scale_activation): Sigmoid()\n",
            "            )\n",
            "            (3): Conv2dNormActivation(\n",
            "              (0): Conv2d(672, 112, kernel_size=(1, 1), stride=(1, 1), bias=False)\n",
            "              (1): BatchNorm2d(112, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n",
            "            )\n",
            "          )\n",
            "          (stochastic_depth): StochasticDepth(p=0.125, mode=row)\n",
            "        )\n",
            "      )\n",
            "      (6): Sequential(\n",
            "        (0): MBConv(\n",
            "          (block): Sequential(\n",
            "            (0): Conv2dNormActivation(\n",
            "              (0): Conv2d(112, 672, kernel_size=(1, 1), stride=(1, 1), bias=False)\n",
            "              (1): BatchNorm2d(672, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n",
            "              (2): SiLU(inplace=True)\n",
            "            )\n",
            "            (1): Conv2dNormActivation(\n",
            "              (0): Conv2d(672, 672, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), groups=672, bias=False)\n",
            "              (1): BatchNorm2d(672, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n",
            "              (2): SiLU(inplace=True)\n",
            "            )\n",
            "            (2): SqueezeExcitation(\n",
            "              (avgpool): AdaptiveAvgPool2d(output_size=1)\n",
            "              (fc1): Conv2d(672, 28, kernel_size=(1, 1), stride=(1, 1))\n",
            "              (fc2): Conv2d(28, 672, kernel_size=(1, 1), stride=(1, 1))\n",
            "              (activation): SiLU(inplace=True)\n",
            "              (scale_activation): Sigmoid()\n",
            "            )\n",
            "            (3): Conv2dNormActivation(\n",
            "              (0): Conv2d(672, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)\n",
            "              (1): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n",
            "            )\n",
            "          )\n",
            "          (stochastic_depth): StochasticDepth(p=0.1375, mode=row)\n",
            "        )\n",
            "        (1): MBConv(\n",
            "          (block): Sequential(\n",
            "            (0): Conv2dNormActivation(\n",
            "              (0): Conv2d(192, 1152, kernel_size=(1, 1), stride=(1, 1), bias=False)\n",
            "              (1): BatchNorm2d(1152, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n",
            "              (2): SiLU(inplace=True)\n",
            "            )\n",
            "            (1): Conv2dNormActivation(\n",
            "              (0): Conv2d(1152, 1152, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=1152, bias=False)\n",
            "              (1): BatchNorm2d(1152, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n",
            "              (2): SiLU(inplace=True)\n",
            "            )\n",
            "            (2): SqueezeExcitation(\n",
            "              (avgpool): AdaptiveAvgPool2d(output_size=1)\n",
            "              (fc1): Conv2d(1152, 48, kernel_size=(1, 1), stride=(1, 1))\n",
            "              (fc2): Conv2d(48, 1152, kernel_size=(1, 1), stride=(1, 1))\n",
            "              (activation): SiLU(inplace=True)\n",
            "              (scale_activation): Sigmoid()\n",
            "            )\n",
            "            (3): Conv2dNormActivation(\n",
            "              (0): Conv2d(1152, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)\n",
            "              (1): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n",
            "            )\n",
            "          )\n",
            "          (stochastic_depth): StochasticDepth(p=0.15000000000000002, mode=row)\n",
            "        )\n",
            "        (2): MBConv(\n",
            "          (block): Sequential(\n",
            "            (0): Conv2dNormActivation(\n",
            "              (0): Conv2d(192, 1152, kernel_size=(1, 1), stride=(1, 1), bias=False)\n",
            "              (1): BatchNorm2d(1152, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n",
            "              (2): SiLU(inplace=True)\n",
            "            )\n",
            "            (1): Conv2dNormActivation(\n",
            "              (0): Conv2d(1152, 1152, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=1152, bias=False)\n",
            "              (1): BatchNorm2d(1152, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n",
            "              (2): SiLU(inplace=True)\n",
            "            )\n",
            "            (2): SqueezeExcitation(\n",
            "              (avgpool): AdaptiveAvgPool2d(output_size=1)\n",
            "              (fc1): Conv2d(1152, 48, kernel_size=(1, 1), stride=(1, 1))\n",
            "              (fc2): Conv2d(48, 1152, kernel_size=(1, 1), stride=(1, 1))\n",
            "              (activation): SiLU(inplace=True)\n",
            "              (scale_activation): Sigmoid()\n",
            "            )\n",
            "            (3): Conv2dNormActivation(\n",
            "              (0): Conv2d(1152, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)\n",
            "              (1): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n",
            "            )\n",
            "          )\n",
            "          (stochastic_depth): StochasticDepth(p=0.1625, mode=row)\n",
            "        )\n",
            "        (3): MBConv(\n",
            "          (block): Sequential(\n",
            "            (0): Conv2dNormActivation(\n",
            "              (0): Conv2d(192, 1152, kernel_size=(1, 1), stride=(1, 1), bias=False)\n",
            "              (1): BatchNorm2d(1152, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n",
            "              (2): SiLU(inplace=True)\n",
            "            )\n",
            "            (1): Conv2dNormActivation(\n",
            "              (0): Conv2d(1152, 1152, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=1152, bias=False)\n",
            "              (1): BatchNorm2d(1152, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n",
            "              (2): SiLU(inplace=True)\n",
            "            )\n",
            "            (2): SqueezeExcitation(\n",
            "              (avgpool): AdaptiveAvgPool2d(output_size=1)\n",
            "              (fc1): Conv2d(1152, 48, kernel_size=(1, 1), stride=(1, 1))\n",
            "              (fc2): Conv2d(48, 1152, kernel_size=(1, 1), stride=(1, 1))\n",
            "              (activation): SiLU(inplace=True)\n",
            "              (scale_activation): Sigmoid()\n",
            "            )\n",
            "            (3): Conv2dNormActivation(\n",
            "              (0): Conv2d(1152, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)\n",
            "              (1): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n",
            "            )\n",
            "          )\n",
            "          (stochastic_depth): StochasticDepth(p=0.17500000000000002, mode=row)\n",
            "        )\n",
            "      )\n",
            "      (7): Sequential(\n",
            "        (0): MBConv(\n",
            "          (block): Sequential(\n",
            "            (0): Conv2dNormActivation(\n",
            "              (0): Conv2d(192, 1152, kernel_size=(1, 1), stride=(1, 1), bias=False)\n",
            "              (1): BatchNorm2d(1152, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n",
            "              (2): SiLU(inplace=True)\n",
            "            )\n",
            "            (1): Conv2dNormActivation(\n",
            "              (0): Conv2d(1152, 1152, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=1152, bias=False)\n",
            "              (1): BatchNorm2d(1152, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n",
            "              (2): SiLU(inplace=True)\n",
            "            )\n",
            "            (2): SqueezeExcitation(\n",
            "              (avgpool): AdaptiveAvgPool2d(output_size=1)\n",
            "              (fc1): Conv2d(1152, 48, kernel_size=(1, 1), stride=(1, 1))\n",
            "              (fc2): Conv2d(48, 1152, kernel_size=(1, 1), stride=(1, 1))\n",
            "              (activation): SiLU(inplace=True)\n",
            "              (scale_activation): Sigmoid()\n",
            "            )\n",
            "            (3): Conv2dNormActivation(\n",
            "              (0): Conv2d(1152, 320, kernel_size=(1, 1), stride=(1, 1), bias=False)\n",
            "              (1): BatchNorm2d(320, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n",
            "            )\n",
            "          )\n",
            "          (stochastic_depth): StochasticDepth(p=0.1875, mode=row)\n",
            "        )\n",
            "      )\n",
            "      (8): Conv2dNormActivation(\n",
            "        (0): Conv2d(320, 1280, kernel_size=(1, 1), stride=(1, 1), bias=False)\n",
            "        (1): BatchNorm2d(1280, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n",
            "        (2): SiLU(inplace=True)\n",
            "      )\n",
            "    )\n",
            "    (avgpool): AdaptiveAvgPool2d(output_size=1)\n",
            "    (classifier): Sequential(\n",
            "      (0): Dropout(p=0.2, inplace=True)\n",
            "      (1): Linear(in_features=1280, out_features=1000, bias=True)\n",
            "    )\n",
            "  )\n",
            "  (cnn_features): Sequential(\n",
            "    (0): Sequential(\n",
            "      (0): Conv2dNormActivation(\n",
            "        (0): Conv2d(3, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)\n",
            "        (1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n",
            "        (2): SiLU(inplace=True)\n",
            "      )\n",
            "      (1): Sequential(\n",
            "        (0): MBConv(\n",
            "          (block): Sequential(\n",
            "            (0): Conv2dNormActivation(\n",
            "              (0): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=32, bias=False)\n",
            "              (1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n",
            "              (2): SiLU(inplace=True)\n",
            "            )\n",
            "            (1): SqueezeExcitation(\n",
            "              (avgpool): AdaptiveAvgPool2d(output_size=1)\n",
            "              (fc1): Conv2d(32, 8, kernel_size=(1, 1), stride=(1, 1))\n",
            "              (fc2): Conv2d(8, 32, kernel_size=(1, 1), stride=(1, 1))\n",
            "              (activation): SiLU(inplace=True)\n",
            "              (scale_activation): Sigmoid()\n",
            "            )\n",
            "            (2): Conv2dNormActivation(\n",
            "              (0): Conv2d(32, 16, kernel_size=(1, 1), stride=(1, 1), bias=False)\n",
            "              (1): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n",
            "            )\n",
            "          )\n",
            "          (stochastic_depth): StochasticDepth(p=0.0, mode=row)\n",
            "        )\n",
            "      )\n",
            "      (2): Sequential(\n",
            "        (0): MBConv(\n",
            "          (block): Sequential(\n",
            "            (0): Conv2dNormActivation(\n",
            "              (0): Conv2d(16, 96, kernel_size=(1, 1), stride=(1, 1), bias=False)\n",
            "              (1): BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n",
            "              (2): SiLU(inplace=True)\n",
            "            )\n",
            "            (1): Conv2dNormActivation(\n",
            "              (0): Conv2d(96, 96, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=96, bias=False)\n",
            "              (1): BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n",
            "              (2): SiLU(inplace=True)\n",
            "            )\n",
            "            (2): SqueezeExcitation(\n",
            "              (avgpool): AdaptiveAvgPool2d(output_size=1)\n",
            "              (fc1): Conv2d(96, 4, kernel_size=(1, 1), stride=(1, 1))\n",
            "              (fc2): Conv2d(4, 96, kernel_size=(1, 1), stride=(1, 1))\n",
            "              (activation): SiLU(inplace=True)\n",
            "              (scale_activation): Sigmoid()\n",
            "            )\n",
            "            (3): Conv2dNormActivation(\n",
            "              (0): Conv2d(96, 24, kernel_size=(1, 1), stride=(1, 1), bias=False)\n",
            "              (1): BatchNorm2d(24, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n",
            "            )\n",
            "          )\n",
            "          (stochastic_depth): StochasticDepth(p=0.0125, mode=row)\n",
            "        )\n",
            "        (1): MBConv(\n",
            "          (block): Sequential(\n",
            "            (0): Conv2dNormActivation(\n",
            "              (0): Conv2d(24, 144, kernel_size=(1, 1), stride=(1, 1), bias=False)\n",
            "              (1): BatchNorm2d(144, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n",
            "              (2): SiLU(inplace=True)\n",
            "            )\n",
            "            (1): Conv2dNormActivation(\n",
            "              (0): Conv2d(144, 144, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=144, bias=False)\n",
            "              (1): BatchNorm2d(144, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n",
            "              (2): SiLU(inplace=True)\n",
            "            )\n",
            "            (2): SqueezeExcitation(\n",
            "              (avgpool): AdaptiveAvgPool2d(output_size=1)\n",
            "              (fc1): Conv2d(144, 6, kernel_size=(1, 1), stride=(1, 1))\n",
            "              (fc2): Conv2d(6, 144, kernel_size=(1, 1), stride=(1, 1))\n",
            "              (activation): SiLU(inplace=True)\n",
            "              (scale_activation): Sigmoid()\n",
            "            )\n",
            "            (3): Conv2dNormActivation(\n",
            "              (0): Conv2d(144, 24, kernel_size=(1, 1), stride=(1, 1), bias=False)\n",
            "              (1): BatchNorm2d(24, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n",
            "            )\n",
            "          )\n",
            "          (stochastic_depth): StochasticDepth(p=0.025, mode=row)\n",
            "        )\n",
            "      )\n",
            "      (3): Sequential(\n",
            "        (0): MBConv(\n",
            "          (block): Sequential(\n",
            "            (0): Conv2dNormActivation(\n",
            "              (0): Conv2d(24, 144, kernel_size=(1, 1), stride=(1, 1), bias=False)\n",
            "              (1): BatchNorm2d(144, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n",
            "              (2): SiLU(inplace=True)\n",
            "            )\n",
            "            (1): Conv2dNormActivation(\n",
            "              (0): Conv2d(144, 144, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), groups=144, bias=False)\n",
            "              (1): BatchNorm2d(144, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n",
            "              (2): SiLU(inplace=True)\n",
            "            )\n",
            "            (2): SqueezeExcitation(\n",
            "              (avgpool): AdaptiveAvgPool2d(output_size=1)\n",
            "              (fc1): Conv2d(144, 6, kernel_size=(1, 1), stride=(1, 1))\n",
            "              (fc2): Conv2d(6, 144, kernel_size=(1, 1), stride=(1, 1))\n",
            "              (activation): SiLU(inplace=True)\n",
            "              (scale_activation): Sigmoid()\n",
            "            )\n",
            "            (3): Conv2dNormActivation(\n",
            "              (0): Conv2d(144, 40, kernel_size=(1, 1), stride=(1, 1), bias=False)\n",
            "              (1): BatchNorm2d(40, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n",
            "            )\n",
            "          )\n",
            "          (stochastic_depth): StochasticDepth(p=0.037500000000000006, mode=row)\n",
            "        )\n",
            "        (1): MBConv(\n",
            "          (block): Sequential(\n",
            "            (0): Conv2dNormActivation(\n",
            "              (0): Conv2d(40, 240, kernel_size=(1, 1), stride=(1, 1), bias=False)\n",
            "              (1): BatchNorm2d(240, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n",
            "              (2): SiLU(inplace=True)\n",
            "            )\n",
            "            (1): Conv2dNormActivation(\n",
            "              (0): Conv2d(240, 240, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=240, bias=False)\n",
            "              (1): BatchNorm2d(240, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n",
            "              (2): SiLU(inplace=True)\n",
            "            )\n",
            "            (2): SqueezeExcitation(\n",
            "              (avgpool): AdaptiveAvgPool2d(output_size=1)\n",
            "              (fc1): Conv2d(240, 10, kernel_size=(1, 1), stride=(1, 1))\n",
            "              (fc2): Conv2d(10, 240, kernel_size=(1, 1), stride=(1, 1))\n",
            "              (activation): SiLU(inplace=True)\n",
            "              (scale_activation): Sigmoid()\n",
            "            )\n",
            "            (3): Conv2dNormActivation(\n",
            "              (0): Conv2d(240, 40, kernel_size=(1, 1), stride=(1, 1), bias=False)\n",
            "              (1): BatchNorm2d(40, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n",
            "            )\n",
            "          )\n",
            "          (stochastic_depth): StochasticDepth(p=0.05, mode=row)\n",
            "        )\n",
            "      )\n",
            "      (4): Sequential(\n",
            "        (0): MBConv(\n",
            "          (block): Sequential(\n",
            "            (0): Conv2dNormActivation(\n",
            "              (0): Conv2d(40, 240, kernel_size=(1, 1), stride=(1, 1), bias=False)\n",
            "              (1): BatchNorm2d(240, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n",
            "              (2): SiLU(inplace=True)\n",
            "            )\n",
            "            (1): Conv2dNormActivation(\n",
            "              (0): Conv2d(240, 240, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=240, bias=False)\n",
            "              (1): BatchNorm2d(240, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n",
            "              (2): SiLU(inplace=True)\n",
            "            )\n",
            "            (2): SqueezeExcitation(\n",
            "              (avgpool): AdaptiveAvgPool2d(output_size=1)\n",
            "              (fc1): Conv2d(240, 10, kernel_size=(1, 1), stride=(1, 1))\n",
            "              (fc2): Conv2d(10, 240, kernel_size=(1, 1), stride=(1, 1))\n",
            "              (activation): SiLU(inplace=True)\n",
            "              (scale_activation): Sigmoid()\n",
            "            )\n",
            "            (3): Conv2dNormActivation(\n",
            "              (0): Conv2d(240, 80, kernel_size=(1, 1), stride=(1, 1), bias=False)\n",
            "              (1): BatchNorm2d(80, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n",
            "            )\n",
            "          )\n",
            "          (stochastic_depth): StochasticDepth(p=0.0625, mode=row)\n",
            "        )\n",
            "        (1): MBConv(\n",
            "          (block): Sequential(\n",
            "            (0): Conv2dNormActivation(\n",
            "              (0): Conv2d(80, 480, kernel_size=(1, 1), stride=(1, 1), bias=False)\n",
            "              (1): BatchNorm2d(480, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n",
            "              (2): SiLU(inplace=True)\n",
            "            )\n",
            "            (1): Conv2dNormActivation(\n",
            "              (0): Conv2d(480, 480, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=480, bias=False)\n",
            "              (1): BatchNorm2d(480, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n",
            "              (2): SiLU(inplace=True)\n",
            "            )\n",
            "            (2): SqueezeExcitation(\n",
            "              (avgpool): AdaptiveAvgPool2d(output_size=1)\n",
            "              (fc1): Conv2d(480, 20, kernel_size=(1, 1), stride=(1, 1))\n",
            "              (fc2): Conv2d(20, 480, kernel_size=(1, 1), stride=(1, 1))\n",
            "              (activation): SiLU(inplace=True)\n",
            "              (scale_activation): Sigmoid()\n",
            "            )\n",
            "            (3): Conv2dNormActivation(\n",
            "              (0): Conv2d(480, 80, kernel_size=(1, 1), stride=(1, 1), bias=False)\n",
            "              (1): BatchNorm2d(80, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n",
            "            )\n",
            "          )\n",
            "          (stochastic_depth): StochasticDepth(p=0.07500000000000001, mode=row)\n",
            "        )\n",
            "        (2): MBConv(\n",
            "          (block): Sequential(\n",
            "            (0): Conv2dNormActivation(\n",
            "              (0): Conv2d(80, 480, kernel_size=(1, 1), stride=(1, 1), bias=False)\n",
            "              (1): BatchNorm2d(480, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n",
            "              (2): SiLU(inplace=True)\n",
            "            )\n",
            "            (1): Conv2dNormActivation(\n",
            "              (0): Conv2d(480, 480, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=480, bias=False)\n",
            "              (1): BatchNorm2d(480, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n",
            "              (2): SiLU(inplace=True)\n",
            "            )\n",
            "            (2): SqueezeExcitation(\n",
            "              (avgpool): AdaptiveAvgPool2d(output_size=1)\n",
            "              (fc1): Conv2d(480, 20, kernel_size=(1, 1), stride=(1, 1))\n",
            "              (fc2): Conv2d(20, 480, kernel_size=(1, 1), stride=(1, 1))\n",
            "              (activation): SiLU(inplace=True)\n",
            "              (scale_activation): Sigmoid()\n",
            "            )\n",
            "            (3): Conv2dNormActivation(\n",
            "              (0): Conv2d(480, 80, kernel_size=(1, 1), stride=(1, 1), bias=False)\n",
            "              (1): BatchNorm2d(80, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n",
            "            )\n",
            "          )\n",
            "          (stochastic_depth): StochasticDepth(p=0.08750000000000001, mode=row)\n",
            "        )\n",
            "      )\n",
            "      (5): Sequential(\n",
            "        (0): MBConv(\n",
            "          (block): Sequential(\n",
            "            (0): Conv2dNormActivation(\n",
            "              (0): Conv2d(80, 480, kernel_size=(1, 1), stride=(1, 1), bias=False)\n",
            "              (1): BatchNorm2d(480, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n",
            "              (2): SiLU(inplace=True)\n",
            "            )\n",
            "            (1): Conv2dNormActivation(\n",
            "              (0): Conv2d(480, 480, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=480, bias=False)\n",
            "              (1): BatchNorm2d(480, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n",
            "              (2): SiLU(inplace=True)\n",
            "            )\n",
            "            (2): SqueezeExcitation(\n",
            "              (avgpool): AdaptiveAvgPool2d(output_size=1)\n",
            "              (fc1): Conv2d(480, 20, kernel_size=(1, 1), stride=(1, 1))\n",
            "              (fc2): Conv2d(20, 480, kernel_size=(1, 1), stride=(1, 1))\n",
            "              (activation): SiLU(inplace=True)\n",
            "              (scale_activation): Sigmoid()\n",
            "            )\n",
            "            (3): Conv2dNormActivation(\n",
            "              (0): Conv2d(480, 112, kernel_size=(1, 1), stride=(1, 1), bias=False)\n",
            "              (1): BatchNorm2d(112, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n",
            "            )\n",
            "          )\n",
            "          (stochastic_depth): StochasticDepth(p=0.1, mode=row)\n",
            "        )\n",
            "        (1): MBConv(\n",
            "          (block): Sequential(\n",
            "            (0): Conv2dNormActivation(\n",
            "              (0): Conv2d(112, 672, kernel_size=(1, 1), stride=(1, 1), bias=False)\n",
            "              (1): BatchNorm2d(672, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n",
            "              (2): SiLU(inplace=True)\n",
            "            )\n",
            "            (1): Conv2dNormActivation(\n",
            "              (0): Conv2d(672, 672, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=672, bias=False)\n",
            "              (1): BatchNorm2d(672, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n",
            "              (2): SiLU(inplace=True)\n",
            "            )\n",
            "            (2): SqueezeExcitation(\n",
            "              (avgpool): AdaptiveAvgPool2d(output_size=1)\n",
            "              (fc1): Conv2d(672, 28, kernel_size=(1, 1), stride=(1, 1))\n",
            "              (fc2): Conv2d(28, 672, kernel_size=(1, 1), stride=(1, 1))\n",
            "              (activation): SiLU(inplace=True)\n",
            "              (scale_activation): Sigmoid()\n",
            "            )\n",
            "            (3): Conv2dNormActivation(\n",
            "              (0): Conv2d(672, 112, kernel_size=(1, 1), stride=(1, 1), bias=False)\n",
            "              (1): BatchNorm2d(112, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n",
            "            )\n",
            "          )\n",
            "          (stochastic_depth): StochasticDepth(p=0.1125, mode=row)\n",
            "        )\n",
            "        (2): MBConv(\n",
            "          (block): Sequential(\n",
            "            (0): Conv2dNormActivation(\n",
            "              (0): Conv2d(112, 672, kernel_size=(1, 1), stride=(1, 1), bias=False)\n",
            "              (1): BatchNorm2d(672, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n",
            "              (2): SiLU(inplace=True)\n",
            "            )\n",
            "            (1): Conv2dNormActivation(\n",
            "              (0): Conv2d(672, 672, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=672, bias=False)\n",
            "              (1): BatchNorm2d(672, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n",
            "              (2): SiLU(inplace=True)\n",
            "            )\n",
            "            (2): SqueezeExcitation(\n",
            "              (avgpool): AdaptiveAvgPool2d(output_size=1)\n",
            "              (fc1): Conv2d(672, 28, kernel_size=(1, 1), stride=(1, 1))\n",
            "              (fc2): Conv2d(28, 672, kernel_size=(1, 1), stride=(1, 1))\n",
            "              (activation): SiLU(inplace=True)\n",
            "              (scale_activation): Sigmoid()\n",
            "            )\n",
            "            (3): Conv2dNormActivation(\n",
            "              (0): Conv2d(672, 112, kernel_size=(1, 1), stride=(1, 1), bias=False)\n",
            "              (1): BatchNorm2d(112, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n",
            "            )\n",
            "          )\n",
            "          (stochastic_depth): StochasticDepth(p=0.125, mode=row)\n",
            "        )\n",
            "      )\n",
            "      (6): Sequential(\n",
            "        (0): MBConv(\n",
            "          (block): Sequential(\n",
            "            (0): Conv2dNormActivation(\n",
            "              (0): Conv2d(112, 672, kernel_size=(1, 1), stride=(1, 1), bias=False)\n",
            "              (1): BatchNorm2d(672, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n",
            "              (2): SiLU(inplace=True)\n",
            "            )\n",
            "            (1): Conv2dNormActivation(\n",
            "              (0): Conv2d(672, 672, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), groups=672, bias=False)\n",
            "              (1): BatchNorm2d(672, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n",
            "              (2): SiLU(inplace=True)\n",
            "            )\n",
            "            (2): SqueezeExcitation(\n",
            "              (avgpool): AdaptiveAvgPool2d(output_size=1)\n",
            "              (fc1): Conv2d(672, 28, kernel_size=(1, 1), stride=(1, 1))\n",
            "              (fc2): Conv2d(28, 672, kernel_size=(1, 1), stride=(1, 1))\n",
            "              (activation): SiLU(inplace=True)\n",
            "              (scale_activation): Sigmoid()\n",
            "            )\n",
            "            (3): Conv2dNormActivation(\n",
            "              (0): Conv2d(672, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)\n",
            "              (1): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n",
            "            )\n",
            "          )\n",
            "          (stochastic_depth): StochasticDepth(p=0.1375, mode=row)\n",
            "        )\n",
            "        (1): MBConv(\n",
            "          (block): Sequential(\n",
            "            (0): Conv2dNormActivation(\n",
            "              (0): Conv2d(192, 1152, kernel_size=(1, 1), stride=(1, 1), bias=False)\n",
            "              (1): BatchNorm2d(1152, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n",
            "              (2): SiLU(inplace=True)\n",
            "            )\n",
            "            (1): Conv2dNormActivation(\n",
            "              (0): Conv2d(1152, 1152, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=1152, bias=False)\n",
            "              (1): BatchNorm2d(1152, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n",
            "              (2): SiLU(inplace=True)\n",
            "            )\n",
            "            (2): SqueezeExcitation(\n",
            "              (avgpool): AdaptiveAvgPool2d(output_size=1)\n",
            "              (fc1): Conv2d(1152, 48, kernel_size=(1, 1), stride=(1, 1))\n",
            "              (fc2): Conv2d(48, 1152, kernel_size=(1, 1), stride=(1, 1))\n",
            "              (activation): SiLU(inplace=True)\n",
            "              (scale_activation): Sigmoid()\n",
            "            )\n",
            "            (3): Conv2dNormActivation(\n",
            "              (0): Conv2d(1152, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)\n",
            "              (1): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n",
            "            )\n",
            "          )\n",
            "          (stochastic_depth): StochasticDepth(p=0.15000000000000002, mode=row)\n",
            "        )\n",
            "        (2): MBConv(\n",
            "          (block): Sequential(\n",
            "            (0): Conv2dNormActivation(\n",
            "              (0): Conv2d(192, 1152, kernel_size=(1, 1), stride=(1, 1), bias=False)\n",
            "              (1): BatchNorm2d(1152, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n",
            "              (2): SiLU(inplace=True)\n",
            "            )\n",
            "            (1): Conv2dNormActivation(\n",
            "              (0): Conv2d(1152, 1152, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=1152, bias=False)\n",
            "              (1): BatchNorm2d(1152, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n",
            "              (2): SiLU(inplace=True)\n",
            "            )\n",
            "            (2): SqueezeExcitation(\n",
            "              (avgpool): AdaptiveAvgPool2d(output_size=1)\n",
            "              (fc1): Conv2d(1152, 48, kernel_size=(1, 1), stride=(1, 1))\n",
            "              (fc2): Conv2d(48, 1152, kernel_size=(1, 1), stride=(1, 1))\n",
            "              (activation): SiLU(inplace=True)\n",
            "              (scale_activation): Sigmoid()\n",
            "            )\n",
            "            (3): Conv2dNormActivation(\n",
            "              (0): Conv2d(1152, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)\n",
            "              (1): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n",
            "            )\n",
            "          )\n",
            "          (stochastic_depth): StochasticDepth(p=0.1625, mode=row)\n",
            "        )\n",
            "        (3): MBConv(\n",
            "          (block): Sequential(\n",
            "            (0): Conv2dNormActivation(\n",
            "              (0): Conv2d(192, 1152, kernel_size=(1, 1), stride=(1, 1), bias=False)\n",
            "              (1): BatchNorm2d(1152, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n",
            "              (2): SiLU(inplace=True)\n",
            "            )\n",
            "            (1): Conv2dNormActivation(\n",
            "              (0): Conv2d(1152, 1152, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=1152, bias=False)\n",
            "              (1): BatchNorm2d(1152, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n",
            "              (2): SiLU(inplace=True)\n",
            "            )\n",
            "            (2): SqueezeExcitation(\n",
            "              (avgpool): AdaptiveAvgPool2d(output_size=1)\n",
            "              (fc1): Conv2d(1152, 48, kernel_size=(1, 1), stride=(1, 1))\n",
            "              (fc2): Conv2d(48, 1152, kernel_size=(1, 1), stride=(1, 1))\n",
            "              (activation): SiLU(inplace=True)\n",
            "              (scale_activation): Sigmoid()\n",
            "            )\n",
            "            (3): Conv2dNormActivation(\n",
            "              (0): Conv2d(1152, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)\n",
            "              (1): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n",
            "            )\n",
            "          )\n",
            "          (stochastic_depth): StochasticDepth(p=0.17500000000000002, mode=row)\n",
            "        )\n",
            "      )\n",
            "      (7): Sequential(\n",
            "        (0): MBConv(\n",
            "          (block): Sequential(\n",
            "            (0): Conv2dNormActivation(\n",
            "              (0): Conv2d(192, 1152, kernel_size=(1, 1), stride=(1, 1), bias=False)\n",
            "              (1): BatchNorm2d(1152, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n",
            "              (2): SiLU(inplace=True)\n",
            "            )\n",
            "            (1): Conv2dNormActivation(\n",
            "              (0): Conv2d(1152, 1152, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=1152, bias=False)\n",
            "              (1): BatchNorm2d(1152, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n",
            "              (2): SiLU(inplace=True)\n",
            "            )\n",
            "            (2): SqueezeExcitation(\n",
            "              (avgpool): AdaptiveAvgPool2d(output_size=1)\n",
            "              (fc1): Conv2d(1152, 48, kernel_size=(1, 1), stride=(1, 1))\n",
            "              (fc2): Conv2d(48, 1152, kernel_size=(1, 1), stride=(1, 1))\n",
            "              (activation): SiLU(inplace=True)\n",
            "              (scale_activation): Sigmoid()\n",
            "            )\n",
            "            (3): Conv2dNormActivation(\n",
            "              (0): Conv2d(1152, 320, kernel_size=(1, 1), stride=(1, 1), bias=False)\n",
            "              (1): BatchNorm2d(320, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n",
            "            )\n",
            "          )\n",
            "          (stochastic_depth): StochasticDepth(p=0.1875, mode=row)\n",
            "        )\n",
            "      )\n",
            "      (8): Conv2dNormActivation(\n",
            "        (0): Conv2d(320, 1280, kernel_size=(1, 1), stride=(1, 1), bias=False)\n",
            "        (1): BatchNorm2d(1280, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n",
            "        (2): SiLU(inplace=True)\n",
            "      )\n",
            "    )\n",
            "  )\n",
            "  (vit): ViT(\n",
            "    (to_patch_embedding): Sequential(\n",
            "      (0): Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=32, p2=32)\n",
            "      (1): LayerNorm((3072,), eps=1e-05, elementwise_affine=True)\n",
            "      (2): Linear(in_features=3072, out_features=768, bias=True)\n",
            "      (3): LayerNorm((768,), eps=1e-05, elementwise_affine=True)\n",
            "    )\n",
            "    (dropout): Dropout(p=0.1, inplace=False)\n",
            "    (transformer): Transformer(\n",
            "      (norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)\n",
            "      (layers): ModuleList(\n",
            "        (0-3): 4 x ModuleList(\n",
            "          (0): Attention(\n",
            "            (norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)\n",
            "            (attend): Softmax(dim=-1)\n",
            "            (dropout): Dropout(p=0.1, inplace=False)\n",
            "            (to_qkv): Linear(in_features=768, out_features=1536, bias=False)\n",
            "            (to_out): Sequential(\n",
            "              (0): Linear(in_features=512, out_features=768, bias=True)\n",
            "              (1): Dropout(p=0.1, inplace=False)\n",
            "            )\n",
            "          )\n",
            "          (1): FeedForward(\n",
            "            (net): Sequential(\n",
            "              (0): LayerNorm((768,), eps=1e-05, elementwise_affine=True)\n",
            "              (1): Linear(in_features=768, out_features=1536, bias=True)\n",
            "              (2): GELU(approximate='none')\n",
            "              (3): Dropout(p=0.1, inplace=False)\n",
            "              (4): Linear(in_features=1536, out_features=768, bias=True)\n",
            "              (5): Dropout(p=0.1, inplace=False)\n",
            "            )\n",
            "          )\n",
            "        )\n",
            "      )\n",
            "    )\n",
            "    (to_latent): Identity()\n",
            "    (mlp_head): Linear(in_features=768, out_features=768, bias=True)\n",
            "  )\n",
            "  (cnn_proj): Linear(in_features=1280, out_features=384, bias=True)\n",
            "  (vit_proj): Linear(in_features=768, out_features=384, bias=True)\n",
            "  (type_classifier): Sequential(\n",
            "    (0): Linear(in_features=384, out_features=256, bias=True)\n",
            "    (1): GELU(approximate='none')\n",
            "    (2): Dropout(p=0.3, inplace=False)\n",
            "    (3): Linear(in_features=256, out_features=4, bias=True)\n",
            "  )\n",
            "  (grade_classifier): Sequential(\n",
            "    (0): Linear(in_features=384, out_features=256, bias=True)\n",
            "    (1): GELU(approximate='none')\n",
            "    (2): Dropout(p=0.3, inplace=False)\n",
            "    (3): Linear(in_features=256, out_features=4, bias=True)\n",
            "  )\n",
            ")\n",
            "\n",
            "Total parameters: 25,027,180\n",
            "Trainable parameters: 25,027,180\n",
            "\n",
            "Starting fine-tuning...\n"
          ]
        },
        {
          "metadata": {
            "tags": null
          },
          "name": "stderr",
          "output_type": "stream",
          "text": [
            "Fine-tuning Epoch 1: 100%|██████████| 230/230 [02:44<00:00,  1.40it/s, Loss=0.402, Type Acc=0.811, LR=0.0001]\n",
            "Evaluating: 100%|██████████| 58/58 [01:26<00:00,  1.49s/it]\n"
          ]
        },
        {
          "metadata": {
            "tags": null
          },
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "\n",
            "Epoch 1/50 Summary:\n",
            "Train Loss: 0.4016 | Type Acc: 81.07% | Grade Acc: 54411.11%\n",
            "Val Loss: 0.1414 | Type Acc: 94.77% | Grade Acc: 15577.78%\n",
            "Saved new best model to /content/drive/MyDrive/BrainTumorModels/best_self_supervised_model_t4.pth\n"
          ]
        },
        {
          "metadata": {
            "tags": null
          },
          "name": "stderr",
          "output_type": "stream",
          "text": [
            "Fine-tuning Epoch 2: 100%|██████████| 230/230 [01:34<00:00,  2.43it/s, Loss=0.141, Type Acc=0.947, LR=0.0001]\n",
            "Evaluating: 100%|██████████| 58/58 [00:15<00:00,  3.79it/s]\n"
          ]
        },
        {
          "metadata": {
            "tags": null
          },
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "\n",
            "Epoch 2/50 Summary:\n",
            "Train Loss: 0.1406 | Type Acc: 94.69% | Grade Acc: 52730.00%\n",
            "Val Loss: 0.0871 | Type Acc: 97.00% | Grade Acc: 15655.56%\n",
            "Saved new best model to /content/drive/MyDrive/BrainTumorModels/best_self_supervised_model_t4.pth\n"
          ]
        },
        {
          "metadata": {
            "tags": null
          },
          "name": "stderr",
          "output_type": "stream",
          "text": [
            "Fine-tuning Epoch 3: 100%|██████████| 230/230 [01:18<00:00,  2.92it/s, Loss=0.0951, Type Acc=0.966, LR=0.0001]\n",
            "Evaluating: 100%|██████████| 58/58 [00:14<00:00,  3.90it/s]\n"
          ]
        },
        {
          "metadata": {
            "tags": null
          },
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "\n",
            "Epoch 3/50 Summary:\n",
            "Train Loss: 0.0951 | Type Acc: 96.57% | Grade Acc: 76514.29%\n",
            "Val Loss: 0.0635 | Type Acc: 97.71% | Grade Acc: 15855.56%\n",
            "Saved new best model to /content/drive/MyDrive/BrainTumorModels/best_self_supervised_model_t4.pth\n"
          ]
        },
        {
          "metadata": {
            "tags": null
          },
          "name": "stderr",
          "output_type": "stream",
          "text": [
            "Fine-tuning Epoch 4: 100%|██████████| 230/230 [01:16<00:00,  3.02it/s, Loss=0.0691, Type Acc=0.972, LR=0.0001]\n",
            "Evaluating: 100%|██████████| 58/58 [00:15<00:00,  3.77it/s]\n"
          ]
        },
        {
          "metadata": {
            "tags": null
          },
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "\n",
            "Epoch 4/50 Summary:\n",
            "Train Loss: 0.0691 | Type Acc: 97.22% | Grade Acc: 107800.00%\n",
            "Val Loss: 0.0468 | Type Acc: 98.09% | Grade Acc: 15966.67%\n",
            "Saved new best model to /content/drive/MyDrive/BrainTumorModels/best_self_supervised_model_t4.pth\n"
          ]
        },
        {
          "metadata": {
            "tags": null
          },
          "name": "stderr",
          "output_type": "stream",
          "text": [
            "Fine-tuning Epoch 5: 100%|██████████| 230/230 [01:12<00:00,  3.16it/s, Loss=0.0562, Type Acc=0.975, LR=0.0001]\n",
            "Evaluating: 100%|██████████| 58/58 [00:14<00:00,  3.91it/s]\n"
          ]
        },
        {
          "metadata": {
            "tags": null
          },
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "\n",
            "Epoch 5/50 Summary:\n",
            "Train Loss: 0.0562 | Type Acc: 97.45% | Grade Acc: 60277.78%\n",
            "Val Loss: 0.0490 | Type Acc: 98.26% | Grade Acc: 15922.22%\n",
            "Saved new best model to /content/drive/MyDrive/BrainTumorModels/best_self_supervised_model_t4.pth\n"
          ]
        },
        {
          "metadata": {
            "tags": null
          },
          "name": "stderr",
          "output_type": "stream",
          "text": [
            "Fine-tuning Epoch 6: 100%|██████████| 230/230 [01:12<00:00,  3.18it/s, Loss=0.0523, Type Acc=0.978, LR=0.0001]\n",
            "Evaluating: 100%|██████████| 58/58 [00:15<00:00,  3.85it/s]\n"
          ]
        },
        {
          "metadata": {
            "tags": null
          },
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "\n",
            "Epoch 6/50 Summary:\n",
            "Train Loss: 0.0523 | Type Acc: 97.81% | Grade Acc: 77485.71%\n",
            "Val Loss: 0.0450 | Type Acc: 98.58% | Grade Acc: 15977.78%\n",
            "Saved new best model to /content/drive/MyDrive/BrainTumorModels/best_self_supervised_model_t4.pth\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "Fine-tuning Epoch 7: 100%|██████████| 230/230 [01:14<00:00,  3.11it/s, Loss=0.0483, Type Acc=0.981, LR=0.0001]\n",
            "Evaluating: 100%|██████████| 58/58 [00:15<00:00,  3.66it/s]\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "Epoch 7/50 Summary:\n",
            "Train Loss: 0.0483 | Type Acc: 98.09% | Grade Acc: 90233.33%\n",
            "Val Loss: 0.0446 | Type Acc: 98.26% | Grade Acc: 16000.00%\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "Fine-tuning Epoch 8: 100%|██████████| 230/230 [01:09<00:00,  3.29it/s, Loss=0.0336, Type Acc=0.985, LR=0.0001]\n",
            "Evaluating: 100%|██████████| 58/58 [00:15<00:00,  3.77it/s]\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "Epoch 8/50 Summary:\n",
            "Train Loss: 0.0336 | Type Acc: 98.53% | Grade Acc: 54610.00%\n",
            "Val Loss: 0.0497 | Type Acc: 98.42% | Grade Acc: 15966.67%\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "Fine-tuning Epoch 9: 100%|██████████| 230/230 [01:09<00:00,  3.29it/s, Loss=0.0336, Type Acc=0.988, LR=0.0001]\n",
            "Evaluating: 100%|██████████| 58/58 [00:15<00:00,  3.71it/s]\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "Epoch 9/50 Summary:\n",
            "Train Loss: 0.0336 | Type Acc: 98.79% | Grade Acc: 54760.00%\n",
            "Val Loss: 0.0425 | Type Acc: 98.86% | Grade Acc: 16000.00%\n",
            "Saved new best model to /content/drive/MyDrive/BrainTumorModels/best_self_supervised_model_t4.pth\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "Fine-tuning Epoch 10: 100%|██████████| 230/230 [01:12<00:00,  3.18it/s, Loss=0.0261, Type Acc=0.99, LR=0.0001]\n",
            "Evaluating: 100%|██████████| 58/58 [00:14<00:00,  3.93it/s]\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "Epoch 10/50 Summary:\n",
            "Train Loss: 0.0261 | Type Acc: 98.99% | Grade Acc: 54660.00%\n",
            "Val Loss: 0.0340 | Type Acc: 99.07% | Grade Acc: 16066.67%\n",
            "Saved new best model to /content/drive/MyDrive/BrainTumorModels/best_self_supervised_model_t4.pth\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "Fine-tuning Epoch 11: 100%|██████████| 230/230 [01:13<00:00,  3.15it/s, Loss=0.0246, Type Acc=0.989, LR=0.0001]\n",
            "Evaluating: 100%|██████████| 58/58 [00:14<00:00,  3.90it/s]\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "Epoch 11/50 Summary:\n",
            "Train Loss: 0.0246 | Type Acc: 98.94% | Grade Acc: 79100.00%\n",
            "Val Loss: 0.0230 | Type Acc: 99.51% | Grade Acc: 16088.89%\n",
            "Saved new best model to /content/drive/MyDrive/BrainTumorModels/best_self_supervised_model_t4.pth\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "Fine-tuning Epoch 12: 100%|██████████| 230/230 [01:13<00:00,  3.14it/s, Loss=0.0233, Type Acc=0.991, LR=0.0001]\n",
            "Evaluating: 100%|██████████| 58/58 [00:15<00:00,  3.79it/s]\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "Epoch 12/50 Summary:\n",
            "Train Loss: 0.0233 | Type Acc: 99.14% | Grade Acc: 91983.33%\n",
            "Val Loss: 0.0364 | Type Acc: 99.29% | Grade Acc: 16000.00%\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "Fine-tuning Epoch 13: 100%|██████████| 230/230 [01:09<00:00,  3.29it/s, Loss=0.023, Type Acc=0.99, LR=0.0001]\n",
            "Evaluating: 100%|██████████| 58/58 [00:14<00:00,  3.95it/s]\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "Epoch 13/50 Summary:\n",
            "Train Loss: 0.0230 | Type Acc: 99.02% | Grade Acc: 78300.00%\n",
            "Val Loss: 0.0301 | Type Acc: 98.96% | Grade Acc: 16044.44%\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "Fine-tuning Epoch 14: 100%|██████████| 230/230 [01:09<00:00,  3.31it/s, Loss=0.0223, Type Acc=0.991, LR=0.0001]\n",
            "Evaluating: 100%|██████████| 58/58 [00:14<00:00,  3.95it/s]\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "Epoch 14/50 Summary:\n",
            "Train Loss: 0.0223 | Type Acc: 99.10% | Grade Acc: 77857.14%\n",
            "Val Loss: 0.0244 | Type Acc: 99.18% | Grade Acc: 16066.67%\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "Fine-tuning Epoch 15: 100%|██████████| 230/230 [01:09<00:00,  3.31it/s, Loss=0.018, Type Acc=0.993, LR=0.0001]\n",
            "Evaluating: 100%|██████████| 58/58 [00:15<00:00,  3.84it/s]\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "Epoch 15/50 Summary:\n",
            "Train Loss: 0.0180 | Type Acc: 99.28% | Grade Acc: 55270.00%\n",
            "Val Loss: 0.0383 | Type Acc: 99.07% | Grade Acc: 16011.11%\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "Fine-tuning Epoch 16: 100%|██████████| 230/230 [01:10<00:00,  3.27it/s, Loss=0.0164, Type Acc=0.995, LR=5e-5]\n",
            "Evaluating: 100%|██████████| 58/58 [00:14<00:00,  3.91it/s]\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "Epoch 16/50 Summary:\n",
            "Train Loss: 0.0164 | Type Acc: 99.52% | Grade Acc: 78400.00%\n",
            "Val Loss: 0.0218 | Type Acc: 99.56% | Grade Acc: 16100.00%\n",
            "Saved new best model to /content/drive/MyDrive/BrainTumorModels/best_self_supervised_model_t4.pth\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "Fine-tuning Epoch 17: 100%|██████████| 230/230 [01:12<00:00,  3.18it/s, Loss=0.00903, Type Acc=0.996, LR=5e-5]\n",
            "Evaluating: 100%|██████████| 58/58 [00:14<00:00,  3.96it/s]\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "Epoch 17/50 Summary:\n",
            "Train Loss: 0.0090 | Type Acc: 99.56% | Grade Acc: 68625.00%\n",
            "Val Loss: 0.0218 | Type Acc: 99.56% | Grade Acc: 16077.78%\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "Fine-tuning Epoch 18: 100%|██████████| 230/230 [01:09<00:00,  3.32it/s, Loss=0.00964, Type Acc=0.996, LR=5e-5]\n",
            "Evaluating: 100%|██████████| 58/58 [00:14<00:00,  3.92it/s]\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "Epoch 18/50 Summary:\n",
            "Train Loss: 0.0096 | Type Acc: 99.56% | Grade Acc: 69612.50%\n",
            "Val Loss: 0.0327 | Type Acc: 99.24% | Grade Acc: 16011.11%\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "Fine-tuning Epoch 19: 100%|██████████| 230/230 [01:07<00:00,  3.38it/s, Loss=0.0109, Type Acc=0.996, LR=5e-5]\n",
            "Evaluating: 100%|██████████| 58/58 [00:14<00:00,  4.06it/s]\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "Epoch 19/50 Summary:\n",
            "Train Loss: 0.0109 | Type Acc: 99.63% | Grade Acc: 50054.55%\n",
            "Val Loss: 0.0222 | Type Acc: 99.46% | Grade Acc: 16088.89%\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "Fine-tuning Epoch 20: 100%|██████████| 230/230 [01:07<00:00,  3.39it/s, Loss=0.00986, Type Acc=0.996, LR=5e-5]\n",
            "Evaluating: 100%|██████████| 58/58 [00:14<00:00,  4.02it/s]\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "Epoch 20/50 Summary:\n",
            "Train Loss: 0.0099 | Type Acc: 99.58% | Grade Acc: 55390.00%\n",
            "Val Loss: 0.0255 | Type Acc: 99.46% | Grade Acc: 16044.44%\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "Fine-tuning Epoch 21: 100%|██████████| 230/230 [01:07<00:00,  3.40it/s, Loss=0.00918, Type Acc=0.996, LR=2.5e-5]\n",
            "Evaluating: 100%|██████████| 58/58 [00:14<00:00,  4.03it/s]\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "Epoch 21/50 Summary:\n",
            "Train Loss: 0.0092 | Type Acc: 99.59% | Grade Acc: 55140.00%\n",
            "Val Loss: 0.0219 | Type Acc: 99.46% | Grade Acc: 16077.78%\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "Fine-tuning Epoch 22: 100%|██████████| 230/230 [01:07<00:00,  3.41it/s, Loss=0.00563, Type Acc=0.998, LR=2.5e-5]\n",
            "Evaluating: 100%|██████████| 58/58 [00:14<00:00,  3.99it/s]\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "Epoch 22/50 Summary:\n",
            "Train Loss: 0.0056 | Type Acc: 99.81% | Grade Acc: 78028.57%\n",
            "Val Loss: 0.0267 | Type Acc: 99.51% | Grade Acc: 16088.89%\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "Fine-tuning Epoch 23: 100%|██████████| 230/230 [01:07<00:00,  3.41it/s, Loss=0.00546, Type Acc=0.998, LR=2.5e-5]\n",
            "Evaluating: 100%|██████████| 58/58 [00:14<00:00,  4.04it/s]\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "Epoch 23/50 Summary:\n",
            "Train Loss: 0.0055 | Type Acc: 99.82% | Grade Acc: 68300.00%\n",
            "Val Loss: 0.0193 | Type Acc: 99.56% | Grade Acc: 16088.89%\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "Fine-tuning Epoch 24: 100%|██████████| 230/230 [01:07<00:00,  3.39it/s, Loss=0.00618, Type Acc=0.998, LR=2.5e-5]\n",
            "Evaluating: 100%|██████████| 58/58 [00:14<00:00,  4.08it/s]\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "Epoch 24/50 Summary:\n",
            "Train Loss: 0.0062 | Type Acc: 99.75% | Grade Acc: 61144.44%\n",
            "Val Loss: 0.0267 | Type Acc: 99.46% | Grade Acc: 16066.67%\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "Fine-tuning Epoch 25: 100%|██████████| 230/230 [01:07<00:00,  3.40it/s, Loss=0.0047, Type Acc=0.998, LR=2.5e-5]\n",
            "Evaluating: 100%|██████████| 58/58 [00:14<00:00,  4.10it/s]\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "Epoch 25/50 Summary:\n",
            "Train Loss: 0.0047 | Type Acc: 99.80% | Grade Acc: 68662.50%\n",
            "Val Loss: 0.0250 | Type Acc: 99.51% | Grade Acc: 16077.78%\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "Fine-tuning Epoch 26: 100%|██████████| 230/230 [01:07<00:00,  3.40it/s, Loss=0.00598, Type Acc=0.998, LR=2.5e-5]\n",
            "Evaluating: 100%|██████████| 58/58 [00:14<00:00,  4.00it/s]\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "Epoch 26/50 Summary:\n",
            "Train Loss: 0.0060 | Type Acc: 99.82% | Grade Acc: 78757.14%\n",
            "Val Loss: 0.0206 | Type Acc: 99.67% | Grade Acc: 16100.00%\n",
            "Saved new best model to /content/drive/MyDrive/BrainTumorModels/best_self_supervised_model_t4.pth\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "Fine-tuning Epoch 27: 100%|██████████| 230/230 [01:10<00:00,  3.27it/s, Loss=0.00337, Type Acc=0.999, LR=2.5e-5]\n",
            "Evaluating: 100%|██████████| 58/58 [00:14<00:00,  3.95it/s]\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "Epoch 27/50 Summary:\n",
            "Train Loss: 0.0034 | Type Acc: 99.85% | Grade Acc: 69075.00%\n",
            "Val Loss: 0.0192 | Type Acc: 99.56% | Grade Acc: 16122.22%\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "Fine-tuning Epoch 28: 100%|██████████| 230/230 [01:09<00:00,  3.33it/s, Loss=0.00562, Type Acc=0.998, LR=2.5e-5]\n",
            "Evaluating: 100%|██████████| 58/58 [00:14<00:00,  3.96it/s]\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "Epoch 28/50 Summary:\n",
            "Train Loss: 0.0056 | Type Acc: 99.77% | Grade Acc: 69537.50%\n",
            "Val Loss: 0.0205 | Type Acc: 99.67% | Grade Acc: 16122.22%\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "Fine-tuning Epoch 29: 100%|██████████| 230/230 [01:09<00:00,  3.33it/s, Loss=0.00457, Type Acc=0.999, LR=2.5e-5]\n",
            "Evaluating: 100%|██████████| 58/58 [00:14<00:00,  4.03it/s]\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "Epoch 29/50 Summary:\n",
            "Train Loss: 0.0046 | Type Acc: 99.88% | Grade Acc: 60633.33%\n",
            "Val Loss: 0.0175 | Type Acc: 99.67% | Grade Acc: 16100.00%\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "Fine-tuning Epoch 30: 100%|██████████| 230/230 [01:08<00:00,  3.37it/s, Loss=0.00775, Type Acc=0.997, LR=2.5e-5]\n",
            "Evaluating: 100%|██████████| 58/58 [00:14<00:00,  3.98it/s]\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "Epoch 30/50 Summary:\n",
            "Train Loss: 0.0078 | Type Acc: 99.74% | Grade Acc: 61211.11%\n",
            "Val Loss: 0.0218 | Type Acc: 99.62% | Grade Acc: 16111.11%\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "Fine-tuning Epoch 31: 100%|██████████| 230/230 [01:08<00:00,  3.34it/s, Loss=0.00623, Type Acc=0.998, LR=2.5e-5]\n",
            "Evaluating: 100%|██████████| 58/58 [00:14<00:00,  3.95it/s]\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "Epoch 31/50 Summary:\n",
            "Train Loss: 0.0062 | Type Acc: 99.78% | Grade Acc: 68562.50%\n",
            "Val Loss: 0.0160 | Type Acc: 99.73% | Grade Acc: 16111.11%\n",
            "Saved new best model to /content/drive/MyDrive/BrainTumorModels/best_self_supervised_model_t4.pth\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "Fine-tuning Epoch 32: 100%|██████████| 230/230 [01:10<00:00,  3.25it/s, Loss=0.00354, Type Acc=0.998, LR=2.5e-5]\n",
            "Evaluating: 100%|██████████| 58/58 [00:14<00:00,  3.94it/s]\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "Epoch 32/50 Summary:\n",
            "Train Loss: 0.0035 | Type Acc: 99.84% | Grade Acc: 78942.86%\n",
            "Val Loss: 0.0251 | Type Acc: 99.62% | Grade Acc: 16100.00%\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "Fine-tuning Epoch 33: 100%|██████████| 230/230 [01:08<00:00,  3.35it/s, Loss=0.00302, Type Acc=0.999, LR=2.5e-5]\n",
            "Evaluating: 100%|██████████| 58/58 [00:14<00:00,  4.00it/s]\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "Epoch 33/50 Summary:\n",
            "Train Loss: 0.0030 | Type Acc: 99.90% | Grade Acc: 68237.50%\n",
            "Val Loss: 0.0192 | Type Acc: 99.73% | Grade Acc: 16111.11%\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "Fine-tuning Epoch 34: 100%|██████████| 230/230 [01:09<00:00,  3.33it/s, Loss=0.00657, Type Acc=0.998, LR=2.5e-5]\n",
            "Evaluating: 100%|██████████| 58/58 [00:14<00:00,  3.98it/s]\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "Epoch 34/50 Summary:\n",
            "Train Loss: 0.0066 | Type Acc: 99.78% | Grade Acc: 60955.56%\n",
            "Val Loss: 0.0265 | Type Acc: 99.51% | Grade Acc: 16077.78%\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "Fine-tuning Epoch 35: 100%|██████████| 230/230 [01:09<00:00,  3.32it/s, Loss=0.00407, Type Acc=0.999, LR=2.5e-5]\n",
            "Evaluating: 100%|██████████| 58/58 [00:14<00:00,  3.94it/s]\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "Epoch 35/50 Summary:\n",
            "Train Loss: 0.0041 | Type Acc: 99.88% | Grade Acc: 68725.00%\n",
            "Val Loss: 0.0230 | Type Acc: 99.56% | Grade Acc: 16100.00%\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "Fine-tuning Epoch 36: 100%|██████████| 230/230 [01:08<00:00,  3.35it/s, Loss=0.00266, Type Acc=0.999, LR=1.25e-5]\n",
            "Evaluating: 100%|██████████| 58/58 [00:14<00:00,  3.97it/s]\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "Epoch 36/50 Summary:\n",
            "Train Loss: 0.0027 | Type Acc: 99.90% | Grade Acc: 78385.71%\n",
            "Val Loss: 0.0241 | Type Acc: 99.51% | Grade Acc: 16077.78%\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "Fine-tuning Epoch 37: 100%|██████████| 230/230 [01:08<00:00,  3.34it/s, Loss=0.00381, Type Acc=0.999, LR=1.25e-5]\n",
            "Evaluating: 100%|██████████| 58/58 [00:14<00:00,  3.98it/s]\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "Epoch 37/50 Summary:\n",
            "Train Loss: 0.0038 | Type Acc: 99.85% | Grade Acc: 69337.50%\n",
            "Val Loss: 0.0222 | Type Acc: 99.62% | Grade Acc: 16111.11%\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "Fine-tuning Epoch 38: 100%|██████████| 230/230 [01:09<00:00,  3.31it/s, Loss=0.00234, Type Acc=0.999, LR=1.25e-5]\n",
            "Evaluating: 100%|██████████| 58/58 [00:14<00:00,  3.92it/s]\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "Epoch 38/50 Summary:\n",
            "Train Loss: 0.0023 | Type Acc: 99.92% | Grade Acc: 90816.67%\n",
            "Val Loss: 0.0193 | Type Acc: 99.67% | Grade Acc: 16122.22%\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "Fine-tuning Epoch 39: 100%|██████████| 230/230 [01:09<00:00,  3.30it/s, Loss=0.00222, Type Acc=0.999, LR=1.25e-5]\n",
            "Evaluating: 100%|██████████| 58/58 [00:14<00:00,  3.91it/s]\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "Epoch 39/50 Summary:\n",
            "Train Loss: 0.0022 | Type Acc: 99.92% | Grade Acc: 61288.89%\n",
            "Val Loss: 0.0349 | Type Acc: 99.46% | Grade Acc: 16077.78%\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "Fine-tuning Epoch 40: 100%|██████████| 230/230 [01:09<00:00,  3.33it/s, Loss=0.00252, Type Acc=0.999, LR=6.25e-6]\n",
            "Evaluating: 100%|██████████| 58/58 [00:14<00:00,  3.95it/s]\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "Epoch 40/50 Summary:\n",
            "Train Loss: 0.0025 | Type Acc: 99.89% | Grade Acc: 91416.67%\n",
            "Val Loss: 0.0218 | Type Acc: 99.62% | Grade Acc: 16077.78%\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "Fine-tuning Epoch 41: 100%|██████████| 230/230 [01:09<00:00,  3.32it/s, Loss=0.00263, Type Acc=0.999, LR=6.25e-6]\n",
            "Evaluating: 100%|██████████| 58/58 [00:14<00:00,  3.96it/s]\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "Epoch 41/50 Summary:\n",
            "Train Loss: 0.0026 | Type Acc: 99.92% | Grade Acc: 60911.11%\n",
            "Val Loss: 0.0296 | Type Acc: 99.56% | Grade Acc: 16100.00%\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "Fine-tuning Epoch 42: 100%|██████████| 230/230 [01:09<00:00,  3.32it/s, Loss=0.00503, Type Acc=0.998, LR=6.25e-6]\n",
            "Evaluating: 100%|██████████| 58/58 [00:14<00:00,  3.91it/s]\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "Epoch 42/50 Summary:\n",
            "Train Loss: 0.0050 | Type Acc: 99.81% | Grade Acc: 77857.14%\n",
            "Val Loss: 0.0200 | Type Acc: 99.67% | Grade Acc: 16100.00%\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "Fine-tuning Epoch 43: 100%|██████████| 230/230 [01:10<00:00,  3.25it/s, Loss=0.00397, Type Acc=0.999, LR=6.25e-6]\n",
            "Evaluating: 100%|██████████| 58/58 [00:14<00:00,  3.88it/s]\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "Epoch 43/50 Summary:\n",
            "Train Loss: 0.0040 | Type Acc: 99.86% | Grade Acc: 109480.00%\n",
            "Val Loss: 0.0285 | Type Acc: 99.51% | Grade Acc: 16066.67%\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "Fine-tuning Epoch 44: 100%|██████████| 230/230 [01:09<00:00,  3.30it/s, Loss=0.00328, Type Acc=0.999, LR=3.13e-6]\n",
            "Evaluating: 100%|██████████| 58/58 [00:15<00:00,  3.71it/s]\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "Epoch 44/50 Summary:\n",
            "Train Loss: 0.0033 | Type Acc: 99.89% | Grade Acc: 78285.71%\n",
            "Val Loss: 0.0187 | Type Acc: 99.56% | Grade Acc: 16111.11%\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "Fine-tuning Epoch 45: 100%|██████████| 230/230 [01:09<00:00,  3.31it/s, Loss=0.00261, Type Acc=0.999, LR=3.13e-6]\n",
            "Evaluating: 100%|██████████| 58/58 [00:15<00:00,  3.77it/s]\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "Epoch 45/50 Summary:\n",
            "Train Loss: 0.0026 | Type Acc: 99.90% | Grade Acc: 91716.67%\n",
            "Val Loss: 0.0263 | Type Acc: 99.62% | Grade Acc: 16100.00%\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "Fine-tuning Epoch 46: 100%|██████████| 230/230 [01:10<00:00,  3.28it/s, Loss=0.00229, Type Acc=0.999, LR=3.13e-6]\n",
            "Evaluating: 100%|██████████| 58/58 [00:14<00:00,  3.90it/s]\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "Epoch 46/50 Summary:\n",
            "Train Loss: 0.0023 | Type Acc: 99.92% | Grade Acc: 69012.50%\n",
            "Val Loss: 0.0221 | Type Acc: 99.56% | Grade Acc: 16077.78%\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "Fine-tuning Epoch 47: 100%|██████████| 230/230 [01:09<00:00,  3.30it/s, Loss=0.00318, Type Acc=0.999, LR=3.13e-6]\n",
            "Evaluating: 100%|██████████| 58/58 [00:14<00:00,  3.93it/s]\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "Epoch 47/50 Summary:\n",
            "Train Loss: 0.0032 | Type Acc: 99.89% | Grade Acc: 61044.44%\n",
            "Val Loss: 0.0222 | Type Acc: 99.62% | Grade Acc: 16100.00%\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "Fine-tuning Epoch 48: 100%|██████████| 230/230 [01:09<00:00,  3.32it/s, Loss=0.00141, Type Acc=1, LR=1.56e-6]\n",
            "Evaluating: 100%|██████████| 58/58 [00:14<00:00,  3.92it/s]\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "Epoch 48/50 Summary:\n",
            "Train Loss: 0.0014 | Type Acc: 99.96% | Grade Acc: 68437.50%\n",
            "Val Loss: 0.0225 | Type Acc: 99.62% | Grade Acc: 16088.89%\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "Fine-tuning Epoch 49: 100%|██████████| 230/230 [01:09<00:00,  3.32it/s, Loss=0.00143, Type Acc=1, LR=1.56e-6]\n",
            "Evaluating: 100%|██████████| 58/58 [00:15<00:00,  3.87it/s]\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "Epoch 49/50 Summary:\n",
            "Train Loss: 0.0014 | Type Acc: 99.97% | Grade Acc: 67325.00%\n",
            "Val Loss: 0.0222 | Type Acc: 99.51% | Grade Acc: 16077.78%\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "Fine-tuning Epoch 50: 100%|██████████| 230/230 [01:09<00:00,  3.33it/s, Loss=0.00291, Type Acc=0.998, LR=1.56e-6]\n",
            "Evaluating: 100%|██████████| 58/58 [00:14<00:00,  3.96it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "Epoch 50/50 Summary:\n",
            "Train Loss: 0.0029 | Type Acc: 99.82% | Grade Acc: 61677.78%\n",
            "Val Loss: 0.0249 | Type Acc: 99.67% | Grade Acc: 16100.00%\n",
            "\n",
            "Fine-tuning completed!\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "# Load best model"
      ],
      "metadata": {
        "id": "U3fg9_jeZXtd"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "model.load_state_dict(torch.load(os.path.join(save_dir, \"best_self_supervised_model_t4.pth\")))\n",
        "model.eval()\n",
        "\n",
        "# Final evaluation\n",
        "test_loss, test_type_acc, test_grade_acc, type_preds, type_labels, grade_preds, grade_labels = evaluate()\n",
        "print(f\"\\nFinal Test Performance:\")\n",
        "print(f\"Type Accuracy: {test_type_acc:.2%}\")\n",
        "print(f\"Grade Accuracy: {test_grade_acc:.2%} (on tumor cases only)\")\n",
        "\n",
        "# Classification reports\n",
        "print(\"\\nType Classification Report:\")\n",
        "print(classification_report(type_labels, type_preds, target_names=classes))\n",
        "\n",
        "print(\"\\nGrade Classification Report (tumor cases only):\")\n",
        "if len(grade_labels) > 0:\n",
        "    unique_grades = np.unique(grade_labels)\n",
        "    print(classification_report(\n",
        "        grade_labels,\n",
        "        grade_preds,\n",
        "        labels=unique_grades,\n",
        "        target_names=[grade_descriptions[i] for i in unique_grades]\n",
        "    ))"
      ],
      "metadata": {
        "id": "wepJCPlXXtuq",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "e9b13100-1198-49c1-f5ea-80dddc7d6956"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "Evaluating: 100%|██████████| 58/58 [00:14<00:00,  4.02it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "Final Test Performance:\n",
            "Type Accuracy: 99.62%\n",
            "Grade Accuracy: 16100.00% (on tumor cases only)\n",
            "\n",
            "Type Classification Report:\n",
            "              precision    recall  f1-score   support\n",
            "\n",
            "      glioma       0.99      1.00      0.99       503\n",
            "  meningioma       1.00      0.99      0.99       473\n",
            "   pituitary       1.00      1.00      1.00       478\n",
            "    no_tumor       1.00      1.00      1.00       381\n",
            "\n",
            "    accuracy                           1.00      1835\n",
            "   macro avg       1.00      1.00      1.00      1835\n",
            "weighted avg       1.00      1.00      1.00      1835\n",
            "\n",
            "\n",
            "Grade Classification Report (tumor cases only):\n",
            "                                       precision    recall  f1-score   support\n",
            "\n",
            "   Grade I (Least aggressive, benign)       1.00      1.00      1.00       951\n",
            "Grade IV (Most aggressive, malignant)       0.99      1.00      1.00       503\n",
            "\n",
            "                             accuracy                           1.00      1454\n",
            "                            macro avg       1.00      1.00      1.00      1454\n",
            "                         weighted avg       1.00      1.00      1.00      1454\n",
            "\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "# Confusion matrices"
      ],
      "metadata": {
        "id": "V_LsO8unZbvA"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "plt.figure(figsize=(15, 6))\n",
        "plt.subplot(1, 2, 1)\n",
        "cm_type = confusion_matrix(type_labels, type_preds)\n",
        "sns.heatmap(cm_type, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)\n",
        "plt.title('Type Confusion Matrix')\n",
        "plt.xlabel('Predicted')\n",
        "plt.ylabel('True')\n",
        "\n",
        "if len(grade_labels) > 0:\n",
        "    plt.subplot(1, 2, 2)\n",
        "    cm_grade = confusion_matrix(grade_labels, grade_preds, labels=unique_grades)\n",
        "    sns.heatmap(cm_grade, annot=True, fmt='d', cmap='Reds',\n",
        "                xticklabels=[grade_descriptions[i] for i in unique_grades],\n",
        "                yticklabels=[grade_descriptions[i] for i in unique_grades])\n",
        "    plt.title('Grade Confusion Matrix')\n",
        "    plt.xlabel('Predicted')\n",
        "    plt.ylabel('True')\n",
        "\n",
        "plt.tight_layout()\n",
        "plt.show()\n",
        "\n",
        "# Training curves\n",
        "plt.figure(figsize=(15, 5))\n",
        "plt.subplot(1, 2, 1)\n",
        "plt.plot(train_type_accs, label='Train Type Acc')\n",
        "plt.plot(val_type_accs, label='Val Type Acc')\n",
        "plt.title('Type Accuracy Over Epochs')\n",
        "plt.xlabel('Epoch')\n",
        "plt.ylabel('Accuracy')\n",
        "plt.legend()\n",
        "\n",
        "plt.subplot(1, 2, 2)\n",
        "plt.plot(train_grade_accs, label='Train Grade Acc')\n",
        "plt.plot(val_grade_accs, label='Val Grade Acc')\n",
        "plt.title('Grade Accuracy Over Epochs')\n",
        "plt.xlabel('Epoch')\n",
        "plt.ylabel('Accuracy')\n",
        "plt.legend()\n",
        "\n",
        "plt.tight_layout()\n",
        "plt.show()"
      ],
      "metadata": {
        "id": "c_JGe0r4Xtrq",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 1000
        },
        "outputId": "777c6bf7-a1d7-48e0-c07b-32da41a45f1b"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 1500x600 with 4 Axes>"
            ],
            "image/png": "iVBORw0KGgoAAAANSUhEUgAABa0AAAJOCAYAAAC0tysEAAAAOnRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjEwLjAsIGh0dHBzOi8vbWF0cGxvdGxpYi5vcmcvlHJYcgAAAAlwSFlzAAAPYQAAD2EBqD+naQAA7BxJREFUeJzs3XdYFFfbBvB7lt4RFBALNkRQbBAVe8eGPUZFRcUKKooVo4IVO5bYC1hj7DF2rFFExRorNhQLRUFQQanz/eHHvq6AssiyIPcv11yXe+bMmWd2Icx59sw5giiKIoiIiIiIiIiIiIiICgCJsgMgIiIiIiIiIiIiIsrApDURERERERERERERFRhMWhMRERERERERERFRgcGkNREREREREREREREVGExaExEREREREREREVGBwaQ1ERERERERERERERUYTFoTERERERERERERUYHBpDURERERERERERERFRhMWhMRERERERERERFRgcGkNRF914IFC1ChQgWoqKigZs2aed5+//79Ua5cuTxvt7A6c+YMBEHAmTNnlB0KEREREVGB9zP2J0JCQlC/fn3o6OhAEATcuHEjT9tnnyOzcuXKoX///soOg4j+H5PWVKQIgpCjraD94U5LS4O/vz+aNm0KIyMjaGhooFy5chgwYACuXLmi0HMfP34cEyZMQIMGDeDv7485c+Yo9Hz56enTp9LPfNasWVnWcXZ2hiAI0NXVzdU5tm/fjiVLlvxAlEREREREBVNYWBhGjBiBypUrQ1tbG9ra2rCxsYG7uzv+++8/ZYf3w27cuIE+ffqgTJky0NDQgJGREVq2bAl/f3+kpaUp7LwpKSn49ddfERsbCz8/P2zZsgUWFhYKO19+a9q0KQRBgKWlZZb7AwMDpf203bt3y93+3bt34ePjg6dPn/5gpESkTKrKDoAoP23ZskXm9ebNmxEYGJip3NraOj/D+qaPHz+ia9euOHr0KBo3bozJkyfDyMgIT58+xc6dO7Fp0yaEh4ejdOnSCjn/qVOnIJFIsGHDBqirqyvkHOvWrUN6erpC2s4JTU1N/Pnnn5gyZYpMeUJCAv7++29oamrmuu3t27fj9u3bGD16dI6Pady4MT5+/Kiw95uIiIiI6EcdPHgQv/32G1RVVeHs7IwaNWpAIpHg/v372Lt3L1atWoWwsLBCm2xdv349hg0bBlNT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/PmIjIzElClT4O7unuNEORERERERkTJICu/gabkwaU1ERER5T1Dcd/ovXrxAr169EBMTgxIlSqBhw4a4ePEiSpQoAQDw8/ODRCJBt27dkJSUBEdHR6xcuVJ6vIqKCg4ePIjhw4fDwcEBOjo6cHFxwYwZMxQWMxERERERUV4ozKOn5cGkNREREeU9BY603rFjxzf3a2pqYsWKFVixYkW2dSwsLHD48OG8Do2IiIiIiEihCvM81fJg0pqIiIjyngJHWhMRERERERVVRaWnxaQ1ERER5b0i8u0/ERERERFRfuKc1kRERES5xZHWREREREREea6o9LSYtCYiIqK8x5HWREREREREeU4oIn0tpSet09LS4Ofnh507dyI8PBzJycky+2NjY5UUGREREeUaR1oTERERERHluaLS01L6dU6fPh2LFy/Gb7/9hvj4eHh6eqJr166QSCTw8fFRdnhERESUG4Ig30ZERERERETfJRHk2worpSett23bhnXr1mHs2LFQVVVFr169sH79ekybNg0XL15UdnhERESUG4JEvo2IiIiIiIi+SyLnVlgpPfbIyEjY2toCAHR1dREfHw8A6NChAw4dOqTM0IiIiCi3mLQmIiIiIiLKcxJBkGsrrJTeSyxdujQiIiIAABUrVsTx48cBACEhIdDQ0FBmaERERJRbReWZNSIiIiIionzEkdb5pEuXLjh58iQAYOTIkZg6dSosLS3Rr18/DBw4UMnRERERUa5wpDUREREREVGeKyrjg1SVHcDcuXOl//7tt99QtmxZBAcHw9LSEk5OTkqMjIiIiHKtED+GRkREREREVFAVlSE/Sk9af83BwQEODg7KDoOIiIh+BEdPExERERER5TkJisYAoQKRtH716hXOnz+P6OhopKeny+wbNWqUkqIiIiKiXONIayIiIiIiojxXmKf8kIfSk9YBAQEYOnQo1NXVYWxsDOGLTq4gCExaExERFUYcaU1ERERERJTnikpPS+lJ66lTp2LatGnw8vKCRFJU3nYiIqKfHEdaExERERER5TmOtM4niYmJ6NmzJxPWREREPxOOtCYiIiIiIspzRWVOa6X3KF1dXbFr1y5lh0FERER5SRDk24iIiIiIiOi7JIJ8W2Gl9JHWvr6+6NChA44ePQpbW1uoqanJ7F+8eLGSIiMiIqJc40hrIiIiIiKiPFdUeloFIml9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Av0gTSV8Rb3t6gu56jvblVZQJERElKcS45UdAREBSHLrpuwQiAiARsCJfD9n2prJctVXGTpHQZEoltJGWn85JciPTA9CREREBQ8HWhMRERERESlAEelsFYg5rYmIiOjnwoUYiYiIiIiIFEDCOa3zTbNmzb7ZuZV3ehAiIiJSrqJxG0VERERERJTPisgAoQKRtK5Zs6bM65SUFNy4cQO3b9+Gi4uLcoIiIiKiXONIayIiIiIiIgUQisYQoQKRtPbz88uy3MfHBx8+fMjnaIiIiOhHMWVNRERERESkAEVkgFCBTs336dMHGzduVHYYREREJCdBEOTaiIiIiIiIKAckEvm2QqpAjLTOTnBwMDQ1NZUdBhEREcmp8N4aERERERERFWBFZNBPgUhad+3aVea1KIqIiIjAlStXMHXqVCVFRURERLnF0dNEREREREQKwDmt84++vr5M51YikcDKygozZsxA69atlRgZERER5QZT1kRERERERAogKRq9rQKRtA4ICFB2CERERJSHONCaiIiIiIhIAYrISOsCcZUVKlRATExMpvK4uDhUqFBBCRERERHRj5BAkGsjIiIiIiKiHBAE+bZCqkCMtH769CnS0tIylSclJeHly5dKiIiIiIh+RCG+NyIiIiIiIiq4ishIa6UmrQ8cOCD997Fjx2BgYCB9nZaWhpMnT6JcuXJKiIyIiIh+hMDR00RERERERHmPc1orXufOnQEAgiDAxcVFZp+amhrKlSuHRYsWKSEyIiIi+hEcaU1ERERERKQARaSzpdSkdXp6OgCgfPnyCAkJQfHixZUZDhEREeURzlNNRERERESkAJweJP+EhYUpOwQiIiLKQ0Xky38iIiIiIqL8xelBFGvZsmUYMmQINDU1sWzZsm/WHTVqVD5FRURERHmBSWsiIiIiIiIF4EhrxfLz84OzszM0NTXh5+eXbT1BEJi0JiIiKmS4ECMREREREZECFJERQkpLWn85JciX/xZFEcDnZDUREREVTkXkiTUiIiIiIqL8VURGWheYq9ywYQOqVasGTU1NaGpqolq1ali/fr2ywyIiIqJcEOT8j4iIiIiIiHJAIsi3FVIFImk9bdo0eHh4wMnJCbt27cKuXbvg5OSEMWPGYNq0acoOj4iIiOQkCPJt9HPq378/OnfurOwwpE6ePAlra2ukpaUpO5QiKSAgAIaGhsoO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Hx8FBwerdevWunLlitUxe/Toof79+ysoKEj58uXTsGHDrO73r+fu5MmTMplMWrx4sWrXri0vLy+VLVtW27Zts7rNF198ocKFC8vLy0svvviioqOjFRAQYNn+v+1B0tPTNWLECBUqVEju7u4qV66cVq1aZdl+736//fZbPfvss/L09FSlSpV09OhR7dy5U0899ZR8fHzUqFEjXb582XK7nTt3qn79+sqdO7f8/f1Vs2ZN7dmz51+cBePUqlVL3bp1U7du3eTv76/cuXNr8ODBljcGf20PEhoaKkl68cUXZTKZLNfvva7+qlevXqpVq5bV/dx7HdeqVUunTp1S7969ZTKZZDKZJElXr17Va6+9poIFC8rLy0uRkZGaP3/+ffP26tVLuXPnVlRUlDp06KDnn3/ear+7d+8qb968mjFjxr9/kuzQP73GTp8+raZNm8rHx0d+fn56+eWXdfHixX887uzZszV8+HDt37/fcm5mz55tea38dfZ1QkKCTCaT1q9fL0lav369TCaTfvzxR5UvX16enp6qU6eOLl26pJUrV6pUqVLy8/NTy5Ytdfv2bctxkpOT1aNHD+XNm1ceHh565plntHPnTsv2e8dduXKlKlasKHd3d23evPlfP4cAAAD3REdHKzQ0VLNmzVK9evUsvzo7evSotm3bpqFDhyo1NVUNGjRQw4YN9fvvvxsdGXBIPXr0UNmyZRUXF6cPPvhAhw4d0vXr15WSkqILFy5oxYoVeuaZZzRkyBCVKVPG6nMD4OwoWttYnz59tGXLFi1ZskSrV6/Wpk2bHroYeOvWLUVFRSkwMFA7d+7Ud999p59//lndunWz2m/t2rX6448/tHHjRkVHR2vo0KF6/vnnFRgYqO3bt+utt95S586ddfbsWcttfH19NXv2bB06dEiTJk3SF198oY8//vixPnYjzZkzR7lz59aOHTvUvXt3denSRS+99JKqVaumPXv2qEGDBmrdurVu376thIQE1alTR+XLl9euXbu0atUqXbx4US+//HKGY3p7e2v79u0aN26cRowYodWrV/9tjvfee0/vvPOO9u3bpyeeeEKvvfaaUlNTJUlbtmzRW2+9pZ49e2rfvn2qX7++Ro0a9bfHmzRpkiZMmKDx48fr119/VVRUlJo0aZLhTebQoUP1/vvva8+ePXJ1dVXLli3Vv39/TZo0SZs2bdKx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\n"
          },
          "metadata": {}
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 1500x500 with 2 Axes>"
            ],
            "image/png": "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7ZywcFnKss4/v/vuO4YMGeLoS1RUFP/617+w2WzF+l3WtcP8+fPR6XTs3r272HKvv/46BoOhxHNNzVtvvUVmZiYff/yxUwAdICAggP/9739kZGTw5ptvArB8+XJ0Oh0bNmwotq7//e9/6HQ6Dhw44Jh25MgRbr/9dvz8/DCbzXTr1o3vv//eaTntmG7YsIGHH36YoKAgGjVqVGqfKyIiIoKhQ4fyyy+/0KlTJ8xmM23atGHlypXF2p46dYo77rgDPz8/3N3d6dmzJ6tWrSrWLjs7m5dffpkWLVpgNpsJDQ3ltttu4+TJk8Xafvzxx45rtOuuu44dO3Y4zY+NjWXixIk0atQIk8lEaGgow4cPL/b5EkJUH8lEF0LUSffccw9ffPEFa9asYejQoY7psbGx/Pbbb7z00ktO7Tds2MCSJUt45JFHMJlMfPjhh9xyyy1s376ddu3aARAXF0fPnj0dQffAwEB+/vlnJk+ejMVi4bHHHrtiv77//nvS09O58847CQkJYcCAASxatIi77rrLqd2CBQuYNGkSbdu2Zfr06fj6+rJ7925Wr17taLt27VqGDh1KaGgojz76KCEhIRw+fJgff/yRRx99tFLHbf78+WRnZzNlyhRMJhN+fn5YLBY+/fRTxo4dy/33309aWhrz5s0jOjqa7du306lTJ8fykydPZsGCBdx6663cd999WK1WNm3axJ9//km3bt245557uP/++zlw4IDjuALs2LGDY8eOMWPGjFL7tnnzZi5fvsyjjz6K0Vjy19P48eOZP38+P/74Iz179mTMmDGMHz+eHTt2cN111znanTlzhj///NMpm/21117jxRdfZPTo0dx3330kJCTw/vvv069fP3bv3u1UdiUpKYlbb72VO++8k7vvvrtc5WOSk5OLTTMajcXKubz22mvodDqeffZZ4uPjmTNnDoMGDWLPnj24ubkB6udj4sSJXHfddcyaNYu4uDjmzp3Lli1bnPq6b98++vbti4uLC1OmTCEiIoKTJ0/yww8/8Nprrzltd/To0URGRjJr1ix27drFp59+SlBQEG+88Qag1rccOnQoHTp04NVXX8VkMnHixAm2bNlyxX0XQgghRN32448/0qxZM3r06FGh5axWK9HR0fTp04e3337bkTm9bNkyMjMzeeihh/D392f79u28//77nD9/nmXLljmW/+WXXxg1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4cKFuLq6Mnr06AqvSwsU3n333UyYMKHENh06dChzHdrJaO/evUucf+rUqSofqV6n0zllYGtKGmgInDNZNAsXLuTee+9lxIgRPP300wQFBWEwGJg1a1aJg95cSXR0NMHBwSxcuJB+/fqxcOFCQkJCGDRoUJnLaSee+/btY8SIESW22bdvH4BT1siwYcNwd3dn6dKlXH/99SxduhS9Xs8dd9zhaGO329HpdPz8888lZq0Uzbwq6TjVdaVl62ifHzc3NzZu3Mj69etZtWoVq1evZsmSJdx444388ssvFcr2EUIIIUTd4ePjQ2hoqNMgjxqtRnppgxeaTCb0eufcPJvNxk033URycjLPPvssrVq1wsPDgwsXLnDvvfdW6ga93W4nKCio1IFAiz5pWdSyZcvIycnhnXfe4Z133ik2f9GiRbzyyisV7ldZSstIr8h5ekpKCv3798fb25tXX32VqKgozGYzu3bt4tlnn63wsTQYDNx111188sknfPjhh2zZsoWLFy86gvKl0T4j2rl4afbt20fDhg0dwWmTycSIESP45ptv+PDDD4mLi2PLli28/vrrjmW0fXjqqaeIjo4ucb3a0wOa+naufqXzdIDHHnuMYcOG8e2337JmzRpefPFFZs2axW+//Vasxr0QonpIEF0IUaeNHz+eJ554gkuXLrF48WKGDBni9Cic5vjx48WmHTt2DHd3d8dJt5eXFzab7YrB3pLExMTwxx9/MG3atGID59jtdu655x4WL17MjBkzHIPIHDhwoNgJoaZwm7L606BBA0dGcmFaBkR5LF++nKZNm7Jy5Uqnk/2iZVuioqJYs2YNycnJZWajayfnCxYs4I033uDbb78t1yOXffr0wdfXl8WLF/PCCy+U2P6LL74AcBpM1sPDg6FDh7Js2TJmz57NkiVL6Nu3L2FhYU59VxSFyMhIRzZJbSn6WVQUhRMnTjhu0jRp0gRQn6y48cYbndoePXrUMV+7IVPSBW9l6fV6Bg4cyMCBA5k9ezavv/46L7zwAuvXr6/U/wshhBBC1A1Dhgzh008/Zfv27XTv3v2q1rV//36OHTvG559/zvjx4x3T165d69ROO6cp6Tz96NGjTu+joqL49ddf6d27d6UCqIsWLaJdu3YlliX83//+x+LFi3nllVcIDAzE29v7iudXUVFRV2yjXZOkpKQ4Ta/Iefrvv/9OUlISK1eupF+/fo7pMTExxfoDV752APX66Z133uGHH37g559/JjAwsNTgdWFDhw7lk08+YfPmzU6lTDSbNm3i9OnTPPDAA07Tx4wZw+eff866des4fPgwiqI4PTGqndO6uLjU+vmmlhVf+JpIe0pBG7yzSZMmxT6fgKO0jva5joqKYtu2beTl5V1x0NvyioqK4sknn+TJJ5/k+PHjdOrUiXfeeafUJ0WEEFVLyrkIIeq0sWPHotPpePTRRzl16lSpWRRbt251qu997tw5vvvuO26++WYMBgMGg4FRo0axYsWKEk+IExISyuyHlhXzzDPPcPvttzu9Ro8eTf/+/R1tbr75Zry8vJg1axbZ2dlO69GyDbp06UJkZCRz5swpduJdOCMhKiqKI0eOOPVv7969bNmypcz+FqYFqwuvd9u2bY7yKJpRo0ahKEqJWTpFs+HvueceLl++zAMPPEB6evoVs1sA3N3deeqppzh69CgvvPBCsfmrVq1iwYIFREdH07NnT6d5Y8aM4eLFi3z66afs3bvX6cQc4LbbbsNgMPDKK68U66uiKCQlJV2xf1Xliy++cHoUdvny5Vy6dIlbb70VgG7duhEUFMRHH31ETk6Oo93PP//M4cOHGTJkCKBmXPXr14/PPvuMs2fPOm2jpKcTriQ5ObnYtE6dOgE49UMIIYQQ9c8zzzyDu7s7kyZNIi4urtj8ipxblHRuqSgKc+fOdWoXGhpKp06d+Pzzz51KCK5du9ZRV1ozevRobDYb//rXv4ptz2q1FjtfLuzcuXNs3LiR0aNHFztPv/3225k4cSInTpxg27Zt6PV6RowYwQ8//MBff/1VbF3aPo0aNYq9e/c6ykqW1EYLbG/cuNExz2az8fHHH5fa16JKOpa5ubl8+OGHTu3Ke+0A6tO1HTp04NNPP2XFihXceeed5So58vTTT+Pm5sYDDzxQ7Nw5OTmZBx98EHd3d55++mmneYMGDcLPz48lS5awZMkSunfv7lSOJSgoiAEDBvC///2PS5cuFdvula7DqtLFixedfqcWi4UvvviCTp06ERISAsDgwYPZvn2707VSRkYGH3/8MREREY4nZkeNGkViYiL/+c9/im2noufqmZmZxa4bo6Ki8PLykvN0IWqQZKILIeq0wMBAbrnlFpYtW4avr68jwFhUu3btiI6O5pFHHsFkMjlOPAsHhP/v//6P9evX06NHD+6//37atGlDcnIyu3bt4tdffy0xyKhZtGgRnTp1onHjxiXO/8c//sE///lPdu3aRZcuXXj33Xe57777uO6667jrrrto0KABe/fuJTMzk88//xy9Xs9///tfhg0bRqdOnZg4cSKhoaEcOXKEgwcPsmbNGgAmTZrE7NmziY6OZvLkycTHx/PRRx/Rtm1bx4A+VzJ06FBWrlzJyJEjGTJkCDExMXz00Ue0adPGqTbmDTfcwD333MN7773H8ePHueWWW7Db7WzatIkbbriBadOmOdp27tyZdu3asWzZMlq3bk2XLl3K1ZfnnnuO3bt388Ybb7B161ZGjRqFm5sbmzdvZuHChbRu3ZrPP/+82HKDBw/Gy8uLp556ynFDpLCoqCj+/e9/M336dE6fPs2IESPw8vIiJiaGb775hilTpvDUU0+Vq4+lWb58eYkDct10000EBwc73vv5+dGnTx8mTpxIXFwcc+bMoVmzZtx///2AmoXzxhtvMHHiRPr378/YsWOJi4tj7ty5RERE8PjjjzvW9d5779GnTx+6dOnClClTiIyM5PTp06xatYo9e/ZUqP+vvvoqGzduZMiQITRp0oT4+Hg+/PBDGjVqVGK2kRBCCCHqj+bNm7N48WLGjh1Ly5YtGTduHB07dkRRFGJiYli8eDF6vb7E+udFtWrViqioKJ566ikuXLiAt7c3K1asKHFAxFmzZjFkyBD69OnDpEmTSE5O5v3336dt27ZO56H9+/fngQceYNasWezZs4ebb74ZFxcXjh8/zrJly5g7dy633357if1ZvHgxiqLwj3/8o8T5gwcPxmg0smjRInr06MHrr7/OL7/8Qv/+/ZkyZQqtW7fm0qVLLFu2jM2bN+Pr68vTTz/N8uXLueOOO5g0aRJdu3YlOTmZ77//no8++oiOHTvStm1bevbsyfTp0x1Pcn799ddYrdZy/lbg+uuvp0GDBkyYMIFHHnkEnU7Hl19+WSwIW95rB8348eMd577lSXYB9TPy+eefM27cONq3b8/kyZMd557z5s0jMTGRr776ynHzQOPi4sJtt93G119/TUZGBm+//XaxdX/wwQf06dOH9u3bc//999O0aVPi4uLYunUr58+fZ+/eveU+ZiXZtWtXidnaUVFRTmMYtWjRgsmTJ7Njxw6Cg4P57LPPiIuLY/78+Y42zz33HF99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pSgKmZmZxMfHAxAaGlrLPRJCXAskiC6EEEIIIYQQQohrks1mcwTQ/f39a7s74m/Czc0NgPj4eIKCgqS0ixBCBhYVQgghhBBCCCHEtUmrge7u7l7LPRF/N9pnTurwCyFAguhCCCGEEEIIIYS4xkkJF1HT5DMnhChMguhCCCGEEEIIIYQQQgghRCkkiC6EEEIIIYQQQghxjYuIiGDOnDm13Y1KGTBgAI899lhtd0MIISqtVoPoGzduZNiwYYSFhaHT6fj222+vuMzvv/9Oly5dMJlMNGvWjAULFhRr88EHHxAREYHZbKZHjx5s377daX52djZTp07F398fT09PRo0aRVxcXBXtlRBCCCGEEEIIIf6udDpdma+XX365UuvdsWMHU6ZMuer+nThxgkmTJhEeHo7JZKJhw4YMHDiQRYsWYbVar3r91W3WrFkYDAbeeuut2u6KEOJvpFaD6BkZGXTs2JEPPvigXO1jYmIYMmQIN9xwA3v27OGxxx7jvvvuY82aNY42S5Ys4YknnuCll15i165ddOzYkejoaOLj4x1tHn/8cX744QeWLVvGhg0buHjxIrfddluV758QQgghhBBCCCH+Xi5duuR4zZkzB29vb6dpTz31lKOtoijlDlwHBgZe9QCr27dvp0uXLhw+fJgPPviAAwcO8Pvvv3Pffffx3//+l4MHD5a67LUywOZnn33GM888w2effVbbXRFC/I3UahD91ltv5d///jcjR44sV/uPPvqIyMhI3nnnHVq3bs20adO4/fbbeffddx1tZs+ezf3338/EiRNp06YNH330Ee7u7o4/rqmpqcybN4/Zs2dz44030rVrV+bPn88ff/zBn3/+WS37KYQQQgghhBBCiL+HkJAQx8vHxwedTud4f+TIEby8vPj555/p2rUrJpOJzZs3c/LkSYYPH05wcDCenp5cd911/Prrr07rLVrORafT8emnnzJy5Ejc3d1p3rw533//fan9UhSFe++9lxYtWrBlyxaGDRtG8+bNad68OWPHjmXz5s106NABgNOnT6PT6ViyZAn9+/fHbDazaNEikpKSGDt2LA0bNsTd3Z327dvz1VdfOW0nIyOD8ePH4+npSWhoKO+8806xvuTk5PDUU0/RsGFDPDw86NGjB7///vsVj+2GDRvIysri1VdfxWKx8McffzjNt9vtvPnmmzRr1gyTyUR4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iieeeAKjRo3y66GiADvRiYiIiDoqV26jt+SZZ57B6NGj8corr2D69Ok4dOgQ/u///g//93//B8Cayfz000/jz3/+M3r06IHU1FS8+OKLSEhIwJQpUwBYO9fvuOMOMQbGYDBg0aJFmDFjBhISEly6vr7kepmQxd3+Blq2lqc70dvzUFGB2IluKwrX77D21R0HUqkE3WOCcTKnApcKq8XOdE/Ksu106BLhm7nxRETkXU4X0V9++WVkZmZi3LhxkMutVzebzZg9e7bTeYv3338/ioqKsGzZMuTn52PQoEHYtm2bOIQoKyvLLu+8trYWS5cuxZUrVxAcHIxJkybhgw8+sBsS2tJtAsAbb7wBqVSKadOmQafTISMjA2+//bazT4XP0eqFTHQW0YmIiIg6Elduo7dk+PDh2Lx5M5YsWYLly5cjNTUVq1atwqxZs8RlnnvuOWg0GixYsADl5eW4+eabsW3bNqhUKnGZjz76CIsWLcK4cePEbfM333zTpevqa3xhqKivETrRPZWJft32GrTnwZFCET27VAuT2SIW0339fScU0S97KRednehERNQcp4voCoUCGzduxJ///GccO3YMgYGB6N+/v3j4prMWLVqERYsWNXrZ7t277X6/7bbbcObMmTbdJmCNg1m9ejVWr17t1Lr6OqETXc3BokREREQdiqu30Vty99134+67727ycolEguXLl2P58uVNLhMREYENGza4Y/V8FovoDXm6E/26rRO9PUfqxIcFIkAmgcFkQV5FjRjn4usd1uJw0ULvFNGFwaL8+yQiosa0utrao0cP9OjRw5XrQm0kZKIHKdmJTkRERNQRcRvdt7GI3lCUmInuqTiX9h+pI5NKkBQRhCtFGlwr0YrZ6L7eYd0t2ntF9GqdESUa69EQvv48ERGRdzg9WHTatGn429/+1uD8V199Fffdd59LVopah53oRERERB0Tt9F9n9ls6RBd0M6q60T3/GDR9izZ9viulWj9Ks4FAC4XVcNstnj0voUdDRFqBUJUvjl8lYiIvMvpIvoPP/yASZMmNTj/zjvvxA8//OCSlaLW0YiZ6CyiExEREXUk3Eb3fYVVOuiNZsikEsSHqVq+QgcRaSuie6ITvVpnRJnWAKB9DxYFgORIa3TL1eJqXC+17rzx9SJ6cmQQ5FIJtHoT8iprPXrfjHIhIqKWOF1Er66uhkKh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/J9QRK81mCGTSvDS5DRIJBIvrxUREXVELKK3E0IneiA70YmIiIiIiKgdEIroADBnVAp6xIZ4cW2IiKgjY9tyO2CxWFDDTHQiIiIiIiJqR/onhkEiASLVSjw9oYe3V4eIiDowVlzbAb3JDKPZAgAIUrITnYiIiIiIiPxf1+hgbFwwCvFhKoSqAry9OkRE1IGxiN4OCHnoABDEKeVERERERETUToxIjfD2KhARETETvT0Q8tCVcinkMr6kRERERERERERERK7Cims7oBXy0JU8sICIiIiIiIiIiIjIlVhEbwc0OmsnepCCUS5EREREREREREREruT1Ivrq1auRkpIClUqF9PR0HDp0qNnlV61ahV69eiEwMBBJSUl45plnUFtbK16ekpICiUTS4LRw4UJxmTFjxjS4/LHHHnPbY3Q3oROdRXQiIiIiIiIiIiIi1/Jq/sfGjRuxePFivPvuu0hPT8eqVauQkZGB8+fPIyYmpsHyGzZswPPPP4+1a9di9OjRuHDhAubOnQuJRIKVK1cCAH7++WeYTHWDNk+dOoUJEybgvvvus7utRx99FMuXLxd/DwoKctOjdL+6IjrjXIiIiIiIiIiIiIhcyatV15UrV+LRRx/FvHnzAADvvvsutm7dirVr1+L5559vsPxPP/2Em266CQ888AAAa9f5zJkzcfDgQXGZ6Ohou+v89a9/Rbdu3XDbbbfZnR8UFIS4uDhXPySv0NoGi6qV7EQnIiIiIiIiIiIiciWvxbno9XocPnwY48ePr1sZqRTjx4/H/v37G73O6NGjcfjwYTHy5cqVK/j6668xadKkJu/jww8/xMMPPwyJRGJ32UcffYSoqCj069cPS5YsgVarbXZ9dTodKisr7U6+QqNjJzoRERERERERERGRO3it6lpcXAyTyYTY2Fi782NjY3Hu3LlGr/PAAw+guLgYN998MywWC4xGIx577DG88MILjS7/+eefo7y8HHPnzm1wO8nJyUhISMCJEyfw+9//HufPn8dnn33W5PquWLECf/rTn5x7kB4idqIzE52IiIiIiIiIiIjIpfyqdXn37t145ZVX8PbbbyM9PR2XLl3CU089hZdffhkvvvhig+XXrFmDO++8EwkJCXbnL1iwQPy5f//+iI+Px7hx43D58mV069at0ftesmQJFi9eLP5eWVmJpKQkFz2ythE70ZV+9XIS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fP31125eMyIiIiKitmMmuh8T8tAVMikUcr6URERERERERERERK7Gyqsf09qK6OxCJyIiIiIiIiIiInIPFtH9mNYW56JmEZ2IiIiIiIiIiIjILVhE92ManbWIHqRktD0RERERERERERGRO7CI7seEOBd2ohMRERERERERERG5B4vofkxji3MJUrATnYiIiIiIiIiIiMgdWET3Y1qdrRNdyU50IiIiIiIiIiIiIndgEd2PsROdiIiIiIiIiIiIyL28XkRfvXo1UlJSoFKpkJ6ejkOHDjW7/KpVq9CrVy8EBgYiKSkJzzzzDGpra8XL//jHP0IikdidevfubXcbtbW1WLhwISIjIxEcHIxp06ahoKDALY/PnYRO9CBmohMRERERERERERG5hVeL6Bs3bsTixYvx0ksv4ciRIxg4cCAyMjJQWFjY6PIbNmzA888/j5deeglnz57FmjVrsHHjRrzwwgt2y/Xt2xd5eXniae/evXaXP/PMM/jyyy+xadMm7NmzB7m5ubj33nvd9jjdRWtgJzoRERERERERERGRO3m1+rpy5Uo8+uijmDdvHgDg3XffxdatW7F27Vo8//zzDZb/6aefcNNNN+GBBx4AAKSkpGDmzJk4ePCg3XJyuRxxcXGN3mdFRQXWrFmDDRs2YOzYsQCAdevWoU+fPjhw4ABGjhzpyofoVsxEJyIiIiIiIiIiInIvr3Wi6/V6HD58GOPHj69bGakU48ePx/79+xu9zujRo3H48GEx8uXKlSv4+uuvMWnSJLvlLl68iISEBHTt2hWzZs1CVlaWeNnhw4dhMBjs7rd3797o0qVLk/frq5iJTkREREREREREROReXqu+FhcXw2QyITY21u782NhYnDt3rtHrPPDAAyguLsbNN98Mi8UCo9GIxx57zC7OJT09HevXr0evXr2Ql5eHP/3pT7jllltw6tQphISEID8/HwqFAuHh4Q3uNz8/v8n11el00Ol04u+VlZWteNSupdWzE52IiIiIiIiIiIjInbw+WNQZu3fvxiuvvIK3334bR44cwWeffYatW7fi5ZdfFpe58847cd9992HAgAHIyMjA119/jfLycnzyySdtuu8VK1YgLCxMPCUlJbX14bSZRsdOdCIiIiIiIiIiIiJ38loRPSoqCjKZDFteNgUAAC/KSURBVAUFBXbnFxQUNJln/uKLL+Khhx7CI488gv79+2Pq1Kl45ZVXsGLFCpjN5kavEx4ejp49e+LSpUsAgLi4OOj1epSXlzt8vwCwZMkSVFRUiKfs7GwnHq17CJ3oQQp2ohMRERERERERERG5g9eK6AqFAkOHDsXOnTvF88xmM3bu3IlRo0Y1eh2tVgup1H6VZTJrAdlisTR6nerqaly+fBnx8fEAgKFDhyIgIMDufs+fP4+srKwm7xcAlEolQkND7U7eVteJziI6ERERERERERERkTt4NQdk8eLFmDNnDoYNG4YRI0Zg1apV0Gg0mDdvHgBg9uzZSExMxIoVKwAAkydPxsqVKzF48GCkp6fj0qVLePHFFzF58mSxmP7ss89i8uTJSE5ORm5uLl566SXIZDLMnDkTABAWFob58+dj8eLFiIiIQGhoKJ544gmMGjUKI0eO9M4T0Uo1BmsRXa1knAsRERERERERERGRO3i1+nr//fejqKgIy5YtQ35+PgYNGoRt27aJw0azsrLsOs+XLl0KiUSCpUuXIicnB9HR0Zg8eTL+8pe/iMtcv34dM2fORElJCaKjo3HzzTfjwIEDiI6OFpd54403IJVKMW3aNOh0OmRkZODtt9/23AN3EY2OcS5ERERERERERERE7iSxNJWDQs2qrKxEWFgYKioqvBbt0u+lb1GtM2L3s2OQEqX2yjoQERERUev5wjalv+JzR0RERERt5eg2pdcy0altLBYLNMJgUSU70YmIiIiIiIiIiIjcgUV0P1VrMEM4hkCtYCY6ERERERERERERkTuwiO6nhC50AAgMYCc6ERERERERERERkTuwiO6ntDoTAGsBXSqVeHltiIiIiIiIiIiIiNonFtH9lNZg7URXMw+diIiIiIiIiIiIyG1YRPdTGlsnehDz0ImIiIiIiIiIiIjchkV0P6W1ZaIHKdiJTkREREREREREROQuLKL7KaETXa1kJzoRERERERERERGRu7CI7qfYiU5ERERERERERETkfiyi+ymNXshEZxGdiIiIiIiIiIiIyF1YRPdTNbZOdDUHixIRERERERERERG5DYvofkrIRA9SshOdiIiIiIiIiIiIyF1YRPdTWnaiExEREREREREREbkdi+h+qi4TnUV0IiIiIiIiIiIiIndhEd1PaXW2TnTGuRARERERERERERG5DYvofoqd6ERERERERERERETuxyK6nxIy0YMU7EQnIiIiIiIiIiIichcW0f2UVuxEZxGdiIiIiIiIiIiIyF1YRPdTWp21iK5WMs6FiIiIiIiIiIiIyF1YRPdTGsa5EBEREREREREREbkdi+h+SohzYSc6ERERERERERERkfuwiO6nNDp2ohMRERERERERERG5G4vofshoMkNnNAMAghTsRCciIiIiIiIiIiJyFxbR/ZDWYBJ/Zic6ERERERERERERkfuwiO6Hamx56DKpBEo5X0IiIiIiIiIiIiIid2EF1g/Vz0OXSCReXhsiIiIiIiIiIiKi9otFdD+ktXWiq5mHTkRERERERERERORWLKL7IbETXck8dCIiIiIiIiIiIiJ3YhHdD7ETnYiIiIiIiIiIiMgzWET3Qxq9tRM9UMFOdCIiIiIiIiIiIiJ3YhHdD2l1Qic6i+hERERERERERERE7sQiuh/S6oVMdMa5EBEREREREREREbkTi+h+SKNnJzoRERERERERERGRJ7CI7ofETnQOFiUiIiIiIiIiIiJyKxbR/ZBGyERXshOdiIiIiIiIiIiIyJ3YyuyH2IlORETkGSaTCQaDwdurQX4sICAAMhkbH4iIiIiI/BmrsH5IyEQPYiY6ERGRW1gsFuTn56O8vNzbq0LtQHh4OOLi4iCRSLy9KkRERERE1Aosovshrc7aia5mJzoREZFbCAX0mJgYBAUFsfhJrWKxWKDValFYWAgAiI+P9/IaERERERFRa3i9Crt69Wr8/e9/R35+PgYOHIi33noLI0aMaHL5VatW4Z133kFWVhaioqLw61//GitWrIBKpQIArFixAp999hnOnTuHwMBAjB49Gn/729/Qq1cv8TbGjBmDPXv22N3ub37zG7z77rvueZAuphU60ZmJTkRE5HImk0ksoEdGRnp7dcjPBQYGAgAKCwsRExPDaBciIiIiIj/k1cGiGzduxOLFi/HSSy/hyJEjGDhwIDIyMsRunRtt2LABzz//PF566SWcPXsWa9aswcaNG/HCCy+Iy+zZswcLFy7EgQMHsH37dhgMBkycOBEajcbuth599FHk5eWJp1dffdWtj9WVhCI6O9GJiIhcT8hADwoK8vKaUHshvJeYr09ERERE5J+8WoVduXIlHn30UcybNw8A8O6772Lr1q1Yu3Ytnn/++QbL//TTT7jpppvwwAMPAABSUlIwc+ZMHDx4UFxm27ZtdtdZv349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\n"
          },
          "metadata": {}
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "# Prediction function for custom images"
      ],
      "metadata": {
        "id": "oiq_oonMZgSd"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "def predict_tumor(image_path):\n",
        "    try:\n",
        "        img = Image.open(image_path).convert('RGB')\n",
        "    except:\n",
        "        print(f\"Error loading image: {image_path}\")\n",
        "        return None\n",
        "\n",
        "    with torch.no_grad():\n",
        "        img_tensor = finetune_transform(img).unsqueeze(0).to(device)\n",
        "\n",
        "        with torch.amp.autocast(device_type='cuda', dtype=torch.float16):\n",
        "            type_logits, grade_logits = model(img_tensor)\n",
        "\n",
        "    type_probs = torch.softmax(type_logits, dim=1)[0]\n",
        "    grade_probs = torch.softmax(grade_logits, dim=1)[0]\n",
        "\n",
        "    type_idx = torch.argmax(type_probs).item()\n",
        "    tumor_type = classes[type_idx]\n",
        "    type_conf = float(type_probs[type_idx])\n",
        "\n",
        "    result = {\n",
        "        'tumor_detected': tumor_type != 'no_tumor',\n",
        "        'type': tumor_type,\n",
        "        'type_confidence': type_conf,\n",
        "        'grade': None,\n",
        "        'grade_confidence': None,\n",
        "        'characteristics': []\n",
        "    }\n",
        "\n",
        "    if result['tumor_detected']:\n",
        "        grade_idx = torch.argmax(grade_probs).item()\n",
        "        result.update({\n",
        "            'grade': grade_descriptions[grade_idx],\n",
        "            'grade_confidence': float(grade_probs[grade_idx]),\n",
        "            'characteristics': [\n",
        "                \"Diffuse growth pattern\" if tumor_type == 'glioma' else \"Well-circumscribed\",\n",
        "                \"High recurrence risk\" if grade_idx >= 2 else \"Low recurrence risk\",\n",
        "                \"Aggressive\" if grade_idx >= 2 else \"Less aggressive\"\n",
        "            ]\n",
        "        })\n",
        "\n",
        "    plt.figure(figsize=(10, 5))\n",
        "    plt.subplot(1, 2, 1)\n",
        "    plt.imshow(img)\n",
        "    plt.title(f\"Input Image\\n{os.path.basename(image_path)}\")\n",
        "    plt.axis('off')\n",
        "\n",
        "    plt.subplot(1, 2, 2)\n",
        "    info_text = f\"Type: {tumor_type} ({type_conf:.1%})\"\n",
        "    if result['tumor_detected']:\n",
        "        info_text += f\"\\nGrade: {result['grade']}\\nCharacteristics:\\n- \" + \"\\n- \".join(result['characteristics'])\n",
        "    plt.text(0.1, 0.5, info_text, fontsize=12, va='center')\n",
        "    plt.axis('off')\n",
        "    plt.show()\n",
        "\n",
        "    return result"
      ],
      "metadata": {
        "id": "xElp_ds1Xto1"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "# Example prediction"
      ],
      "metadata": {
        "id": "_K8EYqr4Zm_-"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "sample_image =\"/content/drive/MyDrive/Augmented/pituitary/Pituitary_0_3060.jpg\"  # Replace with your image\n",
        "prediction = predict_tumor(sample_image)\n",
        "\n",
        "if prediction:\n",
        "    print(\"\\nPrediction Result:\")\n",
        "    for k, v in prediction.items():\n",
        "        print(f\"{k:>20}: {v}\")"
      ],
      "metadata": {
        "id": "gN0ikVigXtmo",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 587
        },
        "outputId": "2ff78f04-c312-40d9-8391-d83e5a9cb224"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 1000x500 with 2 Axes>"
            ],
            "image/png": "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fv7znzfElNx222344x//iNe97nXu2m677YYHH3wQw8PD7trdd9+Nm2++uaEskoWJiYmnt9HzgGqPqjt//OMfceuttzbcd/TRR6NareLSSy9112q1Gr797W833Ldo0SIcccQRuOSSS7B58+Y59emYBAQEBAQEBAQEzI+g5AQ8I9h9993x8pe/HO95z3tQLpfx9a9/HX19ffiXf/kXd8873/lOXHDBBTj66KNxxhlnYMuWLbj44ouxzz77YGpqyt2XSqWwatUqXH311dhzzz3R29uLfffdF/vuu+8z1p9jjz0WP/3pT/HGN74RxxxzDNasWYOLL74Yq1atQi6Xc/cdf/zxOPTQQ/HP//zPePTRR/GiF70I1113HcbGxgA0qlLf/va38fKXvxwvfvGLceaZZ2LXXXfF0NAQbr31VmzYsAF33333M9a/gICAgICAgIDnM4KSE/CM4NRTT8XZZ5+Nb33rW/jc5z6HffbZB7///e+xZMkSd8/ee++NH/7wh5icnMSHP/xhXHfddbjiiitw4IEHzinvu9/9LpYtW4YPfehDeMtb3oJrr732mewOTj/9dHz+85/H3Xffjfe///24/vrrceWVV+Lggw9uuK+trQ2//vWv8eY3vxk/+MEP8MlPfhJLly51Sk4ymXT3rlq1CnfccQeOOeYYXH755TjrrLNw8cUXIxqN4tOf/vQz2r+AgICAgICAgOczInVfpHdAwA7C2rVrsXLlSnzlK1/BRz7ykWe7Oc8Z/PznP8cb3/hG/O///i/+9m//9tluTkBAQEBAQEDACwpByQkIeJpRLBYbPs/OzuLf//3f0dnZ6VWpAgICAgICAgICnhpCTE5AwNOMs88+G8ViEYcddhjK5TJ++tOf4pZbbsHnP/95pFKpZ7t5AQEBAQEBAQEvOASSExDwNOPII4/E+eefj1/96lcolUrYfffd8e///u943/ve92w3LSAgICAgICDgBYkQkxMQEBAQEBAQEBAQ8IJCiMkJCAgICAgICAgICHhBIZCcgICAgICAgICAgIAXFALJeZ7j9NNPxy677LLDy41EIjj33HN3eLkBzwwuv/xyRCIRrF279tluSkBAQEBAQEDAM45Acp7DoKHKn2QyiT333BPve9/7MDQ05H2mUCjg3HPPxY033rhD23LLLbfg3HPPxcTExA4t9+nAxo0bcdJJJ6G7uxudnZ047rjjsHr16gWXc+mll+Lwww/H4OAgEokEVq5ciXe84x1NicP3vvc97L333kgmk9hjjz3w7//+703Lvvrqq3HYYYchk8mgu7sbL3vZy/D73//+KZUZEBAQEBAQEBCwFSG72vMAn/nMZ7By5UqUSiX87//+Ly666CL85je/wX333YdLL70UtVrN3VsoFHDeeecBAI444ognXWexWER7+7blccstt+C8887D6aefju7u7idd7tONXC6HV77ylZicnMQnPvEJxGIxfO1rX8Phhx+OP//5z+jr62u5rLvuugsrV67EG97wBvT09GDNmjW49NJL8atf/Qp33303li5d6u695JJL8E//9E/4h3/4B3z4wx/GH/7wB7z//e9HoVDAxz72sYZyzz33XHzmM5/BiSeeiNNPPx3VahX33XcfNm7c2HDfQsq0OOWUU3DyyScjkUi03N+AgICAgICAgBcM6gHPWVx22WV1APXbb7+94fqHP/zhOoD6j3/84znPDA8P1wHUzznnnB3alq985St1APU1a9bssDJrtVq9UCjssPLq9Xr9S1/6Uh1A/bbbbnPXHnjggXpbW1v94x//+FMu/4477qgDqH/hC19w1wqFQr2vr69+zDHHNNz7tre9rZ7JZOpjY2Pu2q233lqPRCL1Cy64YN56FlJmQEBAQEBAQEBAI4K72vMQRx55JABgzZo1DTE5a9euxcDAAADgvPPOc25ujK054ogjvOqOL65Hnzv33HPx0Y9+FACwcuVKVy7dti677DIceeSRWLRoERKJBFatWoWLLrpoTj277LILjj32WFx//fU4+OCDkUqlcMkll+Dwww/HS17yEm9f99prLxx99NEtj821116LQw45BIcccoi79qIXvQivetWr8B//8R8tl9MMHCd127vhhhswOjqK9773vQ33nnXWWcjn8/j1r3/trn3961/H4sWL8YEPfAD1eh25XM5bz0LK9MEXk8Px/7//9/9i//33RzKZxKpVq/DTn/50zvP33HMPDj/8cKRSKey00074t3/7N1x22WUhzicgICAgICDgeYFAcp6HeOyxxwBgjuvVwMCAIxdvfOMbccUVV+CKK67ACSec8JTqO+GEE/CWt7wFAPC1r33NlUtCddFFF2HFihX4xCc+gfPPPx8777wz3vve9+Lb3/72nLIeeughvOUtb8GrX/1qfOMb38D++++PU045Bffccw/uu+++hntvv/12PPzww3j729/eUjtrtRruueceHHzwwXO+O/TQQ/HYY49henp6od3H6OgotmzZgjvuuAPveMc7AACvetWr3Pd33XUXAMyp96CDDkI0GnXfA8B///d/45BDDsE3v/lNDAwMoKOjA0uWLMG3vvWthmcXUuZC8Mgjj+DNb34zXve61+ELX/gC2tvb8aY3vQm/+93v3D0bN27EK1/5SvzlL3/Bxz/+cXzoQx/Cj370I3zjG994UnUGBAQEBAQEBDzTCDE5zwNMTk5iZGQEpVIJN998Mz7zmc8glUrh2GOPxa233uruy2QyOPHEE/Ge97wH++23X8vkYHvYb7/9cOCBB+Kqq67C8ccfP0f1uemmm5BKpdzn973vfXjta1+LCy64AGeddVbDvY8++ih++9vfNqgzBxxwAM4++2xceeWV+OIXv+iuX3nllchkMi2TtLGxMZTLZSxZsmTOd7y2adMm7LXXXi2VRyxbtgzlchnAVmL5zW9+E69+9avd95s3b0ZbWxsWLVrU8Fw8HkdfXx82bdoEABgfH8fIyAhuvvlm/P73v8c555yD5cuX47LLLsPZZ5+NWCyGd7/73Qsqc6F4+OGH8ZOf/MSN6RlnnIEXvehF+NjHPub69KUvfQnj4+O48847sf/++wMA3vGOd2CPPfZ4UnUGBAQEBAQEBDzTCErO8wBHHXUUBgYGsPPOO+Pkk09GNpvFz372MyxbtuzZbhoANBAcErLDDz8cq1evxuTkZMO9K1eunON+1tXVheOOOw5XXXUV6vU6AGB2dhZXX301jj/+eGQymZbaUSwWAcAbbJ9MJhvuWQj+8z//E7/5zW9w/vnnY/ny5cjn83Pqjcfj3meTyaSrk65po6Oj+O53v4uPfOQjOOmkk/DrX/8aq1atwr/9278tuMyFYunSpXjjG9/oPnd2duLUU0/FXXfdhSeeeAIA8Nvf/haHHXaYIzgA0Nvbi7e97W1Pqs6AgICAgICAgGcaQcl5HuDb3/429txzT7S3t2NwcBB77bUXotHnDj+9+eabcc455+DWW29FoVBo+G5ychJdXV3u88qVK71lnHrqqbj66qvxhz/8Aa94xSvwX//1XxgaGsIpp5zScjtItqi6KEqlUsM9C8ErX/lKAMDrXvc6HHfccdh3332RzWbxvve9z5VZqVS8z5ZKJVcnf8diMZx44onunmg0ije/+c0455xz8Pjjj2P58uUtl7lQ7L777ohEIg3X9txzTwBbY7oWL16MdevW4bDDDvM+GxAQEBAQEBDwfMBzx1IOaIpDDz0URx11FI444gjsvffeT5rgWOOWmJ2dfdJte+yxx/CqV70KIyMjuOCCC/DrX/8av/vd7/ChD30IABrSWwPNScbRRx+NwcFBXHnllQC2uqotXrwYRx11VMtt6e3tRSKRwObNm+d8x2ua9vnJYLfddsMBBxyAH/3oR+7akiVLMDs7iy1btjTcW6lUMDo66urs7e1FMplEX18f2traGu6lW9r4+PiCygwICAgICAgICJiLQHJeYGhGZACgp6fHe5jnunXrnnS5v/zlL1Eul3Hdddfh3e9+N/7+7/8eRx111IKVhra2Nrz1rW/Ftddei/Hxcfz85z/HW97yljlkYD5Eo1G8+MUvxh133DHnuz/+8Y/Ydddd0dHRsaB2+VAsFhvc8OjWZeu94447UKvV3PfRaBT7778/hoeH56g0jLFhModWy1woHn30UecSSDz88MMAtmWOW7FiBR599FHvswEBAQEBAQEBzwcEkvMCQzqdBgAvmdltt93w4IMPYnh42F27++67cfPNN2+3XMbF2HJJQtRwnpycxGWXXbbQpuOUU07B+Pg43v3udyOXyz2pxAknnngibr/99gZy8NBDD+H3v/893vSmN7VczszMjFNVFLfddhvuvffehqxnRx55JHp7e+ekzb7ooouQTqdxzDHHuGtvfvObMTs7ix/84AfuWqlUwo9+9COsWrXKKTQLKXNkZAQPPvjgHFdBHzZt2oSf/exn7vPU1BR++MMfYv/998fixYsBbFXVbr31Vvz5z392942NjTWoVwEBAQEBAQEBz2WEmJwXGFKpFFatWoWrr74ae+65J3p7e7Hvvvti3333xTvf+U5ccMEFOProo3HGGWdgy5YtuPjii7HPPvtgampq3nIPOuggAMAnP/lJnHzyyYjFYnj961+P17zmNYjH43j961/vyMmll16KRYsWed3G5sMBBxyAfffdF9dccw323ntvHHjggQvu/3vf+15ceumlOOaYY/CRj3wEsVgMF1xwAQYHB/HP//zPLZeTy+Ww8847481vfjP22WcfZDIZ3HvvvbjsssvQ1dWFT33qU+7eVCqFz372szjrrLPw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zeZW77XX+Ky2KSDg+YLTTjsN/f397t1VvOY1r8Fee+3lPkciEbzvfe/Dj370I+y1115IJpM46KCDvEblxo0b8c53vhODg4NIJBLYZ5998P3vf3/OfY8//jgefPDB7bbzxhtvRCQSwdVXX41PfOITWLx4MTKZDN7whjdg/fr1DfdqTM7atWsxMDAAADjvvPPcO8rYmiOOOMKr7vjievS5c889Fx/96EcBACtXrnTlrl27FgBw2WWX4cgjj8SiRYuQSCSwatUqXHTRRXPq2WWXXXDsscfi+uuvx8EHH4xUKoVLLrkEhx9+OF7ykpd4x2KvvfbC0UcfPe94/fznP0c8HscrXvGKhuuJRAKLFy+e91niJz/5CY499lhHcADgqKOOwp577on/+I//cNd+8YtfoFqt4r3vfa+7FolE8J73vAcbNmzArbfeOm891157LQYHB3HCCSe4awMDAzjppJPwi1/8wsXQFItF9PT0uHt6e3tRKBTc53/+53/GySefjIMPPrhpXQcddBB6e3vxi1/8ooURCAjY8WhZySmVSg273IlEApFIxKkwdEGi0aEHRpZKJWcA1Wo1t0tNA4qKAd16aPSoscvdIsZsWEPHqgBsqwY+0+BjoDiAOQTH5xaju9D6m+XTAFN1RMmZJWbsE41qS1B8u+hsJw1IX2Yyjo1mIwPQQF6UWFlXLDvmhHVhsi6FNobKEg/eY5MBqBHNMm3SCDVsNeOd3qckjWtHVRA7xiyLSgaNe41VImHzjTOhSg5dy5jCWRUWKjjlctkR05mZmYb3gsoL4914do51/SQ54Tioymbd6UjM6K6mfVEVkYRMkwr4lBU7H6qkNVN57Htm79X3wa5//d6qkPwdXNQCnk845ZRT8MMf/hDXX389jj32WHf9iSeewO9//3ucc845DfffdNNNuPrqq/H+978fiUQCF154IV772tfitttuw7777gsAGBoawt/8zd84UjQwMID//M//xBlnnIGpqSl88IMfdOWdeuqpuOmmm1p+bz73uc8hEongYx/7GLZs2YKvf/3rOOqoo/DnP//Zq34MDAzgoosuwnve8x688Y1vdIb0fvvtt9ChasAJJ5yAhx9+GFdddRW+9rWvob+/39UHABdddBH22WcfvOENb0B7ezt++ctf4r3vfS9qtRrOOuushrIeeughvOUtb8G73/1unHnmmdhr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fPTRR+q4AQMG4KSTTsLf//531NTUKMthshBt+Xvfq1cv3HDDDbjiiiuwYcMGnHDCCfB6vVi/fj1eeOEFTJo0CZdeeinefPNNTJs2DWPHjsWee+6JxsZGLFiwQAFMALj++uuxfPlyHHPMMejWrRu2bduGuXPnonPnzoY9ecyirKwMw4YNw5133olIJIJTTz3V8H5OTg4efPBBjB49Gn379sWECRPQqVMnVFZWYtmyZfD5fHj55ZfV8TabDUOGDMFbrexbxNhzzz1x3nnn4f3330d5eTkefvhhbN261cB+XnbZZfjnP/+JMWPGYPz48RgwYABisRg++eQTPPvss9iwYUO7pGBmsSvGM1ssXboURxxxhHIebEu0GeRI6RITRIae/HAFXNroSvAhExYJZHgtmWTJ65NdkK9JoMTrma08c6U+W2JmVn+gJ83A9pUM2kfzXCb0lA9J9kC2gcmvZH14PFkE6cLlcDgQCARUP/r9fhQXF8PtditTgm3btqG2thahUAjV1dXKWczlcqGpqUnR/WR63G430um0cmYDYHBq0RkMie6ZQNO2WFoUM3k2A44yMSZQlvNCHisZC9ln8jVd1iZd2fS5JueBXivG1+Q9OCZsjw5Yss0b2XaOoQ76CGoksGB9FUEOXfAkKOfcJSCTG53ycybZTdbYSC97mkiw/of1XJRTmj2fXLgw++zJz4YODPWQr5nJ4vSf9e9WWPFLieOPPx6ffPIJ7rjjDixZsgQPP/wwbDYbunXrhmOOOQZTpkxB//79d3id/Px8vPzyy6puwOFw4A9/+AOmTZvW4nzaP1999dW44oor0KtXL8ybNw8vvfRSiyTpvvvuw4ABA3D//ffjyiuvRF5eHrp3744zzzwTRxxxxE49c05ODhYsWICTTjoJ//d//4c5c+agrq4OHo8H/fr1w4033ojzzz+/TTUHw4cPx0MPPYRbbrkFF110EXr06IHZs2djw4YNLayhH374YZSWluLxxx/Hiy++iOHDh2PRokUGK1wAKC8vx3vvvYfrr78ezz//PObOnYvi4mL07du3hVXtj4kBAwZgv/32U/uj6HHvvfeanjd+/Hh07ty5TW0cP348tmzZgvvvvx+vvfYa9tlnHzz22GNYuHChYazPPPNMPPDAA5g7dy6CwSAqKipw6qmnYubMmervdkVFBe677z7cfPPNOO+889DU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jK5du2L16tW44oorsHnzZvz9738HACxduhSnnXYaRowYgdmzZwMAvvjiC6xatQoXXnghAGDmzJm4+eabMXHiRBx88MEIh8P44IMPsGbNGhx11FFZ2/Dtt9/ixRdfxNixY9GjRw9s3boV999/P4YMGYLPP/8cHTt2BNDMxg8fPhybN2/GhRdeiIqKCjzxxBNYtmxZi2vee++9mDZtGgYNGoSLL74YGzZswAknnIBAIIDOnTu3OP6vf/0rCgoKcOmll6rNt998802MHj0aAwYMwHXXXYecnBzMmzcPw4cPx4oVK3DwwQcDAKZMmYJnn30W06ZNwz777IOamhqsXLkSX3zxBQ488EDU19dj5MiRSKVSuOCCC1BRUYHKykq88sorCAaD8Pv9Ldpz0EEHoWfPnnjmmWdwzjnnGN57+umnEQgEMHLkSADA1q1bceihh8Jms2HatGkoLS3F4sWLcd555yEcDuOiiy7awSywwgordibaDHK4ez2lNqzlIBvDAnMm+QQQjY2NKCsrU0wAALWKz2TU7XbD7XYbpG+UlMm6AgAq8WSSKvcAIejRk1sm8k1NTQgGg6iqqkI8HlcJNh27mIwDzRIiWa8TDAZVUTlX5Jn4MunjqjtZqEQioaRcNptNObsREJpJnXT2RiajfF0GX5d9weN4D13uJgvc+XooFFLMnN/vV8m43F+IzIdMnqXZAFkOOT8IZildbGxsVPbGNKdwuVzKUrmwsFCZNsRiMVVvQ1bQ4/Eodk7ft4kgJlvo+8lIcEHZIec4gSn7KBaLqWclg8haMH0e8rput1s9l9vtVn3C5wOgZJIE1pyLcrNaadIga3O4bxAd5giiySDpEjWOtWwr+yUb4yMlcxLgSImj/vxmEkozCSbvrb8uPw9ypVq/thVW/BbizjvvxLp16/Dhhx9ijz32AABMnjwZHTt2xG233YY///nP6NKlCxYtWgSfz4fXXnst69/BRYsW4fe//z0eeOCBdrVh3333xddff234H3PWWWdhr732wkMPPYRrrrkGAHD//fcrQHT88certh5wwAGG69XX1+Oaa67B7373O7z55psqb9hvv/0wfvx4U5CTTCbxwQcfKKl7JpPBlClTMGzYMCxevFj9bZg8eTL69u2Lq6++GkuWLFHPff755+OOO+5Q17v88svVz59//jnWr1/fwgDi2muvbbVfTj31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\n"
          },
          "metadata": {}
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "Prediction Result:\n",
            "      tumor_detected: True\n",
            "                type: pituitary\n",
            "     type_confidence: 1.0\n",
            "               grade: Grade I (Least aggressive, benign)\n",
            "    grade_confidence: 1.0\n",
            "     characteristics: ['Well-circumscribed', 'Low recurrence risk', 'Less aggressive']\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [],
      "metadata": {
        "id": "Oz-J9L0xXtj6"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [],
      "metadata": {
        "id": "hEy84GJvXtg8"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [],
      "metadata": {
        "id": "j-ZtkqwvXteb"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [],
      "metadata": {
        "id": "8Y9kmbe8Xtbx"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [],
      "metadata": {
        "id": "8Kza70CzXtZD"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [],
      "metadata": {
        "id": "tFah5Jq4XtWO"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [],
      "metadata": {
        "id": "U3iv4XCVXtTs"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [],
      "metadata": {
        "id": "3RcAxRIzXtQ-"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [],
      "metadata": {
        "id": "6wVZzZbTXtOE"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [],
      "metadata": {
        "id": "w8L_csn5XtLK"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [],
      "metadata": {
        "id": "Xx7JXPRuXtIY"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [],
      "metadata": {
        "id": "ct9rU-xVXtGE"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [],
      "metadata": {
        "id": "z7IpsEnFXtDl"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [],
      "metadata": {
        "id": "kDu24hfnXtBf"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [],
      "metadata": {
        "id": "psn_AvUwXs_H"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [],
      "metadata": {
        "id": "3GpETrboXs8V"
      },
      "execution_count": null,
      "outputs": []
    }
  ]
}