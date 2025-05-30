import subprocess
import sys

def detect_cuda_version():
    try:
        output = subprocess.check_output(["nvcc", "--version"], text=True, stderr=subprocess.DEVNULL)
        for line in output.splitlines():
            if "release" in line:
                version_str = line.split("release")[-1].strip().split(",")[0]
                return version_str.replace(".", "")  # e.g., "12.8" → "128"
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass
    return None

def get_index_url(cuda_version):
    # Prioritize nightly builds for newer CUDA
    if cuda_version == "128":
        return "https://download.pytorch.org/whl/nightly/cu128"
    elif cuda_version == "121":
        return "https://download.pytorch.org/whl/cu121"
    elif cuda_version == "118":
        return "https://download.pytorch.org/whl/cu118"
    else:
        return "https://download.pytorch.org/whl/cpu"

def install_torch(index_url):
    print(f"Installing from: {index_url}")
    cmd = [
        "uv", "pip", "install", 
        "torch", "torchvision", "torchaudio",
        "--index-url", index_url
    ]
    if "nightly" in index_url:
        cmd.append("--pre")  # Required for nightly builds
    
    try:
        subprocess.run(cmd, check=True)
        print("PyTorch installed successfully")
    except subprocess.CalledProcessError as e:
        print(f"Installation failed (code {e.returncode}). Try manually:")
        print(" ".join(cmd))
        sys.exit(1)

if __name__ == "__main__":
    cuda = detect_cuda_version()
    print(f"Detected CUDA: {cuda or 'None (CPU fallback)'}")
    install_torch(get_index_url(cuda))