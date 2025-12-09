## 📥 Installation for Developers run in your VPS
#### 🐍 Python (using UV)

```bash
# download the code
sudo apt update
sudo apt install -y git

# generate ssh key and add to your github account
ssh-keygen -t ed25519 -C "your_email@gmail.com"
cat ~/.ssh/id_ed25519.pub

# clone the repo
git clone git@github.com:ngochoaphan2004/p2pfl.git
cd p2pfl

# Install UV if you don't have it https://docs.astral.sh/uv/getting-started/installation/#__tabbed_1_1
apt install -y curl
curl -LsSf https://astral.sh/uv/install.sh | sh

# add path to environment variable
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc

# First, ensure the virtual environment is created
uv sync --all-extras

# Then activate it traditionally
# On Unix/macOS:
source .venv/bin/activate

# On Windows:
.venv\Scripts\activate

```

### Example dataset mnist
```bash
python -m p2pfl run mnist

# or if error UnicodeEncodeError: 'charmap' codec can't encode character...
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8; $Env:PYTHONUTF8 = '1'; python -m p2pfl run mnist
```

> **Note:** The above command installs all dependencies (PyTorch, TensorFlow, and Ray). If you only need specific frameworks, you can use:
> - `uv sync` - Install only core dependencies
> - `uv sync --extra torch` - Install with PyTorch support
> - `uv sync --extra tensorflow` - Install with TensorFlow support
> - `uv sync --extra ray` - Install with Ray support
> 
> Use `--no-dev` to exclude development dependencies.

> **⚠️ Important for Ray users:** If you're using Ray, we recommend activating the virtual environment traditionally instead of using `uv run` to avoid dependency issues with Ray workers. See the [installation guide](https://p2pfl.github.io/p2pfl/installation.html#working-with-traditional-virtual-environment-activation) for details.

#### 🐳 Docker

```bash
docker build -t p2pfl .
docker run -it --rm p2pfl bash
```

## 🎬 Quickstart
To start using P2PFL, follow our [quickstart guide](https://p2pfl.github.io/p2pfl/quickstart.html) in the documentation.

## 📜 License
[GNU General Public License, Version 3.0](https://www.gnu.org/licenses/gpl-3.0.en.html)