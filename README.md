## project_forecast

Проект для построения прогноза по данным о количестве пассажиров с использованием библиотек prophet, statsmodels и pmdarima

## Установка и запуск проекта 

Перед началом убедитесь, что у вас настроена среда WSL2 с дистрибутивом Ubuntu 22.04 и установлен Visual Studio Code с расширением WSL, а также скачан Docker Desktop

Установите следующие расширения VS-Code:
code --install-extension ms-python.python
code --install-extension ms-python.vscode-pylance
code --install-extension ms-azuretools.vscode-docker
code --install-extension formulahendry.code-runner
code --install-extension mtxr.sqltools
code --install-extension ms-vscode-remote.remote-wsl

curl -LsSf https://astral.sh/uv/install.sh | sh             установка uv (если не установлен)
echo 'export PATH="$HOME/.cargo/bin:$PATH"' >> ~/.bashrc	добавление uv в PATH
source ~/.bashrc							                применение изменений
uv –version								                    проверка работоспособности

Откройте терминал Ubuntu WSL и выполните:

Клонируйте репозиторий
git clone https://github.com/ad2tebiz/project_forecast.git

Откройте проект в VS Code через WSL
1. ctrl+shift+p connect to WSL
2. ctrl+shift+p open folder in WSL и выбрать папку проекта

Установите зависимости из pyproject.toml
uv sync

Скопируйте и отредактируйте файл .env
cp .env.example .env

## Использование

Запуск скриптов
uv run src/forecast-prophet.py
uv run src/forecast-statsmodels.py

Для запуска скрипта с библиотекой pmdarima надо установить Miniconda (который будет контролировать зависимости пакетов). Контролирование необходимо из-за того, что pmdarima не работает c версиями numpy выше 2.0. Без miniconda при запуске скрипта обычным способом пакеты переустанавливаются и пакет numpy становится выше версии 2.0.

wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
source ~/.bashrc
conda create -n timeseries -c conda-forge python=3.9 pmdarima pandas matplotlib python-dotenv numpy=1.23 -y
conda activate timeseries
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
conda deactivate
conda list | grep -E "(numpy|pmdarima|pandas)"
python src/check_pmdarima.py
conda install numpy=1.23.5 --force-reinstall -y

CREATE TABLE IF NOT EXISTS forecasts (
            id SERIAL PRIMARY KEY,
            forecast_date DATE NOT NULL,
            predicted_value DECIMAL(15,4) NOT NULL,
            confidence_lower DECIMAL(15,4),
            confidence_upper DECIMAL(15,4),
            model_name VARCHAR(100) NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(forecast_date, model_name)
        )
