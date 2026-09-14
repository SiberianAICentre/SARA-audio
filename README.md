# SARA-audio

Пакет извлекает из русской речи:

- frame-level акустические признаки через openSMILE;
- временные признаки и паузы с порогом `300 ms`;
- скорость речи по активной речи;
- MIC-совместимые скалярные признаки по расшифровке и речи без пауз;
- все 92 лингвистических параметра из исходного DOCX.

Файл [exclude_features.txt](C:/Users/ru-lo/PycharmProjects/SARA-audio/exclude_features.txt)
задает признаки, которые вычисляются для аудита, но удаляются из итоговых
wide/delivery-выгрузок.

Реестр параметров не редактируется вручную. Он генерируется из исходного
документа командой:

```powershell
.venv\Scripts\python.exe tools\generate_feature_plan.py
```

Проверка реестра должна всегда давать 92 признака:

```powershell
sara-validate-registry
```

Установка зависимостей:

```powershell
.venv\Scripts\python.exe -m pip install -e .
```

Для ASR:

```powershell
.venv\Scripts\python.exe -m pip install -e .[asr]
```

Для полного серверного пайплайна с ASR, parquet, NLP и text embeddings:

```powershell
.venv\Scripts\python.exe -m pip install -e .[asr,nlp,parquet,embeddings]
```

Также нужен установленный в `PATH` `ffmpeg`, чтобы конвертировать `.m4a` в
mono PCM WAV. Основной запуск принимает только входной путь и, при
необходимости, новый каталог результата:

```powershell
sara-extract data\raw
sara-extract data\raw results\experiment-01
```

Все рабочие параметры задаются только в
[config/default.yaml](C:/Users/ru-lo/PycharmProjects/SARA-audio/config/default.yaml).
Подробный запуск, структура результатов и настройка расшифровки описаны в
[USER_GUIDE.md](C:/Users/ru-lo/PycharmProjects/SARA-audio/USER_GUIDE.md).
Docker-сервис для Linux/NVIDIA сервера находится в
[docker/rtx4090/README.md](C:/Users/ru-lo/PycharmProjects/SARA-audio/docker/rtx4090/README.md).
