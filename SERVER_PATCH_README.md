# SARA server patch

Этот patch подключает полный каскад JSON-моделей методологов к серверному
запуску: фильтр, восемь реконструированных экспертных признаков, пол и
итоговый прогноз выгорания.

## Установка

Распакуйте patch в отдельный каталог рядом с проектом или прямо в каталог
проекта и выполните PowerShell:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\scripts\windows\Install-SaraServerPatch.ps1 -ProjectRoot C:\SARA-audio -PatchRoot C:\sara-server-patch
```

Если patch уже распакован поверх проекта, достаточно запустить основной скрипт.

## Проверочный запуск

Для CUDA 11.8:

```powershell
.\scripts\windows\Invoke-SaraServerPipeline.ps1 -ProjectRoot C:\SARA-audio -CudaConfig cuda11_8
```

Для CUDA 12:

```powershell
.\scripts\windows\Invoke-SaraServerPipeline.ps1 -ProjectRoot C:\SARA-audio -CudaConfig cuda12
```

После успешного запуска в каталоге результата должны появиться:

- `features_wide.csv`;
- `predictions.csv` и `predictions.json`;
- `interpretation_input.csv` и `interpretation_input.json`;
- `filter_predictions.csv` и `filter_predictions.json`;
- `run_summary.json`.

`predictions.*` содержит компактный итог по информативным записям,
`interpretation_input.*` — полную таблицу для интерпретации, а
`filter_predictions.*` сохраняет решение фильтра для каждого входного файла.
Пол и выгорание рассчитываются по средним признакам группы «оператор-год».

Если в логе появляется путь вида `SARA-audio.server-venv` без обратной косой
черты, установите свежий patch поверх проекта. В нём пути виртуального
окружения собираются через `System.IO.Path.Combine`.

Скрипт использует связку с готовыми колёсами для Python 3.11/Windows:
`faster-whisper==1.2.1` и `ctranslate2==4.8.2`. Старая версия
`faster-whisper==0.10.1` требовала `av==10.*`, который на таком окружении
начинал собираться из исходников и падал на Cython.

Patch переносит весь каталог `src\sara_audio`, включая новые MIC-модули,
поэтому его нужно устанавливать поверх проекта целиком, а не копировать только
отдельные изменённые файлы.

Повторный запуск не должен принудительно удалять весь набор пакетов: зависимости
выравниваются обычным `pip install --upgrade`.
