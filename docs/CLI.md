# Команды и скрипты

После editable/package installation команды находятся в `<venv>/Scripts` на
Windows и `<venv>/bin` на Linux. Эквивалентный надежный вызов для диагностики:

```text
<venv-python> -m sara_audio.<module>
```

## Основные команды

### `sara-extract` / `sara-process`

Обработать один файл или рекурсивно каталог:

```text
sara-extract INPUT [OUTPUT_DIRECTORY] [--config CONFIG]
```

Если output не указан, используется `execution.output_root`. Явно указанный
output должен еще не существовать. `sara-process` является полным alias.

Пример CPU:

```powershell
.\.venv\Scripts\sara-extract.exe .\samples .\artifacts\sample `
  --config .\config\native_cpu.yaml
```

Для Windows CUDA production используйте `web_worker` или Trace script, чтобы
обработка была изолирована от native teardown.

### `sara-web`

```text
sara-web [--config CONFIG] [--output-root PATH] [--temp-root PATH]
         [--host HOST] [--port PORT]
```

Запускает Gradio, сериализует GPU jobs и выносит каждый job в child process.
Аргументы переопределяют `SARA_*` environment variables.

### `sara-diagnose-asr`

```text
sara-diagnose-asr [--config CONFIG] [--load-model]
```

Без `--load-model` показывает доступность runtime. С флагом реально создает
WhisperModel и сообщает device. Это проверка ASR, но не полного pipeline.

### `sara-server-smoke`

```text
sara-server-smoke INPUT OUTPUT_DIRECTORY [--config CONFIG]
```

Проверяет CUDA, запускает pipeline и затем валидирует обязательные признаки,
speech-only WAV, batch size и prediction artifacts. Подходит для контейнера и
Linux. На Windows CUDA предпочтительнее `Trace-SaraPipeline.ps1`, который
использует отдельный worker.

### `sara-package-server-run`

```text
sara-package-server-run SOURCE [DESTINATION]
```

Создает delivery CSV/XLSX, manifest, ZIP и SHA-256 из уже завершенного result.
Аудио и модели повторно не запускаются.

## Проверки и утилиты

### `sara-validate-repository`

```text
sara-validate-repository [--root REPOSITORY]
```

Статически проверяет комплект документации, переносимость Markdown links,
конфигурации и их model paths, 11 JSON models с SHA-256 и Docker COPY contract.
Тяжелые модели не загружаются.

### `sara-validate-registry`

Проверяет встроенный реестр лингвистических признаков и выводит их количество.

### `sara-inspect-audio`

```text
sara-inspect-audio NORMALIZED_WAV
```

Выводит duration, sample rate, channels и sample width для WAV. Команда не
конвертирует файл.

### `sara-prepare-corpus`

```text
sara-prepare-corpus SOURCE
```

Готовит специальный исторический layout `train/ + test.zip`, безопасно проверяя
пути ZIP. Для обычного каталога uploads эта команда не нужна.

## Windows PowerShell scripts

| Скрипт | Назначение |
| --- | --- |
| `Install-SaraCuda12Native.ps1` | чистая pinned CUDA 12 native установка |
| `Trace-SaraPipeline.ps1` | один файл через worker с stage markers и log |
| `Install-SaraServerPatch.ps1` | наложить code/config/model patch без pip |
| `Invoke-SaraServerPipeline.ps1` | legacy установка, corpus prepare, smoke и batch |
| `New-SaraWheelhouse.ps1` | вспомогательная сборка Windows wheelhouse |
| `Invoke-SaraExcelDelivery.ps1` | собрать delivery/XLSX из завершенного result |

### Trace

```powershell
.\scripts\windows\Trace-SaraPipeline.ps1 `
  -InputPath D:\SARA-input\smoke.wav `
  -VenvPath .\.cuda12-venv-final `
  -ConfigPath .\config\rtx4090_cuda12.yaml `
  -OutputPath D:\SARA-results\trace-001 `
  -LogPath D:\SARA-logs\trace-001.log
```

Относительный `VenvPath` разрешается относительно `ProjectRoot`. Скрипт печатает
полные output/log paths и возвращает exit code worker.

### Native installer

```powershell
.\scripts\windows\Install-SaraCuda12Native.ps1 `
  [-ProjectRoot PATH] [-VenvPath PATH] [-SmokeInput FILE] [-LaunchWeb]
```

Целевой venv не должен существовать. Флаг `-SmokeInput` запускает обычный
`sara-extract`; для самой надежной Windows приемки после установки отдельно
используйте Trace script.

## Exit codes

| Код | Значение |
| --- | --- |
| `0` | команда завершилась успешно |
| `2` | ожидаемая Python/config/input ошибка CLI или worker exception |
| `3221225477` / `-1073741819` | Windows access violation `0xC0000005` |

Другой ненулевой код считается ошибкой. Для job дополнительно проверяйте
`run_summary.json`, поскольку native process может завершиться вне Python.
