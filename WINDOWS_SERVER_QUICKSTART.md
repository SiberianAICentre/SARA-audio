# Windows Server quick start

Это короткая инструкция для проверенного native-профиля RTX 4090/CUDA 12.
Полные требования, Docker, offline и rollback описаны в
[`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md), ошибки в
[`docs/TROUBLESHOOTING.md`](docs/TROUBLESHOOTING.md).

## 1. Проверить машину

```powershell
nvidia-smi
py -3.11 --version
```

Обе команды должны завершиться успешно. Работайте из корня постоянной копии
репозитория. Не копируйте готовый venv с другого компьютера.

## 2. Создать изолированное окружение

```powershell
cd D:\Services\SARA-audio
Set-ExecutionPolicy -Scope Process Bypass
.\scripts\windows\Install-SaraCuda12Native.ps1
```

Installer сам создает `.cuda12-venv-final`. Если каталог уже существует, он
останавливается, чтобы не смешать зависимости. Для новой попытки укажите новый
путь:

```powershell
.\scripts\windows\Install-SaraCuda12Native.ps1 `
  -VenvPath D:\Services\SARA-audio\.cuda12-venv-next
```

Успешный финал содержит `model_loaded: true` и `model_device: cuda`.

## 3. Проверить полный pipeline

```powershell
$wav = "D:\SARA-input\smoke.wav"
.\scripts\windows\Trace-SaraPipeline.ps1 `
  -InputPath $wav `
  -VenvPath .\.cuda12-venv-final `
  -ConfigPath .\config\rtx4090_cuda12.yaml
```

Требуется `Pipeline exit code: 0`, строка `SUCCESS` в trace log и файл
`run_summary.json` в выведенном output-каталоге. Не переходите к batch, если
проверка завершилась только после загрузки ASR model: это еще не полный pipeline.

## 4. Запустить интерфейс

```powershell
$env:HF_HOME = "D:\SARA-model-cache"
.\.cuda12-venv-final\Scripts\python.exe -m sara_audio.webapp `
  --config .\config\rtx4090_cuda12.yaml `
  --output-root D:\SARA-results `
  --temp-root D:\SARA-temp `
  --host 0.0.0.0 `
  --port 7860
```

Откройте `http://SERVER_IP:7860`, загрузите smoke-файл и проверьте:

- в интерфейсе появился итог или сообщение, что filter исключил записи;
- скачивается `sara-batch.zip`;
- в `D:\SARA-results\job-*\processing.log` есть `SUCCESS`;
- в `results` созданы feature и prediction tables.

Порт 7860 не следует открывать в интернет напрямую. Production-доступ должен
идти через firewall и reverse proxy с TLS и authentication.

## 5. Запустить каталог без интерфейса

На Windows CUDA используйте тот же изолированный worker, что и web:

```powershell
.\.cuda12-venv-final\Scripts\python.exe -u -X faulthandler `
  -m sara_audio.web_worker `
  --config .\config\rtx4090_cuda12.yaml `
  --input D:\SARA-input\batch-001 `
  --output D:\SARA-results\batch-001
```

Output-каталог не должен существовать. Успешный процесс возвращает код `0`.

## Legacy corpus runner

`Invoke-SaraServerPipeline.ps1` умеет подготовить исторический `train + test.zip`
corpus и имеет переключатель `-CudaConfig cuda11_8|cuda12`. Он не является
рекомендуемым установщиком текущего Windows CUDA 12 окружения, потому что его
generic dependency path отличается от проверенного pinned native installer.
