# Разработка и вклад

## Подготовка окружения

Python 3.11 является reference version, хотя package metadata допускает 3.10+.
Для тестов без тяжелых моделей:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev,nlp,web]"
```

Не используйте production CUDA venv как development environment. CUDA 11,
CUDA 12 и CPU профили должны жить в отдельных каталогах.

## Проверки

```powershell
.\.venv\Scripts\python.exe -m ruff check src tests
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\sara-validate-registry.exe
.\.venv\Scripts\sara-validate-repository.exe
```

`sara-validate-repository` дополнительно проверяет комплект документов и links,
конфигурации, ссылки на модели, JSON schema/checksums и Docker content.

Тесты подменяют тяжелые модели и проверяют orchestration, математику, форматы,
защиту путей, manifests, web worker и delivery. Они не подтверждают работу
конкретного GPU/DLL stack. Для релиза дополнительно обязателен server smoke-test.

## Стиль изменений

- Сохраняйте публичные feature codes и семантику существующих колонок.
- Добавляйте типизированный объект или extractor вместо логики в web layer.
- Тяжелые optional libraries импортируйте внутри runtime-метода.
- Пишите deterministic outputs: стабильный порядок, UTF-8, явный разделитель.
- Не скрывайте missing features заменой на ноль.
- Не меняйте пользовательские артефакты задним числом без версии формата.
- Не помещайте corpus, model cache, output ZIP или venv в Git.

## Добавление признака

1. Определите источник и единицы измерения.
2. Реализуйте extractor с `FeatureValue` и evidence.
3. Подключите его в `FeaturePipeline`.
4. Для лингвистического признака обновите registry JSON.
5. Добавьте unit test и проверку wide output.
6. Проверьте конфликт имени с identity и существующими признаками.
7. Обновите `docs/ARCHITECTURE.md`, `docs/OUTPUTS.md` и словарь delivery.

## Обновление MIC

Следуйте checklist в `docs/METHODOLOGY.md`. Изменение JSON считается изменением
поведения даже без изменения Python-кода. В commit должны попасть канонические
models, provenance handoff, тест и release note.

## Изменение зависимостей

У проекта несколько сознательно разных dependency profiles. Изменение версии
нужно внести во все относящиеся к ней места:

- `pyproject.toml` для generic extras;
- соответствующий `requirements-cuda*.txt`;
- installer или Dockerfile;
- таблицу совместимости в документации;
- server smoke evidence.

После `pip install` сохраняйте `python -m pip freeze` в release evidence, но не
коммитьте автоматически сгенерированный freeze как универсальный lock для другой
ОС. Windows и Linux GPU wheels различаются.

## Commit и review

Рекомендуемый commit перед публикацией:

```powershell
git status --short
git diff --check
git diff
git add README.md docs src tests config models docker scripts pyproject.toml
git commit -m "docs: document architecture and reproducible deployment"
```

Перед `git add` проверьте, что нет аудио, результатов, cache, credentials и
локальных абсолютных путей. Не используйте `git add -A` вслепую на сервере.

Review должен отдельно ответить:

- не изменился ли порядок pipeline;
- совпадают ли model feature names и wide columns;
- сохраняется ли backward compatibility outputs;
- обрабатываются ли пустой input, все отфильтрованные строки и mono fallback;
- остается ли web-процесс жив после падения worker;
- воспроизводится ли установка в чистом окружении.

## Definition of done

- Ruff и pytest проходят.
- Все JSON models загружаются.
- Документация и `--help` согласованы.
- Docker image содержит config и models.
- На целевом сервере diagnostic загружает модель на нужном device.
- Реальный end-to-end trace завершается `SUCCESS`.
- Результат содержит manifests, wide table и ожидаемые prediction tables.
- Изменение зафиксировано отдельным commit с понятным release note.
