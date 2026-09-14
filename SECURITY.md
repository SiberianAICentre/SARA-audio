# Security and data handling

SARA-audio processes audio, transcripts, inferred sex, and burnout-related
scores. Treat every input and output as potentially sensitive personal data.

## Deployment boundary

The bundled Gradio server does not provide application authentication, role
based access control, TLS, malware scanning, quotas, or a retention policy. Do
not expose port 7860 directly to the public internet. Place it behind an
authenticated reverse proxy and firewall, and restrict filesystem access to the
service account.

## Data handling

- Store uploads, generated WAV files, transcripts, predictions, and logs on an
  approved encrypted volume.
- Define deletion periods for job directories, temporary files, and archives.
- Avoid sharing full job ZIP files when only compact prediction tables are
  required.
- Remember that manifests contain original paths and logs may contain filenames.
- Do not commit corpora, outputs, model caches, credentials, or service logs.

## Input and resource controls

The service accepts WAV, M4A, MP3, FLAC, and OGG by extension and passes media
to ffmpeg. Production ingress should enforce file size, request size, timeout,
concurrency, and available disk space. The in-process lock serializes GPU jobs
but does not limit the length of the Gradio queue.

## Reporting

Report suspected vulnerabilities privately to the repository owner. Include the
affected commit, deployment profile, reproduction steps, and whether sensitive
data may have been exposed. Do not attach real customer audio to a public issue.

## Model governance

Predicted sex and burnout can be sensitive and may be incorrect. Access,
retention, human review, and permitted use must be defined by the deploying
organization. The output is not a medical diagnosis and must not be the sole
basis for consequential employment decisions.
