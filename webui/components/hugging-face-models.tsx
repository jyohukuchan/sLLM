'use client';

import {
  Check,
  Copy,
  Download,
  FileBox,
  LoaderCircle,
  Search,
  TriangleAlert,
} from 'lucide-react';
import { useEffect, useState } from 'react';

import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import {
  fetchHuggingFaceDownloadJob,
  fetchHuggingFaceFiles,
  fetchHuggingFaceStatus,
  searchHuggingFaceModels,
  startHuggingFaceDownload,
  type ApiConfig,
  type HuggingFaceDownloadJob,
  type HuggingFaceFiles,
  type HuggingFaceGgufFile,
  type HuggingFaceModel,
  type HuggingFaceStatus,
} from '@/lib/sllm-api';

interface HuggingFaceModelsProps {
  config: ApiConfig;
  live: boolean;
  selectedPath?: string | null;
  onDownloadComplete: () => void | Promise<void>;
}

const compactNumber = new Intl.NumberFormat('en', {
  notation: 'compact',
  maximumFractionDigits: 1,
});

function formatBytes(bytes: number): string {
  if (!Number.isFinite(bytes) || bytes <= 0) return 'unknown size';
  const units = ['B', 'KB', 'MB', 'GB', 'TB'];
  let value = bytes;
  let unit = 0;
  while (value >= 1000 && unit < units.length - 1) {
    value /= 1000;
    unit += 1;
  }
  return `${value >= 10 || unit === 0 ? value.toFixed(0) : value.toFixed(1)} ${units[unit]}`;
}

function terminal(job: HuggingFaceDownloadJob): boolean {
  return job.state === 'completed' || job.state === 'failed';
}

export function HuggingFaceModels({
  config,
  live,
  selectedPath,
  onDownloadComplete,
}: HuggingFaceModelsProps) {
  const [status, setStatus] = useState<HuggingFaceStatus | null>(null);
  const [query, setQuery] = useState('');
  const [models, setModels] = useState<HuggingFaceModel[]>([]);
  const [selectedModel, setSelectedModel] = useState<HuggingFaceModel | null>(
    null,
  );
  const [files, setFiles] = useState<HuggingFaceFiles | null>(null);
  const [job, setJob] = useState<HuggingFaceDownloadJob | null>(null);
  const [busy, setBusy] = useState<'search' | 'files' | ''>('');
  const [copied, setCopied] = useState('');
  const [error, setError] = useState('');

  useEffect(() => {
    let current = true;
    if (!live) {
      return () => {
        current = false;
      };
    }
    void fetchHuggingFaceStatus(config)
      .then((next) => {
        if (current) setStatus(next);
      })
      .catch((cause: unknown) => {
        if (current)
          setError(
            cause instanceof Error
              ? cause.message
              : 'Hugging Face status could not be read.',
          );
      });
    return () => {
      current = false;
    };
  }, [config, live]);

  async function runSearch() {
    if (!live || !query.trim()) return;
    setBusy('search');
    setError('');
    setSelectedModel(null);
    setFiles(null);
    try {
      const result = await searchHuggingFaceModels(config, query);
      setModels(result.models);
    } catch (cause) {
      setModels([]);
      setError(cause instanceof Error ? cause.message : 'Model search failed.');
    } finally {
      setBusy('');
    }
  }

  async function openModel(model: HuggingFaceModel) {
    setSelectedModel(model);
    setFiles(null);
    setBusy('files');
    setError('');
    try {
      setFiles(
        await fetchHuggingFaceFiles(config, model.repo_id, model.revision),
      );
    } catch (cause) {
      setError(
        cause instanceof Error
          ? cause.message
          : 'Repository files could not be read.',
      );
    } finally {
      setBusy('');
    }
  }

  async function copyCommand(file: HuggingFaceGgufFile) {
    setError('');
    try {
      await navigator.clipboard.writeText(file.download_command);
      setCopied(file.path);
      window.setTimeout(
        () => setCopied((current) => (current === file.path ? '' : current)),
        1600,
      );
    } catch {
      setError('The download command could not be copied.');
    }
  }

  async function download(file: HuggingFaceGgufFile) {
    if (!selectedModel || !selectedPath) return;
    setError('');
    try {
      let next = await startHuggingFaceDownload(
        config,
        selectedModel.repo_id,
        selectedModel.revision,
        file,
      );
      setJob(next);
      while (!terminal(next)) {
        await new Promise((resolve) => window.setTimeout(resolve, 1000));
        next = await fetchHuggingFaceDownloadJob(config, next.id);
        setJob(next);
      }
      setStatus(await fetchHuggingFaceStatus(config));
      if (next.state === 'completed') await onDownloadComplete();
    } catch (cause) {
      setError(
        cause instanceof Error ? cause.message : 'Model download failed.',
      );
    }
  }

  const downloading = job && !terminal(job);
  const currentStatus = live ? status : null;
  const cliUnavailable = currentStatus !== null && !currentStatus.cli_available;

  return (
    <section className="hf-panel" aria-labelledby="hf-heading">
      <div className="hf-heading">
        <div>
          <span className="eyebrow">HUGGING FACE</span>
          <strong id="hf-heading">Find GGUF models</strong>
        </div>
        {!live ? (
          <Badge variant="outline">offline</Badge>
        ) : currentStatus?.authenticated ? (
          <Badge variant="outline">
            {currentStatus.username || 'authenticated'}
          </Badge>
        ) : currentStatus?.auth_state === 'unauthenticated' ? (
          <Badge variant="outline">anonymous</Badge>
        ) : (
          <Badge variant="outline">checking</Badge>
        )}
      </div>

      {currentStatus?.auth_state === 'unauthenticated' && (
        <output className="hf-auth-warning">
          <TriangleAlert />
          <p>
            Hugging Face is unauthenticated. Anonymous requests have lower rate
            limits, and gated or private files may fail.
          </p>
        </output>
      )}
      {currentStatus?.auth_state === 'unknown' &&
        currentStatus.cli_available && (
          <p className="hf-status-note">
            Hugging Face authentication status could not be confirmed.
          </p>
        )}
      {cliUnavailable && (
        <div className="error-banner hf-error" role="alert">
          <span>The `hf` CLI is not available on the sLLM server.</span>
        </div>
      )}
      {!selectedPath && live && (
        <p className="hf-status-note">
          Select a model folder before preparing a download.
        </p>
      )}
      <p className="hf-status-note">
        Remote results are unverified. sLLM compatibility is checked only after
        download and model-folder scanning.
      </p>

      <form
        className="hf-search-form"
        onSubmit={(event) => {
          event.preventDefault();
          void runSearch();
        }}
      >
        <Input
          aria-label="Search Hugging Face models"
          value={query}
          maxLength={128}
          disabled={!live || cliUnavailable}
          onChange={(event) => setQuery(event.target.value)}
          placeholder="Model name, author, or architecture"
        />
        <Button
          type="submit"
          variant="outline"
          disabled={
            !live ||
            cliUnavailable ||
            !query.trim() ||
            busy === 'search' ||
            Boolean(downloading)
          }
        >
          {busy === 'search' ? (
            <LoaderCircle className="hf-spin" />
          ) : (
            <Search />
          )}
          Search
        </Button>
      </form>

      {error && (
        <div className="error-banner hf-error" role="alert">
          <span>{error}</span>
        </div>
      )}

      {!!models.length && (
        <div className="hf-results" aria-label="Hugging Face model results">
          {models.map((model) => (
            <button
              type="button"
              key={`${model.repo_id}:${model.revision}`}
              className={
                selectedModel?.repo_id === model.repo_id
                  ? 'hf-result-row hf-result-selected'
                  : 'hf-result-row'
              }
              disabled={busy === 'files' || Boolean(downloading)}
              onClick={() => void openModel(model)}
            >
              <span>
                <strong>{model.repo_id}</strong>
                <small>
                  {compactNumber.format(model.downloads)} downloads ·{' '}
                  {compactNumber.format(model.likes)} likes
                </small>
              </span>
              <span className="hf-result-badges">
                {model.gated && <Badge variant="outline">gated</Badge>}
                {model.private && <Badge variant="outline">private</Badge>}
                <Badge variant="outline">GGUF</Badge>
              </span>
            </button>
          ))}
        </div>
      )}

      {busy === 'files' && (
        <p className="hf-status-note hf-loading">
          <LoaderCircle className="hf-spin" /> Reading repository files…
        </p>
      )}

      {files && (
        <div className="hf-files">
          <div className="hf-revision">
            <span>{files.files.length} root-level GGUF files</span>
            <code title={files.revision}>{files.revision.slice(0, 12)}</code>
          </div>
          {files.files.map((file) => (
            <article className="hf-file-row" key={file.path}>
              <div className="hf-file-title">
                <FileBox />
                <span>
                  <strong>{file.path}</strong>
                  <small>
                    {formatBytes(file.size_bytes)}
                    {file.derived_lock_path ? ' · sLLM lock included' : ''}
                  </small>
                </span>
              </div>
              <code className="hf-command">{file.download_command}</code>
              <div className="hf-file-actions">
                <Button
                  type="button"
                  size="sm"
                  variant="outline"
                  disabled={Boolean(downloading)}
                  onClick={() => void copyCommand(file)}
                >
                  {copied === file.path ? <Check /> : <Copy />}
                  {copied === file.path ? 'Copied' : 'Copy command'}
                </Button>
                <Button
                  type="button"
                  size="sm"
                  className="connect-button"
                  disabled={!selectedPath || Boolean(downloading)}
                  onClick={() => void download(file)}
                >
                  {job?.file_path === file.path && downloading ? (
                    <LoaderCircle className="hf-spin" />
                  ) : (
                    <Download />
                  )}
                  Download
                </Button>
              </div>
              {job?.file_path === file.path && (
                <output
                  className={`hf-download-status hf-download-${job.state}`}
                >
                  <strong>{job.state}</strong>
                  {job.message && <span>{job.message}</span>}
                </output>
              )}
            </article>
          ))}
          {!files.files.length && (
            <p className="empty-note">
              No repository-root GGUF files were found. Nested files are not
              shown because the current sLLM model folder scan is non-recursive.
            </p>
          )}
        </div>
      )}
    </section>
  );
}
