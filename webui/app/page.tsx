'use client';

import {
  Activity,
  ArrowLeft,
  ArrowUp,
  BarChart3,
  BrainCircuit,
  Cable,
  Check,
  ChevronRight,
  CircleStop,
  Cpu,
  Folder,
  FolderOpen,
  Gauge,
  HardDrive,
  Menu,
  MessageSquare,
  Moon,
  PanelRight,
  Play,
  RefreshCw,
  Server,
  Settings2,
  Sparkles,
  Sun,
  Timer,
  Unplug,
  Zap,
} from 'lucide-react';
import { useEffect, useMemo, useRef, useState } from 'react';

import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { HuggingFaceModels } from '@/components/hugging-face-models';
import {
  NativeSelect,
  NativeSelectOption,
} from '@/components/ui/native-select';
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetHeader,
  SheetTitle,
} from '@/components/ui/sheet';
import { Textarea } from '@/components/ui/textarea';
import {
  browseModelLibrary,
  fetchModelLibrary,
  fetchServerMetrics,
  fetchServerSnapshot,
  modelAction,
  rescanModelLibrary,
  selectModelLibraryFolder,
  streamChatCompletion,
  type ApiConfig,
  type ChatMessage,
  type ModelLibraryBrowse,
  type ModelLibraryModel,
  type ModelLibrarySnapshot,
  type ServerModel,
  type ServerProps,
} from '@/lib/sllm-api';
import { modelMetricSnapshot, subtractMetrics } from '@/lib/metrics';
import { fetchIntegratedRuntimeConfig } from '@/lib/runtime-config';

type ConnectionState = 'demo' | 'connecting' | 'live' | 'error';
type Theme = 'light' | 'dark';
type View = 'performance' | 'chat';
type BenchmarkState = 'idle' | 'running' | 'complete' | 'error';
type UiMessage = ChatMessage & { id: string };

interface HardwareInfo {
  vendor: string;
  name: string;
  target: string;
  memoryBytes?: number;
  source: 'fixture' | 'server';
}

interface BenchmarkResult {
  id: string;
  model: string;
  prefillTps: number;
  decodeTps: number;
  ttftMs: number;
  e2eMs: number;
  promptTokens: number;
  completionTokens: number;
  source: 'fixture' | 'server-metrics';
  createdAt: string;
}

const demoModels: ServerModel[] = [
  { id: 'qwen35-4b-bf16', lifecycle: 'loaded', residentBytes: 4_980_000_000 },
  { id: 'qwen35-4b-fp16', lifecycle: 'configured' },
];

const demoLibrary: ModelLibrarySnapshot = {
  schema_version: 'sllm-model-library-v1',
  selected_path: '/srv/models',
  models: [
    {
      alias: 'qwen35-4b-bf16',
      file_name: 'qwen35-4b-bf16.gguf',
      size_bytes: 4_980_000_000,
      architecture: 'qwen35',
      supported_architecture: true,
      compatible: true,
    },
    {
      alias: 'llama-example',
      file_name: 'llama-example.gguf',
      size_bytes: 3_120_000_000,
      architecture: 'llama',
      supported_architecture: false,
      compatible: false,
      reason: 'This GGUF architecture is not implemented by sLLM.',
    },
  ],
};

const demoProps: ServerProps = {
  schema_version: '1',
  state: 'ready',
  scheduler: { active_requests: 0, queued_requests: 0 },
  features: { reasoning: true, structured_output: true },
};

const demoHardware: HardwareInfo = {
  vendor: 'AMD',
  name: 'Radeon PRO R9700',
  target: 'gfx1201',
  memoryBytes: 32_000_000_000,
  source: 'fixture',
};

const demoBenchmark: BenchmarkResult = {
  id: 'fixture-run',
  model: demoModels[0].id,
  prefillTps: 3611.8,
  decodeTps: 128.4,
  ttftMs: 410,
  e2eMs: 1380,
  promptTokens: 1480,
  completionTokens: 124,
  source: 'fixture',
  createdAt: 'DEMO FIXTURE',
};

const defaultApiConfig: ApiConfig = {
  baseUrl: 'http://127.0.0.1:8080',
  userKey: '',
  adminKey: '',
};

const seedMessages: UiMessage[] = [
  {
    id: 'seed-user',
    role: 'user',
    content: 'Summarize the active inference configuration.',
  },
  {
    id: 'seed-assistant',
    role: 'assistant',
    reasoning:
      'Use only the prototype fixture. Keep the GPU and KV-cache details explicit.',
    content:
      'Qwen3.5-4B BF16 is shown on gfx1201. The standard OCP KV path uses MXFP8 E4M3; FP16 remains an explicit rollback option. This is fixture data, not live GPU evidence.',
  },
];

function formatBytes(value?: number): string {
  if (value === undefined) return 'not resident';
  return `${(value / 1_000_000_000).toFixed(1)} GB`;
}

function formatRate(value?: number): string {
  if (value === undefined) return '—';
  return value >= 1000
    ? value.toLocaleString(undefined, { maximumFractionDigits: 0 })
    : value.toFixed(1);
}

function stringField(
  value: Record<string, unknown>,
  ...keys: string[]
): string {
  for (const key of keys) {
    if (typeof value[key] === 'string') return value[key];
  }
  return '';
}

function numberField(
  value: Record<string, unknown>,
  ...keys: string[]
): number | undefined {
  for (const key of keys) {
    if (typeof value[key] === 'number') return value[key];
  }
  return undefined;
}

function hardwareFromProps(props: ServerProps | null): HardwareInfo | null {
  if (!props) return null;
  const hardware = (props as ServerProps & Record<string, unknown>).hardware;
  if (!hardware || typeof hardware !== 'object' || Array.isArray(hardware))
    return null;
  const record = hardware as Record<string, unknown>;
  const target = stringField(record, 'target', 'architecture', 'gfx');
  const name = stringField(record, 'name', 'device_name', 'device');
  if (!target && !name) return null;
  return {
    vendor: stringField(record, 'vendor') || 'Unknown vendor',
    name: name || 'Unnamed GPU',
    target: target || 'unknown target',
    memoryBytes: numberField(record, 'memory_bytes', 'vram_bytes'),
    source: 'server',
  };
}

function StatusDot({ state }: { state: ConnectionState }) {
  return <span className={`status-dot status-${state}`} aria-hidden="true" />;
}

function ModelRow({
  model,
  selected,
  busy,
  onSelect,
  onAction,
}: {
  model: ServerModel;
  selected: boolean;
  busy: boolean;
  onSelect: () => void;
  onAction: (action: 'load' | 'unload') => void;
}) {
  const lifecycle = model.lifecycle?.toLowerCase() || '';
  const loaded = lifecycle.includes('loaded') || lifecycle === 'ready';
  return (
    <div className={`model-row ${selected ? 'model-selected' : ''}`}>
      <button
        type="button"
        className="model-main"
        aria-label={`Select ${model.id}`}
        onClick={onSelect}
      >
        <span className={`model-light ${loaded ? 'model-loaded' : ''}`} />
        <span>
          <strong>{model.id}</strong>
          <small>
            {model.lifecycle || 'configured'} ·{' '}
            {formatBytes(model.residentBytes)}
          </small>
        </span>
      </button>
      <Button
        variant="ghost"
        size="icon-xs"
        disabled={busy}
        aria-label={`${loaded ? 'Unload' : 'Load'} ${model.id}`}
        onClick={() => onAction(loaded ? 'unload' : 'load')}
      >
        {loaded ? <Unplug /> : <Cable />}
      </Button>
    </div>
  );
}

function LibraryModelRow({ model }: { model: ModelLibraryModel }) {
  const mtpAssistant = model.architecture === 'gemma4mtp';
  return (
    <div
      className={`library-model-row ${model.compatible ? '' : 'library-model-disabled'}`}
      title={model.reason || undefined}
    >
      <span className="library-file-icon">
        <HardDrive />
      </span>
      <span>
        <strong>{model.file_name}</strong>
        <small>
          {model.architecture} · {formatBytes(model.size_bytes)}
        </small>
        {model.mtp_companion_file_name && (
          <small>MTP assistant: {model.mtp_companion_file_name}</small>
        )}
        {mtpAssistant && (
          <small>
            {model.mtp_companion_for
              ? `MTP companion for: ${model.mtp_companion_for}`
              : 'MTP assistant companion only · requires its matching target'}
          </small>
        )}
        {!model.compatible && model.reason && <em>{model.reason}</em>}
      </span>
      <Badge variant="outline">
        {mtpAssistant
          ? 'companion only'
          : model.compatible
            ? 'ready'
            : 'unsupported'}
      </Badge>
    </div>
  );
}

function MessageCard({ message }: { message: UiMessage }) {
  const assistant = message.role === 'assistant';
  return (
    <article
      className={`message ${assistant ? 'message-assistant' : 'message-user'}`}
    >
      <div className="message-meta">
        <span className="message-avatar">
          {assistant ? <Sparkles /> : <span>YOU</span>}
        </span>
        <strong>{assistant ? 'sLLM' : 'Operator'}</strong>
      </div>
      {message.reasoning && (
        <details className="reasoning" open>
          <summary>
            <BrainCircuit /> Reasoning stream
          </summary>
          <p>{message.reasoning}</p>
        </details>
      )}
      <p className="message-content">
        {message.content || (assistant ? '…' : '')}
      </p>
    </article>
  );
}

export default function Home() {
  const [view, setView] = useState<View>('performance');
  const [messages, setMessages] = useState<UiMessage[]>(seedMessages);
  const [input, setInput] = useState('');
  const [models, setModels] = useState<ServerModel[]>(demoModels);
  const [props, setProps] = useState<ServerProps | null>(demoProps);
  const [selectedModel, setSelectedModel] = useState(demoModels[0].id);
  const [connection, setConnection] = useState<ConnectionState>('demo');
  const [config, setConfig] = useState<ApiConfig>(defaultApiConfig);
  const [connectionOpen, setConnectionOpen] = useState(false);
  const [libraryOpen, setLibraryOpen] = useState(false);
  const [library, setLibrary] = useState<ModelLibrarySnapshot>(demoLibrary);
  const [libraryBrowse, setLibraryBrowse] = useState<ModelLibraryBrowse | null>(
    null,
  );
  const [libraryBusy, setLibraryBusy] = useState(false);
  const [libraryError, setLibraryError] = useState('');
  const [navOpen, setNavOpen] = useState(false);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [streaming, setStreaming] = useState(false);
  const [error, setError] = useState('');
  const [busyModel, setBusyModel] = useState('');
  const [maxTokens, setMaxTokens] = useState(512);
  const [theme, setTheme] = useState<Theme>('light');
  const [benchmarkPrompt, setBenchmarkPrompt] = useState(
    'Explain in practical terms how KV-cache precision affects memory bandwidth during transformer inference.',
  );
  const [benchmarkTokens, setBenchmarkTokens] = useState(128);
  const [benchmarkState, setBenchmarkState] =
    useState<BenchmarkState>('complete');
  const [benchmark, setBenchmark] = useState<BenchmarkResult | null>(
    demoBenchmark,
  );
  const [benchmarkHistory, setBenchmarkHistory] = useState<BenchmarkResult[]>([
    demoBenchmark,
  ]);
  const [benchmarkError, setBenchmarkError] = useState('');
  const abortRef = useRef<AbortController | null>(null);
  const benchmarkAbortRef = useRef<AbortController | null>(null);

  const queue = Number(props?.scheduler?.queued_requests ?? 0);
  const active = Number(props?.scheduler?.active_requests ?? 0);
  const live = connection === 'live';
  const hardware = live ? hardwareFromProps(props) : demoHardware;
  const selectedServerModel = models.find(
    (model) => model.id === selectedModel,
  );
  const connectionLabel = useMemo(() => {
    if (connection === 'live') return 'Live server';
    if (connection === 'connecting') return 'Connecting';
    if (connection === 'error') return 'Connection error';
    return 'Safe demo';
  }, [connection]);

  async function connectLive(
    targetConfig: ApiConfig = config,
    openOnFailure = false,
  ) {
    benchmarkAbortRef.current?.abort();
    setConnection('connecting');
    setError('');
    try {
      const snapshot = await fetchServerSnapshot(targetConfig);
      setModels(snapshot.models);
      setProps(snapshot.props);
      setSelectedModel(snapshot.models[0]?.id || '');
      setBenchmark(null);
      setBenchmarkHistory([]);
      setBenchmarkState('idle');
      setBenchmarkError('');
      try {
        setLibrary(await fetchModelLibrary(targetConfig));
        setLibraryError('');
      } catch (cause) {
        setLibrary({
          schema_version: 'sllm-model-library-v1',
          models: [],
          error: null,
        });
        setLibraryError(
          cause instanceof Error
            ? cause.message
            : 'Model library is unavailable.',
        );
      }
      setConnection('live');
      setConnectionOpen(false);
    } catch (cause) {
      setConnection('error');
      setError(cause instanceof Error ? cause.message : 'Connection failed.');
      if (openOnFailure) setConnectionOpen(true);
    }
  }

  useEffect(() => {
    const controller = new AbortController();
    void fetchIntegratedRuntimeConfig(controller.signal)
      .then((runtime) => {
        if (!runtime || controller.signal.aborted) return;
        const integratedConfig = {
          ...defaultApiConfig,
          baseUrl: runtime.apiBaseUrl,
        };
        setConfig(integratedConfig);
        void connectLive(integratedConfig, true);
      })
      .catch((cause) => {
        if (controller.signal.aborted) return;
        setConnection('error');
        setError(
          cause instanceof Error
            ? cause.message
            : 'Integrated runtime configuration failed.',
        );
        setConnectionOpen(true);
      });
    return () => controller.abort();
  }, []);

  function returnToDemo() {
    abortRef.current?.abort();
    benchmarkAbortRef.current?.abort();
    setConnection('demo');
    setModels(demoModels);
    setProps(demoProps);
    setSelectedModel(demoModels[0].id);
    setBenchmark(demoBenchmark);
    setBenchmarkHistory([demoBenchmark]);
    setBenchmarkState('complete');
    setLibrary(demoLibrary);
    setLibraryBrowse(null);
    setLibraryError('');
    setBenchmarkError('');
    setError('');
    setConnectionOpen(false);
  }

  async function refreshLive() {
    if (!live) return;
    setError('');
    try {
      const snapshot = await fetchServerSnapshot(config);
      setModels(snapshot.models);
      setProps(snapshot.props);
      setLibrary(await fetchModelLibrary(config));
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'Refresh failed.');
    }
  }

  async function openModelLibrary() {
    setLibraryOpen(true);
    setLibraryError('');
    if (!live) {
      setLibrary(demoLibrary);
      setLibraryBrowse(null);
      return;
    }
    setLibraryBusy(true);
    try {
      const [snapshot, listing] = await Promise.all([
        fetchModelLibrary(config),
        browseModelLibrary(config, library.selected_path || undefined),
      ]);
      setLibrary(snapshot);
      setLibraryBrowse(listing);
    } catch (cause) {
      setLibraryError(
        cause instanceof Error
          ? cause.message
          : 'Model folder could not be opened.',
      );
    } finally {
      setLibraryBusy(false);
    }
  }

  async function browseFolder(path?: string) {
    if (!live) return;
    setLibraryBusy(true);
    setLibraryError('');
    try {
      setLibraryBrowse(await browseModelLibrary(config, path));
    } catch (cause) {
      setLibraryError(
        cause instanceof Error ? cause.message : 'Folder could not be opened.',
      );
    } finally {
      setLibraryBusy(false);
    }
  }

  async function selectCurrentFolder() {
    if (!live || !libraryBrowse) return;
    setLibraryBusy(true);
    setLibraryError('');
    try {
      const snapshot = await selectModelLibraryFolder(
        config,
        libraryBrowse.current_path,
      );
      setLibrary(snapshot);
      const server = await fetchServerSnapshot(config);
      setModels(server.models);
      setProps(server.props);
      if (!server.models.some((model) => model.id === selectedModel))
        setSelectedModel(server.models[0]?.id || '');
    } catch (cause) {
      setLibraryError(
        cause instanceof Error
          ? cause.message
          : 'Model folder could not be selected.',
      );
    } finally {
      setLibraryBusy(false);
    }
  }

  async function rescanLibrary() {
    if (!live) return;
    setLibraryBusy(true);
    setLibraryError('');
    try {
      setLibrary(await rescanModelLibrary(config));
      const server = await fetchServerSnapshot(config);
      setModels(server.models);
      setProps(server.props);
    } catch (cause) {
      setLibraryError(
        cause instanceof Error ? cause.message : 'Model folder rescan failed.',
      );
    } finally {
      setLibraryBusy(false);
    }
  }

  async function runModelAction(alias: string, action: 'load' | 'unload') {
    setBusyModel(alias);
    setError('');
    try {
      if (live) {
        await modelAction(config, alias, action);
        await refreshLive();
      } else {
        setModels((current) =>
          current.map((model) =>
            model.id === alias
              ? {
                  ...model,
                  lifecycle: action === 'load' ? 'loaded' : 'configured',
                  residentBytes: action === 'load' ? 4_980_000_000 : undefined,
                }
              : model,
          ),
        );
      }
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'Model action failed.');
    } finally {
      setBusyModel('');
    }
  }

  async function simulate(id: string) {
    const reasoningText =
      'Inspect the request, use the selected model fixture, and separate internal reasoning from the answer.';
    const answer = `Prototype response from ${selectedModel}. Connect a server to replace this deterministic fixture with a live SSE stream.`;
    for (const value of reasoningText) {
      await new Promise((resolve) => setTimeout(resolve, 8));
      if (abortRef.current?.signal.aborted)
        throw new DOMException('Aborted', 'AbortError');
      setMessages((current) =>
        current.map((item) =>
          item.id === id
            ? { ...item, reasoning: (item.reasoning ?? '') + value }
            : item,
        ),
      );
    }
    for (const value of answer) {
      await new Promise((resolve) => setTimeout(resolve, 6));
      if (abortRef.current?.signal.aborted)
        throw new DOMException('Aborted', 'AbortError');
      setMessages((current) =>
        current.map((item) =>
          item.id === id ? { ...item, content: item.content + value } : item,
        ),
      );
    }
  }

  async function send() {
    const prompt = input.trim();
    if (!prompt || streaming || !selectedModel) return;
    const user: UiMessage = {
      id: crypto.randomUUID(),
      role: 'user',
      content: prompt,
    };
    const assistant: UiMessage = {
      id: crypto.randomUUID(),
      role: 'assistant',
      content: '',
    };
    const requestMessages: ChatMessage[] = [
      ...messages.map(({ role, content, reasoning: priorReasoning }) => ({
        role,
        content,
        reasoning: priorReasoning,
      })),
      user,
    ];
    setMessages((current) => [...current, user, assistant]);
    setInput('');
    setError('');
    setStreaming(true);
    const controller = new AbortController();
    abortRef.current = controller;
    try {
      if (live) {
        await streamChatCompletion(
          config,
          {
            model: selectedModel,
            messages: requestMessages,
            temperature: 0,
            topP: 1,
            maxTokens,
            responseFormat: 'text',
            reasoning: false,
            reasoningBudget: 1,
          },
          controller.signal,
          {
            onContent: (value) =>
              setMessages((current) =>
                current.map((item) =>
                  item.id === assistant.id
                    ? { ...item, content: item.content + value }
                    : item,
                ),
              ),
            onReasoning: (value) =>
              setMessages((current) =>
                current.map((item) =>
                  item.id === assistant.id
                    ? { ...item, reasoning: (item.reasoning ?? '') + value }
                    : item,
                ),
              ),
          },
        );
      } else {
        await simulate(assistant.id);
      }
    } catch (cause) {
      if (!(cause instanceof DOMException && cause.name === 'AbortError')) {
        setError(cause instanceof Error ? cause.message : 'Generation failed.');
      }
    } finally {
      setStreaming(false);
      abortRef.current = null;
    }
  }

  function stop() {
    abortRef.current?.abort();
  }

  async function runBenchmark() {
    const prompt = benchmarkPrompt.trim();
    if (
      !prompt ||
      !selectedModel ||
      benchmarkState === 'running' ||
      benchmarkTokens < 2
    )
      return;

    const controller = new AbortController();
    benchmarkAbortRef.current = controller;
    setBenchmarkState('running');
    setBenchmarkError('');
    setError('');

    try {
      let result: BenchmarkResult;
      if (!live) {
        await new Promise((resolve) => setTimeout(resolve, 480));
        if (controller.signal.aborted)
          throw new DOMException('Aborted', 'AbortError');
        result = {
          ...demoBenchmark,
          id: crypto.randomUUID(),
          model: selectedModel,
          createdAt: `DEMO · ${new Date().toLocaleTimeString()}`,
        };
      } else {
        const before = modelMetricSnapshot(
          await fetchServerMetrics(config),
          selectedModel,
        );
        if (controller.signal.aborted)
          throw new DOMException('Aborted', 'AbortError');

        await streamChatCompletion(
          config,
          {
            model: selectedModel,
            messages: [{ role: 'user', content: prompt }],
            temperature: 0,
            topP: 1,
            maxTokens: benchmarkTokens,
            responseFormat: 'text',
            reasoning: false,
            reasoningBudget: 1,
          },
          controller.signal,
          { onContent: () => undefined, onReasoning: () => undefined },
        );

        const after = modelMetricSnapshot(
          await fetchServerMetrics(config),
          selectedModel,
        );
        const delta = subtractMetrics(after, before);
        const decodeSeconds = delta.e2eSeconds - delta.ttftSeconds;
        const decodedTokens = delta.completionTokens - 1;
        if (delta.successes !== 1)
          throw new Error(
            'The metric window included another successful request. Retry while this model is idle.',
          );
        if (
          delta.promptTokens <= 0 ||
          decodedTokens <= 0 ||
          delta.ttftSeconds <= 0 ||
          decodeSeconds <= 0
        )
          throw new Error(
            'The server metrics did not contain a complete throughput window.',
          );
        result = {
          id: crypto.randomUUID(),
          model: selectedModel,
          prefillTps: delta.promptTokens / delta.ttftSeconds,
          decodeTps: decodedTokens / decodeSeconds,
          ttftMs: delta.ttftSeconds * 1000,
          e2eMs: delta.e2eSeconds * 1000,
          promptTokens: delta.promptTokens,
          completionTokens: delta.completionTokens,
          source: 'server-metrics',
          createdAt: new Date().toLocaleTimeString(),
        };
      }

      setBenchmark(result);
      setBenchmarkHistory((current) => [result, ...current].slice(0, 6));
      setBenchmarkState('complete');
    } catch (cause) {
      if (cause instanceof DOMException && cause.name === 'AbortError') {
        setBenchmarkState((current) =>
          current === 'running' ? (benchmark ? 'complete' : 'idle') : current,
        );
      } else {
        const message =
          cause instanceof Error ? cause.message : 'Benchmark failed.';
        setBenchmarkError(
          message.includes('404')
            ? 'The server does not expose /metrics. Enable metrics before running a live benchmark.'
            : message,
        );
        setBenchmarkState('error');
      }
    } finally {
      benchmarkAbortRef.current = null;
    }
  }

  function stopBenchmark() {
    benchmarkAbortRef.current?.abort();
  }

  function toggleTheme() {
    const next = theme === 'light' ? 'dark' : 'light';
    document.documentElement.classList.toggle('dark', next === 'dark');
    document.documentElement.style.colorScheme = next;
    setTheme(next);
  }

  const modelList = (
    <>
      <button
        type="button"
        className="library-card"
        onClick={() => void openModelLibrary()}
      >
        <FolderOpen />
        <span>
          <small>Model folder</small>
          <strong>
            {library.selected_path
              ? library.selected_path.split('/').filter(Boolean).at(-1) || '/'
              : 'Choose folder'}
          </strong>
        </span>
        <ChevronRight />
      </button>
      <div className="section-label">
        <span>Models</span>
        <Badge variant="outline">{models.length}</Badge>
      </div>
      <div className="model-list">
        {models.map((model) => (
          <ModelRow
            key={model.id}
            model={model}
            selected={model.id === selectedModel}
            busy={busyModel === model.id}
            onSelect={() => setSelectedModel(model.id)}
            onAction={(action) => void runModelAction(model.id, action)}
          />
        ))}
        {!models.length && (
          <p className="empty-note">No configured models were returned.</p>
        )}
      </div>
      {!!library.models.length && (
        <>
          <div className="section-label library-section-label">
            <span>GGUF files</span>
            <Badge variant="outline">{library.models.length}</Badge>
          </div>
          <div className="library-model-list">
            {library.models.map((model) => (
              <LibraryModelRow key={model.file_name} model={model} />
            ))}
          </div>
        </>
      )}
    </>
  );

  const settings = (
    <div className="settings-stack">
      <div className="setting-heading">
        <Settings2 />
        <span>Chat</span>
      </div>
      <label className="setting-control" htmlFor="max-tokens">
        <span id="max-tokens-label">Output token limit</span>
        <Input
          aria-labelledby="max-tokens-label"
          id="max-tokens"
          type="number"
          min={1}
          max={32768}
          value={maxTokens}
          onChange={(event) => setMaxTokens(Number(event.target.value))}
        />
        <small className="setting-help">
          Maximum response length. The model may finish earlier.
        </small>
      </label>
    </div>
  );

  const benchmarkSettings = (
    <div className="settings-stack benchmark-settings">
      <div className="setting-heading">
        <BarChart3 />
        <span>Performance run</span>
      </div>
      <div className="benchmark-guide">
        <strong>How workload size is chosen</strong>
        <p>
          Type or paste the workload into Prompt. Its input token count is
          measured automatically; there is no separate input-token setting.
        </p>
      </div>
      <label className="setting-control" htmlFor="benchmark-model">
        <span id="benchmark-model-label">Model</span>
        <NativeSelect
          className="w-full"
          id="benchmark-model"
          aria-labelledby="benchmark-model-label"
          value={selectedModel}
          onChange={(event) => setSelectedModel(event.target.value)}
        >
          {models.map((model) => (
            <NativeSelectOption key={model.id} value={model.id}>
              {model.id}
            </NativeSelectOption>
          ))}
        </NativeSelect>
      </label>
      <label
        className="setting-control benchmark-prompt"
        htmlFor="benchmark-prompt"
      >
        <span id="benchmark-prompt-label">Prompt</span>
        <Textarea
          aria-labelledby="benchmark-prompt-label"
          id="benchmark-prompt"
          value={benchmarkPrompt}
          onChange={(event) => setBenchmarkPrompt(event.target.value)}
        />
        <small className="setting-help">
          Longer text produces a larger prefill workload.
        </small>
      </label>
      <label className="setting-control" htmlFor="benchmark-tokens">
        <span id="benchmark-tokens-label">Output token limit</span>
        <Input
          aria-labelledby="benchmark-tokens-label"
          id="benchmark-tokens"
          type="number"
          min={2}
          max={4096}
          value={benchmarkTokens}
          onChange={(event) => setBenchmarkTokens(Number(event.target.value))}
        />
        <small className="setting-help">
          This is only a maximum. Generation can stop earlier when the model
          reaches its end token.
        </small>
      </label>
      {benchmarkState === 'running' ? (
        <Button
          variant="destructive"
          onClick={stopBenchmark}
          className="benchmark-button"
        >
          <CircleStop /> Stop run
        </Button>
      ) : (
        <Button
          onClick={() => void runBenchmark()}
          className="benchmark-button connect-button"
          disabled={
            !selectedModel || !benchmarkPrompt.trim() || benchmarkTokens < 2
          }
        >
          <Play /> {live ? 'Run live measurement' : 'Run demo fixture'}
        </Button>
      )}
      <div className="measurement-method">
        <strong>{live ? 'Server metrics' : 'Fixture mode'}</strong>
        <p>
          {live
            ? 'Runs one streaming request and subtracts the model’s /metrics counters before and after it.'
            : 'Uses fixed sample values to exercise the interface. Changing the prompt or output limit does not change demo results.'}
        </p>
      </div>
    </div>
  );

  const chartMaximum = Math.max(
    benchmark?.prefillTps ?? 0,
    benchmark?.decodeTps ?? 0,
    1,
  );

  return (
    <main className="console-shell">
      <header className="topbar">
        <div className="brand">
          <span>sLLM</span>
          <em>console</em>
        </div>
        <div className="topbar-center">
          <span className="eyebrow">
            {view === 'performance'
              ? 'PERFORMANCE WORKSPACE'
              : 'CHAT WORKSPACE'}
          </span>
          <strong>
            {view === 'performance'
              ? hardware?.target || 'GPU identity unavailable'
              : selectedModel || 'No model selected'}
          </strong>
        </div>
        <div className="topbar-actions">
          <Button
            className="theme-button"
            variant="ghost"
            size="icon"
            onClick={toggleTheme}
            aria-label={`Switch to ${theme === 'light' ? 'dark' : 'light'} mode`}
            aria-pressed={theme === 'dark'}
          >
            {theme === 'light' ? <Moon /> : <Sun />}
          </Button>
          <button
            type="button"
            className="connection-pill"
            onClick={() => setConnectionOpen(true)}
          >
            <StatusDot state={connection} />
            <span>{connectionLabel}</span>
            <ChevronRight />
          </button>
          <Button
            className="mobile-button"
            variant="ghost"
            size="icon"
            onClick={() => setNavOpen(true)}
            aria-label="Open models"
          >
            <Menu />
          </Button>
          <Button
            className="mobile-button"
            variant="ghost"
            size="icon"
            onClick={() => setSettingsOpen(true)}
            aria-label={`Open ${view === 'performance' ? 'benchmark' : 'chat'} settings`}
          >
            <PanelRight />
          </Button>
        </div>
      </header>

      <nav className="side-nav" aria-label="Workspace navigation">
        <div className="nav-items">
          <button
            type="button"
            className={`nav-item ${view === 'performance' ? 'nav-active' : ''}`}
            onClick={() => setView('performance')}
          >
            <Gauge />
            <span>Performance</span>
          </button>
          <button
            type="button"
            className={`nav-item ${view === 'chat' ? 'nav-active' : ''}`}
            onClick={() => setView('chat')}
          >
            <MessageSquare />
            <span>Chat</span>
            <small>secondary</small>
          </button>
          <button type="button" className="nav-item" disabled>
            <Server />
            <span>Sessions</span>
            <small>soon</small>
          </button>
        </div>
        <div className="nav-models">{modelList}</div>
        <button
          type="button"
          className="runtime-card"
          onClick={() => setConnectionOpen(true)}
        >
          <Cpu />
          <span>
            <small>Runtime</small>
            <strong>{props?.state || (live ? 'unknown' : 'fixture')}</strong>
          </span>
          <ChevronRight />
        </button>
      </nav>

      {view === 'performance' ? (
        <section className="dashboard" aria-label="GPU performance overview">
          <div className="conversation-heading dashboard-heading">
            <div>
              <span className="eyebrow">RUNTIME / PERFORMANCE</span>
              <h1>GPU &amp; throughput</h1>
            </div>
            <div className="runtime-stats">
              <span>
                <i className={active ? 'stat-hot' : ''} />
                {active} active
              </span>
              <span>
                <i />
                {queue} queued
              </span>
              <Button
                variant="ghost"
                size="icon-sm"
                onClick={() => void refreshLive()}
                disabled={!live}
                aria-label="Refresh runtime"
              >
                <RefreshCw />
              </Button>
            </div>
          </div>

          <div className="dashboard-scroll" aria-live="polite">
            {error && (
              <div className="error-banner dashboard-error" role="alert">
                <span>{error}</span>
                <button type="button" onClick={() => setError('')}>
                  Dismiss
                </button>
              </div>
            )}
            {benchmarkError && (
              <div className="error-banner dashboard-error" role="alert">
                <span>{benchmarkError}</span>
                <button type="button" onClick={() => setBenchmarkError('')}>
                  Dismiss
                </button>
              </div>
            )}

            <div className={`measurement-notice ${live ? 'notice-live' : ''}`}>
              <Activity />
              <div>
                <strong>
                  {live
                    ? 'LIVE SERVER MODE'
                    : 'DEMO FIXTURE — NOT GPU EVIDENCE'}
                </strong>
                <p>
                  {live
                    ? 'Runtime values come from the connected sLLM server. Throughput appears after a measurement run.'
                    : 'The values below are fixed samples for shaping the interface. Connect sLLM for a real run.'}
                </p>
              </div>
            </div>

            <article className="hardware-card">
              <div className="hardware-glyph">
                <Cpu />
              </div>
              <div className="hardware-copy">
                <span className="eyebrow">DETECTED COMPUTE</span>
                {hardware ? (
                  <>
                    <h2>{hardware.name}</h2>
                    <p>
                      {hardware.vendor} · {hardware.target}
                    </p>
                  </>
                ) : (
                  <>
                    <h2>GPU identity not reported</h2>
                    <p>
                      The current /props contract has no hardware identity
                      field. This panel is ready to consume one when the server
                      exposes it.
                    </p>
                  </>
                )}
              </div>
              <div className="hardware-facts">
                <span>
                  <HardDrive />
                  <small>VRAM</small>
                  <strong>
                    {hardware?.memoryBytes !== undefined
                      ? formatBytes(hardware.memoryBytes)
                      : hardware
                        ? 'not reported'
                        : '—'}
                  </strong>
                </span>
                <span>
                  <Zap />
                  <small>Model resident</small>
                  <strong>
                    {formatBytes(selectedServerModel?.residentBytes)}
                  </strong>
                </span>
                <Badge variant="outline">
                  {hardware?.source === 'server'
                    ? 'server'
                    : hardware
                      ? 'fixture'
                      : 'unavailable'}
                </Badge>
              </div>
            </article>

            <div className="metric-grid">
              <article className="metric-card metric-primary">
                <div className="metric-label">
                  <Activity /> Prefill
                </div>
                <div className="metric-number">
                  {formatRate(benchmark?.prefillTps)} <small>tok/s</small>
                </div>
                <p>Prompt tokens ÷ accumulated TTFT</p>
              </article>
              <article className="metric-card">
                <div className="metric-label">
                  <Gauge /> Decode
                </div>
                <div className="metric-number">
                  {formatRate(benchmark?.decodeTps)} <small>tok/s</small>
                </div>
                <p>Tokens after first ÷ post-TTFT time</p>
              </article>
              <article className="metric-card">
                <div className="metric-label">
                  <Timer /> TTFT
                </div>
                <div className="metric-number">
                  {benchmark ? benchmark.ttftMs.toFixed(0) : '—'}{' '}
                  <small>ms</small>
                </div>
                <p>Queue, prefill, and first streamed delta</p>
              </article>
              <article className="metric-card">
                <div className="metric-label">
                  <BarChart3 /> Tokens in this run
                </div>
                <div className="metric-token-pair">
                  <span>
                    <small>Input tokens</small>
                    <strong>
                      {benchmark
                        ? benchmark.promptTokens.toLocaleString()
                        : '—'}
                    </strong>
                  </span>
                  <span>
                    <small>Actual output tokens</small>
                    <strong>
                      {benchmark
                        ? benchmark.completionTokens.toLocaleString()
                        : '—'}
                    </strong>
                  </span>
                </div>
                <p>Measured token counts for the latest completed run</p>
              </article>
            </div>

            <div className="performance-panels">
              <article className="throughput-panel">
                <div className="panel-title">
                  <div>
                    <span className="eyebrow">LATEST RUN</span>
                    <h2>Throughput profile</h2>
                  </div>
                  <Badge variant="outline">
                    {benchmarkState === 'running'
                      ? 'measuring…'
                      : benchmark?.source === 'server-metrics'
                        ? 'server metrics'
                        : 'fixture'}
                  </Badge>
                </div>
                <div className="rate-chart">
                  <div className="rate-row">
                    <span>Prefill</span>
                    <div>
                      <i
                        style={{
                          width: `${((benchmark?.prefillTps ?? 0) / chartMaximum) * 100}%`,
                        }}
                      />
                    </div>
                    <strong>{formatRate(benchmark?.prefillTps)}</strong>
                  </div>
                  <div className="rate-row">
                    <span>Decode</span>
                    <div>
                      <i
                        style={{
                          width: `${((benchmark?.decodeTps ?? 0) / chartMaximum) * 100}%`,
                        }}
                      />
                    </div>
                    <strong>{formatRate(benchmark?.decodeTps)}</strong>
                  </div>
                </div>
                <p className="method-footnote">
                  These are estimates derived from server counters, not a
                  kernel-only benchmark. Run with no concurrent traffic for a
                  clean window.
                </p>
              </article>

              <article className="history-panel">
                <div className="panel-title">
                  <div>
                    <span className="eyebrow">RECENT</span>
                    <h2>Measurement runs</h2>
                  </div>
                  <span className="history-count">
                    {benchmarkHistory.length}
                  </span>
                </div>
                <div className="run-list">
                  {benchmarkHistory.map((run) => (
                    <div className="run-row" key={run.id}>
                      <div>
                        <strong>{run.model}</strong>
                        <small>
                          {run.createdAt} ·{' '}
                          {run.source === 'fixture' ? 'fixture' : 'live'}
                        </small>
                      </div>
                      <span>
                        <small>prefill</small>
                        <strong>{formatRate(run.prefillTps)}</strong>
                      </span>
                      <span>
                        <small>decode</small>
                        <strong>{formatRate(run.decodeTps)}</strong>
                      </span>
                    </div>
                  ))}
                  {!benchmarkHistory.length && (
                    <p className="empty-note">
                      No live measurements yet. Configure a prompt and run one.
                    </p>
                  )}
                </div>
              </article>
            </div>
          </div>
        </section>
      ) : (
        <section className="conversation" aria-label="Chat">
          <div className="conversation-heading">
            <div>
              <span className="eyebrow">SESSION / 001</span>
              <h1>Model workspace</h1>
            </div>
            <div className="runtime-stats">
              <span>
                <i className={active ? 'stat-hot' : ''} />
                {active} active
              </span>
              <span>
                <i />
                {queue} queued
              </span>
              <Button
                variant="ghost"
                size="icon-sm"
                onClick={() => void refreshLive()}
                disabled={!live}
                aria-label="Refresh runtime"
              >
                <RefreshCw />
              </Button>
            </div>
          </div>

          {error && (
            <div className="error-banner" role="alert">
              <span>{error}</span>
              <button type="button" onClick={() => setError('')}>
                Dismiss
              </button>
            </div>
          )}

          <div className="messages" aria-live="polite">
            <div className="demo-notice">
              <Check />{' '}
              {live
                ? 'Responses and runtime state come from the connected server.'
                : 'Safe demo uses deterministic fixture data and is not GPU evidence.'}
            </div>
            {messages.map((message) => (
              <MessageCard key={message.id} message={message} />
            ))}
          </div>

          <div className="composer-wrap">
            <div className="composer">
              <Textarea
                aria-label="Message"
                placeholder="Ask the selected model…"
                value={input}
                onChange={(event) => setInput(event.target.value)}
                onKeyDown={(event) => {
                  if (event.key === 'Enter' && !event.shiftKey) {
                    event.preventDefault();
                    void send();
                  }
                }}
              />
              <div className="composer-footer">
                <div className="composer-tags">
                  <span>{selectedModel || 'no model'}</span>
                  <span>
                    <BrainCircuit /> reasoning
                  </span>
                </div>
                {streaming ? (
                  <Button
                    className="send-button stop-button"
                    size="icon"
                    onClick={stop}
                    aria-label="Stop generation"
                  >
                    <CircleStop />
                  </Button>
                ) : (
                  <Button
                    className="send-button"
                    size="icon"
                    onClick={() => void send()}
                    disabled={!input.trim() || !selectedModel}
                    aria-label="Send message"
                  >
                    <ArrowUp />
                  </Button>
                )}
              </div>
            </div>
            <p>
              Enter to send · Shift+Enter for a new line · Keys remain in memory
              only
            </p>
          </div>
        </section>
      )}

      <aside
        className="inspector"
        aria-label={
          view === 'performance' ? 'Benchmark settings' : 'Chat settings'
        }
      >
        {view === 'performance' ? benchmarkSettings : settings}
      </aside>

      <Sheet open={navOpen} onOpenChange={setNavOpen}>
        <SheetContent side="left" className="mobile-sheet">
          <SheetHeader>
            <SheetTitle>Models</SheetTitle>
            <SheetDescription>
              Change workspace or model lifecycle state.
            </SheetDescription>
          </SheetHeader>
          <div className="sheet-body">
            <div className="mobile-view-switch">
              <button
                type="button"
                className={`nav-item ${view === 'performance' ? 'nav-active' : ''}`}
                onClick={() => {
                  setView('performance');
                  setNavOpen(false);
                }}
              >
                <Gauge /> Performance
              </button>
              <button
                type="button"
                className={`nav-item ${view === 'chat' ? 'nav-active' : ''}`}
                onClick={() => {
                  setView('chat');
                  setNavOpen(false);
                }}
              >
                <MessageSquare /> Chat
              </button>
            </div>
            {modelList}
          </div>
        </SheetContent>
      </Sheet>
      <Sheet open={settingsOpen} onOpenChange={setSettingsOpen}>
        <SheetContent className="mobile-sheet">
          <SheetHeader>
            <SheetTitle>
              {view === 'performance' ? 'Performance run' : 'Chat settings'}
            </SheetTitle>
            <SheetDescription>
              {view === 'performance'
                ? 'Configure a single throughput measurement.'
                : 'Settings for the next request.'}
            </SheetDescription>
          </SheetHeader>
          <div className="sheet-body">
            {view === 'performance' ? benchmarkSettings : settings}
          </div>
        </SheetContent>
      </Sheet>
      <Sheet open={libraryOpen} onOpenChange={setLibraryOpen}>
        <SheetContent className="library-sheet">
          <SheetHeader>
            <SheetTitle>Model library</SheetTitle>
            <SheetDescription>
              Select a server folder, discover local GGUF files, or acquire a
              revision-pinned model from Hugging Face.
            </SheetDescription>
          </SheetHeader>
          <div className="sheet-body library-sheet-body">
            {libraryError && (
              <div className="error-banner" role="alert">
                <span>{libraryError}</span>
              </div>
            )}
            {!live ? (
              <div className="connection-warning">
                <strong>Demo fixture</strong>
                <p>
                  Connect a local sLLM server to browse its filesystem. The
                  sample GGUF entries are display-only.
                </p>
              </div>
            ) : (
              <>
                <div className="folder-location">
                  <Folder />
                  <code>
                    {libraryBrowse?.current_path ||
                      library.selected_path ||
                      'Loading…'}
                  </code>
                </div>
                <div className="folder-actions">
                  <Button
                    variant="outline"
                    disabled={libraryBusy || !libraryBrowse?.parent_path}
                    onClick={() =>
                      void browseFolder(libraryBrowse?.parent_path || undefined)
                    }
                  >
                    <ArrowLeft /> Parent
                  </Button>
                  <Button
                    className="connect-button"
                    disabled={libraryBusy || !libraryBrowse}
                    onClick={() => void selectCurrentFolder()}
                  >
                    <Check /> Use this folder
                  </Button>
                </div>
                <div className="folder-list" aria-busy={libraryBusy}>
                  {libraryBrowse?.directories.map((directory) => (
                    <button
                      type="button"
                      key={directory.path}
                      disabled={libraryBusy}
                      onClick={() => void browseFolder(directory.path)}
                    >
                      <Folder />
                      <span>{directory.name}</span>
                      <ChevronRight />
                    </button>
                  ))}
                  {!libraryBusy && !libraryBrowse?.directories.length && (
                    <p className="empty-note">No child folders are visible.</p>
                  )}
                </div>
              </>
            )}

            <HuggingFaceModels
              config={config}
              live={live}
              selectedPath={library.selected_path}
              onDownloadComplete={rescanLibrary}
            />

            <div className="library-results-heading">
              <div>
                <span className="eyebrow">DISCOVERED GGUF</span>
                <strong>{library.models.length} files</strong>
              </div>
              <Button
                variant="outline"
                size="sm"
                disabled={!live || libraryBusy || !library.selected_path}
                onClick={() => void rescanLibrary()}
              >
                <RefreshCw /> Rescan
              </Button>
            </div>
            <div className="library-sheet-models">
              {library.models.map((model) => (
                <LibraryModelRow key={model.file_name} model={model} />
              ))}
              {!library.models.length && (
                <p className="empty-note">
                  Select a folder containing .gguf files. Converted sLLM bundles
                  also include a matching .derived-lock.json file.
                </p>
              )}
            </div>
          </div>
        </SheetContent>
      </Sheet>
      <Sheet open={connectionOpen} onOpenChange={setConnectionOpen}>
        <SheetContent className="connection-sheet">
          <SheetHeader>
            <SheetTitle>Server connection</SheetTitle>
            <SheetDescription>
              Credentials are held in React memory and are never persisted by
              this prototype.
            </SheetDescription>
          </SheetHeader>
          <div className="sheet-body connection-form">
            <label htmlFor="server-endpoint">
              <span id="endpoint-label">Endpoint</span>
              <Input
                aria-labelledby="endpoint-label"
                id="server-endpoint"
                value={config.baseUrl}
                onChange={(event) =>
                  setConfig((current) => ({
                    ...current,
                    baseUrl: event.target.value,
                  }))
                }
                placeholder="http://127.0.0.1:8080"
              />
            </label>
            <label htmlFor="user-key">
              <span id="user-key-label">User bearer key</span>
              <Input
                aria-labelledby="user-key-label"
                id="user-key"
                type="password"
                autoComplete="off"
                value={config.userKey}
                onChange={(event) =>
                  setConfig((current) => ({
                    ...current,
                    userKey: event.target.value,
                  }))
                }
              />
            </label>
            <label htmlFor="admin-key">
              <span>
                Admin bearer key{' '}
                <small>optional on a credential-free loopback server</small>
              </span>
              <Input
                aria-label="Admin bearer key"
                id="admin-key"
                type="password"
                autoComplete="off"
                value={config.adminKey}
                onChange={(event) =>
                  setConfig((current) => ({
                    ...current,
                    adminKey: event.target.value,
                  }))
                }
              />
            </label>
            <div className="connection-warning">
              <strong>Browser boundary</strong>
              <p>
                The server must allow this page’s exact CORS origin. An
                HTTPS-hosted page cannot call a plain HTTP endpoint.
              </p>
            </div>
            <Button
              className="connect-button"
              onClick={() => void connectLive()}
              disabled={connection === 'connecting'}
            >
              <Cable />
              {connection === 'connecting'
                ? 'Connecting…'
                : 'Connect live server'}
            </Button>
            <Button variant="outline" onClick={returnToDemo}>
              Use safe demo
            </Button>
          </div>
          <div className="connection-scope">
            <span>Prototype API surface</span>
            <code>/healthz</code>
            <code>/readyz</code>
            <code>/v1/models</code>
            <code>/props</code>
            <code>/metrics</code>
            <code>/v1/chat/completions</code>
            <code>/admin/model-library</code>
          </div>
        </SheetContent>
      </Sheet>
    </main>
  );
}
