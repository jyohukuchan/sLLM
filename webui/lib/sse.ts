export interface SseEvent {
  data: string;
  event?: string;
  id?: string;
}

function parseEventBlock(block: string): SseEvent | null {
  const data: string[] = [];
  let event: string | undefined;
  let id: string | undefined;

  for (const rawLine of block.split('\n')) {
    const line = rawLine.endsWith('\r') ? rawLine.slice(0, -1) : rawLine;
    if (!line || line.startsWith(':')) continue;
    const separator = line.indexOf(':');
    const field = separator === -1 ? line : line.slice(0, separator);
    let value = separator === -1 ? '' : line.slice(separator + 1);
    if (value.startsWith(' ')) value = value.slice(1);
    if (field === 'data') data.push(value);
    if (field === 'event') event = value;
    if (field === 'id' && !value.includes('\0')) id = value;
  }

  return data.length ? { data: data.join('\n'), event, id } : null;
}

export class SseDecoder {
  private buffer = '';

  push(chunk: string): SseEvent[] {
    this.buffer += chunk;
    this.buffer = this.buffer
      .replaceAll('\r\n', '\n')
      .replace(/\r(?!$)/g, '\n');
    const events: SseEvent[] = [];
    let boundary = this.buffer.indexOf('\n\n');
    while (boundary !== -1) {
      const event = parseEventBlock(this.buffer.slice(0, boundary));
      if (event) events.push(event);
      this.buffer = this.buffer.slice(boundary + 2);
      boundary = this.buffer.indexOf('\n\n');
    }
    return events;
  }

  finish(): SseEvent[] {
    this.buffer = this.buffer.replaceAll('\r\n', '\n').replaceAll('\r', '\n');
    const event = parseEventBlock(this.buffer);
    this.buffer = '';
    return event ? [event] : [];
  }
}
