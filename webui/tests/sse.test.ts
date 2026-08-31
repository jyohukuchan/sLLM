import assert from 'node:assert/strict';
import test from 'node:test';

import { SseDecoder } from '../lib/sse.ts';

void test('decodes fragmented events', () => {
  const decoder = new SseDecoder();
  assert.deepEqual(decoder.push('data: {"choices":'), []);
  assert.deepEqual(decoder.push('[]}\n\ndata: [DONE]\n\n'), [
    { data: '{"choices":[]}', event: undefined, id: undefined },
    { data: '[DONE]', event: undefined, id: undefined },
  ]);
});

void test('joins multiline data and ignores comments', () => {
  const decoder = new SseDecoder();
  assert.deepEqual(
    decoder.push(': ping\nid: 7\nevent: token\ndata: a\ndata: b\n\n'),
    [{ data: 'a\nb', event: 'token', id: '7' }],
  );
});

void test('flushes a final event without a separator', () => {
  const decoder = new SseDecoder();
  decoder.push('data: [DONE]');
  assert.deepEqual(decoder.finish(), [
    { data: '[DONE]', event: undefined, id: undefined },
  ]);
});

void test('preserves a CRLF boundary split across chunks', () => {
  const decoder = new SseDecoder();
  assert.deepEqual(decoder.push('data: first\r'), []);
  assert.deepEqual(decoder.push('\n\r\ndata: second\r\n\r\n'), [
    { data: 'first', event: undefined, id: undefined },
    { data: 'second', event: undefined, id: undefined },
  ]);
});
