import assert from 'node:assert/strict';
import { test } from 'node:test';
import { ApiClient } from '../web/cml/api.js';

function browserFetch(response, calls) {
  return async function (address, options) {
    if (this !== globalThis) throw new TypeError('Illegal invocation');
    calls.push({ address, options });
    return response;
  };
}

test('default browser fetch loads the catalogue with its global receiver', async () => {
  const original = globalThis.fetch;
  const calls = [];
  globalThis.fetch = browserFetch({ ok: true, status: 200, json: async () => ({ tasks: ['regression'] }) }, calls);
  try {
    const api = new ApiClient();
    assert.deepEqual(await api.request('/cml/catalogue'), { tasks: ['regression'] });
    assert.equal(calls[0].address, '/api/cml/catalogue');
    assert.equal(calls[0].options.method, 'GET');
  } finally {
    globalThis.fetch = original;
  }
});

test('browser fetch downloads an export with its global receiver', async () => {
  const originalFetch = globalThis.fetch;
  const originalDocument = globalThis.document;
  const calls = [];
  const link = { click() { this.clicked = true; } };
  globalThis.fetch = browserFetch({
    ok: true,
    headers: new Headers({ 'Content-Disposition': 'attachment; filename="model.joblib"' }),
    blob: async () => new Blob(['trained model']),
  }, calls);
  globalThis.document = { createElement: () => link };
  try {
    await new ApiClient().download('/cml/models/example/export');
    assert.equal(calls[0].address, '/api/cml/models/example/export');
    assert.equal(link.download, 'model.joblib');
    assert.equal(link.clicked, true);
    assert.match(link.href, /^blob:/);
  } finally {
    globalThis.fetch = originalFetch;
    if (originalDocument === undefined) delete globalThis.document;
    else globalThis.document = originalDocument;
  }
});

test('project download sends the JSON snapshot by POST with the browser receiver', async () => {
  const originalFetch = globalThis.fetch, originalDocument = globalThis.document;
  const calls = [], link = { click() {} };
  globalThis.fetch = browserFetch({ ok: true, headers: new Headers(), blob: async () => new Blob(['project']) }, calls);
  globalThis.document = { createElement: () => link };
  const body = { spec: { dataset_id: 'snapshot', preprocessing: { declarative_pipeline: { source: 'pipeline = StandardScaler()' } } } };
  try {
    await new ApiClient().download('/cml/project-exports?format=py', 'project.py', { method: 'POST', body });
    assert.equal(calls[0].options.method, 'POST');
    assert.equal(calls[0].options.headers['Content-Type'], 'application/json');
    assert.deepEqual(JSON.parse(calls[0].options.body), body);
    assert.equal(link.download, 'project.py');
  } finally {
    globalThis.fetch = originalFetch;
    if (originalDocument === undefined) delete globalThis.document; else globalThis.document = originalDocument;
  }
});
