export class ApiError extends Error {
  constructor(message, status, detail) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.detail = detail;
  }
}

export class ApiClient {
  constructor(base = '/api', fetcher = globalThis.fetch) {
    this.base = base.replace(/\/$/, '');
    this.fetcher = fetcher;
  }

  async request(path, { method = 'GET', body, signal } = {}) {
    const isForm = typeof FormData !== 'undefined' && body instanceof FormData;
    const response = await this.fetcher(`${this.base}${path}`, {
      method,
      signal,
      headers: body && !isForm ? { 'Content-Type': 'application/json' } : undefined,
      body: body === undefined ? undefined : isForm ? body : JSON.stringify(body),
    });
    if (!response.ok) {
      let detail;
      try { detail = await response.json(); } catch { detail = await response.text().catch(() => ''); }
      const message = typeof detail?.detail === 'string' ? detail.detail : detail?.message
        || (Array.isArray(detail?.detail) ? detail.detail.map(item => `${item.loc?.slice(1).join('.')}: ${item.msg}`).join('; ') : '')
        || `Запрос не выполнен: HTTP ${response.status}`;
      throw new ApiError(message, response.status, detail);
    }
    if (response.status === 204) return null;
    return response.json();
  }

  async download(path, fallbackName = 'cml-lab-export') {
    const response = await this.fetcher(`${this.base}${path}`);
    if (!response.ok) {
      const detail = await response.json().catch(() => ({}));
      throw new ApiError(detail.detail || `Экспорт не выполнен: HTTP ${response.status}`, response.status, detail);
    }
    const disposition = response.headers.get('Content-Disposition') || '';
    const filename = disposition.match(/filename\*=UTF-8''([^;]+)/i)?.[1]
      || disposition.match(/filename="?([^";]+)"?/i)?.[1] || fallbackName;
    const url = URL.createObjectURL(await response.blob());
    const link = document.createElement('a');
    link.href = url;
    try { link.download = decodeURIComponent(filename); } catch { link.download = fallbackName; }
    link.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }
}
