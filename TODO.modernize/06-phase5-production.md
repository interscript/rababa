# Phase 5 — Production deployment

## Goal
Every release is reproducible, rollback-safe, observable, and shippable
to both the website and the Ruby gem without manual steps.

## Tasks

### 5.1 CDN

- jsDelivr auto-mirrors GitHub releases: `https://cdn.jsdelivr.net/gh/interscript/<repo>@<tag>/<path>`.
- Pin version in `npm/models/manifest.json`; runner reads version → resolves CDN URL.
- SHA256 in manifest. Runner verifies hash before loading ONNX.

### 5.2 Browser caching

Mirror HTTP loader's persistent cache in `src/isc/loader.ts`:
```typescript
class IscStrategy implements LoadStrategy {
  async load(code: SystemCode): Promise<CompiledMap | undefined> {
    // 1. In-memory cache (instant)
    if (this.cache.has(code)) return this.cache.get(code)!
    // 2. localStorage cache (survives reload)
    const cached = this.readLocalStorage(code)
    if (cached) { this.cache.set(code, cached); return cached }
    // 3. Network fetch → SHA256 verify → compile → cache
    return this.fetchAndCompile(code)
  }
}
```

### 5.3 Server-side fallback

For browsers without ONNX runtime support (very old Safari):
- Server endpoint on Cloudflare Workers + ONNX runtime.
- Route ML funcalls via `transliterateAsync()` server-side.
- Manifest flag: `server_only: true` for models too big for browsers.

### 5.4 A/B rollout

- Manifest supports `rollout_percentage: 0..100` per model version.
- Runner honors it: clients randomly include themselves in cohort.
- Bump `1% → 10% → 50% → 100%` over a week.
- Rollback: revert manifest in `ml-models/npm/models/manifest.json`.

### 5.5 Telemetry

Opt-in, anonymous, aggregate-only:
- Inference latency per model.
- DER / CER if a labeled input is supplied.
- Crash stack traces (sanitized).

**NO input content is collected.** Verified by code review + open source tooling.

Hosted on:
- Local collection point (browser → Cloudflare Worker → Logflare).
- Or simple Prometheus + Grafana on a side channel.

### 5.6 CI: release pipeline

`.github/workflows/release.yml`:
```yaml
name: release
on:
  workflow_dispatch:
    inputs:
      task: { description: rababa_arabic / rababa_hebrew / secryst_thai_ipa }
      version: { description: "semver" }
jobs:
  train-and-release:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.11" }
      - run: pip install modal
      - run: modal token set --token=${{ secrets.MODAL_TOKEN }}
      - run: modal run modal_app.py::train_student --task ${{ inputs.task }}
      - run: modal run modal_app.py::export_onnx --task ${{ inputs.task }} --version ${{ inputs.version }}
      - run: modal run modal_app.py::evaluate --task ${{ inputs.task }} --version ${{ inputs.version }}
      - uses: softprops/action-gh-release@v2
        with:
          tag_name: ${{ inputs.task }}-v${{ inputs.version }}
          files: |
            models/${{ inputs.task }}-v${{ inputs.version }}-*.onnx
            models/${{ inputs.task }}-v${{ inputs.version }}-vocab.json
            SHA256SUMS
      - run: |
          # Update manifest
          python scripts/bump_manifest.py --task ${{ inputs.task }} --version ${{ inputs.version }} \
            --status stable --sha256 models/${{ inputs.task }}-v${{ inputs.version }}-SHA256SUMS
      - uses: peter-evans/create-pull-request-action@v6
        with:
          commit-message: "release: ${{ inputs.task }} v${{ inputs.version }}"
```

## Acceptance

- [ ] `manifest.json` has all v1.0.0 entries with `status: stable`.
- [ ] jsDelivr CDN serves each `q8` ONNX.
- [ ] SHA256 verification in runner.
- [ ] Rollback via manifest version bump (manual test).
- [ ] CI release workflow green end-to-end on a test cut.

## Open questions
1. **Telemetry vendor**: roll our own, or use PostHog / Plausible / etc? Recommend: keep it dead simple — Logflare + Cloudflare Worker.
2. **A/B cohort sampling**: cookie-based or session-based? Session-based for privacy.
