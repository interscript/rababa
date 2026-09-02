# Contributing

Thanks for your interest in contributing!

## Development setup

```bash
git clone <this repo>
cd <repo>
bundle install         # Ruby projects
# or
npm ci                 # JS projects
```

## Workflow

1. Fork → branch from `main`
2. Make changes with tests
3. Run `bundle exec rspec` (Ruby) or `npm test` (JS) locally
4. Run `bundle exec standardrb` (Ruby) or `npm run lint` (JS)
5. Open a PR with a clear description

## Code style

- Ruby: enforced by [StandardRB](https://github.com/standardrb/standard)
- JavaScript: enforced by ESLint + Prettier
- Python: enforced by ruff

## Commit messages

Use [Conventional Commits](https://www.conventionalcommits.org/):

```
feat: add new transliteration system
fix: correct off-by-one in CALT lookup
chore: bump dependencies
docs: clarify README
```

## Releases

Maintainers tag releases following semver. CI publishes on tag push.
