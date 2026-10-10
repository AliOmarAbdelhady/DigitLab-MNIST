#!/usr/bin/env bash
# Deploy the web build (+ APK if present) to GitHub Pages using a git worktree,
# so the working tree is never disturbed.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "==> building web"
( cd web && npm run sync >/dev/null && npm run build >/dev/null )

WT=$(mktemp -d /tmp/digitlab-pages.XXXXXX)
if git show-ref --verify --quiet refs/heads/gh-pages; then
  git worktree add "$WT" gh-pages >/dev/null
  ( cd "$WT" && git rm -rq . 2>/dev/null || true )
else
  git worktree add --orphan -b gh-pages "$WT" >/dev/null
fi
cp -r web/dist/. "$WT"/

# include the release APK as a download, when it has been built
APK=mobile/android/app/build/outputs/apk/release/app-release.apk
if [ -f "$APK" ]; then
  cp "$APK" "$WT/digitlab.apk"
fi

( cd "$WT" && git add -A && git commit -qm "deploy: DigitLab web" && git push -q origin gh-pages )
git worktree remove --force "$WT"
echo "==> deployed to GitHub Pages"
