#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
release_dir="$project_root/release-linux"
build_dir="$release_dir/build"
app_name="botanic-spectrum-analyser"
app_dir="$build_dir/AppDir"
tool_dir="$build_dir/appimagetool"
rpm_url='https://download.opensuse.org/pub/opensuse/repositories/home:/kimi:/AppImage/16.0/x86_64/appimagetool-app-continuous-lp160.3.2.x86_64.rpm'
rpm_sha256='6bd8017310ba120c0929584dedabb72ce2c326e63f94532429c608ebdce8dbbf'

mkdir -p "$build_dir"
docker build -f "$project_root/packaging/linux/Dockerfile" -t bsa-linux-build:bullseye "$project_root"
docker run --rm --user "$(id -u):$(id -g)" \
  -e HOME=/tmp -e PYINSTALLER_CONFIG_DIR=/out/cache \
  -e MPLCONFIGDIR=/tmp/mpl -e CUDA_VISIBLE_DEVICES=-1 \
  -v "$project_root:/source:ro" -v "$build_dir:/out" -w /source \
  bsa-linux-build:bullseye \
  pyinstaller --noconfirm --clean --onedir --windowed \
    --name "$app_name" --paths /source/src \
    --hidden-import PIL._tkinter_finder \
    --runtime-hook /source/packaging/linux/runtime_hook.py \
    --add-binary /usr/bin/fc-list:. \
    --add-data /source/packaging/linux/fonts.conf:. \
    --add-data /source/src/bsa/assets:bsa/assets \
    --distpath /out/dist --workpath /out/work --specpath /out \
    /source/packaging/linux/entrypoint.py

cp -a "$project_root/Models" "$build_dir/dist/$app_name/Models"
install -m 755 "$project_root/packaging/linux/bsa-label-creator" \
  "$build_dir/dist/$app_name/bsa-label-creator"

mkdir -p "$tool_dir"
if [[ ! -f "$tool_dir/appimagetool.rpm" ]]; then
  curl --fail --location --silent --show-error "$rpm_url" -o "$tool_dir/appimagetool.rpm"
fi
printf '%s  %s\n' "$rpm_sha256" "$tool_dir/appimagetool.rpm" | sha256sum -c -
if [[ ! -x "$tool_dir/squashfs-root/AppRun" ]]; then
  (cd "$tool_dir" && bsdtar -xf appimagetool.rpm && ./usr/bin/appimagetool --appimage-extract >/dev/null)
fi
runtime_offset="$($tool_dir/usr/bin/appimagetool --appimage-offset)"
head -c "$runtime_offset" "$tool_dir/usr/bin/appimagetool" > "$tool_dir/runtime"

rm -rf "$app_dir"
mkdir -p "$app_dir/usr/lib/bsa" "$app_dir/usr/share/applications"
cp -a "$build_dir/dist/$app_name" "$app_dir/usr/lib/bsa/"
cp "$project_root/src/bsa/assets/BSA_logo.png" "$app_dir/$app_name.png"
cp "$project_root/packaging/linux/$app_name.desktop" "$app_dir/$app_name.desktop"
cp "$project_root/packaging/linux/$app_name.desktop" "$app_dir/usr/share/applications/"
cp "$project_root/packaging/linux/AppRun" "$app_dir/AppRun"
chmod 755 "$app_dir/AppRun"

ARCH=x86_64 "$tool_dir/squashfs-root/AppRun" \
  --runtime-file "$tool_dir/runtime" \
  "$app_dir" "$release_dir/Botanic-Spectrum-Analyser-Linux-x86_64.AppImage"
chmod 755 "$release_dir/Botanic-Spectrum-Analyser-Linux-x86_64.AppImage"

tar --sort=name --mtime='@0' --owner=0 --group=0 --numeric-owner \
  -C "$build_dir/dist" -cf - "$app_name" | gzip -n > \
  "$release_dir/Botanic-Spectrum-Analyser-Linux-x86_64.tar.gz"

(cd "$release_dir" && sha256sum \
  Botanic-Spectrum-Analyser-Linux-x86_64.AppImage \
  Botanic-Spectrum-Analyser-Linux-x86_64.tar.gz > SHA256SUMS.txt)

du -h "$release_dir"/Botanic-Spectrum-Analyser-Linux-x86_64.*
