#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
source "${SCRIPT_DIR}/env.sh"

APPTAINER_BIN="${APPTAINER_BIN:-}"
if [[ -z "${APPTAINER_BIN}" ]]; then
  if command -v apptainer >/dev/null 2>&1; then
    APPTAINER_BIN=apptainer
  elif command -v singularity >/dev/null 2>&1; then
    APPTAINER_BIN=singularity
  else
    echo "Neither apptainer nor singularity is available on PATH." >&2
    exit 1
  fi
fi

container_path_exists() {
  local path="$1"
  if [[ -d "${SLIME_SIF}" ]]; then
    [[ -e "${SLIME_SIF}${path}" ]]
  else
    "${APPTAINER_BIN}" exec --cleanenv "${SLIME_SIF}" test -e "${path}"
  fi
}

container_has_py_or_legacy_pyc() {
  local module_path="$1"
  container_path_exists "${module_path}.py" || container_path_exists "${module_path}.pyc"
}

validate_stdlib_files() {
  local missing=()
  for module_path in \
    /usr/lib/python3.12/encodings/__init__ \
    /usr/lib/python3.12/os \
    /usr/lib/python3.12/site; do
    if ! container_has_py_or_legacy_pyc "${module_path}"; then
      missing+=("${module_path}.py or ${module_path}.pyc")
    fi
  done

  if [[ "${#missing[@]}" -gt 0 ]]; then
    {
      echo "Container Python stdlib is incomplete or pycache-only."
      echo "Missing source or legacy sourceless .pyc paths:"
      printf "  %s\n" "${missing[@]}"
    } >&2
    return 1
  fi
}

materialize_sourceless_stdlib() {
  if [[ ! -d "${SLIME_SIF}" ]]; then
    echo "Skipping sourceless stdlib repair for non-sandbox container: ${SLIME_SIF}"
    return 0
  fi

  local stdlib_root="${SLIME_SIF}/usr/lib/python3.12"
  if [[ ! -d "${stdlib_root}" ]]; then
    echo "No Python stdlib directory found in sandbox: ${stdlib_root}" >&2
    return 1
  fi

  local created=0
  local pyc pycache_dir target_dir pyc_name module_name target
  while IFS= read -r pyc; do
    pycache_dir="$(dirname "${pyc}")"
    target_dir="$(dirname "${pycache_dir}")"
    pyc_name="$(basename "${pyc}")"
    module_name="${pyc_name%.cpython-312.pyc}"
    target="${target_dir}/${module_name}.pyc"

    if [[ -e "${target_dir}/${module_name}.py" || -e "${target}" ]]; then
      continue
    fi
    cp -p "${pyc}" "${target}"
    created=$((created + 1))
  done < <(find "${stdlib_root}" -path '*/__pycache__/*.cpython-312.pyc' -type f | sort)

  echo "Sourceless stdlib repair materialized ${created} legacy .pyc files."
}

validate_container() {
  validate_stdlib_files
  echo "Validating container Python/imports"
  "${APPTAINER_BIN}" exec --cleanenv "${SLIME_SIF}" \
    python3 -c "import encodings, os, site, sglang, torch; print('container ok')"
  echo "Validating container through wrapper"
  "${SCRIPT_DIR}/container_exec.sh" \
    python3 -c "import encodings, os, site, sglang, torch; print('container wrapper ok')"
}

mkdir -p "$(dirname "${SLIME_SIF}")" "${APPTAINER_CACHEDIR}" "${APPTAINER_TMPDIR}"
export APPTAINER_CACHEDIR APPTAINER_TMPDIR

cat <<EOF
Container setup
  image uri: ${SLIME_IMAGE_URI}
  format   : ${SLIME_CONTAINER_FORMAT}
  path     : ${SLIME_SIF}
  cache    : ${APPTAINER_CACHEDIR}
  tmp      : ${APPTAINER_TMPDIR}
EOF

if [[ -e "${SLIME_SIF}" && "${FORCE_PULL:-0}" != "1" ]]; then
  echo "Container already exists. Validating it now."
  if [[ -d "${SLIME_SIF}" ]]; then
    materialize_sourceless_stdlib
  fi
  if validate_container; then
    echo "Existing container is valid: ${SLIME_SIF}"
  else
    cat >&2 <<EOF
Existing container failed validation:
  ${SLIME_SIF}

Move it aside or remove it manually, then rerun this script to rebuild it.
EOF
    exit 1
  fi
  exit 0
fi

if [[ -e "${SLIME_SIF}" && "${FORCE_PULL:-0}" == "1" ]]; then
  echo "Refusing to overwrite existing ${SLIME_SIF} automatically." >&2
  echo "Move it aside or remove it manually, then rerun this script." >&2
  exit 1
fi

if [[ "${SLIME_CONTAINER_FORMAT}" == "sandbox" ]]; then
  "${APPTAINER_BIN}" build --sandbox "${SLIME_SIF}" "${SLIME_IMAGE_URI}"
  materialize_sourceless_stdlib
else
  "${APPTAINER_BIN}" pull --force "${SLIME_SIF}" "${SLIME_IMAGE_URI}"
fi

validate_container
echo "Wrote and validated ${SLIME_SIF}"
