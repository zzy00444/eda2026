#!/usr/bin/env bash
# Normalise a competition request CSV so its first line is the required "From,To"
# header.  The official input carries that header; locally produced From,To files
# (for example a cut of delay_estimate_ans.csv) often start directly at the first
# row, which makes the estimator exit with "expected From,To header" after the
# expensive PGO compile.  Sourced by build_candidate_wsl.sh and
# validate_stage5_wsl.sh so both paths accept either form.
ensure_request_header() {
  local file=${1:?request CSV path required}
  if [[ ! -s "$file" ]]; then
    echo "Empty or missing request CSV: $file" >&2
    return 2
  fi
  local first=''
  IFS= read -r first <"$file" || true
  first=${first%$'\r'}
  if [[ "$first" == "From,To" ]]; then
    return 0
  fi
  local tmp="$file.header.tmp"
  if ! { printf 'From,To\n'; cat -- "$file"; } >"$tmp"; then
    rm -f "$tmp"
    echo "Cannot prepend header to $file" >&2
    return 2
  fi
  mv -- "$tmp" "$file"
  echo "Prepended From,To header to $file" >&2
}
