#!/usr/bin/env bash
# Run a bounded AQ4_0 direct/grouped full-model profile window as root.
#
# The caller supplies an already validated candidate manifest and a profile
# binary.  This script contains no credentials: invoke it through the approved
# privileged wrapper.  It stops only ullm-openai.service, acquires the shared
# R9700 flock after the service releases it, and always tries to restore the
# service before returning.
set -Eeuo pipefail

readonly SERVICE='ullm-openai.service'
readonly LOCK_FILE='/run/ullm/r9700.lock'
readonly ACTIVE_MANIFEST='/etc/ullm/served-models/active.json'
readonly FILTER='ullm-sq8-r9700|run_measurements.py|llama-bench|llama-server|promote-served-model'

if [[ ${EUID} -ne 0 || $# -ne 3 ]]; then
    echo "usage: run-aq4-grouped-profile-window.sh CANDIDATE_MANIFEST PROFILE_BINARY OUTPUT_DIR" >&2
    exit 64
fi

readonly CANDIDATE_MANIFEST=$1
readonly PROFILE_BINARY=$2
readonly OUTPUT_DIR=$3

if [[ ! -f ${CANDIDATE_MANIFEST} || ! -x ${PROFILE_BINARY} || -e ${OUTPUT_DIR} ]]; then
    echo 'candidate manifest/profile binary/output directory precondition failed' >&2
    exit 65
fi
if [[ $(systemctl show "${SERVICE}" -p ActiveState --value) != active ]]; then
    echo 'refusing to create a service window unless the gateway is already active' >&2
    exit 66
fi

umask 027
mkdir --mode=750 "${OUTPUT_DIR}"
service_stopped=0
lock_held=0

record_service() {
    systemctl show "${SERVICE}" -p ActiveState -p SubState -p Result -p NRestarts -p MainPID
}

record_pgrep_pids() {
    local label=$1
    local raw
    raw=$(mktemp /tmp/bq-aq4-profile-pgrep.XXXXXX)
    pgrep -af "${FILTER}" >"${raw}" || true
    : >"${OUTPUT_DIR}/${label}-pgrep-pids.txt"
    while IFS=' ' read -r pid _; do
        [[ -n ${pid} ]] || continue
        local command
        command=$(ps -p "${pid}" -o comm= 2>/dev/null | tr -d '[:space:]' || true)
        printf '%s\t%s\n' "${pid}" "${command:-gone}" >>"${OUTPUT_DIR}/${label}-pgrep-pids.txt"
    done <"${raw}"
    rm -f -- "${raw}"
}

restore_service() {
    local restored=0
    systemctl start "${SERVICE}" >"${OUTPUT_DIR}/service-restore.stdout" 2>"${OUTPUT_DIR}/service-restore.stderr" && restored=1
    if [[ ${restored} -eq 0 ]] && [[ $(systemctl show "${SERVICE}" -p Result --value) == start-limit-hit ]]; then
        printf '%s\n' 'start-limit recovery: reset-failed then one start' >>"${OUTPUT_DIR}/service-restore.stderr"
        systemctl reset-failed "${SERVICE}" >>"${OUTPUT_DIR}/service-restore.stdout" 2>>"${OUTPUT_DIR}/service-restore.stderr"
        systemctl start "${SERVICE}" >>"${OUTPUT_DIR}/service-restore.stdout" 2>>"${OUTPUT_DIR}/service-restore.stderr" && restored=1
    fi
    printf '%s\n' "${restored}" >"${OUTPUT_DIR}/service-restore.exit-status"
    record_service >"${OUTPUT_DIR}/post-window-service.txt"
    sha256sum "${ACTIVE_MANIFEST}" >"${OUTPUT_DIR}/post-window-active-manifest.sha256"
    fuser -v "${LOCK_FILE}" >"${OUTPUT_DIR}/post-window-fuser.txt" 2>&1 || true
    record_pgrep_pids post-window
    [[ ${restored} -eq 1 ]]
}

cleanup() {
    local status=$?
    set +e
    if [[ ${lock_held} -eq 1 ]]; then
        flock -u 9
        exec 9>&-
    fi
    if [[ ${service_stopped} -eq 1 ]]; then
        restore_service || status=1
    fi
    chown -R homelab1:homelab1 "${OUTPUT_DIR}"
    trap - EXIT
    exit "${status}"
}
trap cleanup EXIT

record_service >"${OUTPUT_DIR}/pre-window-service.txt"
sha256sum "${ACTIVE_MANIFEST}" >"${OUTPUT_DIR}/pre-window-active-manifest.sha256"
fuser -v "${LOCK_FILE}" >"${OUTPUT_DIR}/pre-window-fuser.txt" 2>&1 || true
record_pgrep_pids pre-window
sha256sum "${PROFILE_BINARY}" >"${OUTPUT_DIR}/profile-binary.sha256"
sha256sum "${CANDIDATE_MANIFEST}" >"${OUTPUT_DIR}/candidate-manifest.sha256"

systemctl stop "${SERVICE}" >"${OUTPUT_DIR}/service-stop.stdout" 2>"${OUTPUT_DIR}/service-stop.stderr"
service_stopped=1
record_service >"${OUTPUT_DIR}/post-stop-service.txt"
sha256sum "${ACTIVE_MANIFEST}" >"${OUTPUT_DIR}/post-stop-active-manifest.sha256"
fuser -v "${LOCK_FILE}" >"${OUTPUT_DIR}/post-stop-fuser.txt" 2>&1 || true
record_pgrep_pids post-stop
if fuser -s "${LOCK_FILE}"; then
    echo 'R9700 lock remained held after the gateway stopped' >&2
    exit 67
fi

exec 9>"${LOCK_FILE}"
flock -n 9
lock_held=1
printf '%s\n' "$(date --iso-8601=seconds)" >"${OUTPUT_DIR}/lock-acquired.txt"

mapfile -t REQUIRED_ENVIRONMENT < <(jq -r '.worker.required_environment[]' "${CANDIDATE_MANIFEST}")
for name in "${REQUIRED_ENVIRONMENT[@]}"; do
    [[ ${name} =~ ^[A-Z_][A-Z0-9_]*$ ]] || { echo "unsafe required environment name: ${name}" >&2; exit 68; }
done

run_profile() {
    local label=$1
    local grouped=$2
    local -a environment=(
        'PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/bin'
        'HOME=/var/cache/ullm'
        'XDG_CACHE_HOME=/var/cache/ullm'
        'HIP_VISIBLE_DEVICES=1'
    )
    local name
    for name in "${REQUIRED_ENVIRONMENT[@]}"; do
        environment+=("${name}=1")
    done
    if [[ ${grouped} == 1 ]]; then
        environment+=('ULLM_EXPERIMENTAL_PAGED_DECODE_GQA_GROUPED_SPLIT=1')
    fi
    printf '%s\n' "$(date --iso-8601=seconds)" >"${OUTPUT_DIR}/${label}.started-at"
    runuser -u homelab1 -- env -i "${environment[@]}" "${PROFILE_BINARY}" 1339 --warmup 6 --measured 32 \
        >"${OUTPUT_DIR}/${label}.jsonl" 2>"${OUTPUT_DIR}/${label}.stderr"
    printf '%s\n' "$(date --iso-8601=seconds)" >"${OUTPUT_DIR}/${label}.finished-at"
}

# Alternate modes so a slowly changing device temperature cannot become a one-sided result.
run_profile direct-a 0
run_profile grouped-a 1
run_profile direct-b 0
run_profile grouped-b 1
printf '%s\n' 'completed' >"${OUTPUT_DIR}/window-work-complete.txt"
