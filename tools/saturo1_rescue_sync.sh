#!/bin/bash
# saturo1_rescue_sync.sh
# saturo_1 处于周期性高负载（sshd 仅在负载低谷能响应），本脚本持续重试，
# 一旦抓到可用窗口就用 rsync（断点续传）把远端实验结果拉到本地备份目录。
# 只读远端，不做任何远端写操作。

set -u

HOST="saturo_1"
BACKUP_ROOT="/Users/dongqx/code/l20a_saturo/remote_backup/saturo_1"
LOG="/Users/dongqx/code/l20a_saturo/remote_backup/rescue_sync.log"
REPO_CACHE="/Users/dongqx/code/l20a_saturo/remote_backup/.remote_repo_path"
DONE_FLAG="/Users/dongqx/code/l20a_saturo/remote_backup/SYNC_COMPLETE"

# 每次尝试都用全新连接（避免复用被 D 状态进程毒化的会话）
SSH_OPTS="-o BatchMode=yes -o ConnectTimeout=45 -o ServerAliveInterval=10 -o ServerAliveCountMax=3 -o ControlMaster=no -o ControlPath=none"

mkdir -p "$BACKUP_ROOT"

log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

# macOS 无 timeout 命令，用 perl alarm 做看门狗，防止会话挂死
with_to() { local t=$1; shift; perl -e 'alarm shift; exec @ARGV' "$t" "$@"; }

discover_repo() {
    # 只探测固定候选路径，避免 find 全盘扫描碰到坏挂载点挂死
    with_to 90 ssh $SSH_OPTS "$HOST" \
        'ls -d /root/l20a_saturo /root/code/l20a_saturo /home/*/l20a_saturo /workspace/l20a_saturo /data/l20a_saturo /mnt/*/l20a_saturo /root/saturo /root/code/saturo 2>/dev/null | head -1' \
        2>>"$LOG"
}

attempt=0
while true; do
    attempt=$((attempt + 1))

    # 1) 确定远端仓库路径（成功一次后缓存，之后不再探测）
    if [[ -s "$REPO_CACHE" ]]; then
        REPO=$(cat "$REPO_CACHE")
    else
        log "attempt #$attempt: probing remote repo path..."
        REPO=$(discover_repo | tr -d '\r')
        if [[ -z "$REPO" ]]; then
            log "attempt #$attempt: no window or repo not found, retry in 30s"
            sleep 30
            continue
        fi
        echo "$REPO" > "$REPO_CACHE"
        log "attempt #$attempt: remote repo = $REPO"
    fi

    # 2) 拉取实验结果（断点续传；rsync --timeout 防止 IO 挂死时僵住）
    log "attempt #$attempt: rsync experiments/ ..."
    with_to 3600 rsync -az --partial --timeout=90 \
        --exclude '__pycache__' --exclude '*.pyc' --exclude '.venv' \
        -e "ssh $SSH_OPTS" \
        "$HOST:$REPO/experiments/" "$BACKUP_ROOT/experiments/" >>"$LOG" 2>&1
    rc=$?

    if [[ $rc -eq 0 ]]; then
        log "attempt #$attempt: rsync experiments/ COMPLETE"
        # 3) saturo/results/ 是快速搬迁手册标注的核心实验结果目录，必须同步
        with_to 3600 rsync -az --partial --timeout=90 \
            --exclude '__pycache__' --exclude '*.pyc' \
            -e "ssh $SSH_OPTS" \
            "$HOST:$REPO/saturo/results/" "$BACKUP_ROOT/saturo_results/" >>"$LOG" 2>&1 \
            && log "saturo/results/ synced" || { log "saturo/results/ sync FAILED, will retry"; sleep 30; continue; }
        # 4) 顺带拉调试数据（尽力而为，失败不影响主目标）
        with_to 1800 rsync -az --partial --timeout=90 \
            --exclude '__pycache__' --exclude '*.pyc' \
            -e "ssh $SSH_OPTS" \
            "$HOST:$REPO/debug_info/" "$BACKUP_ROOT/debug_info/" >>"$LOG" 2>&1 \
            && log "debug_info/ synced" || log "debug_info/ sync skipped/failed (non-fatal)"
        touch "$DONE_FLAG"
        log "ALL DONE. Backup at: $BACKUP_ROOT"
        break
    else
        log "attempt #$attempt: rsync interrupted (rc=$rc), partial data kept, retry in 30s"
        sleep 30
    fi
done
