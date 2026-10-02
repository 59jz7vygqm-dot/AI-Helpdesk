#!/bin/sh
# Render the configuration, then hand the container over to Asterisk in the
# foreground so docker logs and `docker compose restart` behave normally.
set -eu

: "${PROVIDER_HOST:?PROVIDER_HOST is required -- the managed PBX}"
: "${PROVIDER_USER:?PROVIDER_USER is required -- the number the provider gave this Asterisk}"
: "${PROVIDER_PASSWORD:?PROVIDER_PASSWORD is required}"
: "${AGENT_PASSWORD:?AGENT_PASSWORD is required -- the agent's own SIP password}"
: "${HUMAN_EXTEN:?HUMAN_EXTEN is required -- where unanswered and transferred calls go}"

# Exported, not merely set: envsubst reads the environment, so a variable that
# only exists in the shell renders as an empty string. That produced a silent
# `Dial(PJSIP/,)` and a `server_uri` with no port -- a config that loads and
# does nothing.
export PROVIDER_HOST PROVIDER_USER PROVIDER_PASSWORD AGENT_PASSWORD HUMAN_EXTEN
export PROVIDER_PORT="${PROVIDER_PORT:-5060}"
export AGENT_EXTEN="${AGENT_EXTEN:-7000}"
export AGENT_RING_SECONDS="${AGENT_RING_SECONDS:-20}"
export ASTERISK_RTP_START="${ASTERISK_RTP_START:-10000}"
export ASTERISK_RTP_END="${ASTERISK_RTP_END:-15999}"
export SIP_BIND_PORT="${SIP_BIND_PORT:-5060}"

# Only these names are substituted. Asterisk's own ${EXTEN} and friends must
# survive untouched -- an unrestricted envsubst would silently empty them and
# leave a dialplan that dials nowhere.
VARS='${PROVIDER_HOST} ${PROVIDER_PORT} ${PROVIDER_USER} ${PROVIDER_PASSWORD} ${AGENT_EXTEN} ${AGENT_PASSWORD} ${HUMAN_EXTEN} ${AGENT_RING_SECONDS} ${ASTERISK_RTP_START} ${ASTERISK_RTP_END} ${SIP_BIND_PORT}'

for template in /opt/asterisk-templates/*.tmpl; do
    name=$(basename "$template" .tmpl)
    envsubst "$VARS" < "$template" > "/etc/asterisk/$name"
done

# Refuse to start on a config that renders to something that cannot work. Both
# of these failed silently once: Asterisk loads, registers, and no call reaches
# anything. Only this script's own placeholders are checked -- Asterisk's
# ${EXTEN} and ${DIALSTATUS} belong in the dialplan and must survive.
leftover=$(printf '%s' "$VARS" | tr -d '${}' | tr ' ' '|' | sed 's/||*/|/g; s/^|//; s/|$//')
for rendered in pjsip.conf extensions.conf rtp.conf; do
    if grep -nE "\\\$\\{($leftover)\\}" "/etc/asterisk/$rendered"; then
        echo "FATAL: /etc/asterisk/$rendered still holds a placeholder (above)." >&2
        echo "envsubst reads the environment, so check the variable is exported." >&2
        exit 1
    fi
    if grep -nE '(PJSIP/,|PJSIP/@|^[a-z_]+=$|:$)' "/etc/asterisk/$rendered"; then
        echo "FATAL: /etc/asterisk/$rendered rendered an empty value (above)." >&2
        exit 1
    fi
done

echo "asterisk config rendered: registering as $PROVIDER_USER at $PROVIDER_HOST:$PROVIDER_PORT,"
echo "agent extension $AGENT_EXTEN, humans at $HUMAN_EXTEN, rtp ${ASTERISK_RTP_START}-${ASTERISK_RTP_END}"

exec asterisk -f -vvv
