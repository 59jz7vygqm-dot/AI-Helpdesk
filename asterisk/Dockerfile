# A local Asterisk in front of the agent.
#
# Built from Debian's own package rather than a community image: the version is
# whatever the distro ships and security updates come from the distro, so there
# is no third party in the path of a telephone system. Debian 12 carries
# Asterisk 20 (LTS), which has PJSIP and AudioSocket.
FROM debian:12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        asterisk \
        gettext-base \
    && rm -rf /var/lib/apt/lists/*

# Credentials live in .env, never in an image layer, so the configuration is
# rendered from templates at start.
COPY templates /opt/asterisk-templates
COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
