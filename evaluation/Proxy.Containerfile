FROM docker.io/library/debian:bookworm-slim
RUN apt-get update && apt-get install -y --no-install-recommends squid ca-certificates && rm -rf /var/lib/apt/lists/*
COPY squid.conf /etc/squid/squid.conf
RUN /usr/sbin/squid -k parse -f /etc/squid/squid.conf
USER 1000:1000
ENTRYPOINT ["/usr/sbin/squid", "-N", "-f", "/etc/squid/squid.conf"]
