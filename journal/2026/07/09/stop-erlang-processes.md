# Stop Erlang processes

- Date: 2026-07-09
- Matched command line: `erlang`
- Finding: The Erlang processes belonged to Docker container `firecrawl-rabbitmq-1` using image `rabbitmq:3-management`.
- Assessment: RabbitMQ is an application service and is not required for normal OS operation.
- Action: Stopped container `firecrawl-rabbitmq-1`.
- Verification:
  - `pgrep -afi '[e]rlang'` returned no remaining processes.
  - `docker ps --all --filter name=firecrawl-rabbitmq-1` showed `Exited (0)`.
