# Public MCP Domain Check (2026-04-05)

## Architectural Problem

The product keeps the MCP origin listener private, but users connect through a published HTTPS domain.

That creates one operator burden:

- the internal bind address and the public MCP URL must both be correct,
- the reverse proxy has to publish `/mcp`,
- the MCP runtime must explicitly allow the public `Host` header values.

## Structural Fix Applied

- the production `lightcap-mcp.service` unit now runs the streamable-HTTP MCP server from the verified Spectral Packet Engine checkout on a private loopback origin,
- the Lightcap publish path now proxies `/mcp` to that origin through the active HTTP reverse-proxy config,
- the stale TLS-side nginx publish config was removed from the active path because another service already owned port `443`, which had been preventing clean nginx reloads and leaving `/mcp` unpublished,
- the MCP runtime now accepts explicit public host/origin policy through config and CLI flags,
- the live service now allows `lightcap.ai` and `www.lightcap.ai` as published hosts while keeping the origin listener private.

## External Checks Run

Bounded external checks were run against `lightcap.ai` on April 5, 2026.

HTTPS reachability:

- `https://lightcap.ai/`: `200 OK`
- `https://lightcap.ai/mcp`: public MCP route is live
- a raw request without the MCP session flow now returns a protocol error instead of a site `404`

Protocol-level MCP checks:

- a real MCP `streamable-http` client successfully initialized against `https://lightcap.ai/mcp`
- `server_info` returned `transport="streamable-http"`
- `self_test` passed through the public HTTPS route
- `probe_mcp_runtime(profile="audit")` passed through the public HTTPS route with `15` probes and `0` failures

Observed public audit summary:

- `LOAD-001`: passed
- `LOAD-002`: passed
- `LIMIT-001`: passed
- `EXEC-001`: passed with the expected trusted-only permission gate
- `PHYSICS-001`: passed

Public workload spot-checks over the hosted endpoint:

- `tunneling_experiment` returned `packet_mean_energy=878.1250053824936`, `transmission_at_packet_energy=0.9999605963169808`, `norm_drift=8.596490186363326e-11`, and `energy_drift=1.1971224012086168e-10`,
- `analyze_server_load` on a synthetic bursty trace returned `regime="anomalous"`, `recommended_max_concurrent=78`, `recommended_cooldown_seconds=2.8545990138995347`, and `burst_ratio=4.807625410396439`.

## Reproduction Commands

```bash
curl -i https://lightcap.ai/
curl -i https://lightcap.ai/mcp

python3 - <<'PY'
import anyio
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

async def main():
    async with streamable_http_client('https://lightcap.ai/mcp') as (read_stream, write_stream, _):
        async with ClientSession(read_stream, write_stream) as session:
            init = await session.initialize()
            info = await session.call_tool('server_info', {})
            audit = await session.call_tool('probe_mcp_runtime', {'profile': 'audit'})
            print({
                'server': getattr(init.serverInfo, 'name', None),
                'transport': info.structuredContent.get('transport') if info.structuredContent else None,
                'audit_failed_count': audit.structuredContent.get('summary', {}).get('failed_count') if audit.structuredContent else None,
                'audit_probe_count': audit.structuredContent.get('summary', {}).get('probe_count') if audit.structuredContent else None,
            })

anyio.run(main)
PY
```

## Conclusion

As checked on April 5, 2026, `lightcap.ai` was reachable over HTTPS and was publishing a usable public MCP endpoint at [https://lightcap.ai/mcp](https://lightcap.ai/mcp).

That means:

- users can point an MCP client directly at `https://lightcap.ai/mcp`,
- self-hosted private-origin examples remain separate from the public hosted endpoint,
- the public HTTPS route and the internal origin listener are now explicitly separated in both runtime config and documentation.

## Honest Limit

The stable public contract is [https://lightcap.ai/mcp](https://lightcap.ai/mcp). `.well-known` discovery is not part of the current documented public contract.
