"""Remote launcher for federated ChemGraph Academy campaigns.

Submit-mode only: qsub a fresh PBS job per --site via the login node,
poll qstat until R, poll for daemon placement.json, exit. Multi-site
launches run concurrently (asyncio.gather over N sites).
"""
