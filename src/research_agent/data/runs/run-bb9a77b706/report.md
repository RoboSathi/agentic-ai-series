# MCP + DuckDB + Rill Dashboard Integration

**Status:** PARTIAL

## Summary

The Model Context Protocol (MCP) follows a client-host-server architecture where each host can run multiple client instances [E1][E4]. MCP is open and enables integration between LLM applications and external data sources and tools [E12][E23]. The DuckDB MCP server is distributed via the mcp-server-motherduck package. Install it using uv (preferred for Python CLI tools) or pip [E8]. The server acts as a bridge between DuckDB and AI-driven environments, such as Cursor or Claude, allowing these tools to execute SQL commands and return results in real time. It supports both local DuckDB instances and cloud-based MotherDuck databases [E10]. Rill's default embedded engine is DuckDB [E27]. Rill Developer is an open-source, code-first BI tool that enables software engineers to profile data sets, design data models, and deploy dashboards to Rill Cloud [E31]. Users will follow the same basic steps that include downloading and installing Rill, configuring MotherDuck as a source, entering a SQL expression to return a single table and clicking import [E39][E40]. Together, these components allow AI assistants to query DuckDB databases and surface results in Rill-powered dashboards [E13][E30].

## Concepts

MCP follows a client-host-server architecture with stateful sessions; each client connects to exactly one server [E3][E6][E23]. Servers expose resources, tools, and prompts via MCP primitives [E7][E23]. The server acts as a bridge between DuckDB and AI-driven environments, such as Cursor or Claude, allowing these tools to execute SQL commands and return results in real time. It supports both local DuckDB instances and cloud-based MotherDuck databases [E10]. The DuckDB/MotherDuck MCP server allows the AI copilot (Cursor) to directly run queries against local DuckDB databases and/or remote MotherDuck databases and interpret the results [E13]. This guide assumes you'll be using Rill's default embedded engine, DuckDB [E27]. Rill Developer is an open-source, code-first BI tool that enables software engineers to profile data sets, design data models, and deploy dashboards to Rill Cloud [E31]. By default, Rill includes DuckDB as an embedded OLAP engine that ingests data from data sources and powers your dashboards. When you start a new project, you'll see a connectors/duckdb.yaml file alongside other project files. No additional configuration is needed to use DuckDB with Rill Developer or Rill Cloud [E36]. Because Rill is already built with DuckDB embedded, no additional type wrangling is needed, the data in MotherDuck is "production-ready" for Rill dashboards [E41]. MCP primitives (tools, prompts, resources) connect AI models to external data [E23][E25][E26].

## Examples

Install the DuckDB MCP server with uv pip install mcp-server-motherduck or pip install mcp-server-motherduck [E8]. Configure an IDE such as Cursor by adding a JSON entry with command uvx and args mcp-server-motherduck, --db-path md:, and optionally --motherduck-token [E9][E14][E15]. Test the Configuration: Restart your IDE and run a test query: SELECT 1 AS test; If successful, the server returns the result, confirming the setup [E11]. SELECT mcp_publish_table('products', 'data://tables/products', 'json'); [E17] mcp_publish_execution_tool(name, desc, sql, props, required, bindings) - Publish execution tools with prepared parameter binding [E18]. Create a Rill project with curl https://rill.sh | sh and rill start my-rill-project, then ingest data from MotherDuck by configuring MotherDuck as a source and entering a SQL expression [E28][E40]. Generate an AI-powered dashboard in Rill via the side menu option "Generate dashboard with AI" [E30].

## Known unknowns

- None declared during scoping.

## Coverage & Limits

- R1: Every factual sentence in the report cites collected evidence. — 23/28, scored 0.82 of the 1 required; found: The DuckDB MCP server is distributed via the mcp-server-motherduck package., The server acts as a bridge between DuckDB and AI-driven environments, such as Cursor or Claude, allowing these tools to execute SQL commands and return results in real time., The server acts as a bridge between DuckDB and AI-driven environments, such as Cursor or Claude, allowing these tools to execute SQL commands and return results in real time., By default, Rill includes DuckDB as an embedded OLAP engine that ingests data from data sources and powers your dashboards.; run stopped: every sub-concept is covered in two sources; the critique found the answer incomplete: The claims provide solid coverage of MCP architecture, DuckDB/MotherDuck MCP server setup, and Rill Developer's DuckDB integration for dashboards. However, the critical intersection — how to place text-to-SQL in a chat UI that calls the MCP server and then visualizes results in Rill — is largely absent. There are no claims describing the chat UI component, the natural-language-to-SQL flow via MCP tools, or the end-to-end wiring between chat → MCP server → DuckDB → Rill dashboard.. Tried: Model Context Protocol MCP server client architecture specification, MCP protocol specification tools resources prompts, build MCP server DuckDB SQL execution example, MCP server DuckDB integration tutorial Python, Rill Developer DuckDB dashboard analytics integration, Rill dashboard DuckDB data source configuration, chat UI MCP client text-to-SQL implementation, MCP client chat interface natural language SQL.
- R2: Cited claims are supported by their quoted spans. — 21/23, scored 0.91 of the 1 required; found: It supports both local DuckDB instances and cloud-based MotherDuck databases [E10]., It supports both local DuckDB instances and cloud-based MotherDuck databases [E10].; run stopped: every sub-concept is covered in two sources; the critique found the answer incomplete: The claims provide solid coverage of MCP architecture, DuckDB/MotherDuck MCP server setup, and Rill Developer's DuckDB integration for dashboards. However, the critical intersection — how to place text-to-SQL in a chat UI that calls the MCP server and then visualizes results in Rill — is largely absent. There are no claims describing the chat UI component, the natural-language-to-SQL flow via MCP tools, or the end-to-end wiring between chat → MCP server → DuckDB → Rill dashboard.. Tried: Model Context Protocol MCP server client architecture specification, MCP protocol specification tools resources prompts, build MCP server DuckDB SQL execution example, MCP server DuckDB integration tutorial Python, Rill Developer DuckDB dashboard analytics integration, Rill dashboard DuckDB data source configuration, chat UI MCP client text-to-SQL implementation, MCP client chat interface natural language SQL.
- 3 source(s) rejected — blocked or paywalled. e.g. https://deepwiki.com/modelcontextprotocol/servers/1.1-mcp-protocol-and-architecture, https://medium.com/@msapeker/building-a-resilient-multi-agent-text-to-sql-chatbot-with-mcp-langgraph-chainlit-f5a5bf593329 (+1 more in the archived report)
- Budget: Q5: searches exhausted.
- Budget: Q6: searches exhausted.

## Citations

- [E1] https://modelcontextprotocol.io/specification/2025-11-25/architecture — retrieved 2026-10-05T16:20:27.992519+00:00 — page sha256 b16ac0c7a12c969e… — “The Model Context Protocol (MCP) follows a client-host-server architecture where each
host can run multiple client instances.”
- [E4] https://modelcontextprotocol.io/specification/2026-07-28/architecture — retrieved 2026-10-05T16:20:28.028375+00:00 — page sha256 0ee52ffe40f4399a… — “The Model Context Protocol (MCP) follows a client-host-server architecture where each
host can run multiple client instances.”
- [E12] https://modelcontextprotocol.info/specification/ — retrieved 2026-10-05T16:20:28.383782+00:00 — page sha256 daf220b267e7129e… — “The Model Context Protocol (MCP) is an open protocol that enables seamless integration between LLM applications and external data sources and tools.”
- [E23] https://dev.to/aws-heroes/mcp-prompts-and-resources-the-primitives-youre-not-using-3oo1 — retrieved 2026-10-05T16:20:55.671173+00:00 — page sha256 ceacfd27b198ccdb… — “The Model Context Protocol (MCP, spec 2025-11-25) defines three primitives for connecting AI models to external services: tools, prompts, and resources.”
- [E8] https://apidog.com/blog/duckdb-mcp-server — retrieved 2026-10-05T16:20:32.344247+00:00 — page sha256 070d7f02ab610ea4… — “The DuckDB MCP server is distributed via the mcp-server-motherduck package. Install it using uv (preferred for Python CLI tools) or pip:
Install uv (Optional):pip install uv

Install the MCP Server:uv pip install mcp-server-motherduck

Or, with pip:pip install mcp-server-motherduck”
- [E10] https://apidog.com/blog/duckdb-mcp-server — retrieved 2026-10-05T16:20:32.344247+00:00 — page sha256 070d7f02ab610ea4… — “The server acts as a bridge between DuckDB and AI-driven environments, such as Cursor or Claude, allowing these tools to execute SQL commands and return results in real time. It supports both local DuckDB instances and cloud-based MotherDuck databases, offering flexibility for diverse use cases. Additionally, the DuckDB MCP server includes security features like read-only mode and read-scaling tokens, ensuring safe integration with third-party tools.”
- [E27] https://docs.rilldata.com/developers/get-started/quickstart — retrieved 2026-10-05T16:21:53.961031+00:00 — page sha256 503239116851bd9b… — “This guide assumes you'll be using Rill's default embedded engine, DuckDB.”
- [E31] https://www.rilldata.com/blog/why-we-built-rill-with-duckdb — retrieved 2026-10-05T16:21:54.410365+00:00 — page sha256 5e91b70d28561be3… — “Rill Developer is an open-source, code-first BI tool that enables software engineers to profile data sets, design data models, and deploy dashboards to Rill Cloud.”
- [E39] http://www.rilldata.com/blog/fast-duckdb-powered-dashboards — retrieved 2026-10-05T16:21:56.455196+00:00 — page sha256 38272cad3c5df48d… — “At Rill, we’ve been longtime fans of DuckDB, in fact it’s the engine that powers auto-profiling and interactive modeling in Rill Developer, and the dashboards in Rill Cloud.”
- [E40] http://www.rilldata.com/blog/fast-duckdb-powered-dashboards — retrieved 2026-10-05T16:21:56.455196+00:00 — page sha256 38272cad3c5df48d… — “Users will follow the same basic steps that include
downloading and installing Rill 
 configuring MotherDuck as a source
entering a SQL expression to return a single table and clicking import”
- [E13] https://motherduck.com/blog/faster-data-pipelines-with-mcp-duckdb-ai — retrieved 2026-10-05T16:20:42.418537+00:00 — page sha256 d6b4eac69ee400d9… — “The DuckDB/MotherDuck MCP server allows the AI copilot (Cursor) to directly run queries against local DuckDB databases and/or remote MotherDuck databases and interpret the results.”
- [E30] https://docs.rilldata.com/developers/get-started/quickstart — retrieved 2026-10-05T16:21:53.961031+00:00 — page sha256 503239116851bd9b… — “Select the model's side menu and select "Generate dashboard with AI". Watch as Rill configures your metrics view and creates an Explore dashboard.”
- [E3] https://modelcontextprotocol.io/specification/2025-11-25/architecture — retrieved 2026-10-05T16:20:27.992519+00:00 — page sha256 b16ac0c7a12c969e… — “Each client is created by the host and maintains an isolated server connection:

Establishes one stateful session per server”
- [E6] https://modelcontextprotocol.io/specification/2026-07-28/architecture — retrieved 2026-10-05T16:20:28.028375+00:00 — page sha256 0ee52ffe40f4399a… — “Each client is created by the host and communicates with exactly one server:”
- [E7] https://modelcontextprotocol.io/specification/2026-07-28/architecture — retrieved 2026-10-05T16:20:28.028375+00:00 — page sha256 0ee52ffe40f4399a… — “Servers provide specialized context and capabilities:

Expose resources, tools and prompts via MCP primitives

Operate independently with focused responsibilities

Request client input (sampling, elicitation, roots) via InputRequiredResult within a reply

Must respect security constraints

Can be local processes or remote services”
- [E36] https://docs.rilldata.com/developers/build/connectors/olap/duckdb — retrieved 2026-10-05T16:21:53.920245+00:00 — page sha256 da574bcdf697ffda… — “By default, Rill includes DuckDB as an embedded OLAP engine that ingests data from data sources and powers your dashboards. When you start a new project, you'll see a connectors/duckdb.yaml file alongside other project files. No additional configuration is needed to use DuckDB with Rill Developer or Rill Cloud.”
- [E41] http://www.rilldata.com/blog/fast-duckdb-powered-dashboards — retrieved 2026-10-05T16:21:56.455196+00:00 — page sha256 38272cad3c5df48d… — “Because Rill is already built with DuckDB embedded, no additional type wrangling is needed, the data in MotherDuck is “production-ready” for Rill dashboards.”
- [E25] https://dev.to/aws-heroes/mcp-prompts-and-resources-the-primitives-youre-not-using-3oo1 — retrieved 2026-10-05T16:20:55.671173+00:00 — page sha256 ceacfd27b198ccdb… — “Clients discover available prompts via prompts/list -- parallel to tools/list -- and present them to users as slash commands, menu items, or quick actions.”
- [E26] https://dev.to/aws-heroes/mcp-prompts-and-resources-the-primitives-youre-not-using-3oo1 — retrieved 2026-10-05T16:20:55.671173+00:00 — page sha256 ceacfd27b198ccdb… — “At the protocol level, a prompt is simple: the client calls prompts/get with a name and arguments, and the server returns a GetPromptResult containing a sequence of PromptMessage values.”
- [E9] https://apidog.com/blog/duckdb-mcp-server — retrieved 2026-10-05T16:20:32.344247+00:00 — page sha256 070d7f02ab610ea4… — “To integrate the DuckDB MCP server with your IDE or AI tool, configure it using a JSON file. Below is an example for Cursor:
Access Cursor Settings: Navigate to Settings > Cursor Settings > MCP > Add a new Global MCP Server.
Create Configuration: Add the following JSON configuration:{
 "mcpServers": {
 "mcp-server-motherduck": {
 "command": "uvx",
 "args": [
 "mcp-server-motherduck",
 "--db-path",
 "md:",
 "--motherduck-token",
 "<YOUR_MOTHERDUCK_TOKEN_HERE>"
 ]
 }
 }
}

Replace <YOUR_MOTHERDUCK_TOKEN_HERE> with your MotherDuck token for cloud access. For local databases, use:{
 "mcpServers": {
 "mcp-server-motherduck": {
 "command": "uvx",
 "args": [
 "mcp-server-motherduck",
 "--db-path",
 "/path/to/your/local.db"
 ]
 }
 }
}

For in-memory databases, use --db-path :memory:.
Enable Read-Only Mode (Optional): To prevent data modifications, add the --read-only flag:{
 "mcpServers": {
 "mcp-server-motherduck": {
 "command": "uvx",
 "args": [
 "mcp-server-motherduck",
 "--db-path",
 "/path/to/your/local.db",
 "--read-only"
 ]
 }
 }
}”
- [E14] https://motherduck.com/blog/faster-data-pipelines-with-mcp-duckdb-ai — retrieved 2026-10-05T16:20:42.418537+00:00 — page sha256 d6b4eac69ee400d9… — “To install the DuckDB/MotherDuck MCP server in Cursor, go to Settings > Cursor Settings > MCP > Add a new Global MCP Server and add the following JSON configuration:”
- [E15] https://motherduck.com/blog/faster-data-pipelines-with-mcp-duckdb-ai — retrieved 2026-10-05T16:20:42.418537+00:00 — page sha256 d6b4eac69ee400d9… — “{
 "mcpServers": {
 "mcp-server-motherduck": {
 "command": "uvx", // Assumes uvx is installed and in PATH for running Python CLI tools
 "args": [
 "mcp-server-motherduck",
 "--db-path",
 "md:", // Connects to MotherDuck by default. Use a local file path like "my_local_db.duckdb" for local DBs.
 "--motherduck-token",
 "<YOUR_MOTHERDUCK_TOKEN_HERE>" // Required if connecting to MotherDuck
 ]
 }
 }
}”
- [E11] https://apidog.com/blog/duckdb-mcp-server — retrieved 2026-10-05T16:20:32.344247+00:00 — page sha256 070d7f02ab610ea4… — “Test the Configuration: Restart your IDE and run a test query:SELECT 1 AS test;

If successful, the server returns the result, confirming the setup.”
- [E17] https://duckdb.org/community_extensions/extensions/duckdb_mcp.html — retrieved 2026-10-05T16:20:38.215949+00:00 — page sha256 0c14689aefe69574… — “SELECT mcp_publish_table('products', 'data://tables/products', 'json');”
- [E18] https://duckdb.org/community_extensions/extensions/duckdb_mcp.html — retrieved 2026-10-05T16:20:38.215949+00:00 — page sha256 0c14689aefe69574… — “mcp_publish_execution_tool(name, desc, sql, props, required, bindings) - Publish execution tools with prepared parameter binding”
- [E28] https://docs.rilldata.com/developers/get-started/quickstart — retrieved 2026-10-05T16:21:53.961031+00:00 — page sha256 503239116851bd9b… — “Get started with just two commands:

curl https://rill.sh | sh
rill start my-rill-project”
