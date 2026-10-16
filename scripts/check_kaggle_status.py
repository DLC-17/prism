"""Check status of the Kaggle notebook training session via Kaggle MCP.
"""

import os
import sys
import json
import urllib.request
import urllib.error

MCP_URL = "https://www.kaggle.com/mcp"
USER_NAME = "davidc17"
KERNEL_SLUG = "prism-ptcg-ai-training-pipeline-run-1792110379"

def call_mcp_tool(bearer_token: str, tool_name: str, arguments: dict):
    req_body = json.dumps({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {
            "name": tool_name,
            "arguments": arguments
        }
    }).encode("utf-8")

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {bearer_token}",
        "Accept": "text/event-stream, application/json"
    }

    req = urllib.request.Request(MCP_URL, data=req_body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            raw = resp.read().decode("utf-8")
            for line in raw.split("\n"):
                if line.startswith("data:"):
                    data_str = line[len("data:"):].strip()
                    if data_str:
                        return json.loads(data_str)
            return {"raw": raw}
    except urllib.error.HTTPError as e:
        err_msg = e.read().decode("utf-8", errors="replace")
        return {"error": f"HTTP {e.code}: {e.reason}", "details": err_msg}
    except Exception as e:
        return {"error": str(e)}

def main():
    token = os.environ.get("KAGGLE_API", "").strip()
    if not token:
        print("[!] KAGGLE_API is not set.")
        sys.exit(1)

    print(f"[*] Checking status for kernel {USER_NAME}/{KERNEL_SLUG}...")
    res = call_mcp_tool(token, "get_notebook_session_status", {
        "request": {
            "userName": USER_NAME,
            "kernelSlug": KERNEL_SLUG
        }
    })
    print(f"[*] Response: {json.dumps(res, indent=2)}")

if __name__ == "__main__":
    main()
