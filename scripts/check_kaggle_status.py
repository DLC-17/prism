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
    status_res = call_mcp_tool(token, "get_notebook_session_status", {
        "request": {
            "userName": USER_NAME,
            "kernelSlug": KERNEL_SLUG
        }
    })
    print(f"[*] Session Status: {json.dumps(status_res, indent=2)}")

    print(f"[*] Listing output files for {USER_NAME}/{KERNEL_SLUG}...")
    out_res = call_mcp_tool(token, "list_notebook_session_output", {
        "request": {
            "userName": USER_NAME,
            "kernelSlug": KERNEL_SLUG
        }
    })

    if isinstance(out_res, dict) and "result" in out_res:
        content = out_res["result"].get("content", [])
        if content:
            blob = content[0].get("text", "")
            try:
                data = json.loads(blob)
                files = data.get("files", [])
                print(f"[*] Found {len(files)} output files:")
                for f in files:
                    fname = f.get("file_name", "")
                    furl = f.get("url", "")
                    print(f"    - {fname}")
                    if fname in ["execution.log", "error.log"]:
                        print(f"[*] Downloading and displaying {fname}...")
                        try:
                            with urllib.request.urlopen(furl, timeout=30) as r:
                                text_data = r.read().decode("utf-8", errors="replace")
                                print(f"=== {fname} ===\n{text_data}\n=== END {fname} ===")
                                with open(fname, "w", encoding="utf-8") as out_f:
                                    out_f.write(text_data)
                        except Exception as e:
                            print(f"[!] Failed to fetch {fname}: {e}")
                    elif fname in ["submission.tar.gz", "prism_model.pt"]:
                        print(f"[*] Downloading {fname}...")
                        try:
                            with urllib.request.urlopen(furl, timeout=60) as r:
                                with open(fname, "wb") as out_f:
                                    out_f.write(r.read())
                            print(f"[SUCCESS] Downloaded {fname} ({os.path.getsize(fname) / (1024*1024):.2f} MB)")
                        except Exception as e:
                            print(f"[!] Failed to download {fname}: {e}")
            except Exception as e:
                print(f"[!] JSON parse error: {e}")

if __name__ == "__main__":
    main()
