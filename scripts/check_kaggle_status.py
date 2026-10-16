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
                    if fname in ["__notebook__.ipynb", "__results__.html"]:
                        print(f"[*] Fetching {fname} from {furl[:60]}...")
                        try:
                            with urllib.request.urlopen(furl, timeout=30) as r:
                                content_bytes = r.read()
                                if fname == "__notebook__.ipynb":
                                    nb = json.loads(content_bytes.decode("utf-8"))
                                    for idx, c in enumerate(nb.get("cells", [])):
                                        for out in c.get("outputs", []):
                                            if out.get("output_type") == "error":
                                                ename = out.get("ename")
                                                evalue = out.get("evalue")
                                                print(f"[!] ERROR in Cell {idx}: {ename}: {evalue}")
                                                for tb in out.get("traceback", []):
                                                    print("    ", tb)
                                            elif "text" in out:
                                                print(f"[*] Cell {idx} stdout: {''.join(out['text']).strip()}")
                        except Exception as e:
                            print(f"[!] Failed to fetch {fname}: {e}")
            except Exception as e:
                print(f"[!] JSON parse error: {e}")

if __name__ == "__main__":
    main()
