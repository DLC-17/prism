"""Unified Kaggle Notebook Launcher and Training Trigger.
Supports:
1. Kaggle MCP Server (Bearer Token / KGAT_...)
2. Kaggle API / CLI (kaggle.json, username/key)
"""

import os
import sys
import json
import time
import urllib.request
import urllib.error
from pathlib import Path
import subprocess

MCP_URL = "https://www.kaggle.com/mcp"
COMPETITION_SLUG = "the-pokemon-company-ptcg-ai-battle-challenge-playground"
KERNEL_SLUG = "prism-ptcg-ai-training-pipeline"
KERNEL_TITLE = "Prism PTCG AI Training Pipeline"

def get_notebook_content() -> str:
    root = Path(__file__).resolve().parent.parent
    nb_path = root / "notebooks" / "kaggle_training_pipeline.ipynb"
    if not nb_path.exists():
        raise FileNotFoundError(f"Notebook not found at {nb_path}")
    with open(nb_path, "r", encoding="utf-8") as f:
        return f.read()

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
            # Parse SSE response
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

def run_via_mcp(token: str) -> bool:
    print(f"[*] Attempting to launch notebook via Kaggle MCP ({MCP_URL})...")
    nb_content = get_notebook_content()

    # Step 1: Search for existing user notebooks to find any existing Prism notebook
    existing_id = None
    existing_slug = None
    for group in ["Profile", "PublicAndUsersPrivate"]:
        search_res = call_mcp_tool(token, "search_notebooks", {
            "request": {
                "group": group,
                "hasGroup": True,
                "search": "prism",
                "hasSearch": True,
                "pageSize": 10,
                "hasPageSize": True
            }
        })
        print(f"[*] Search ({group}) result: {json.dumps(search_res, indent=2)}")
        if isinstance(search_res, dict) and "result" in search_res:
            content = search_res["result"].get("content", [])
            for item in content:
                text_blob = item.get("text", "")
                try:
                    data = json.loads(text_blob)
                    kernels = data.get("kernels") or data.get("notebooks") or []
                    for k in kernels:
                        k_title = (k.get("title") or k.get("newTitle") or "").lower()
                        k_slug = (k.get("slug") or k.get("scriptUrl") or "").lower()
                        if "prism" in k_title or "prism" in k_slug:
                            existing_id = k.get("id")
                            existing_slug = k.get("slug") or k.get("scriptUrl")
                            print(f"[*] Found existing notebook: ID={existing_id}, Slug={existing_slug}, Title={k.get('title')}")
                            break
                except Exception:
                    pass
        if existing_id:
            break

    # Step 2: Formulate save_notebook request
    save_request = {
        "text": nb_content,
        "hasText": True,
        "language": "python",
        "hasLanguage": True,
        "kernelType": "notebook",
        "hasKernelType": True,
        "isPrivate": True,
        "hasIsPrivate": True,
        "enableGpu": True,
        "hasEnableGpu": True,
        "enableInternet": True,
        "hasEnableInternet": True,
        "competitionDataSources": [COMPETITION_SLUG],
        "kernelExecutionType": "SaveAndRunAll",
        "hasKernelExecutionType": True
    }

    if existing_id:
        print(f"[*] Updating existing notebook (ID={existing_id})...")
        save_request["id"] = int(existing_id)
        save_request["hasId"] = True
    else:
        # Generate a unique title with run timestamp to guarantee freshness
        run_title = f"{KERNEL_TITLE} (Run {int(time.time())})"
        print(f"[*] Creating new notebook with title: '{run_title}'...")
        save_request["newTitle"] = run_title
        save_request["hasNewTitle"] = True

    res = call_mcp_tool(token, "save_notebook", {"request": save_request})
    print(f"[*] Kaggle MCP Response: {json.dumps(res, indent=2)}")

    if isinstance(res, dict) and "result" in res:
        if res["result"].get("isError"):
            err_text = "".join(c.get("text", "") for c in res["result"].get("content", []))
            print(f"[!] Kaggle MCP reported error: {err_text}")
            return False

        content = res["result"].get("content", [])
        if content:
            msg = content[0].get("text", "")
            print(f"[*] Notebook Status: {msg}")

    print("[SUCCESS] Notebook successfully submitted and training started on Kaggle!")
    return True

def run_via_kaggle_cli(username: str, key: str) -> bool:
    print(f"[*] Setting up Kaggle CLI credentials for user: {username}...")
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_dir.mkdir(parents=True, exist_ok=True)
    kaggle_json = kaggle_dir / "kaggle.json"
    with open(kaggle_json, "w") as f:
        json.dump({"username": username, "key": key}, f)
    kaggle_json.chmod(0o600)

    # Ensure kaggle is installed
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "kaggle"], check=False)

    work_dir = Path("/tmp/kaggle_kernel_push")
    work_dir.mkdir(parents=True, exist_ok=True)

    nb_content = get_notebook_content()
    (work_dir / "kaggle_training_pipeline.ipynb").write_text(nb_content, encoding="utf-8")

    metadata = {
        "id": f"{username}/{KERNEL_SLUG}",
        "title": KERNEL_TITLE,
        "code_file": "kaggle_training_pipeline.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": "false",
        "enable_gpu": "true",
        "enable_tpu": "false",
        "enable_internet": "true",
        "dataset_sources": [],
        "competition_sources": [COMPETITION_SLUG],
        "kernel_sources": [],
        "model_sources": []
    }
    with open(work_dir / "kernel-metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"[*] Pushing notebook to Kaggle: {metadata['id']}...")
    env = os.environ.copy()
    env["KAGGLE_USERNAME"] = username
    env["KAGGLE_KEY"] = key

    ret = subprocess.run(["kaggle", "kernels", "push", "-p", str(work_dir)], env=env, capture_output=True, text=True)
    print(ret.stdout)
    if ret.returncode != 0:
        print(f"[!] Error pushing kernel: {ret.stderr}")
        return False

    print("[SUCCESS] Kernel pushed to Kaggle!")
    print(f"[*] Tracking URL: https://www.kaggle.com/code/{username}/{KERNEL_SLUG}")
    return True

def main():
    api_secret = os.environ.get("KAGGLE_API", "").strip()
    if not api_secret:
        print("[!] Error: KAGGLE_API environment variable is empty or not set.")
        sys.exit(1)

    # 1. Check if JSON format
    parsed_json = None
    try:
        parsed_json = json.loads(api_secret)
    except Exception:
        pass

    if isinstance(parsed_json, dict) and "username" in parsed_json and "key" in parsed_json:
        print("[*] Detected Kaggle JSON credential.")
        success = run_via_kaggle_cli(parsed_json["username"], parsed_json["key"])
        if not success:
            sys.exit(1)
        return

    # 2. Check if username:key format
    if ":" in api_secret and not api_secret.startswith("http"):
        parts = api_secret.split(":", 1)
        if len(parts) == 2 and not parts[0].startswith("KGAT_"):
            print("[*] Detected username:key format.")
            success = run_via_kaggle_cli(parts[0].strip(), parts[1].strip())
            if success:
                return

    # 3. Default to Kaggle MCP Bearer Token
    print("[*] Attempting Kaggle MCP authentication with Bearer Token...")
    success = run_via_mcp(api_secret)
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main()
