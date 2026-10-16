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

    # First, let's see if we can get user info or check user profile
    user_res = call_mcp_tool(token, "get_user_profile", {"request": {}})
    print(f"[*] Kaggle user profile response: {json.dumps(user_res, indent=2)}")

    username = None
    if isinstance(user_res, dict) and "result" in user_res:
        content = user_res["result"].get("content", [])
        if content and isinstance(content[0], dict):
            try:
                user_data = json.loads(content[0].get("text", "{}"))
                username = user_data.get("userName") or user_data.get("username")
            except Exception:
                pass

    print(f"[*] Resolved username: {username}")

    # Test 1: Try without slug (None), allowing Kaggle to generate slug from newTitle
    print("[*] Trying save_notebook with slug=None...")
    args = {
        "request": {
            "newTitle": KERNEL_TITLE,
            "text": nb_content,
            "language": "python",
            "kernelType": "notebook",
            "isPrivate": False,
            "enableGpu": True,
            "enableInternet": True,
            "competitionDataSources": [COMPETITION_SLUG],
            "kernelExecutionType": "SaveAndRunAll"
        }
    }
    if username:
        args["request"]["slug"] = f"{username}/{KERNEL_SLUG}"

    res = call_mcp_tool(token, "save_notebook", args)
    print(f"[*] Kaggle MCP Response: {json.dumps(res, indent=2)}")

    # If error and we haven't tried username/slug, try username/slug
    if isinstance(res, dict) and "result" in res:
        content = res["result"].get("content", [])
        if content and "error" in content[0].get("text", ""):
            err_text = content[0].get("text")
            print(f"[!] Error detected in response: {err_text}")
            if not username:
                # Try getting username from another tool like search_notebooks
                nb_search = call_mcp_tool(token, "search_notebooks", {"request": {"pageSize": 1}})
                print(f"[*] Notebook search response: {json.dumps(nb_search, indent=2)}")
            return False

    print("[SUCCESS] Notebook successfully submitted to Kaggle via MCP!")
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
