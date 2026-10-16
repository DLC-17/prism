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

    # Step 1: Discover or verify existing notebook
    target_slug = "prism-ptcg-ai-training-pipeline-run-1792110379"
    existing_id = 138940832
    existing_slug = target_slug

    print(f"[*] Searching for notebook with slug '{target_slug}' or query 'prism'...")
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
        if isinstance(search_res, dict) and "result" in search_res:
            content = search_res["result"].get("content", [])
            for item in content:
                text_blob = item.get("text", "")
                try:
                    data = json.loads(text_blob)
                    kernels = data.get("kernels") or data.get("notebooks") or []
                    for k in kernels:
                        print(f"[*] Raw candidate: {json.dumps(k)}")
                        cand_id = k.get("id") or k.get("scriptId") or k.get("kernelId") or k.get("script_id")
                        cand_slug = k.get("slug") or k.get("scriptUrl") or k.get("url") or ""
                        cand_title = k.get("title") or k.get("newTitle") or ""
                        if "1792110379" in cand_title or "1792110379" in str(cand_slug) or target_slug in str(cand_slug):
                            if cand_id:
                                existing_id = cand_id
                                existing_slug = cand_slug or target_slug
                                print(f"[SUCCESS] Matched target kernel: ID={existing_id}, Slug={existing_slug}")
                                break
                except Exception as e:
                    pass
        if existing_id and existing_id != 138940832:
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
        print(f"[*] Updating existing notebook (Kernel ID={existing_id}, Slug={existing_slug})...")
        save_request["id"] = int(existing_id)
        save_request["hasId"] = True
    else:
        run_title = "Prism PTCG AI Training Pipeline (Run 1792110379)"
        print(f"[*] Creating notebook with title: '{run_title}'...")
        save_request["newTitle"] = run_title
    res = call_mcp_tool(token, "save_notebook", {"request": save_request})
    print(f"[*] Kaggle MCP Response: {json.dumps(res, indent=2)}")

    def check_response_error(r):
        if not isinstance(r, dict) or "result" not in r:

            return True, "Invalid response"
        if r["result"].get("isError"):
            return True, "".join(c.get("text", "") for c in r["result"].get("content", []))
        content = r["result"].get("content", [])
        if content:
            t = content[0].get("text", "")
            try:
                p = json.loads(t)
                if isinstance(p, dict) and "error" in p:
                    return True, str(p["error"])
            except Exception:
                pass
        return False, ""

    is_err, err_msg = check_response_error(res)
    if is_err:
        print(f"[!] Initial save attempt failed: {err_msg}")
        print(f"[*] Attempting save with newTitle '{target_slug}'...")
        title_request = dict(save_request)
        title_request.pop("id", None)
        title_request.pop("hasId", None)
        title_request["newTitle"] = "Prism PTCG AI Training Pipeline (Run 1792110379)"
        title_request["hasNewTitle"] = True
        res = call_mcp_tool(token, "save_notebook", {"request": title_request})
        print(f"[*] Title fallback response: {json.dumps(res, indent=2)}")
        is_err, err_msg = check_response_error(res)

    if is_err:
        print(f"[!] Target title failed: {err_msg}. Attempting fresh run title...")
        fresh_request = dict(save_request)
        fresh_request.pop("id", None)
        fresh_request.pop("hasId", None)
        fresh_request["newTitle"] = f"{KERNEL_TITLE} (Run {int(time.time())})"
        fresh_request["hasNewTitle"] = True
        res = call_mcp_tool(token, "save_notebook", {"request": fresh_request})
        print(f"[*] Fresh run response: {json.dumps(res, indent=2)}")
        is_err, err_msg = check_response_error(res)

    if is_err:
        print(f"[!] Kaggle MCP reported error: {err_msg}")
        return False

    content = res["result"].get("content", [])
    if content:
        msg = content[0].get("text", "")
        print(f"[*] Notebook Status: {msg}")
        try:
            parsed_info = json.loads(msg)
            if isinstance(parsed_info, dict) and "url" in parsed_info:
                existing_slug = parsed_info["url"].split("/")[-1]
                print(f"[*] Updated tracking slug to: {existing_slug}")
        except Exception:
            pass

    print("[SUCCESS] Notebook successfully submitted and training started on Kaggle!")
    print(f"[*] Tracking URL: https://www.kaggle.com/code/davidc17/{existing_slug}")

    # Monitor status for up to 150 seconds
    print("[*] Monitoring initial execution status...")
    for i in range(15):

        time.sleep(10)
        status_res = call_mcp_tool(token, "get_notebook_session_status", {
            "request": {
                "userName": "davidc17",
                "kernelSlug": existing_slug
            }
        })
        status_text = ""
        if isinstance(status_res, dict) and "result" in status_res:
            content = status_res["result"].get("content", [])
            if content:
                status_text = content[0].get("text", "")
        print(f"[*] [{i+1}/15] Status check: {status_text}")
        if "RUNNING" in status_text:
            print("[*] Training is actively running on Kaggle GPU!")
        elif "COMPLETE" in status_text:
            print("[SUCCESS] Training completed successfully!")
            out_res = call_mcp_tool(token, "list_notebook_session_output", {
                "request": {
                    "userName": "davidc17",
                    "kernelSlug": existing_slug
                }
            })
            if isinstance(out_res, dict) and "result" in out_res:
                content = out_res["result"].get("content", [])
                if content:
                    try:
                        data = json.loads(content[0].get("text", ""))
                        for f in data.get("files", []):
                            if f.get("file_name") == "execution.log":
                                print("[*] Execution Log:")
                                with urllib.request.urlopen(f.get("url"), timeout=20) as r:
                                    print(r.read().decode("utf-8", errors="replace"))
                    except Exception:
                        pass
            break
        elif "ERROR" in status_text:
            print(f"[!] Error encountered: {status_text}")
            time.sleep(5)
            out_res = call_mcp_tool(token, "list_notebook_session_output", {
                "request": {
                    "userName": "davidc17",
                    "kernelSlug": existing_slug
                }
            })
            print(f"[*] Raw out_res: {json.dumps(out_res, indent=2)}")
            if isinstance(out_res, dict) and "result" in out_res:
                content = out_res["result"].get("content", [])
                if content:
                    try:
                        data = json.loads(content[0].get("text", ""))
                        for f in data.get("files", []):
                            if f.get("file_name") in ["execution.log", "error.log"]:
                                print(f"[*] Output file '{f.get('file_name')}':")
                                with urllib.request.urlopen(f.get("url"), timeout=20) as r:
                                    print(r.read().decode("utf-8", errors="replace"))
                    except Exception as e:
                        print(f"[!] Log fetch error: {e}")
            return False

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
