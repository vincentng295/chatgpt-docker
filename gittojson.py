import os
import json
import zipfile
import requests
from requests.auth import HTTPBasicAuth
import io
import chardet
import re

MAX_FILE_SIZE = 100_000 # 100 KB per file
MAX_JSON_SIZE = 600_000 # 600 KB total JSON

def is_binary(content_bytes):
    if b'\0' in content_bytes:
        return True
    result = chardet.detect(content_bytes[:512])
    return result["encoding"] is None

def get_default_branch(owner, repo, token=None, auth=None):
    url = f"https://api.github.com/repos/{owner}/{repo}"
    headers = {"Authorization": f"token {token}"} if token else {}
    r = requests.get(url, headers=headers, auth=auth)
    r.raise_for_status()
    return r.json().get("default_branch", "main")

def make_download_url(owner, repo, branch, path):
    return f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{path}"

def parse_username_token(github_url):  
    """
    Parse GitHub URL to extract username and token if present.
    Example: https://username:token@github.com/owner/repo/...
    """
    m = re.match(r"https://([^:/]+):([^@]+)@github\.com/.*", github_url)
    if m:
        username, token = m.groups()
        return username, token
    return None, None

def parse_repo_branch(github_url):
    """
    Parse GitHub URL to extract owner, repo, and branch (if specified).
    Supports URLs in the following formats:
    - https://github.com/owner/repo
    - https://github.com/owner/repo/tree/branch
    - https://github.com/owner/repo.git@branch
    """
    url = github_url.rstrip("/")

    # Case 1: .../tree/<branch>
    m = re.match(r".*github\.com/([^/]+)/([^/]+)/tree/([^/]+)", url)
    if m:
        owner, repo, branch = m.groups()
        return owner, repo, branch

    # Case 2: ...git@branch
    m = re.match(r".*github\.com/([^/]+)/([^/]+)\.git@(.+)", url)
    if m:
        owner, repo, branch = m.groups()
        return owner, repo, branch

    # Case 3: .../owner/repo or .../owner/repo.git
    parts = url.split("/")
    owner, repo = parts[-2], parts[-1].removesuffix(".git")
    return owner, repo, None


def repo_to_json(github_url, output_json=None, token=None):
    auth = None
    # Parse token from URL if not provided
    if token is None:
        # username, token (or password)
        username, password = parse_username_token(github_url)
        auth = HTTPBasicAuth(username, password)
    # Parse owner/repo/branch from URL
    owner, repo, branch = parse_repo_branch(github_url)
    if branch is None:
        branch = get_default_branch(owner, repo, token, auth)

    # download and process ZIP
    zip_url = f"https://api.github.com/repos/{owner}/{repo}/zipball/{branch}"
    headers = {"Authorization": f"token {token}"} if token else {}
    r = requests.get(zip_url, headers=headers, auth=auth)
    r.raise_for_status()

    json_size_estimate = 0 # estimate total JSON size
    repo_data = [] # list of file entries
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        for fname in z.namelist():
            if fname.endswith("/"):  # folder
                continue
            try:
                content_bytes = z.read(fname)
                size = len(content_bytes)

                # GitHub ZIP will have a top-level folder named repo-branch/
                # We want to strip that part
                rel_path = "/".join(fname.split("/")[1:])

                raw_url = make_download_url(owner, repo, branch, rel_path)

                if size > MAX_FILE_SIZE:
                    entry = {
                        "path": rel_path,
                        "type": "skipped",
                        "content": f"[file skipped, size={size} bytes]",
                        "download_url": raw_url
                    }
                elif is_binary(content_bytes):
                    entry = {
                        "path": rel_path,
                        "type": "binary",
                        "content": f"[binary file, size={size} bytes]",
                        "download_url": raw_url
                    }
                else:
                    content = content_bytes.decode("utf-8", errors="replace")
                    entry = {
                        "path": rel_path,
                        "type": "text",
                        "content": content,
                        "download_url": raw_url
                    }

                # Check total JSON size
                entry_json = json.dumps(entry, ensure_ascii=False)
                if json_size_estimate + len(entry_json.encode("utf-8")) > MAX_JSON_SIZE:
                    repo_data.append({
                        "path": rel_path,
                        "type": "skipped",
                        "content": "[skipped due to JSON size limit]",
                        "download_url": raw_url
                    })
                    break  # Stop processing more files
                else:
                    repo_data.append(entry)
                    json_size_estimate += len(entry_json.encode("utf-8"))

            except Exception as e:
                repo_data.append({
                    "path": fname,
                    "type": "error",
                    "content": str(e)
                })

    if output_json:
        file_io = open(output_json, "wb+")
    else: # if output_json is None then return BytesIO
        file_io = io.BytesIO()
    json_str = json.dumps(repo_data, ensure_ascii=False, indent=2)
    file_io.write(json_str.encode("utf-8"))
    file_io.seek(0)
    return file_io