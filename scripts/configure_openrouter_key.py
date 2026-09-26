#!/usr/bin/env python3
"""Run interactively on Quest. Input is hidden; no credential enters this repo."""
import getpass
import os
from pathlib import Path

path=Path.home()/'.config/crashbench/openrouter.key'
path.parent.mkdir(parents=True,exist_ok=True)
if path.exists() or path.is_symlink():raise SystemExit('Credential file already exists; leave it intact or update it yourself securely.')
key=getpass.getpass('OpenRouter API key (hidden): ').strip()
if not key.startswith('sk-or-') or any(c.isspace() for c in key):raise SystemExit('Unexpected OpenRouter key format; no file written.')
fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
with os.fdopen(fd,'w') as file:file.write(key+'\n')
del key
print('Configured private credential file:',path)
