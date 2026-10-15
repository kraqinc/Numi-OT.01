# Learn folder — knowledge Numi can use **without** re-training

Drop `.md` / `.txt` / `.jsonl` here. On the next chat, RAG picks matching chunks.

```bash
# one-shot fact
python scripts/learn_fact.py "Colombia UTC offset is UTC-5 year-round."

# from a file
python scripts/learn_fact.py --file notes.md

# list
ls knowledge/learn/
```

Env:
- `NUMI_RAG=1` (default) inject context
- `NUMI_KNOWLEDGE_DIRS=/path/a:/path/b` extra folders
