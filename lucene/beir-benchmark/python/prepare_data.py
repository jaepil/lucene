#!/usr/bin/env python3
"""
Download BEIR datasets and generate dense embeddings for hybrid search benchmarking.

Outputs per dataset:
  <output_dir>/<dataset>/corpus.jsonl   - {"_id": str, "title": str, "text": str, "embedding": [float...]}
  <output_dir>/<dataset>/queries.jsonl  - {"_id": str, "text": str, "embedding": [float...]}
  <output_dir>/<dataset>/qrels.tsv      - query_id\tdoc_id\tscore
"""

import argparse
import json
import os
import sys

import numpy as np
from beir import util as beir_util
from beir.datasets.data_loader import GenericDataLoader
from sentence_transformers import SentenceTransformer


BEIR_DATASETS = [
    "scifact",
    "nfcorpus",
    "fiqa",
    "arguana",
    "trec-covid",
    "nq",
    "hotpotqa",
    "dbpedia-entity",
    "fever",
    "climate-fever",
    "scidocs",
    "quora",
]

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
BATCH_SIZE = 256


def download_dataset(dataset_name, data_dir):
    dataset_path = os.path.join(data_dir, dataset_name)
    if os.path.exists(dataset_path):
        print(f"  Dataset {dataset_name} already downloaded at {dataset_path}")
        return dataset_path

    urls = [
        f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset_name}.zip",
        f"https://huggingface.co/datasets/BeIR/{dataset_name}/resolve/main/{dataset_name}.zip",
    ]

    for url in urls:
        try:
            print(f"  Downloading {dataset_name} from {url}...")
            dataset_path = beir_util.download_and_unzip(url, data_dir)
            return dataset_path
        except Exception as e:
            print(f"  Failed: {e}")
            continue

    raise RuntimeError(f"All download URLs failed for {dataset_name}")


def detect_device():
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def encode_texts(model, texts, batch_size=BATCH_SIZE):
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    return embeddings.astype(np.float32)


def prepare_dataset(dataset_name, data_dir, output_dir, model):
    print(f"\n{'='*60}")
    print(f"Processing: {dataset_name}")
    print(f"{'='*60}")

    dataset_path = download_dataset(dataset_name, data_dir)
    corpus, queries, qrels = GenericDataLoader(data_folder=dataset_path).load(split="test")

    print(f"  Corpus: {len(corpus)} documents")
    print(f"  Queries: {len(queries)} queries")
    print(f"  Qrels: {sum(len(v) for v in qrels.values())} judgments")

    out_path = os.path.join(output_dir, dataset_name)
    os.makedirs(out_path, exist_ok=True)

    # Encode corpus
    print("  Encoding corpus...")
    corpus_ids = list(corpus.keys())
    corpus_texts = []
    for cid in corpus_ids:
        title = corpus[cid].get("title", "")
        text = corpus[cid].get("text", "")
        corpus_texts.append(f"{title} {text}".strip())

    corpus_embeddings = encode_texts(model, corpus_texts)

    corpus_file = os.path.join(out_path, "corpus.jsonl")
    print(f"  Writing {corpus_file}...")
    with open(corpus_file, "w", encoding="utf-8") as f:
        for i, cid in enumerate(corpus_ids):
            doc = {
                "_id": cid,
                "title": corpus[cid].get("title", ""),
                "text": corpus[cid].get("text", ""),
                "embedding": corpus_embeddings[i].tolist(),
            }
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")

    # Encode queries
    print("  Encoding queries...")
    query_ids = list(queries.keys())
    query_texts = [queries[qid] for qid in query_ids]
    query_embeddings = encode_texts(model, query_texts)

    queries_file = os.path.join(out_path, "queries.jsonl")
    print(f"  Writing {queries_file}...")
    with open(queries_file, "w", encoding="utf-8") as f:
        for i, qid in enumerate(query_ids):
            q = {
                "_id": qid,
                "text": queries[qid],
                "embedding": query_embeddings[i].tolist(),
            }
            f.write(json.dumps(q, ensure_ascii=False) + "\n")

    # Write qrels
    qrels_file = os.path.join(out_path, "qrels.tsv")
    print(f"  Writing {qrels_file}...")
    with open(qrels_file, "w", encoding="utf-8") as f:
        for qid, doc_scores in qrels.items():
            for doc_id, score in doc_scores.items():
                f.write(f"{qid}\t{doc_id}\t{score}\n")

    print(f"  Done: {dataset_name}")
    return len(corpus), len(queries)


def main():
    parser = argparse.ArgumentParser(description="Prepare BEIR datasets with embeddings")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help=f"Dataset names (default: all). Available: {', '.join(BEIR_DATASETS)}",
    )
    parser.add_argument(
        "--data-dir",
        default="/data/beir-raw",
        help="Directory for raw BEIR downloads",
    )
    parser.add_argument(
        "--output-dir",
        default="/data/beir-prepared",
        help="Directory for prepared output files",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Sentence transformer model (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help=f"Encoding batch size (default: {BATCH_SIZE})",
    )
    args = parser.parse_args()

    datasets = args.datasets if args.datasets else BEIR_DATASETS

    device = detect_device()
    print(f"Model: {args.model}")
    print(f"Device: {device}")
    print(f"Datasets: {', '.join(datasets)}")
    print(f"Output: {args.output_dir}")

    model = SentenceTransformer(args.model, device=device)

    summary = []
    for ds in datasets:
        if ds not in BEIR_DATASETS:
            print(f"WARNING: Unknown dataset '{ds}', skipping.")
            continue
        n_docs, n_queries = prepare_dataset(ds, args.data_dir, args.output_dir, model)
        summary.append((ds, n_docs, n_queries))

    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    print(f"{'Dataset':<20} {'Documents':>10} {'Queries':>10}")
    print(f"{'-'*40}")
    for ds, nd, nq in summary:
        print(f"{ds:<20} {nd:>10,} {nq:>10,}")
    print(f"\nAll data saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
