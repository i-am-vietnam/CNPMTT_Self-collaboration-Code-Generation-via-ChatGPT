"""
Ngay 3 - Phan tich loi: so sanh tung bai giua 2 file ket qua va kiem tra Tester.
Cach chay:
    python3 analyze.py --a out_E1_direct.jsonl --b out_E2_selfcollab.jsonl
"""
import argparse
import csv
import json
import multiprocessing as mp
import os
import subprocess
import sys

mp.set_start_method("fork", force=True)

from datasets import load_dataset
from core.utils import find_method_name
from evaluate.execute.execution import evaluate_with_test_code


def load(path):
    with open(path) as f:
        return {s["task_id"]: s for s in map(json.loads, f)}


def check(samples):
    prepared = []
    for s in samples.values():
        s = dict(s)
        gen = s["prompt"] + s["completion"]
        s.update(generation=gen, prompt="", completion=gen,
                 entry_point=find_method_name(gen) or "candidate")
        prepared.append(s)
    res = evaluate_with_test_code(prepared, timeout=10)
    return {r["task_id"]: bool(r["passed"]) for r in res}


def tester_verdict(work_dir, task_id):
    d = os.path.join(work_dir, task_id.replace("/", "_"))
    runner = os.path.join(d, "_test_runner.py")
    if not os.path.exists(runner) or not os.path.exists(os.path.join(d, "solution.py")):
        return False, None, 0
    n_assert = open(runner).read().count("assert ")
    try:
        r = subprocess.run([sys.executable, "_test_runner.py"], cwd=d,
                           capture_output=True, text=True, timeout=30)
        return True, r.returncode == 0, n_assert
    except subprocess.TimeoutExpired:
        return True, False, n_assert


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--a", default="out_E1_direct.jsonl")
    p.add_argument("--b", default="out_E2_selfcollab.jsonl")
    p.add_argument("--work_dir", default="humaneval_workdir")
    p.add_argument("--out", default="phan_tich.csv")
    args = p.parse_args()

    all_ids = [t["task_id"] for t in load_dataset("openai/openai_humaneval", split="test")]
    A, B = load(args.a), load(args.b)
    pa, pb = check(A), check(B)
    pa = {t: pa.get(t, False) for t in all_ids}
    pb = {t: pb.get(t, False) for t in all_ids}

    rows, missed, n_assert_total, n_with_test = [], [], 0, 0
    for t in all_ids:
        has_test, t_pass, n_assert = tester_verdict(args.work_dir, t)
        n_with_test += has_test
        n_assert_total += n_assert
        if has_test and t_pass and not pb[t]:
            missed.append(t)
        rows.append({"task_id": t, "A_dung": pa[t], "B_dung": pb[t],
                     "tester_co_test": has_test, "tester_bao_pass": t_pass,
                     "so_assert_cua_tester": n_assert, "loai_loi": ""})

    with open(args.out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    both = [t for t in all_ids if pa[t] and pb[t]]
    only_a = [t for t in all_ids if pa[t] and not pb[t]]
    only_b = [t for t in all_ids if pb[t] and not pa[t]]
    none = [t for t in all_ids if not pa[t] and not pb[t]]

    print("\n================ KET QUA PHAN TICH ================")
    print(f"A = {args.a}: dung {sum(pa.values())}/{len(all_ids)}")
    print(f"B = {args.b}: dung {sum(pb.values())}/{len(all_ids)}")
    print(f"Ca hai deu dung : {len(both)}")
    print(f"Chi A dung      : {len(only_a)}  {only_a}")
    print(f"Chi B dung      : {len(only_b)}  {only_b}")
    print(f"Ca hai deu sai  : {len(none)}  {none}")
    print("--------------- Tester cua B ---------------")
    print(f"So bai Tester co viet test          : {n_with_test}")
    print(f"Tong so lenh assert Tester da viet  : {n_assert_total}")
    print(f"Tester bao PASS nhung thuc te SAI   : {len(missed)}/{len(all_ids) - sum(pb.values())} bai sai  {missed}")
    print(f"\nDa ghi chi tiet vao {args.out}")


if __name__ == "__main__":
    main()
