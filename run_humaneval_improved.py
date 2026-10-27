"""
E3 - Self-collaboration CAI TIEN (giu nguyen Analyst va Coder cua ban goc, chi thay Tester).

Diem yeu cua ban goc (phat hien o Ngay 3):
  - Tester chi print, khong assert  -> 21/22 bai sai van bao PASSED.
  - Chu thich "Expected output" cua Tester nhieu khi cung sai -> khong the tin mu quang.

Cai tien:
  1. Test "cung" (hard): lay cac vi du >>> trong de bai, so sanh dap an that.
     Day la thong tin cong khai trong de, KHONG dung bo test cham diem.
  2. Test "mem" (soft): Tester viet assert cho truong hop bien, nhung duoc danh dau
     "co the sai" -> Coder phai tu kiem tra lai theo de bai truoc khi sua.
  3. Bao loi cu the: dau vao, ket qua mong doi, ket qua thuc te.
  4. Chong thoai lui: sau moi lan Coder viet, cham lai test cung (chay tren may,
     khong ton API). Giu phien ban qua nhieu test cung nhat.
So vong toi da giong ban goc (--max_round 2) de so sanh cong bang.

Cach chay:
    bash run.sh ... (KHONG dung)  ->  dung:
    set -a; source .env; set +a
    python3 run_humaneval_improved.py --quiet --work_dir workdir_E3 --output_path out_E3_improved.jsonl
"""
import ast
import doctest
import json
import os
import re
import subprocess
import sys

import run_humaneval as base
from core.backend import call_llm_with_tools
from core.agent import ToolAgent as BaseAgent, HUMANEVAL_TOOL_SCHEMAS

# ---------------------------------------------------------------------------
# Prompt moi cho Tester
# ---------------------------------------------------------------------------
TESTER_PROMPT_V2 = base.TEAM_CONTEXT + """

You are the **Tester** on this team. Write test cases for the function `{entry_point}`.

Rules:
1. Write 3 to 6 test cases, each as ONE line of the form:
   assert {entry_point}(<arguments>) == <expected value>
2. Derive every expected value ONLY from the requirement, by reasoning step by step.
   Do NOT run the developer's code in your head to get the expected value.
3. Focus on edge cases (empty input, single element, negative numbers, ties, boundaries)
   that are VALID according to the requirement. Never test inputs the requirement forbids.
4. Output only the assert lines inside one ```python code block.
"""

FEEDBACK_V2 = """
## Test report from the Tester

{hard_part}{soft_part}
Fix the code so that it satisfies the requirement. Do NOT repeat the same mistake."""

HARD_HEADER = """### MUST FIX - examples written in the problem statement (these are certainly correct)
"""
SOFT_HEADER = """### Tester's own tests failed (the Tester may be WRONG)
For each case below, first re-derive the expected value from the requirement yourself.
Change your code ONLY if you are confident the requirement really implies that value.
"""

# ---------------------------------------------------------------------------
# Bo chay test (chay trong tien trinh rieng, co timeout tung test)
# ---------------------------------------------------------------------------
RUNNER = r'''
import ast, json, signal, sys
sys.path.insert(0, ".")
spec = json.load(open("_tests_spec.json"))
try:
    import solution as M
    G = dict(vars(M))
except Exception as e:
    print(json.dumps([{"ok": False, "got": "import error: %r" % e} for _ in spec]))
    sys.exit(0)

def _alarm(s, f):
    raise TimeoutError("timeout (>5s)")
signal.signal(signal.SIGALRM, _alarm)

def norm(s):
    return " ".join(str(s).split())

out = []
for t in spec:
    r = {"ok": False, "got": ""}
    signal.alarm(5)
    try:
        if t["kind"] == "hard":
            val = eval(t["src"], G)
            got = repr(val)
            r["got"] = got
            ok = norm(got) == norm(t["want"])
            if not ok:
                try:
                    ok = eval(t["want"], G) == val
                except Exception:
                    pass
            r["ok"] = bool(ok)
        else:
            node = ast.parse(t["src"]).body[0]
            if (isinstance(node, ast.Assert) and isinstance(node.test, ast.Compare)
                    and len(node.test.ops) == 1 and isinstance(node.test.ops[0], ast.Eq)):
                try:
                    rhs = eval(compile(ast.Expression(node.test.comparators[0]), "<t>", "eval"), G)
                except Exception:
                    r.update(ok=True, invalid=True)   # dap an cua Tester loi -> bo qua test nay
                    out.append(r)
                    continue
                lhs = eval(compile(ast.Expression(node.test.left), "<t>", "eval"), G)
                r["got"] = repr(lhs)
                r["want"] = repr(rhs)
                r["ok"] = bool(lhs == rhs)
            else:
                exec(t["src"], G)
                r["ok"] = True
    except Exception as e:
        r["got"] = "%s: %s" % (type(e).__name__, e)
    finally:
        signal.alarm(0)
    out.append(r)
print(json.dumps(out))
'''


def doc_examples(prompt, entry_point=None):
    """Lay cac vi du >>> trong docstring cua de bai -> test cung."""
    text = prompt
    try:  # chi lay phan docstring (bo dau nhay """ o cuoi)
        docs = [ast.get_docstring(n) for n in ast.walk(ast.parse(prompt))
                if isinstance(n, ast.FunctionDef) and (entry_point is None or n.name == entry_point)]
        if docs and docs[-1]:
            text = docs[-1]
    except Exception:
        pass
    try:
        exs = doctest.DocTestParser().get_examples(text)
    except Exception:
        return []
    tests = []
    for e in exs:
        src, want = e.source.strip(), e.want.strip()
        if src and want and "\n" not in src:
            tests.append({"kind": "hard", "src": src, "want": want})
    if not tests and entry_point:
        # nhieu de viet vi du dang: f(3) == 5  /  f(3) ➞ 5  /  f(3) => 5
        pat = re.compile(r"(%s\(.*\))\s*(?:==|➞|=>|->)\s*(.+?)\s*$" % re.escape(entry_point))
        for line in text.splitlines():
            m = pat.search(line)
            if not m:
                continue
            src, want = m.group(1).strip(), m.group(2).strip().rstrip(".,;")
            try:
                ast.parse(src, mode="eval"); ast.parse(want, mode="eval")
            except SyntaxError:
                continue
            tests.append({"kind": "hard", "src": src, "want": want})
    return tests


def tester_asserts(config, requirement, entry_point):
    """Goi LLM Tester, lay toi da 6 dong assert -> test mem."""
    messages = [
        {"role": "system", "content": TESTER_PROMPT_V2.format(entry_point=entry_point)},
        {"role": "user", "content": f"## Requirement\n{requirement}"},
    ]
    try:
        resp = call_llm_with_tools(messages, config, tools=None)
        text = resp.choices[0].message.content or ""
    except Exception:
        return []
    lines = [l.strip() for l in text.splitlines() if l.strip().startswith("assert ")]
    return [{"kind": "soft", "src": l} for l in lines[:6]]


def run_tests(task_dir, tests):
    """Chay danh sach test tren solution.py hien tai. Tra ve list ket qua."""
    if not tests:
        return []
    with open(os.path.join(task_dir, "_tests_spec.json"), "w") as f:
        json.dump(tests, f)
    with open(os.path.join(task_dir, "_check_runner.py"), "w") as f:
        f.write(RUNNER)
    try:
        r = subprocess.run([sys.executable, "_check_runner.py"], cwd=task_dir,
                           capture_output=True, text=True, timeout=60)
        res = json.loads(r.stdout.strip().splitlines()[-1])
    except Exception as e:
        res = [{"ok": False, "got": f"runner error: {e}"} for _ in tests]
    for t, x in zip(tests, res):
        x.update(kind=t["kind"], src=t["src"], want=x.get("want", t.get("want", "")))
    return res


def read_solution(task_dir):
    p = os.path.join(task_dir, "solution.py")
    return open(p).read() if os.path.exists(p) else ""


def make_report(results):
    hard = [r for r in results if r["kind"] == "hard" and not r["ok"]]
    soft = [r for r in results if r["kind"] == "soft" and not r["ok"]]
    if not hard and not soft:
        return None
    hp = sp = ""
    if hard:
        hp = HARD_HEADER + "".join(
            f"- `{r['src']}` -> expected `{r['want']}`, got `{r['got']}`\n" for r in hard) + "\n"
    if soft:
        sp = SOFT_HEADER + "".join(
            f"- `{r['src']}` -> your code returned `{r['got']}`\n" for r in soft) + "\n"
    return FEEDBACK_V2.format(hard_part=hp, soft_part=sp)


# ---------------------------------------------------------------------------
# Quy trinh moi cho 1 bai
# ---------------------------------------------------------------------------
def run_task_improved(task, config, work_dir, max_round=2, max_steps=10, verbose=True):
    task_id, prompt, entry_point = task["task_id"], task["prompt"], task["entry_point"]
    task_dir = os.path.join(work_dir, task_id.replace("/", "_"))
    os.makedirs(task_dir, exist_ok=True)
    requirement = f"```python\n{prompt}\n```\n\nThe function name must be `{entry_point}`."
    log = {"task_id": task_id, "rounds": []}

    # Phase 1: ANALYST (giu nguyen ban goc)
    resp = call_llm_with_tools(
        [{"role": "system", "content": base.ANALYST_PROMPT},
         {"role": "user", "content": f"Analyze this requirement and create an implementation plan:\n\n{requirement}"}],
        config, tools=None)
    analyst_plan = resp.choices[0].message.content or ""

    hard_tests = doc_examples(prompt, entry_point)
    soft_tests = None          # Tester chi viet 1 lan, dung lai cho cac vong sau
    test_report = None
    best_code, best_score = "", -1

    for round_idx in range(max_round):
        # Phase 2: CODER (giu nguyen ban goc)
        if test_report:
            ctx = " You receive a test report from the Tester"
            coder_task = (f"## Plan from the Analyst\n{analyst_plan}\n\n## Requirement\n{requirement}\n\n"
                          f"{test_report}\n\nFix the implementation based on the test feedback.")
        else:
            ctx = " You receive a plan from the Analyst"
            coder_task = (f"## Plan from the Analyst\n{analyst_plan}\n\n## Requirement\n{requirement}\n\n"
                          f"Implement the function based on the Analyst's plan.")
        BaseAgent(config=config,
                  system_prompt=base.CODER_PROMPT.format(context_source=ctx, entry_point=entry_point),
                  repo_path=task_dir, tool_schemas=HUMANEVAL_TOOL_SCHEMAS,
                  max_steps=max_steps, verbose=verbose).run(coder_task)

        code = read_solution(task_dir)
        # Chong thoai lui: cham test cung (mien phi) va giu ban tot nhat
        hard_res = run_tests(task_dir, hard_tests)
        score = sum(r["ok"] for r in hard_res) if code else -1
        rlog = {"round": round_idx + 1, "hard_total": len(hard_tests), "hard_pass": max(score, 0)}
        if code and score >= best_score:
            best_code, best_score = code, score
            rlog["kept"] = True
        else:
            rlog["kept"] = False   # ban moi te hon -> giu ban cu

        if round_idx == max_round - 1:
            log["rounds"].append(rlog)
            break

        # Phase 3: TESTER moi
        if not code:
            test_report = "## Test report\nNo solution.py file was written. Use write_file to create it."
            log["rounds"].append(rlog)
            continue
        if soft_tests is None:
            # --no_soft: bo Tester LLM, chi dung test cung (ablation)
            soft_tests = [] if NO_SOFT else tester_asserts(config, requirement, entry_point)
        results = hard_res + run_tests(task_dir, soft_tests)
        rlog["soft_total"] = len(soft_tests)
        rlog["soft_pass"] = sum(r["ok"] for r in results if r["kind"] == "soft")
        log["rounds"].append(rlog)
        test_report = make_report(results)
        if verbose:
            print(f"  Tester: hard {rlog['hard_pass']}/{rlog['hard_total']}, "
                  f"soft {rlog['soft_pass']}/{rlog['soft_total']}")
        if test_report is None:
            break              # tat ca test deu dat

    log["final_hard_pass"] = max(best_score, 0)
    with open(os.path.join(task_dir, "improved_log.json"), "w") as f:
        json.dump(log, f, indent=1)
    return best_code


NO_SOFT = False

if __name__ == "__main__":
    # Tham so rieng cua ban cai tien, lay ra truoc khi chuyen phan con lai cho base.main()
    if "--no_soft" in sys.argv:
        NO_SOFT = True
        sys.argv.remove("--no_soft")
        print("Config: --no_soft (chi test cung, khong co test mem cua Tester)")
    base.run_task = run_task_improved   # thay quy trinh, giu nguyen phan doc du lieu/ghi ket qua
    base.main()
